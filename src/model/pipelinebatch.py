from typing import Union, List, Tuple
import PIL.Image
import numpy as np
import torch
import tqdm
import time
from diffusers.utils.torch_utils import randn_tensor
from model.trt_model_runner import InferenceEngine

from model.pipeline import CatVTONPipeline
from utils import (
    compute_vae_encodings, numpy_to_pil,
    resize_and_crop, resize_and_padding
)


class BatchPipeline(CatVTONPipeline):
    def __init__(
        self,
        base_ckpt,
        attn_ckpt,
        attn_ckpt_version="mix",
        vae_ckpt="stabilityai/sd-vae-ft-mse",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=True,
        use_tf32=True,
        use_trt=True,
        engine_path=None,
    ):
        load_torch_unet = not use_trt
        super().__init__(
            base_ckpt,
            attn_ckpt,
            attn_ckpt_version=attn_ckpt_version,
            vae_ckpt=vae_ckpt,
            weight_dtype=weight_dtype,
            device=device,
            compile=compile,
            skip_safety_check=skip_safety_check,
            use_tf32=use_tf32,
            load_unet=load_torch_unet
        )
        self.use_trt = use_trt
        if use_trt:
            self.engine = InferenceEngine(engine_path)

    def check_inputs(
        self, images: Union[torch.Tensor, List[PIL.Image.Image]],
        condition_images: Union[torch.Tensor, List[PIL.Image.Image]],
        masks: Union[torch.Tensor, List[PIL.Image.Image]],
        width: int,
        height: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(images, torch.Tensor) and isinstance(condition_images, torch.Tensor) \
                and isinstance(masks, torch.Tensor):
            return images, condition_images, masks
        # Type is PIL.Image
        assert all(image.size == mask.size for image, mask in zip(images, masks)), "Image and mask must have the same size"
        images = [resize_and_crop(image, (width, height)) for image in images]
        masks = [resize_and_crop(mask, (width, height)) for mask in masks]
        condition_images = [resize_and_padding(condition_image, (width, height)) for condition_image in condition_images]
        images = torch.Tensor(images)
        condition_images = torch.Tensor(condition_images)
        masks = torch.Tensor(masks)
        return images, condition_images, masks

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        condition_images: torch.Tensor,
        masks: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        self_guidance_scale: float = 0,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        images, condition_images, masks = self.check_inputs(images, condition_images, masks, width, height)
        images = images.to(self.device, dtype=self.weight_dtype)
        condition_images = condition_images.to(self.device, dtype=self.weight_dtype)
        masks = masks.to(self.device, dtype=self.weight_dtype)
        concat_dim = -2  # FIXME: y axis concat
        # Mask image
        use_self_guidance = self_guidance_scale > 0

        masked_images = images * (masks < 0.5)
        # VAE encoding
        s = time.time()
        masked_latents = compute_vae_encodings(masked_images, self.vae)
        condition_latents = compute_vae_encodings(condition_images, self.vae)
        self.vae.delete_encoder()
        etime = (time.time() - s) * 1000
        print(f"Etime torch {etime}")
        mask_latents = torch.nn.functional.interpolate(masks, size=masked_latents.shape[-2:], mode="nearest")
        del images, masks, condition_images

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latents, condition_latents], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latents, torch.zeros_like(mask_latents)], dim=concat_dim)
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat_guidance = torch.cat(
                [
                    torch.cat([masked_latents, torch.zeros_like(condition_latents)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat_guidance = torch.cat([mask_latent_concat] * 2)
        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        prev_noise_pred = None
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                # prepare the input for the inpainting model      (2, 4, ., .)                  (2, 1, ., .)       (2, 4, ., .)
                if do_classifier_free_guidance:
                    inpainting_latent_model_input = torch.cat(
                        [
                            non_inpainting_latent_model_input,
                            mask_latent_concat_guidance,
                            masked_latent_concat_guidance], dim=1
                    )
                else:
                    inpainting_latent_model_input = torch.cat(
                        [non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1
                    )

                # predict the noise residual
                if self.use_trt:
                    # Predict with tensorrt if it's enabled
                    t = t.detach().cpu().numpy()
                    batch_size = inpainting_latent_model_input.size()[0]
                    ts = np.repeat(t, batch_size)
                    noise_pred = self.engine.infer(
                        [inpainting_latent_model_input.detach().cpu().numpy(), ts],
                        True, self.device,
                    )[0]
                else:
                    # Fallback to the pytorch model if trt is not enabled
                    noise_pred = self.unet(
                        inpainting_latent_model_input,
                        t.to(self.device),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                orig_noise_pred = noise_pred.detach()
                if prev_noise_pred is not None and i > num_warmup_steps and use_self_guidance:
                    sg_term = noise_pred - prev_noise_pred
                    noise_pred = noise_pred + self_guidance_scale * sg_term
                if use_self_guidance:
                    prev_noise_pred = orig_noise_pred
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        s = time.time()
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.scaling_factor * latents
        images = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        self.vae.delete_decoder()
        etime = (time.time() - s) * 1000
        print(f"Etime torch decode {etime}")
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images