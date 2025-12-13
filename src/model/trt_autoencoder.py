import os
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from model.trt_model_runner import InferenceEngine

class AutoencoderKL:
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        latent_channels: int = 4,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = None

        # pass init params to Decoder
        self.decoder = None
        self.device = "cuda"
        self.dtype = dtype
        self.quant_conv = nn.Conv2d(
            2 * latent_channels, 2 * latent_channels, 1
        ).to(device=self.device, dtype=self.dtype) if use_quant_conv else None
        if use_quant_conv:
            self.quant_conv.load_state_dict(torch.load(os.environ.get("QUANT_CONV_PATH")))

        self.post_quant_conv = nn.Conv2d(
            latent_channels, latent_channels, 1
        ).to(device=self.device, dtype=self.dtype) if use_post_quant_conv else None
        if use_post_quant_conv:
            self.post_quant_conv.load_state_dict(torch.load(os.environ.get("POST_QUANT_CONV_PATH")))

        self.use_slicing = False
        self.use_tiling = False
        self.scaling_factor = 0.18215
        self.sample_size = 256
        self.block_out_channels = [128, 256, 512, 512]
        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.sample_size
        sample_size = (
            self.sample_size[0]
            if isinstance(self.sample_size, (list, tuple))
            else self.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def init_encoder(self):
        self.encoder = InferenceEngine(os.environ.get("ENCODER_TRT_PATH"))

    def delete_encoder(self):
        if self.encoder is None:
            return
        del self.encoder
        self.encoder = None

    def init_decoder(self):
        self.decoder = InferenceEngine(os.environ.get("DECODER_TRT_PATH"))

    def delete_decoder(self):
        if self.decoder is None:
            return
        del self.decoder
        self.decoder = None

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.encoder is None:
            self.init_encoder()
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [
                self.encoder.mini_batch_infer(
                    [x_slice.detach().cpu().numpy()]
                ).to(device=self.device, dtype=self.dtype)
                for x_slice in x.split(1)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder.mini_batch_infer(
                [x.detach().cpu().numpy()]
            ).to(
                device=self.device, dtype=self.dtype
            )

        print(h.dtype)
        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        if self.decoder is None:
            self.init_decoder()
        z = z.detach().cpu().numpy()
        dec = self.decoder.mini_batch_infer([z]).to(
            device=self.device, dtype=self.dtype
        )

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput:
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)