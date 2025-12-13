import os
from datetime import datetime

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# ------------------------------
# User Inputs (Modify as needed)
# ------------------------------

# Paths to model checkpoints
base_model_path = "booksforcharlie/stable-diffusion-inpainting"  # Base model path
resume_path = "zhengchong/CatVTON"  # Checkpoint of trained try-on model

# Output directory
output_dir = "resource/demo/output"

# Image resolution
width = 768
height = 1024

# Mixed precision settings
allow_tf32 = True
mixed_precision = "no"  # Choose from "no", "fp16", "bf16"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inference parameters
person_image_path = "../model.jpeg"  # Path to the person image
cloth_image_path = "../20241111_143132188_iOS.jpg"    # Path to the cloth image
cloth_type = "overall"  # Options: "upper", "lower", "overall"
num_inference_steps = 100
guidance_scale = 2.5
seed = 42  # Set to -1 for random seed
# show_type = "input & mask & result"  # Options: "result only", "input & result", "input & mask & result"

# Optional mask path (set to None if not using a custom mask)
mask_path = None  # e.g., "path_to_mask_image.png"

# ------------------------------
# End of User Inputs
# ------------------------------

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Download the model repository
repo_path = snapshot_download(repo_id=resume_path)

# Initialize the pipeline
pipeline = CatVTONPipeline(
    base_ckpt=base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(mixed_precision),
    use_tf32=allow_tf32,
    device=device
)

# Initialize the AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)

def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    # show_type
):
    person_image_input = person_image
    mask = None

    # Load and process the person image
    person_image = Image.open(person_image_input).convert("RGB")
    person_image = resize_and_crop(person_image, (width, height))

    # Load and process the cloth image
    cloth_image = Image.open(cloth_image).convert("RGB")
    cloth_image = resize_and_padding(cloth_image, (width, height))

    # Load and process the mask if provided
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        mask = resize_and_crop(mask, (width, height))
    else:
        # Generate the mask automatically
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Set up the generator for reproducibility
    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Perform inference
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]

    # Save the result
    tmp_folder = output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    # Post-process and compose the result image
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)

    # Prepare the final result based on show_type
    # if show_type == "result only":
    return result_image


if __name__ == "__main__":
    # Run the inference
    result = submit_function(
        person_image_path,
        cloth_image_path,
        cloth_type,
        num_inference_steps,
        guidance_scale,
        seed,
        # show_type
    )

    # Display or save the result
    result.show()
    result_save_path = os.path.join(output_dir, "result.png")
    result.save(result_save_path)
    print(f"Result saved to {result_save_path}")

