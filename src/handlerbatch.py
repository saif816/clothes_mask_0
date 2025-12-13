import runpod
import os
import io
import sys
import math
import time
import threading
import aiohttp
import asyncio
from huggingface_hub import snapshot_download
import nest_asyncio
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import boto3
from diffusers.image_processor import VaeImageProcessor
from model.cloth_masker import AutoMasker
from model.pipelinebatch import BatchPipeline
from utils import (
    init_weight_dtype, resize_and_crop, resize_and_padding,
    prepare_image, prepare_mask_image, format_error, load_image_from_url
)

# ------------------------------
# Environment Variables and AWS Configuration
# ------------------------------

# Load environment variables (these will be set in RunPod)
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", 'eu-north-1')
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "YOUR_S3_BUCKET_NAME")

# Initialize AWS S3 client
# Initialize in thread
thread_local = threading.local()
nest_asyncio.apply()
loop = asyncio.get_event_loop()

# ------------------------------
# Model Initialization
# ------------------------------

# Paths to model checkpoints
modeldir = Path(__file__).parent
# modeldir = Path(__file__).parent.parent / "Models"  # Local debug
base_model_path = f"booksforcharlie/stable-diffusion-inpainting"
repo_path = snapshot_download("zhengchong/CatVTON")
vae_model_path = f"stabilityai/sd-vae-ft-mse"

# Image resolution
width = 768
height = 1024

# Mixed precision settings
allow_tf32 = False
mixed_precision = "fp16"
# set it according to precision
MAX_BATCH_SIZE = 8

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline
pipeline = BatchPipeline(
    base_ckpt=base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    vae_ckpt=vae_model_path,
    weight_dtype=init_weight_dtype(mixed_precision),
    use_tf32=allow_tf32,
    device=device,
    engine_path=os.environ.get("UNET_TRT_PATH"),
)

# Initialize the AutoMasker
mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True
)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)
print("successfully load...")

# ------------------------------
# Helper Functions
# ------------------------------
async def download_image_pair(session, user_image_url: str, garment_image_url: str):
    try:
        if "https://" in user_image_url:
            person_image = await load_image_from_url(session, user_image_url)
        else:
            person_image = Image.open(user_image_url).convert("RGB")
        if "https://" in garment_image_url:
            garment_image = await load_image_from_url(session, garment_image_url)
        else:
            garment_image = Image.open(garment_image_url).convert("RGB")
        return person_image, garment_image
    except Exception as e:
        print(f"Download failed for {user_image_url}, {garment_image_url}: {e}")
        return None


async def download_images(person_urls, cloth_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image_pair(session, pl, cl) for pl, cl in zip(person_urls, cloth_urls)
        ]
        batch = await asyncio.gather(*tasks)    
    return batch


def get_s3_client():
    if not hasattr(thread_local, "s3"):
        thread_local.s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    return thread_local.s3


def save_to_s3(image):
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_data.seek(0)

    s3 = get_s3_client()
    date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"results/{date_str}.png"
    print('start upload single...')
    s3.upload_fileobj(
        image_data,
        S3_BUCKET_NAME,
        filename,
        ExtraArgs={'ContentType': 'image/png'}
    )
    print('finish upload single...')
    result_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    return result_url


async def upload_images(images):
    urls = await asyncio.gather(*(asyncio.to_thread(save_to_s3, image) for image in images))
    return urls


def _submit_function(
    person_image_urls,
    cloth_image_urls,
    cloth_types,
    num_inference_steps,
    guidance_scale,
    seed
) -> List[str]:
    # Download and process the person cloth images
    try:
        downloads = loop.run_until_complete(download_images(person_image_urls, cloth_image_urls))
        person_images, cloth_images = zip(*downloads)
        person_images = [resize_and_crop(img, (width, height)) for img in person_images]
        cloth_images = [resize_and_padding(img, (width, height)) for img in cloth_images]
    except Exception as e:
        raise RuntimeError(f"Failed to process person cloth images: {str(e)}")

    # Generate the mask automatically
    try:
        # TODO in parallel or batch
        mask_images = [
            mask_processor.blur(automasker(pi, ct)['mask'], blur_factor=9)
            for pi, ct in zip(person_images, cloth_types)
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to generate mask: {str(e)}")

    person_tensor = prepare_image(person_images).to(pipeline.device, dtype=pipeline.weight_dtype)
    cloth_tensor = prepare_image(cloth_images).to(pipeline.device, dtype=pipeline.weight_dtype)
    mask_tensor = prepare_mask_image(mask_images).to(pipeline.device, dtype=pipeline.weight_dtype)
    # Set up the generator for reproducibility
    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)
    # Perform inference
    try:
        result_images = pipeline(
            images=person_tensor,
            condition_images=cloth_tensor,
            masks=mask_tensor,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    except Exception as e:
        e_type, e_value, e_traceback = sys.exc_info()
        error_text = format_error(e_type, e_value, e_traceback)
        raise RuntimeError(f"Failed during inference: {error_text}")

    # Save to S3 and get URL
    try:
        if os.environ.get("RUNTIME_MODE", "dev") == "dev":
            for i, image in enumerate(result_images):
                image.save(f"{os.environ.get('PATH_TO_LOCAL_IMAGE_SAVE', '.')}/{i}.jpg")
        result_urls = loop.run_until_complete(upload_images(result_images))
    except Exception as e:
        raise RuntimeError(f"Failed to upload image: {str(e)}")

    return result_urls


def packbatch(iterable):
    for i in range(0, len(iterable), MAX_BATCH_SIZE):
        batch = iterable[i:i + MAX_BATCH_SIZE]

        metaid = []
        profile_image_urls = []
        garment_urls = []
        cloth_types = []

        for item in batch:
            try:
                metaid.append({
                    "user_id": item["user_id"],
                    "fitcheck_id": item["fitcheck_id"],
                })
                profile_image_urls.append(item["profile_image_url"])
                garment_urls.append(item["garment_url"])
                cloth_types.append(item["cloth_type"])
            except KeyError as e:
                raise ValueError(f"Missing key {e} in item: {item}") from e

        yield metaid, profile_image_urls, garment_urls, cloth_types


def unpackbatch(metainfo: Dict, urls: List[str]):
    for info, url in zip(metainfo, urls):
        info.update({"vton_image_url": url})

    return metainfo


def submit_function(
    useitems: List[Dict],
    num_inference_steps,
    guidance_scale,
    seed
):
    # in case of over batch
    results = []
    for metainfo, profile_image_urls, garment_urls, cloth_types in packbatch(useitems):
        # clear cuda
        torch.cuda.empty_cache()
        try:
            vton_urls = _submit_function(
                profile_image_urls,
                garment_urls,
                cloth_types,
                num_inference_steps,
                guidance_scale,
                seed
            )
            # unpack
            results.extend(unpackbatch(metainfo, vton_urls))
        except Exception as e:
             results.extend(unpackbatch(metainfo, [str(e)] * len(metainfo)))
             return {"data": results, "status_code": 500}
    return {"data": results, "status_code": 200}

# ------------------------------
# Handler Function
# ------------------------------

def handler(job) -> Dict[str, Dict]:
    """
    Example:
        {
            "data": [
                {
                    "user_id": "a1f3c2e4",
                    "fitcheck_id": "d49fd42c",
                    "vton_image_url": "https://"
                },
                {
                    "user_id": "b6e78c2d",
                    "fitcheck_id": "92f2938f",
                    "vton_image_url": "https://"
                }
            ],
            "status_code": 200
        }
    """
    event: Dict = job["input"]
    useitems: List = event["data"]
    num_inference_steps = event.get('num_inference_steps', 10)
    guidance_scale = event.get('guidance_scale', 2.5)
    seed = event.get('seed', 42)

    result = submit_function(
        useitems,
        num_inference_steps,
        guidance_scale,
        seed
    )

    return result

print("start service ...")
runpod.serverless.start({"handler": handler})