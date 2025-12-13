from flask import Flask, request, jsonify
import requests
import os
import io
from datetime import datetime
import torch
from PIL import Image
import boto3
from dotenv import load_dotenv  # Added import
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# ------------------------------
# Model Initialization
# ------------------------------

# Paths to model checkpoints
base_model_path = "booksforcharlie/stable-diffusion-inpainting"
resume_path = "zhengchong/CatVTON"

# Image resolution
width = 768
height = 1024

# Mixed precision settings
allow_tf32 = True
mixed_precision = "no"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# ------------------------------
# AWS S3 Configuration
# ------------------------------

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

s3_bucket_name = os.getenv('S3_BUCKET_NAME')

# ------------------------------
# Helper Functions
# ------------------------------

def save_to_s3(image_data):
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"results/{date_str}.png"
    s3.upload_fileobj(
        image_data,
        s3_bucket_name,
        filename,
        ExtraArgs={'ContentType': 'image/png'}
    )
    result_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{filename}"
    return result_url

def submit_function(
    person_image_url,
    cloth_image_url,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed
):
    # Download and process the person image
    try:
        response = requests.get(person_image_url)
        response.raise_for_status()
        person_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        person_image = resize_and_crop(person_image, (width, height))
    except Exception as e:
        return {'error': f"Failed to process person image: {str(e)}"}, 400

    # Download and process the cloth image
    try:
        response = requests.get(cloth_image_url)
        response.raise_for_status()
        cloth_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        cloth_image = resize_and_padding(cloth_image, (width, height))
    except Exception as e:
        return {'error': f"Failed to process cloth image: {str(e)}"}, 400

    # Generate the mask automatically
    try:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
        mask = mask_processor.blur(mask, blur_factor=9)
    except Exception as e:
        return {'error': f"Failed to generate mask: {str(e)}"}, 500

    # Set up the generator for reproducibility
    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Perform inference
    try:
        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )[0]
    except Exception as e:
        return {'error': f"Failed during inference: {str(e)}"}, 500

    # Save the result image to in-memory file
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Save to S3 and get URL
    try:
        result_url = save_to_s3(img_byte_arr)
    except Exception as e:
        return {'error': f"Failed to upload image: {str(e)}"}, 500

    return {'result_url': result_url}, 200

# ------------------------------
# API Routes
# ------------------------------

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    person_image_url = data.get('person_image_url')
    cloth_image_url = data.get('cloth_image_url')
    cloth_type = data.get('cloth_type')
    num_inference_steps = data.get('num_inference_steps', 50)
    guidance_scale = data.get('guidance_scale', 2.5)
    seed = data.get('seed', 42)

    if not person_image_url or not cloth_image_url or not cloth_type:
        return jsonify({'error': 'Missing required parameters'}), 400

    result, status_code = submit_function(
        person_image_url,
        cloth_image_url,
        cloth_type,
        num_inference_steps,
        guidance_scale,
        seed
    )

    return jsonify(result), status_code

# ------------------------------
# Main Function
# ------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
