import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from utils import resize_and_crop

base_model_path = f"booksforcharlie/stable-diffusion-inpainting"
repo_path = snapshot_download("zhengchong/CatVTON")
vae_model_path = f"stabilityai/sd-vae-ft-mse"

width = 768
height = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device=device
)

path_to_image = "/mnt/c/users/razaa/downloads/WhatsApp Image 2025-08-13 at 14.13.34_a5114ef0.jpg"

img = Image.open(path_to_image)
img = resize_and_crop(img, (width, height))
out = automasker(img, "overall")

out["densepose"].show()
out["mask"].show()