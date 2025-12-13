'''Make mask data to quantize model
'''
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')

import os
from pathlib import Path
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from model.cloth_masker import AutoMasker


repo_path = "/media/data/VTonModels/CatVTON/"
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)
mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True
)

def _make_single(personpath):
    person_image = Image.open(personpath)
    mask = automasker(
        person_image,
        mask_type="upper"
    )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)
    return mask


def main(ps:Path, max=256):
    cnt = 0
    for imgname in ps.iterdir():
        imgpath = ps / imgname
        mask = _make_single(imgpath)
        tp = imgpath.parent.parent / "mask" / (imgpath.stem + "_fitcheckmask" + imgpath.suffix)
        mask.save(tp)
        print(tp)
        cnt += 1
        if cnt > max:
            break    


if __name__ == "__main__":
    imagedir = Path("/media/data/datasets/VITON-HD/test/image")
    main(imagedir)