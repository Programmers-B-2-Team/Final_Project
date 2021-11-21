import os
from binarization import *
from resize import resizing
from tqdm import tqdm


def resizing_human(image_file, model=None, temp_size=512):
    temp_img, temp_mask = human_binarization(image_file, model, temp_size)
    resized_img, _ = resizing(temp_img, temp_mask)
    return resized_img


def resizing_cloth(image_path, save_path, temp_size=512):
    os.makedirs(f'{save_path}/img', exist_ok=True)
    os.makedirs(f'{save_path}/mask', exist_ok=True)
    results = cloth_binarization(image_path, temp_size)

    for name, img, msk in tqdm(results, desc="Clothes Resizing ..."):
        img.save(f'{save_path}/{name.split(".")[0]}.jpg')
        resized_img, resized_msk = resizing(img, msk)
        resized_img.save(f'{save_path}/img/{name.split(".")[0]}.jpg')
        resized_msk.save(f'{save_path}/mask/{name.split(".")[0]}.jpg')


