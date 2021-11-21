from torch.utils import model_zoo
import albumentations as albu
import torch
from skimage import io
from resize import *
from create_model import *
from utils import *
import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm


def human_binarization(image_file, model, temp_size):
    # 이미 사전에 서버에서 create_model()을 사용하여 모델을 선언하고,
    # model.eval()로 모델을 evaluation mode로 전환시키고
    # model parameter로 제공한 경우
    if model:
        net = model
    else:
        net = create_model("Unet_human")
        net.eval()

    image = io.imread(image_file)
    resized_image = temp_resize(temp_size, image)
    tmp = (resized_image*255).astype(np.uint8)
    orig_img = Image.fromarray(tmp)

    image_tensor = image_to_tensor(resized_image)

    d1, _, _, _, _, _, _ = net(image_tensor)
    prediction = d1[:, 0, :, :]
    max_p = torch.max(prediction)
    min_p = torch.min(prediction)
    normalized_pred = (prediction - min_p) / (max_p - prediction)
    pred = normalized_pred.squeeze()
    pred_np = pred.cpu().data.numpy()

    masked_image = Image.fromarray(pred_np*255).convert('RGB')
    del d1

    return orig_img, masked_image


def cloth_binarization(image_path, temp_size):
    net = create_model("Unet_cloth")
    net.eval()

    image_list = os.listdir(image_path)
    results = []

    for image in tqdm(image_list, desc="Clothes Binary Masking ..."):
        img = load_rgb(f'{image_path}/{image}')
        temp_image = temp_resize(temp_size, img, lib="cv2")
        # temp_image = img

        orig_resized_img = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
        orig_img = Image.fromarray(orig_resized_img)

        padded_image, pads = pad(
            temp_image, factor=32, border=cv2.BORDER_CONSTANT
        )
        transform = albu.Compose([albu.Normalize(p=1)], p=1)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

        with torch.no_grad():
            prediction = net(x)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255
        masked_image = Image.fromarray(mask)

        results.append((image, orig_img, masked_image))

    return results
