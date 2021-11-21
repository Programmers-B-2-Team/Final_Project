from PIL import Image, ImageChops
from skimage import transform
import cv2


def trim(mask_im):
    background = Image.new(mask_im.mode, mask_im.size, mask_im.getpixel((0, 0)))
    diff = ImageChops.difference(mask_im, background)
    diff = ImageChops.add(diff, diff, 12, -17)
    bbox = diff.getbbox()
    if bbox:
        ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        if ratio > 0.75:
            x1 = bbox[0] - 10
            x2 = bbox[2] + 10
            w = x2 - x1
            c = (bbox[3] - bbox[1]) / 2 + bbox[1]
            h = 4 / 3 * w
            y1 = c - (h / 2) if c - (h / 2) >= 0 else 0
            y2 = c + (h / 2) if c + (h / 2) <= mask_im.size[1] else mask_im.size[1]
            bbox = (x1, y1, x2, y2)

        elif ratio < 0.75:
            y1 = bbox[1] - 10
            y2 = bbox[3] + 10
            h = y2 - y1
            c = (bbox[2] - bbox[0]) / 2 + bbox[0]
            w = 3 / 4 * h
            x1 = c - (w / 2) if c - (w / 2) >= 0 else 0
            x2 = c + (w / 2) if c + (w / 2) <= mask_im.size[0] else mask_im.size[0]
            bbox = (x1, y1, x2, y2)

        return bbox

    else:
        print('Failure!')
        return None


def trim_hub(mask_im):
    background = Image.new(mask_im.mode, mask_im.size, mask_im.getpixel((0, 0)))
    diff = ImageChops.difference(mask_im, background)
    diff = ImageChops.add(diff, diff, 12, -17)
    bbox = diff.getbbox()
    if bbox:
        ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        if ratio > 0.75:
            x1 = bbox[0] - 20
            x2 = bbox[2] + 20
            w = x2 - x1
            c = (bbox[3] - bbox[1]) / 2 + bbox[1]
            h = 4 / 3 * w
            y1 = c - (h / 2)
            y2 = c + (h / 2)
            bbox = (x1, y1, x2, y2)

        elif ratio < 0.75:
            y1 = bbox[1] - 20
            y2 = bbox[3] + 20
            h = y2 - y1
            c = (bbox[2] - bbox[0]) / 2 + bbox[0]
            w = 3 / 4 * h
            x1 = c - (w / 2)
            x2 = c + (w / 2)
            bbox = (x1, y1, x2, y2)

        return bbox

    else:
        print('Failure!')
        return None


def crop_and_resize(images, bbox, target_size):
    crop_images = [image.crop(bbox) for image in images]
    return (img.resize(target_size, resample=Image.NEAREST) for img in crop_images)


def temp_resize(output_size, image, lib="ski"):
    if lib == "ski":
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = output_size * h // w, output_size
        else:
            new_h, new_w = output_size, output_size * w // h

        resized_image = transform.resize(image, (new_h, new_w), mode='constant')
    else:
        w, h = image.shape[:2]
        if h > w:
            new_h, new_w = output_size * h // w, output_size
        else:
            new_h, new_w = output_size, output_size * w // h

        resized_image = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_AREA)
    return resized_image


def resizing(image, masked_image):
    # target_size 설정
    target_size = (192, 256)

    # bbox = trim(masked_image)
    bbox = trim_hub(masked_image)  # AI Hub 이미지 처리에 좀 더 최적화
    resized_img, resized_mask = crop_and_resize(
        [image, masked_image], bbox, target_size
    )

    return resized_img, resized_mask