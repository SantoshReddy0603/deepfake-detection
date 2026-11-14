"""Training randomness and a fixed, named 16-condition validation contract."""
import hashlib
import io
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

CONDITIONS = ("clean", "jpeg_30", "jpeg_50", "jpeg_70", "blur_1", "blur_2", "noise_5", "noise_10",
              "downscale_112", "downscale_160", "brightness_0.8", "brightness_1.2",
              "contrast_0.8", "contrast_1.2", "center_crop_0.8", "combined")
CACHE_RECIPE_VERSION = 1


def jpeg(image, quality):
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=int(quality))
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")


def noise(image, sigma, rng):
    values = np.asarray(image).astype(np.float32)
    values += rng.normal(0, sigma, values.shape)
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def degrade(image, condition, image_id, seed=42):
    if condition not in CONDITIONS:
        raise ValueError("Unknown validation condition")
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{image_id}:{condition}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(derived)
    if condition == "clean":
        return image.copy()
    kind, _, value = condition.partition("_")
    if kind == "jpeg":
        return jpeg(image, int(value))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(float(value)))
    if kind == "noise":
        return noise(image, float(value), rng)
    if kind == "downscale":
        return image.resize((int(value), int(value)), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    if kind in {"brightness", "contrast"}:
        factory = ImageEnhance.Brightness if kind == "brightness" else ImageEnhance.Contrast
        return factory(image).enhance(float(value))
    if condition == "center_crop_0.8":
        w, h = image.size
        cw, ch = round(w * .8), round(h * .8)
        return image.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize(image.size, Image.Resampling.BILINEAR)
    image = image.resize((112, 112), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    image = image.filter(ImageFilter.GaussianBlur(1))
    image = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(.8)).enhance(.8)
    return jpeg(noise(image, 5, rng), 30)


def pixelate(image, block):
    w, h = image.size
    small = image.resize((max(1, w // block), max(1, h // block)), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


class TrainAugment:
    def __init__(self, degradation=False, colour="strong", mosaic=0.0, mosaic_max=16):
        if colour not in {"strong", "none", "grayscale"}:
            raise ValueError("Invalid colour recipe")
        if not 0 <= mosaic <= 1:
            raise ValueError("Mosaic probability must be in [0, 1]")
        self.degradation, self.colour, self.mosaic = degradation, colour, mosaic
        self.mosaic_max = int(mosaic_max)
        self.geometry = T.Compose([T.RandomResizedCrop(224, scale=(.85, 1.0), ratio=(.9, 1.1)),
                                   T.RandomHorizontalFlip(), T.RandomRotation(5, fill=127)])
        self.jitter = T.ColorJitter(brightness=.45, contrast=.45, saturation=.45, hue=.08)

    def __call__(self, image):
        image = self.geometry(image)
        if self.degradation:
            if self.colour != "none" and random.random() < .8:
                image = self.jitter(image)
            if self.colour == "grayscale" and random.random() < .3:
                image = image.convert("L").convert("RGB")
            if random.random() < .5:
                image = jpeg(image, random.randint(30, 95))
            if random.random() < .3:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, 1.5)))
            if random.random() < .3:
                side = random.randint(112, 224)
                method = random.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.NEAREST])
                image = image.resize((side, side), method).resize((224, 224), method)
            if random.random() < .3:
                image = noise(image, random.uniform(1, 10), np.random)
            if random.random() < .1:
                values = np.asarray(image, dtype=np.float32) / 255
                rate = random.uniform(80, 200)
                image = Image.fromarray(np.clip(np.random.poisson(values * rate) / rate * 255, 0, 255).astype(np.uint8))
            if random.random() < .15:
                array = np.array(image, copy=True)
                width, height = random.randint(12, 40), random.randint(12, 40)
                x, y = random.randint(0, 224 - width), random.randint(0, 224 - height)
                array[y:y + height, x:x + width] = 127
                image = Image.fromarray(array)
        if self.mosaic and random.random() < self.mosaic:
            image = pixelate(image, random.randint(4, self.mosaic_max))
        return np.array(image, dtype=np.uint8, copy=True)
