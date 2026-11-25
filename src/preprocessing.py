"""Online image augmentation and deterministic evaluation preprocessing.

Compute normalization statistics on the training split first, then pass them to
Preprocessor for both training and evaluation. Pixel values are RGB in [0, 1]
prior to normalization. Randomness uses Python and PyTorch worker-local RNGs.
"""

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


@dataclass
class AugmentationConfig:
    size: tuple[int, int] = (224, 224)  # height, width
    max_ops: int = 3
    weights: dict[str, float] = field(default_factory=lambda: {
        'flip': 0.5, 'affine': 0.4, 'color_jitter': 0.5,
        'gamma': 0.2, 'blur': 0.1, 'noise': 0.1, 'crop': 0.2, 'erase': 0.15,
    })
    enabled: dict[str, bool] = field(default_factory=lambda: {
        'flip': True, 'affine': True, 'color_jitter': True,
        'gamma': True, 'blur': True, 'noise': True, 'crop': True, 'erase': True,
    })
    rotation: float = 5.0
    translation: float = 0.03
    scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    gamma: tuple[float, float] = (0.9, 1.1)
    blur_radius: tuple[float, float] = (0.1, 1.0)
    noise_std: tuple[float, float] = (0.01, 0.02)
    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = (0.9, 1.1)
    erase_count: tuple[int, int] = (1, 3)
    erase_area: tuple[float, float] = (0.01, 0.03)
    erase_max_total: float = 0.06


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker); seed its generator too."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def training_statistics(train_dir: str | Path, size: tuple[int, int] = (224, 224)):
    """Return pixel-weighted RGB mean/std of resized, unaugmented training images."""
    paths = sorted(p for p in Path(train_dir).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError(f'No images found in {train_dir}')
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    for path in paths:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize((size[1], size[0]), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float64) / 255.0
        sums += pixels.sum(axis=(0, 1))
        squares += (pixels ** 2).sum(axis=(0, 1))
        count += pixels.shape[0] * pixels.shape[1]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    return tuple(mean.tolist()), tuple(std.tolist())


class Preprocessor:
    """Callable for datasets. Set training=False for validation and test."""

    def __init__(self, mean, std, config=None, training=False):
        self.config = config or AugmentationConfig()
        self.training = training
        if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError('mean and positive std must contain three RGB values')
        if self.config.max_ops < 0 or any(v < 0 for v in self.config.weights.values()):
            raise ValueError('max_ops and weights must be non-negative')
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.last_applied = []  # worker-local; use transform_with_details for auditing

    def select(self):
        cfg = self.config
        available = [name for name in ('flip', 'affine', 'crop', 'color_jitter', 'gamma', 'blur', 'noise', 'erase')
                     if cfg.enabled.get(name, False) and cfg.weights.get(name, 0) > 0]
        n = random.randint(0, cfg.max_ops)
        selected = []
        conflicts = ({'color_jitter', 'gamma'}, {'affine', 'crop'}, {'crop', 'erase'}, {'blur', 'noise', 'erase'})
        while available and len(selected) < n:
            name = random.choices(available, weights=[cfg.weights[x] for x in available], k=1)[0]
            selected.append(name)
            available.remove(name)
            for group in conflicts:
                if name in group:
                    available = [x for x in available if x not in group]
        order = ('flip', 'affine', 'crop', 'color_jitter', 'gamma', 'blur', 'noise', 'erase')
        return [name for name in order if name in selected]

    def transform_with_details(self, image):
        cfg = self.config
        image = ImageOps.exif_transpose(image).convert('RGB').resize((cfg.size[1], cfg.size[0]), Image.Resampling.BILINEAR)
        details = []
        selected = self.select() if self.training else []
        for name in selected:
            if name == 'flip':
                image = ImageOps.mirror(image)
                params = {}
            elif name == 'affine':
                angle = random.uniform(-cfg.rotation, cfg.rotation)
                scale = random.uniform(*cfg.scale)
                dx = random.uniform(-cfg.translation, cfg.translation) * image.width
                dy = random.uniform(-cfg.translation, cfg.translation) * image.height
                theta = math.radians(angle)
                a, b = math.cos(theta) / scale, math.sin(theta) / scale
                cx, cy = image.width / 2, image.height / 2
                matrix = (a, b, cx - a * (cx + dx) - b * (cy + dy), -b, a, cy + b * (cx + dx) - a * (cy + dy))
                image = image.transform(image.size, Image.Transform.AFFINE, matrix, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
                params = {'angle': angle, 'scale': scale, 'dx': dx, 'dy': dy}
            elif name == 'crop':
                width, height = image.size
                box = (0, 0, width, height)
                for _ in range(10):
                    area = width * height * random.uniform(*cfg.crop_scale)
                    ratio = math.exp(random.uniform(math.log(cfg.crop_ratio[0]), math.log(cfg.crop_ratio[1])))
                    crop_width = round(math.sqrt(area * ratio))
                    crop_height = round(math.sqrt(area / ratio))
                    if 0 < crop_width <= width and 0 < crop_height <= height:
                        left = random.randint(0, width - crop_width)
                        top = random.randint(0, height - crop_height)
                        box = (left, top, left + crop_width, top + crop_height)
                        break
                image = image.crop(box).resize((width, height), Image.Resampling.BILINEAR)
                params = {'box': box}
            elif name == 'color_jitter':
                changes = [('brightness', cfg.brightness, ImageEnhance.Brightness), ('contrast', cfg.contrast, ImageEnhance.Contrast), ('saturation', cfg.saturation, ImageEnhance.Color)]
                random.shuffle(changes)
                params = {}
                for key, amount, enhancer in changes:
                    factor = random.uniform(1 - amount, 1 + amount)
                    image = enhancer(image).enhance(factor)
                    params[key] = factor
                params['order'] = [key for key, _, _ in changes]
            elif name == 'gamma':
                gamma = random.uniform(*cfg.gamma)
                image = image.point([round(255 * ((i / 255) ** gamma)) for i in range(256)] * 3)
                params = {'gamma': gamma}
            elif name == 'blur':
                radius = random.uniform(*cfg.blur_radius)
                image = image.filter(ImageFilter.GaussianBlur(radius))
                params = {'radius': radius}
            elif name == 'noise':
                # Noise is applied after tensor conversion below.
                params = {'std': random.uniform(*cfg.noise_std)}
            elif name == 'erase':
                width, height = image.size
                image_area = width * height
                remaining = int(image_area * cfg.erase_max_total)
                boxes = []
                draw = ImageDraw.Draw(image)
                for _ in range(random.randint(*cfg.erase_count)):
                    minimum = math.ceil(image_area * cfg.erase_area[0])
                    maximum = min(int(image_area * cfg.erase_area[1]), remaining)
                    if minimum > maximum:
                        break
                    for _ in range(10):
                        area = random.randint(minimum, maximum)
                        ratio = random.uniform(0.5, 2.0)
                        rect_width = max(1, int(math.sqrt(area * ratio)))
                        rect_height = max(1, int(area / rect_width))
                        actual_area = rect_width * rect_height
                        if rect_width <= width and rect_height <= height and minimum <= actual_area <= maximum:
                            left = random.randint(0, width - rect_width)
                            top = random.randint(0, height - rect_height)
                            box = (left, top, left + rect_width, top + rect_height)
                            draw.rectangle((left, top, box[2] - 1, box[3] - 1), fill=(128, 128, 128))
                            boxes.append(box)
                            remaining -= actual_area
                            break
                params = {'boxes': boxes}
            details.append((name, params))
        tensor = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.0
        for name, params in details:
            if name == 'noise':
                tensor = (tensor + torch.randn_like(tensor) * params['std']).clamp(0, 1)
        return (tensor - self.mean) / self.std, details

    def __call__(self, image):
        tensor, self.last_applied = self.transform_with_details(image)
        return tensor
