"""Data responsibilities: transforms, datasets, samplers, download."""

import json
from collections import defaultdict

import kagglehub
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, Sampler
from wildlife_datasets.datasets import WildlifeDataset

# Transformations
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train=False, img_size=512):
    transforms = []

    # Resize is always needed first
    transforms.append(T.Resize([img_size, img_size]))

    if train:
        # Augmentations for training
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
        ])

    # Final normalization
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

    return T.Compose(transforms)


def denormalize(tensor, mean=MEAN, std=STD):
    """Denormalize a tensor image with mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def download_czechlynx():
    path = kagglehub.dataset_download('picekl/czechlynx', force_download=False)
    return path


class MaskableDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_mask = kwargs.pop('enable_mask', False)

    def apply_segmentation(self, img: Image, idx: int) -> Image:
        if self.enable_mask:
            mask_data = json.loads(self.df.loc[idx, 'mask'])
            mask = mask_utils.decode(mask_data).astype(np.uint8)
            mask = np.expand_dims(np.asfortranarray(mask), axis=-1)

            # Ensure mask is a 2D array, as PIL expects shape (H, W) or (H, W, 3)
            mask_2d = mask.squeeze()
            mask_image = Image.fromarray(mask_2d * 255).convert("L")

            # Create background with noise sampled from normal distribution
            # Sample from N(MEAN, STD) for each pixel, preserving statistical properties
            h, w = img.size[1], img.size[0]  # PIL Image.size is (width, height)
            background = np.zeros((h, w, 3), dtype=np.float32)

            for i, (mean_val, std_val) in enumerate(zip(MEAN, STD)):
                # Sample from normal distribution: N(mean, std^2)
                background[:, :, i] = np.random.normal(mean_val, std_val, size=(h, w))

            # Clamp to [0, 1] range (normalized ImageNet range)
            background = np.clip(background, 0.0, 1.0)

            # Convert to [0, 255] range and uint8
            background_rgb = (background * 255).astype(np.uint8)
            background_image = Image.fromarray(background_rgb, mode='RGB')

            # Combine mask with image
            img = Image.composite(img, background_image, mask_image)

        # Apply transform after mask composition
        if self.transform:
            img = self.transform(img)

        return img


class PKSampler(Sampler):
    """
    Sampler for P classes, K instances each.

    Args:
        dataset: Your CzechLynx dataset (must have .labels or similar)
        p: Number of classes per batch
        k: Number of instances per class
        shuffle: Shuffle classes each epoch
    """
    def __init__(self, dataset: Dataset, p: int, k: int, shuffle: bool = True):
        self.p = p
        self.k = k
        self.shuffle = shuffle
        self.batch_size = p * k

        # Build index map: {label: [indices]}
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.label_to_indices[label].append(idx)

        # Filter classes with at least K instances
        self.valid_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 1]
        
        # Calculate number of samples based on dataset size, not just classes
        self.n_samples = len(dataset)
        # Ensure we have enough batches
        self.n_batches = self.n_samples // self.batch_size

    def __iter__(self):
        # For each batch, pick P classes at random
        for _ in range(self.n_batches):
            batch_indices = []
            
            # Select P classes (replace=False if enough classes, else True)
            replace_classes = len(self.valid_labels) < self.p
            selected_classes = np.random.choice(self.valid_labels, self.p, replace=replace_classes)
            
            for lbl in selected_classes:
                indices = self.label_to_indices[lbl]
                # Sample K instances from this class
                replace_instances = len(indices) < self.k
                selected_indices = np.random.choice(indices, self.k, replace=replace_instances)
                batch_indices.extend(selected_indices)
                
            yield from batch_indices

    def __len__(self):
        return self.n_batches * self.batch_size
