"""
Data loading and preprocessing module for Global Dog Breed Identification.
Handles 356 dog breed classes using reproducible stratified splits (data/processed/dataset_splits.json).

Applies data augmentation ONLY to training data (RandomResizedCrop, RandomHorizontalFlip, Rotation, ColorJitter).
Validation and Test sets receive clean 224x224 resize + ImageNet normalization without augmentation.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

NUM_CLASSES = 356
DOG_BREED_JSON = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'dog_breeds_356.json')
SPLITS_JSON = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'dataset_splits.json')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFER_IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

def load_dog_breed_classes():
    if os.path.exists(DOG_BREED_JSON):
        with open(DOG_BREED_JSON, 'r', encoding='utf-8') as f:
            classes = json.load(f)
            if len(classes) == NUM_CLASSES:
                return classes
    return [f"Dog Breed {i+1}" for i in range(NUM_CLASSES)]

DOG_BREEDS_CLASSES = load_dog_breed_classes()

# Data Augmentation transforms for Training ONLY
def get_train_transforms(img_size=TRANSFER_IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Clean evaluation transforms for Validation and Test (No augmentation)
def get_eval_transforms(img_size=TRANSFER_IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Preprocessing for inference on arbitrary uploaded images
def get_inference_transforms(img_size=TRANSFER_IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DogBreedSplitDataset(Dataset):
    """Dataset class that loads images according to data/processed/dataset_splits.json."""
    def __init__(self, samples_list, transform=None):
        self.samples = samples_list
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['path']
        label = item['label']
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label


def create_dataloaders(splits_file=SPLITS_JSON, img_size=TRANSFER_IMG_SIZE, batch_size=BATCH_SIZE):
    """Creates train, val, and test DataLoaders from data/processed/dataset_splits.json."""
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Missing split file: {splits_file}. Run Phase 2 splitting first.")

    with open(splits_file, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)

    train_tf = get_train_transforms(img_size)
    eval_tf = get_eval_transforms(img_size)

    train_dataset = DogBreedSplitDataset(splits_data['train_samples'], transform=train_tf)
    val_dataset = DogBreedSplitDataset(splits_data['val_samples'], transform=eval_tf)
    test_dataset = DogBreedSplitDataset(splits_data['test_samples'], transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"✓ DataLoaders successfully created from dataset_splits.json:")
    print(f"  • Train DataLoader:      {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  • Validation DataLoader: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  • Test DataLoader:       {len(test_dataset)} samples ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader
