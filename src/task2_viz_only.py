"""
Task 2: Load best model and generate segmentation sample visualization only.
Run this after training completes — no re-training needed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

import albumentations as A
from albumentations.pytorch import ToTensorV2

from task2_segmentation import (
    SegmentationDataset,
    plot_segmentation_samples,
    SEED,
)

DATA_DIR = "./data"
OUTPUT_DIR = "./output/task2"
IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_CLASSES = 2
MODEL_NAME = "DeepLabV3Plus"
ENCODER_NAME = "resnet50"


def val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Data: val split only --
    trainval_base = OxfordIIITPet(root=DATA_DIR, split="trainval",
                                   target_types="segmentation", download=True)
    n = len(trainval_base)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    split = int(n * 0.8)
    val_idx = indices[split:]
    val_ds = SegmentationDataset(trainval_base, val_idx, val_transform(), NUM_CLASSES)
    print(f"Val samples: {len(val_ds)}")

    # -- Model --
    model = smp.create_model(arch=MODEL_NAME, encoder_name=ENCODER_NAME,
                              encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    model = model.to(device)
    model.eval()

    # -- Load checkpoint --
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch", "?")
    print(f"Loaded best model: epoch {best_epoch} from {ckpt_path}")

    # -- Generate sample visualization --
    plot_segmentation_samples(model, val_ds, OUTPUT_DIR, device, NUM_CLASSES)


if __name__ == "__main__":
    main()
