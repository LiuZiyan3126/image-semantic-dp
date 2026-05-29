"""
Task 2: Oxford-IIIT Pet — Image Segmentation (Evaluation Only)
Loads a trained model and evaluates on the test set.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from torchvision.datasets import OxfordIIITPet

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = "./data"
OUTPUT_DIR = "./output/task2"
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = 384
NUM_CLASSES = 2  # binary segmentation (read from checkpoint if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_pil, mask_pil = self.base[self.indices[idx]]
        image = np.array(image_pil)
        mask = np.array(mask_pil)
        # Binary remapping: 1=fg, 2=bg, 3=ignore
        mask = np.where(mask == 1, 1,
                np.where(mask == 2, 0, 255)).astype(np.int64)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        return image, mask


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    intersect_fg = torch.tensor(0.0, device=device)
    pred_fg_sum = torch.tensor(0.0, device=device)
    target_fg_sum = torch.tensor(0.0, device=device)
    correct_pixels = torch.tensor(0.0, device=device)
    total_valid_pixels = torch.tensor(0.0, device=device)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        valid = masks != 255

        for c in range(num_classes):
            pred_c = (preds == c) & valid
            target_c = (masks == c) & valid
            intersection[c] += (pred_c & target_c).sum()
            union[c] += (pred_c | target_c).sum()

        pred_fg = (preds == 1)
        target_fg = (masks == 1)
        intersect_fg += (pred_fg & target_fg & valid).sum()
        pred_fg_sum += (pred_fg & valid).sum()
        target_fg_sum += (target_fg & valid).sum()

        correct_pixels += (preds[valid] == masks[valid]).sum()
        total_valid_pixels += valid.sum()

    ious = intersection / (union + 1e-6)
    miou = ious.mean().item()
    bg_iou = ious[0].item()
    fg_iou = ious[1].item()
    dice = (2.0 * intersect_fg / (pred_fg_sum + target_fg_sum + 1e-6)).item()
    pixel_acc = (correct_pixels / (total_valid_pixels + 1e-6)).item()

    return {
        "miou": miou, "bg_iou": bg_iou, "fg_iou": fg_iou,
        "dice": dice, "pixel_acc": pixel_acc,
        "ious": ious.cpu().numpy().tolist(),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_segmentation_samples(model, dataset, output_dir, device):
    model.eval()
    val_transform = get_val_transform()

    fig, axes = plt.subplots(8, 3, figsize=(12, 32))
    rng = np.random.RandomState(SEED)
    sample_indices = rng.choice(len(dataset), min(8, len(dataset)), replace=False)

    for row, ds_idx in enumerate(sample_indices):
        idx = dataset.indices[ds_idx]
        image_pil, mask_pil = dataset.base[idx]
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        transformed = val_transform(image=image_np, mask=mask_np)
        inp = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp).argmax(1).squeeze(0).cpu().numpy()

        # Un-normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_disp = transformed["image"].cpu().numpy().transpose(1, 2, 0)
        image_disp = image_disp * std + mean
        image_disp = np.clip(image_disp, 0, 1)

        # GT display
        gt_disp = np.zeros((*pred.shape, 3), dtype=np.float32)
        gt_disp[mask_np == 1] = [0, 1, 0]
        gt_disp[mask_np == 2] = [0, 0, 0]
        gt_disp[mask_np == 3] = [0.5, 0.5, 0.5]

        # Pred display
        pred_disp = np.zeros((*pred.shape, 3), dtype=np.float32)
        pred_disp[pred == 1] = [0, 1, 0]
        pred_disp[pred == 0] = [0, 0, 0]

        axes[row, 0].imshow(image_disp)
        axes[row, 0].set_title("Image" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_disp)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_disp)
        axes[row, 2].set_title("Prediction" if row == 0 else "")
        axes[row, 2].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "segmentation_samples.png"), dpi=120)
    plt.close()
    print(f"[Plot] Saved segmentation_samples.png to {output_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load test set (official test split, 3669 images)
    test_base = OxfordIIITPet(root=DATA_DIR, split="test",
                              target_types="segmentation", download=True)
    test_indices = list(range(len(test_base)))
    test_ds = SegmentationDataset(test_base, test_indices, get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    # Load checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Run task2_segmentation.py first to train a model.")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    args = checkpoint.get("args", {})
    model_name = args.get("model_name", "DeepLabV3Plus")
    encoder_name = args.get("encoder_name", "resnet50")
    num_classes = args.get("num_classes", NUM_CLASSES)

    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"Model: {model_name} / {encoder_name}")
    print(f"Saved val mIoU: {checkpoint.get('val_miou', 'N/A')}")

    # Build model and load weights
    model = smp.create_model(
        arch=model_name,
        encoder_name=encoder_name,
        encoder_weights=None,  # weights loaded from checkpoint
        in_channels=3,
        classes=num_classes,
    )
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    metrics = evaluate(model, test_loader, num_classes, DEVICE)

    print(f"\nTest Set Results:")
    print(f"  mIoU          : {metrics['miou']:.4f}")
    print(f"  Background IoU: {metrics['bg_iou']:.4f}")
    print(f"  Foreground IoU: {metrics['fg_iou']:.4f}")
    print(f"  Dice (FG)     : {metrics['dice']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_acc']:.4f}")

    # Save results
    results = {
        "model": model_name,
        "encoder": encoder_name,
        "num_classes": num_classes,
        "best_epoch": checkpoint.get("epoch", -1),
        "test_miou": float(metrics["miou"]),
        "test_bg_iou": float(metrics["bg_iou"]),
        "test_fg_iou": float(metrics["fg_iou"]),
        "test_dice": float(metrics["dice"]),
        "test_pixel_acc": float(metrics["pixel_acc"]),
        "per_class_ious": metrics["ious"],
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")

    # Plot samples
    plot_segmentation_samples(model, test_ds, OUTPUT_DIR, DEVICE)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
