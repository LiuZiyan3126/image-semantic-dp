"""
Task 2: Oxford-IIIT Pet — Image Segmentation
Model: DeepLabV3+ with ResNet-50 encoder (smp, ImageNet pretrained)
Framework: PyTorch + segmentation-models-pytorch + albumentations
Target GPU: AutoDL RTX 4090 D
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from torchvision.datasets import OxfordIIITPet

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet — Segmentation Fine-tuning")
    # Paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./output/task2",
                        help="Output directory for checkpoints & plots")
    # Data
    parser.add_argument("--img_size", type=int, default=384,
                        help="Input image size (square)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    # Model
    parser.add_argument("--model_name", type=str, default="DeepLabV3Plus",
                        help="smp architecture name (e.g. Unet, UnetPlusPlus, DeepLabV3Plus)")
    parser.add_argument("--encoder_name", type=str, default="resnet50",
                        help="smp encoder name (e.g. resnet34, resnet50, efficientnet-b0)")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes (2=binary, 3=trimap)")
    # Training
    parser.add_argument("--epochs", type=int, default=80,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    # Scheduler
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Linear warm-up epochs")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum LR for cosine annealing")
    # Misc
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable AMP")
    parser.add_argument("--loss_type", type=str, default="ce",
                        choices=["ce", "dice", "combined"],
                        help="Loss type: ce, dice, or combined")
    return parser.parse_args()


# ── Dataset ──────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """Wrap OxfordIIITPet with trimap→label remapping and albumentations transforms."""

    def __init__(self, base_dataset, indices, transform, num_classes):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_pil, mask_pil = self.base[self.indices[idx]]
        image = np.array(image_pil)
        mask = np.array(mask_pil)

        if self.num_classes == 2:
            # 1=foreground→1, 2=background→0, 3=border→255 (ignore)
            mask = np.where(mask == 1, 1,
                    np.where(mask == 2, 0, 255)).astype(np.int64)
        else:
            # 3-class mode: keep trimap values as-is (1/2/3)
            mask = mask.astype(np.int64)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()

        return image, mask


def get_transforms(img_size, mode="train"):
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def build_datasets(data_dir, img_size, num_classes, batch_size, num_workers):
    """Load Oxford-IIIT Pet, split trainval→train/val, return DataLoaders."""
    # Load trainval pool (3680 images) and split 80/20
    trainval_base = OxfordIIITPet(root=data_dir, split="trainval",
                                  target_types="segmentation", download=True)
    n = len(trainval_base)
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    split = int(n * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = SegmentationDataset(trainval_base, train_idx,
                                   get_transforms(img_size, "train"), num_classes)
    val_ds = SegmentationDataset(trainval_base, val_idx,
                                  get_transforms(img_size, "val"), num_classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"Training samples: {len(train_ds)}")
    print(f"Val samples:      {len(val_ds)}")
    print(f"Classes:          {num_classes}")
    return train_loader, val_loader, val_ds


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(model_name, encoder_name, num_classes):
    model = smp.create_model(
        arch=model_name,
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


# ── Loss ─────────────────────────────────────────────────────────────────────

def build_loss(loss_type, num_classes, device):
    if loss_type == "ce":
        ignore_idx = 255 if num_classes == 2 else None
        return nn.CrossEntropyLoss(ignore_index=ignore_idx).to(device)
    elif loss_type == "dice":
        ignore_idx = 255 if num_classes == 2 else None
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_idx).to(device)
    else:  # combined
        ignore_idx = 255 if num_classes == 2 else None
        ce = nn.CrossEntropyLoss(ignore_index=ignore_idx).to(device)
        dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_idx).to(device)
        return lambda pred, target: ce(pred, target) + dice(pred, target)


# ── Train / Val ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


@torch.no_grad()
def validate(model, loader, criterion, num_classes, device):
    model.eval()
    running_loss = 0.0
    total = 0

    # Per-class intersection / union (accumulated across all batches)
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    # For Dice
    intersect_fg = torch.tensor(0.0, device=device)
    pred_fg_sum = torch.tensor(0.0, device=device)
    target_fg_sum = torch.tensor(0.0, device=device)
    # For Pixel Accuracy
    correct_pixels = torch.tensor(0.0, device=device)
    total_valid_pixels = torch.tensor(0.0, device=device)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        preds = outputs.argmax(1)  # (B, H, W)
        valid = masks != 255

        # Per-class IoU accumulation
        for c in range(num_classes):
            pred_c = (preds == c) & valid
            target_c = (masks == c) & valid
            intersection[c] += (pred_c & target_c).sum()
            union[c] += (pred_c | target_c).sum()

        # Dice for foreground (class 1)
        pred_fg = (preds == 1)
        target_fg = (masks == 1)
        intersect_fg += (pred_fg & target_fg & valid).sum()
        pred_fg_sum += (pred_fg & valid).sum()
        target_fg_sum += (target_fg & valid).sum()

        # Pixel accuracy (valid pixels only)
        correct_pixels += (preds[valid] == masks[valid]).sum()
        total_valid_pixels += valid.sum()

    # Compute metrics
    ious = intersection / (union + 1e-6)
    miou = ious.mean().item()
    fg_iou = ious[1].item() if num_classes >= 2 else 0.0
    dice = (2.0 * intersect_fg / (pred_fg_sum + target_fg_sum + 1e-6)).item()
    pixel_acc = (correct_pixels / (total_valid_pixels + 1e-6)).item()

    return {
        "loss": running_loss / total,
        "miou": miou,
        "fg_iou": fg_iou,
        "dice": dice,
        "pixel_acc": pixel_acc,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    # mIoU
    axes[1].plot(epochs, history["val_miou"], "g-", label="mIoU")
    axes[1].plot(epochs, history["val_fg_iou"], "g--", label="FG IoU", alpha=0.6)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU (Validation)"); axes[1].legend(); axes[1].grid(True)

    # Dice + Pixel Accuracy
    axes[2].plot(epochs, history["val_dice"], "c-", label="Dice (FG)")
    axes[2].plot(epochs, history["val_pixel_acc"], "m-", label="Pixel Acc")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
    axes[2].set_title("Dice & Pixel Accuracy (Val)")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved training_curves.png to {output_dir}")


def plot_segmentation_samples(model, dataset, output_dir, device, num_classes):
    """Plot 8 fixed-index images with their ground-truth and predicted masks."""
    model.eval()

    # Get val transform (no augmentation) for visualization
    val_transform = get_transforms(384, mode="val")  # use fixed size for consistency
    # We need to apply this to raw PIL images from the base dataset

    fig, axes = plt.subplots(8, 3, figsize=(12, 32))
    rng = np.random.RandomState(SEED)
    sample_indices = rng.choice(len(dataset), min(8, len(dataset)), replace=False)

    for row, ds_idx in enumerate(sample_indices):
        idx = dataset.indices[ds_idx]
        image_pil, mask_pil = dataset.base[idx]
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        # Apply val transform to image for model input
        transformed = val_transform(image=image_np, mask=mask_np)
        inp = transformed["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            pred = model(inp).argmax(1).squeeze(0).cpu().numpy()  # (H, W)

        # Un-normalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_disp = transformed["image"].cpu().numpy().transpose(1, 2, 0)
        image_disp = image_disp * std + mean
        image_disp = np.clip(image_disp, 0, 1)

        # Build ground-truth mask display
        if num_classes == 2:
            # 1=fg→green, 0=bg→black, 255=ignore→gray
            gt_disp = np.zeros((*pred.shape, 3), dtype=np.float32)
            gt_disp[mask_np == 1] = [0, 1, 0]       # green = foreground
            gt_disp[mask_np == 2] = [0, 0, 0]       # black = background
            gt_disp[mask_np == 3] = [0.5, 0.5, 0.5] # gray = border
        else:
            gt_disp = np.zeros((*pred.shape, 3), dtype=np.float32)
            colors = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}
            for v, c in colors.items():
                gt_disp[mask_np == v] = c

        # Build prediction mask display
        pred_disp = np.zeros((*pred.shape, 3), dtype=np.float32)
        if num_classes == 2:
            pred_disp[pred == 1] = [0, 1, 0]
            pred_disp[pred == 0] = [0, 0, 0]
        else:
            for v, c in colors.items():
                pred_disp[pred == v] = c

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
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    train_loader, val_loader, val_ds = build_datasets(
        args.data_dir, args.img_size, args.num_classes,
        args.batch_size, args.num_workers)

    # Model
    model = build_model(args.model_name, args.encoder_name, args.num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_name} / Encoder: {args.encoder_name}")
    print(f"Total params: {total_params:,}  |  Trainable: {trainable:,}")

    # Loss
    criterion = build_loss(args.loss_type, args.num_classes, device)
    print(f"Loss: {args.loss_type}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Scheduler: linear warm-up → cosine annealing
    def warmup_cosine(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.lr_min / args.lr + 0.5 * (1 - args.lr_min / args.lr) * (1 + np.cos(np.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == "cuda") else None
    print(f"AMP: {'enabled' if scaler else 'disabled'}")

    # History
    history = {
        "train_loss": [], "val_loss": [],
        "val_miou": [], "val_fg_iou": [],
        "val_dice": [], "val_pixel_acc": [],
        "lr": [],
    }
    best_miou = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training {args.model_name} ({args.encoder_name}) for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        metrics = validate(
            model, val_loader, criterion, args.num_classes, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_miou"].append(metrics["miou"])
        history["val_fg_iou"].append(metrics["fg_iou"])
        history["val_dice"].append(metrics["dice"])
        history["val_pixel_acc"].append(metrics["pixel_acc"])
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"LR: {current_lr:.2e} | "
              f"TrLoss: {train_loss:.4f} | "
              f"VaLoss: {metrics['loss']:.4f} | "
              f"mIoU: {metrics['miou']:.4f} | FG-IoU: {metrics['fg_iou']:.4f} | "
              f"Dice: {metrics['dice']:.4f} | PixAcc: {metrics['pixel_acc']:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best (by mIoU)
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            best_epoch = epoch + 1
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": metrics["miou"],
                "val_fg_iou": metrics["fg_iou"],
                "val_dice": metrics["dice"],
                "val_pixel_acc": metrics["pixel_acc"],
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  >>> Best model saved (mIoU: {best_miou:.4f})")

    # ── Final best model eval ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete. Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Load best model & re-evaluate
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = validate(model, val_loader, criterion, args.num_classes, device)

    print(f"\nFinal Results (best checkpoint on validation set):")
    print(f"  mIoU         : {final_metrics['miou']:.4f}")
    print(f"  Foreground IoU: {final_metrics['fg_iou']:.4f}")
    print(f"  Dice (FG)    : {final_metrics['dice']:.4f}")
    print(f"  Pixel Acc    : {final_metrics['pixel_acc']:.4f}")
    print(f"  Val Loss     : {final_metrics['loss']:.4f}")

    # Save results JSON
    results = {
        "model": args.model_name,
        "encoder": args.encoder_name,
        "num_classes": args.num_classes,
        "loss_type": args.loss_type,
        "best_epoch": best_epoch,
        "best_val_miou": float(best_miou),
        "final_val_miou": float(final_metrics["miou"]),
        "final_val_fg_iou": float(final_metrics["fg_iou"]),
        "final_val_dice": float(final_metrics["dice"]),
        "final_val_pixel_acc": float(final_metrics["pixel_acc"]),
        "final_val_loss": float(final_metrics["loss"]),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    plot_curves(history, args.output_dir)
    plot_segmentation_samples(model, val_ds, args.output_dir, device, args.num_classes)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
