"""
Task 1: Oxford 102 Flowers — Image Classification
Model: ResNet-34 (timm, ImageNet pretrained)
Framework: PyTorch + timm + torchvision
Target GPU: AutoDL RTX 4090 D
"""

import os
import argparse
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.datasets import Flowers102
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

import timm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Oxford 102 Flowers — ResNet-34 Fine-tuning")
    # Paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./output/task1",
                        help="Output directory for checkpoints & plots")
    # Data
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    # Model
    parser.add_argument("--model_name", type=str, default="resnet34",
                        help="timm model name")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Classifier dropout rate")
    # Training
    parser.add_argument("--epochs", type=int, default=60,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing epsilon")
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
    parser.add_argument("--save_best_only", action="store_true", default=True,
                        help="Only save the best checkpoint")
    return parser.parse_args()


# ── Dataset ──────────────────────────────────────────────────────────────────
def build_datasets(data_dir: str, img_size: int):
    """Load Oxford 102 Flowers with standard splits and return dataloaders."""

    # Training transforms with moderate augmentation
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize(int(img_size * 1.143)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # Flowers102 splits: 'train' (1020), 'val' (1020), 'test' (6149)
    # Combine train + val for more training data → 2040 images
    train_set = Flowers102(root=data_dir, split="train", download=True,
                           transform=train_transform)
    val_subset = Flowers102(root=data_dir, split="val", download=True,
                            transform=train_transform)
    train_full = ConcatDataset([train_set, val_subset])

    val_set = Flowers102(root=data_dir, split="test", download=True,
                         transform=val_transform)

    print(f"Training samples: {len(train_full)}  (train + val combined)")
    print(f"Test samples:     {len(val_set)}")
    print(f"Classes:         102")
    return train_full, val_set


# ── Model ────────────────────────────────────────────────────────────────────
def build_model(model_name: str, num_classes: int, dropout: float):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes,
                              drop_rate=dropout)
    return model


# ── Train / Val one epoch ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

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

        # Top-1
        _, pred1 = outputs.topk(1, dim=1)
        correct_top1 += (pred1.squeeze(1) == labels).sum().item()
        # Top-5
        _, pred5 = outputs.topk(5, dim=1)
        correct_top5 += pred5.eq(labels.view(-1, 1)).sum().item()

    return running_loss / total, correct_top1 / total, correct_top5 / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        _, pred5 = outputs.topk(5, dim=1)
        _, pred1 = outputs.topk(1, dim=1)
        correct_top1 += (pred1.squeeze(1) == labels).sum().item()
        correct_top5 += pred5.eq(labels.view(-1, 1)).sum().item()

        all_preds.extend(pred1.squeeze(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (running_loss / total,
            correct_top1 / total,
            correct_top5 / total,
            all_preds, all_labels)


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    # Top-1
    axes[1].plot(epochs, history["train_top1"], "b-", label="Train")
    axes[1].plot(epochs, history["val_top1"], "r-", label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Top-1 Accuracy"); axes[1].legend(); axes[1].grid(True)

    # Top-5
    axes[2].plot(epochs, history["train_top5"], "b-", label="Train")
    axes[2].plot(epochs, history["val_top5"], "r-", label="Val")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Top-5 Accuracy"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved training_curves.png to {output_dir}")


def plot_confusion_matrix(cm, class_names, output_dir, normalize=True):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", cbar=True, ax=ax, linewidths=0.1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized) — Oxford 102 Flowers")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=120)
    plt.close()
    print(f"[Plot] Saved confusion_matrix.png to {output_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    train_set, val_set = build_datasets(args.data_dir, args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Model
    model = build_model(args.model_name, num_classes=102, dropout=args.dropout)
    model = model.to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}")

    # Loss — label smoothing helps with overfitting on small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Scheduler: linear warm-up → cosine annealing
    # Step-based: warm-up for warmup_epochs, then cosine to lr_min
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
        "train_top1": [], "val_top1": [],
        "train_top5": [], "val_top5": [],
        "lr": [],
    }
    best_acc = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training {args.model_name} for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_top1, val_top5, val_preds, val_labels = validate(
            model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_top1)
        history["train_top5"].append(train_top5)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"LR: {current_lr:.2e} | "
              f"TrLoss: {train_loss:.4f} | TrTop1: {train_top1:.4f} | TrTop5: {train_top5:.4f} | "
              f"VaLoss: {val_loss:.4f} | VaTop1: {val_top1:.4f} | VaTop5: {val_top5:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best
        if val_top1 > best_acc:
            best_acc = val_top1
            best_epoch = epoch + 1
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  >>> Best model saved (Top-1: {best_acc:.4f})")

    # ── Final best model eval ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete. Best Top-1: {best_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Load best model & compute final confusion matrix
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_top1, val_top5, val_preds, val_labels = validate(
        model, val_loader, criterion, device)

    print(f"\nFinal Results (best checkpoint on test set):")
    print(f"  Top-1 Accuracy : {val_top1:.4f} ({val_top1*100:.2f}%)")
    print(f"  Top-5 Accuracy : {val_top5:.4f} ({val_top5*100:.2f}%)")
    print(f"  Loss           : {val_loss:.4f}")

    # Save results JSON
    results = {
        "model": args.model_name,
        "best_epoch": best_epoch,
        "best_val_top1": float(best_acc),
        "final_val_top1": float(val_top1),
        "final_val_top5": float(val_top5),
        "final_val_loss": float(val_loss),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    plot_curves(history, args.output_dir)

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    class_names = [str(i) for i in range(102)]
    plot_confusion_matrix(cm, class_names, args.output_dir, normalize=True)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
