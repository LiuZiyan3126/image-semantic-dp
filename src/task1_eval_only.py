"""
Task 1 Eval-Only: Load saved best_model.pth and generate final results.
Run this AFTER task1_classification.py has completed training.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import Flowers102
from sklearn.metrics import confusion_matrix
import seaborn as sns

import timm

DATA_DIR = "./data"
OUTPUT_DIR = "./output/task1"
BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=102):
    model = timm.create_model("resnet34", pretrained=False, num_classes=num_classes)
    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        _, pred5 = outputs.topk(5, dim=1)
        _, pred1 = outputs.topk(1, dim=1)
        correct_top1 += (pred1.squeeze(1) == labels).sum().item()
        correct_top5 += pred5.eq(labels.view(-1, 1)).sum().item()
        total += images.size(0)

        all_preds.extend(pred1.squeeze(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return correct_top1 / total, correct_top5 / total, all_preds, all_labels


def main():
    print(f"Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────
    val_transform = T.Compose([
        T.Resize(int(IMG_SIZE * 1.143)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_set = Flowers102(root=DATA_DIR, split="test", download=True,
                         transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test samples: {len(val_set)}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])

    best_epoch = ckpt.get("epoch", "?")
    best_val_top1 = ckpt.get("val_top1", 0.0)
    print(f"Checkpoint from epoch {best_epoch}, saved Top-1: {best_val_top1:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    top1, top5, preds, labels = evaluate(model, val_loader)

    print(f"\n{'='*50}")
    print(f"Final Results on Test Set (6149 images)")
    print(f"{'='*50}")
    print(f"  Top-1 Accuracy : {top1:.4f} ({top1*100:.2f}%)")
    print(f"  Top-5 Accuracy : {top5:.4f} ({top5*100:.2f}%)")

    # ── Save results.json ─────────────────────────────────────────────────
    results = {
        "model": "resnet34",
        "best_epoch": best_epoch,
        "saved_val_top1": float(best_val_top1),
        "final_val_top1": float(top1),
        "final_val_top5": float(top5),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] results.json saved")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(cm_norm, cmap="Blues", cbar=True, ax=ax, linewidths=0.1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized) — Oxford 102 Flowers")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    print(f"[OK] confusion_matrix.png saved")

    # ── Per-class accuracy ────────────────────────────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    worst5 = np.argsort(per_class_acc)[:5]
    print(f"\nTop-5 hardest classes (lowest per-class accuracy):")
    for idx in worst5:
        print(f"  Class {idx:3d}: {per_class_acc[idx]:.4f} ({cm.sum(axis=1)[idx]:.0f} samples)")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
