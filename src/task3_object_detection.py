"""
Task 3: PASCAL VOC — Object Detection
Model: Faster R-CNN + ResNet-50-FPN (torchvision, COCO pretrained)
Framework: PyTorch + torchvision
Target GPU: AutoDL RTX 4090 D
"""

import os
import argparse
import time
import json
import copy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
import torchvision.transforms.functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Constants ────────────────────────────────────────────────────────────────
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = 21  # 20 VOC + background


def parse_args():
    parser = argparse.ArgumentParser(description="PASCAL VOC — Faster R-CNN Fine-tuning")
    # Paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./output/task3",
                        help="Output directory for checkpoints & plots")
    # Data
    parser.add_argument("--img_max_size", type=int, default=800,
                        help="Max image side length (keep aspect ratio)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU (Faster R-CNN is memory-heavy)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    # Scheduler
    parser.add_argument("--warmup_epochs", type=int, default=2,
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

def voc_target_to_boxes(target, width, height):
    """Parse VOCDetection annotation dict → boxes(N,4) & labels(N) in xyxy format."""
    objs = target["annotation"].get("object", [])
    if not isinstance(objs, list):
        objs = [objs]

    boxes = []
    labels = []
    for obj in objs:
        # Skip difficult objects during training
        if int(obj.get("difficult", "0")) == 1:
            continue
        name = obj["name"]
        if name not in VOC_CLASSES:
            continue
        bbox = obj["bndbox"]
        x1 = float(bbox["xmin"]); y1 = float(bbox["ymin"])
        x2 = float(bbox["xmax"]); y2 = float(bbox["ymax"])
        # Clip to image bounds
        x1 = max(0, min(x1, width));  y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width));  y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(VOC_CLASSES.index(name) + 1)  # 1-indexed, 0 = background

    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
    return boxes, labels


def resize_image_and_boxes(image, boxes, max_size):
    """Resize image so longer side = max_size, keeping aspect ratio, and scale boxes."""
    w, h = image.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = F.resize(image, (new_h, new_w))
        boxes = boxes * scale
    return image, boxes


class VOCDataset(Dataset):
    """Wrap VOCDetection to return (image_tensor, target_dict) for Faster R-CNN."""

    def __init__(self, detection_dataset, max_size, augment=False):
        self.base = detection_dataset
        self.max_size = max_size
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_pil, target_raw = self.base[idx]
        w, h = image_pil.size

        boxes, labels = voc_target_to_boxes(target_raw, w, h)

        # Data augmentation: random horizontal flip
        if self.augment and np.random.rand() < 0.5:
            image_pil = F.hflip(image_pil)
            if boxes.shape[0] > 0:
                flipped_boxes = boxes.clone()
                flipped_boxes[:, 0] = w - boxes[:, 2]  # new x1 = w - old x2
                flipped_boxes[:, 2] = w - boxes[:, 0]  # new x2 = w - old x1
                boxes = flipped_boxes

        # Resize
        if self.augment:
            # Random scale: max side between 480 and max_size
            random_max = np.random.randint(480, self.max_size + 1)
            image_pil, boxes = resize_image_and_boxes(image_pil, boxes, random_max)
        else:
            image_pil, boxes = resize_image_and_boxes(image_pil, boxes, self.max_size)

        # To tensor
        image_tensor = F.to_tensor(image_pil)
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        target = {"boxes": boxes, "labels": labels}
        return image_tensor, target


def collate_fn(batch):
    """Group images and targets into lists (variable sizes allowed)."""
    return tuple(zip(*batch))


def build_datasets(data_dir, max_size, batch_size, num_workers):
    """Load PASCAL VOC 2007+2012 trainval → train, 2007 test → val."""
    # Training set: VOC 2007 trainval + VOC 2012 trainval
    print("Loading VOC 2007 trainval ...")
    train2007_full = VOCDetection(root=data_dir, year="2007", image_set="trainval",
                                   download=True)
    print("Loading VOC 2012 trainval ...")
    train2012_full = VOCDetection(root=data_dir, year="2012", image_set="trainval",
                                   download=True)

    # Wrap for detection format with augmentation
    train2007 = VOCDataset(train2007_full, max_size, augment=True)
    train2012 = VOCDataset(train2012_full, max_size, augment=True)
    train_set = ConcatDataset([train2007, train2012])

    # Validation set: VOC 2007 test
    print("Loading VOC 2007 test ...")
    test2007_full = VOCDetection(root=data_dir, year="2007", image_set="test",
                                  download=True)
    val_set = VOCDataset(test2007_full, max_size, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn)

    print(f"Training samples: {len(train_set)}  (VOC07+12 trainval)")
    print(f"Test samples:     {len(val_set)}  (VOC07 test)")
    print(f"Classes:          {len(VOC_CLASSES)}")
    return train_loader, val_loader


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── Train / Val ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    total = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        running_loss += losses.item() * len(images)
        total += len(images)

    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference on validation set, collect predictions and ground truths.
    Returns raw tensors on cpu for mAP computation, and summary dict."""
    model.eval()

    all_preds = []  # list of {boxes, scores, labels}
    all_gts = []    # list of {boxes, labels}

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)  # list[dict] with boxes, labels, scores

        for output, target in zip(outputs, targets):
            # Move to cpu and filter low-score predictions
            keep = output["scores"] > 0.05
            all_preds.append({
                "boxes": output["boxes"][keep].cpu(),
                "scores": output["scores"][keep].cpu(),
                "labels": output["labels"][keep].cpu(),
            })
            all_gts.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

    return all_preds, all_gts


# ── mAP Computation (VOC-style) ──────────────────────────────────────────────

def compute_voc_ap(recall, precision):
    """Compute VOC-style AP using all-point interpolation."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing (take max from right)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    # Compute area under PR curve
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return float(ap)


def compute_map(all_preds, all_gts, iou_thresh=0.5, num_classes=NUM_CLASSES):
    """Compute mAP@IoU_thresh given lists of predictions and ground truths."""
    # Organize by class
    gt_per_class = defaultdict(list)   # class → list of (image_idx, boxes)
    pred_per_class = defaultdict(list) # class → list of (image_idx, box, score)

    for img_idx, gt in enumerate(all_gts):
        for box, label in zip(gt["boxes"], gt["labels"]):
            gt_per_class[int(label.item())].append((img_idx, box))

    for img_idx, pred in enumerate(all_preds):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            pred_per_class[int(label.item())].append((img_idx, box, float(score.item())))

    aps = {}
    for cls_id in range(1, num_classes):  # skip background (class 0)
        preds_c = pred_per_class[cls_id]
        gts_c = gt_per_class[cls_id]

        if len(preds_c) == 0 and len(gts_c) == 0:
            aps[cls_id] = 1.0
            continue
        if len(preds_c) == 0:
            aps[cls_id] = 0.0
            continue

        # Sort predictions by confidence (descending)
        preds_c = sorted(preds_c, key=lambda x: x[2], reverse=True)

        # For each image, track which GT boxes haven't been matched yet
        gt_matched = defaultdict(set)  # image_idx -> set of matched gt indices

        num_gt = sum(1 for g in gts_c)

        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for i, (p_img_idx, p_box, _) in enumerate(preds_c):
            # Find best-matching GT box in same image
            image_gts = [(j, g_box) for j, (g_img_idx, g_box) in enumerate(gts_c)
                         if g_img_idx == p_img_idx and j not in gt_matched[p_img_idx]]

            best_iou = 0.0
            best_j = -1
            for j, g_box in image_gts:
                iou = box_iou_single(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1.0
                gt_matched[p_img_idx].add(best_j)
            else:
                fp[i] = 1.0

        # Compute precision-recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(num_gt, 1)
        precision = tp_cum / (tp_cum + fp_cum + 1e-12)

        ap = compute_voc_ap(recall, precision)
        aps[cls_id] = ap

    return aps


def box_iou_single(box_a, box_b):
    """Compute IoU for two boxes in xyxy format."""
    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter / (area_a + area_b - inter + 1e-12)
    return float(iou)


def summarize_metrics(aps, iou_thresh):
    """Convert per-class APs to summary metrics."""
    ap_values = list(aps.values())
    mAP = np.mean(ap_values) if ap_values else 0.0
    return {
        f"mAP@{iou_thresh}": float(mAP),
        f"per_class_AP@{iou_thresh}": {VOC_CLASSES[c - 1]: aps.get(c, 0.0)
                                       for c in range(1, NUM_CLASSES)},
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True)

    # mAP@0.5
    axes[1].plot(epochs, history.get("val_map50", []), "g-", label="mAP@0.5")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].set_title("mAP@0.5 (Validation)"); axes[1].legend(); axes[1].grid(True)

    # mAP@0.5:0.95
    axes[2].plot(epochs, history.get("val_map5095", []), "c-", label="mAP@0.5:0.95")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("mAP")
    axes[2].set_title("mAP@0.5:0.95 (Validation)")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved training_curves.png to {output_dir}")


def plot_detection_samples(model, val_loader, device, output_dir, num_samples=8):
    """Plot prediction samples with bounding boxes."""
    model.eval()

    # Collect samples without materializing entire dataset
    samples = []
    rng = np.random.RandomState(SEED)
    # Randomly skip batches to get diverse samples
    skip_interval = max(1, len(val_loader) // (num_samples // 2 + 1))
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx % skip_interval != 0:
            continue
        images_gpu = [img.to(device) for img in images]
        with torch.no_grad():
            preds = model(images_gpu)
        for img, pred, gt in zip(images, preds, targets):
            if len(samples) >= num_samples:
                break
            samples.append((img, pred, gt))
        if len(samples) >= num_samples:
            break

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for i, (img_tensor, pred, gt) in enumerate(samples):
        ax = axes[i]
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)

        h, w = img_np.shape[:2]

        # Draw GT boxes in green (dashed style)
        gt_boxes = gt["boxes"].cpu().numpy()
        gt_labels = gt["labels"].cpu().numpy()
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1.5, edgecolor="lime",
                                     facecolor="none", linestyle="--")
            ax.add_patch(rect)

        # Draw predicted boxes in red, score > 0.5 only
        keep = pred["scores"].cpu() > 0.5
        pred_boxes = pred["boxes"].cpu()[keep].numpy()
        pred_scores = pred["scores"].cpu()[keep].numpy()
        pred_labels = pred["labels"].cpu()[keep].numpy()
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1.5, edgecolor="red",
                                     facecolor="none")
            ax.add_patch(rect)
            cls_name = VOC_CLASSES[label - 1] if 1 <= label <= 20 else "?"
            ax.text(x1, y1 - 2, f"{cls_name} {score:.2f}",
                    fontsize=6, color="white", backgroundcolor="red")

        ax.axis("off")
        # Legend patches
        green_patch = patches.Patch(color="lime", alpha=0.5, label="GT (green dash)")
        red_patch = patches.Patch(color="red", alpha=0.5, label="Pred (red)")
        ax.legend(handles=[green_patch, red_patch], fontsize=6, loc="lower right")

    # Hide unused subplots
    for i in range(len(samples), len(axes)):
        axes[i].axis("off")

    plt.suptitle("Faster R-CNN — Detection Samples (green dash=GT, red=Pred)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "detection_samples.png"), dpi=120)
    plt.close()
    print(f"[Plot] Saved detection_samples.png to {output_dir}")


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
    train_loader, val_loader = build_datasets(
        args.data_dir, args.img_max_size, args.batch_size, args.num_workers)

    # Model
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Faster R-CNN + ResNet-50-FPN")
    print(f"Total params: {total_params:,}  |  Trainable: {trainable:,}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: linear warm-up → cosine annealing
    def warmup_cosine(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.lr_min / args.lr + 0.5 * (1 - args.lr_min / args.lr) * (1 + np.cos(np.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

    # AMP
    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and device.type == "cuda") else None
    print(f"AMP: {'enabled' if scaler else 'disabled'}")

    # History
    history = {
        "train_loss": [],
        "val_map50": [],
        "val_map5095": [],
        "lr": [],
    }
    best_map = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training Faster R-CNN + ResNet-50-FPN for {args.epochs} epochs")
    print(f"Dataset: PASCAL VOC 2007+2012 trainval → train, VOC07 test → val")
    print(f"Batch size: {args.batch_size}, Max image size: {args.img_max_size}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)

        # Evaluate (every epoch, but could do every N epochs to save time)
        all_preds, all_gts = evaluate(model, val_loader, device)

        # Compute mAP
        aps_50 = compute_map(all_preds, all_gts, iou_thresh=0.5)
        aps_5095_list = [compute_map(all_preds, all_gts, iou_thresh=t)
                         for t in np.arange(0.5, 1.0, 0.05)]
        map_5095 = np.mean([list(a.values()) for a in aps_5095_list])

        metrics_50 = summarize_metrics(aps_50, 0.5)
        map50 = metrics_50["mAP@0.5"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(train_loss)
        history["val_map50"].append(map50)
        history["val_map5095"].append(float(map_5095))
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"LR: {current_lr:.2e} | "
              f"TrLoss: {train_loss:.4f} | "
              f"mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map_5095:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best (by mAP@0.5)
        if map50 > best_map:
            best_map = map50
            best_epoch = epoch + 1
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_map50": float(map50),
                "val_map5095": float(map_5095),
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  >>> Best model saved (mAP@0.5: {best_map:.4f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete. Best mAP@0.5: {best_map:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Load best model & re-evaluate
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    all_preds, all_gts = evaluate(model, val_loader, device)

    # Detailed per-class mAP
    aps_50 = compute_map(all_preds, all_gts, iou_thresh=0.5)
    metrics_50 = summarize_metrics(aps_50, 0.5)

    print(f"\nFinal Results (best checkpoint on VOC07 test):")
    print(f"  mAP@0.5     : {metrics_50['mAP@0.5']:.4f}")
    # Per-class summary
    per_class = metrics_50["per_class_AP@0.5"]
    print(f"\nPer-class AP@0.5:")
    for cls_name, ap in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:>15s}: {ap:.4f}")

    # Save results JSON
    results = {
        "model": "Faster R-CNN + ResNet-50-FPN",
        "best_epoch": best_epoch,
        "best_val_map50": float(best_map),
        "final_val_map50": float(metrics_50["mAP@0.5"]),
        "per_class_AP": per_class,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save history
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plots
    plot_curves(history, args.output_dir)
    plot_detection_samples(model, val_loader, device, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
