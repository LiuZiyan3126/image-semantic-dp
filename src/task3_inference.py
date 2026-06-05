"""
Task 3 Inference: Run object detection on a single image or a folder of images.
Usage:
    python src/task3_inference.py --image your_photo.jpg
    python src/task3_inference.py --dir ./images/ --conf_thresh 0.3
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ── Constants ────────────────────────────────────────────────────────────────
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = 21  # 20 VOC + background

# Distinct colors for 20 classes
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Task3 Object Detection Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Process all images in a folder")
    parser.add_argument("--output_dir", type=str, default="./output/task3/inference",
                        help="Where to save results")
    parser.add_argument("--model_path", type=str, default="./output/task3/best_model.pth",
                        help="Path to trained checkpoint")
    parser.add_argument("--img_max_size", type=int, default=800,
                        help="Max image side length")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold for displaying boxes")
    return parser.parse_args()


def load_model(args, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    map50 = ckpt.get("val_map50", "N/A")
    print(f"Loaded model: epoch {epoch}, mAP@0.5: {map50}")
    return model


def preprocess(image_path, max_size):
    """Load image, resize keeping aspect ratio, return tensor and original PIL."""
    image_pil = Image.open(image_path).convert("RGB")
    w, h = image_pil.size

    # Resize so longer side = max_size
    scale = max_size / max(w, h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = F.resize(image_pil, (new_h, new_w))
    else:
        image_resized = image_pil

    image_tensor = F.to_tensor(image_resized)
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    return image_tensor, image_pil, scale if scale < 1.0 else 1.0


@torch.no_grad()
def predict(model, image_tensor, device):
    output = model([image_tensor.to(device)])[0]
    return {k: v.cpu() for k, v in output.items()}


def draw_boxes(image_pil, pred, conf_thresh, scale):
    """Draw bounding boxes on image, returns annotated PIL Image."""
    img_np = np.array(image_pil)
    h, w = img_np.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(max(1, w / 100), max(1, h / 100)), dpi=100)
    ax.imshow(img_np)
    ax.axis("off")

    keep = pred["scores"] > conf_thresh
    boxes = pred["boxes"][keep].numpy()
    scores = pred["scores"][keep].numpy()
    labels = pred["labels"][keep].numpy()

    # Scale boxes back to original image size
    boxes = boxes / scale

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        cls_idx = int(label) - 1
        cls_name = VOC_CLASSES[cls_idx] if 0 <= cls_idx < 20 else "?"
        color = CLASS_COLORS[cls_idx % 20]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), f"{cls_name} {score:.2f}",
                fontsize=8, color="white", backgroundcolor=color,
                fontweight="bold")

    plt.tight_layout(pad=0)
    return fig


def process_single(model, image_path, output_dir, args, device):
    print(f"Processing: {image_path}")
    image_tensor, original_pil, scale = preprocess(image_path, args.img_max_size)
    pred = predict(model, image_tensor, device)

    n_det = (pred["scores"] > args.conf_thresh).sum().item()
    print(f"  Detections (conf>{args.conf_thresh}): {n_det}")

    # Print detections
    keep = pred["scores"] > args.conf_thresh
    for box, score, label in zip(
        pred["boxes"][keep].numpy(),
        pred["scores"][keep].numpy(),
        pred["labels"][keep].numpy()
    ):
        box = box / scale
        cls_name = VOC_CLASSES[int(label) - 1] if 1 <= int(label) <= 20 else "?"
        print(f"    {cls_name:>15s} | conf={score:.3f} | "
              f"box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")

    # Draw and save
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.splitext(os.path.basename(image_path))[0]
    fig = draw_boxes(original_pil, pred, args.conf_thresh, scale)
    save_path = os.path.join(output_dir, f"{out_base}_detected.png")
    fig.savefig(save_path, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args, device)

    if args.image:
        image_paths = [args.image]
    elif args.dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = sorted([
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not image_paths:
            print(f"No images found in {args.dir}")
            return
        print(f"Found {len(image_paths)} image(s)")
    else:
        print("Specify --image or --dir")
        return

    for img_path in image_paths:
        process_single(model, img_path, args.output_dir, args, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
