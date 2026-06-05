"""
Task 2 Inference: Run segmentation on a single image or a folder of images.
Usage:
    python src/task2_inference.py --image your_photo.jpg
    python src/task2_inference.py --dir ./images/ --save_overlay
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser(description="Task2 Segmentation Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Process all images in a folder")
    parser.add_argument("--output_dir", type=str, default="./output/task2/inference",
                        help="Where to save results")
    parser.add_argument("--model_path", type=str, default="./output/task2/best_model.pth",
                        help="Path to trained checkpoint")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--save_overlay", action="store_true", default=True,
                        help="Save alpha-blended overlay in addition to mask")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="DeepLabV3Plus")
    parser.add_argument("--encoder_name", type=str, default="resnet50")
    return parser.parse_args()


def load_model(args, device):
    model = smp.create_model(
        arch=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=args.num_classes,
    )
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded model: epoch {epoch}, mIoU: {ckpt.get('best_miou', 'N/A')}")
    return model


def preprocess(image_path, img_size):
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    result = transform(image=image_np)
    return result["image"].unsqueeze(0), image_pil


def predict(model, input_tensor, device):
    with torch.no_grad():
        logits = model(input_tensor.to(device))  # (1, C, H, W)
        mask = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def save_results(mask, original_pil, out_base, save_overlay, img_size):
    os.makedirs(os.path.dirname(out_base) if os.path.dirname(out_base) else ".", exist_ok=True)

    # Save mask as PNG (resize back to original size)
    mask_img = Image.fromarray(mask * 255)  # 0=bg, 1=fg → 0, 255
    mask_img = mask_img.resize(original_pil.size, Image.NEAREST)
    mask_path = out_base + "_mask.png"
    mask_img.save(mask_path)
    print(f"  Mask saved: {mask_path}")

    if not save_overlay:
        return

    # Build color mask: green=fg, transparent=bg
    mask_np = np.array(mask_img)  # 0 or 255
    fg = mask_np >= 128

    # Resize original to match
    original_rgba = original_pil.convert("RGBA")
    original_resized = np.array(original_pil.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32)

    # Color mask at model resolution then resize back
    color = np.zeros((img_size, img_size, 3), dtype=np.float32)
    color[mask == 1] = [0, 1, 0]  # green for foreground

    # Blend
    alpha = 0.4
    blended = original_resized * (1 - alpha) + color * 255 * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended).resize(original_pil.size, Image.BILINEAR)

    overlay_path = out_base + "_overlay.png"
    blended_pil.save(overlay_path)
    print(f"  Overlay saved: {overlay_path}")


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
        print(f"\nProcessing: {img_path}")
        inp, original_pil = preprocess(img_path, args.img_size)
        mask = predict(model, inp, device)
        out_base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0])
        save_results(mask, original_pil, out_base, args.save_overlay, args.img_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
