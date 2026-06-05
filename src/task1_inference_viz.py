"""
Task 1 — 可视化推理脚本
在本地运行，随机抽取测试集图片，展示模型预测结果（Top-5 概率 + 图片）
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import Flowers102

import timm

# ── 配置 ──────────────────────────────────────────────────────────────
DATA_DIR = "./data"
CKPT_PATH = "./output/task1/best_model.pth"
OUTPUT_DIR = "./output/task1"
IMG_SIZE = 224
NUM_SAMPLES = 10  # 展示多少张图片
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    model = timm.create_model("resnet34", pretrained=False, num_classes=102)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, images):
    """返回 batch 中每张图的 top5 概率和类别索引"""
    images = images.to(DEVICE)
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    top5_prob, top5_idx = probs.topk(5, dim=1)
    return top5_prob.cpu().numpy(), top5_idx.cpu().numpy()


def main():
    print(f"Device: {DEVICE}")

    # ── 加载模型 ──
    print(f"Loading model from {CKPT_PATH} ...")
    model = load_model()

    # ── 加载测试集 ──
    val_transform = T.Compose([
        T.Resize(int(IMG_SIZE * 1.143)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = Flowers102(root=DATA_DIR, split="test", download=True,
                               transform=val_transform)
    print(f"Test set: {len(full_dataset)} images")

    # ── 随机采样 ──
    indices = random.sample(range(len(full_dataset)), NUM_SAMPLES)
    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=NUM_SAMPLES, shuffle=False)

    images, labels = next(iter(loader))
    top5_prob, top5_idx = predict(model, images)
    labels = labels.numpy()

    # ── 反归一化以便显示 ──
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # ── 可视化：每个样本一张子图 ──
    ncols = 5
    nrows = (NUM_SAMPLES + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 4, nrows * 5))

    for i in range(NUM_SAMPLES):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        # 反归一化图像
        img = images[i].permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis("off")

        # 标题：正确答案 + Top-5 预测
        true_label = labels[i]
        ax.set_title(f"True: Class {true_label}", fontsize=10, fontweight="bold", color="green", loc="left")

        # 子图下方打印预测文字
        pred_text = ""
        for k in range(5):
            marker = " ✓" if top5_idx[i, k] == true_label else ""
            pred_text += f"  #{k+1}: cls {top5_idx[i,k]:3d} ({top5_prob[i,k]*100:.1f}%){marker}\n"
        ax.set_xlabel(pred_text, fontsize=7, family="monospace")

    plt.suptitle("Oxford 102 Flowers — ResNet-34 Inference (Top-5 per sample)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "inference_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Saved visualization: {save_path}")

    # ── 终端输出概要 ──
    top1_correct = 0
    for i in range(NUM_SAMPLES):
        if top5_idx[i, 0] == labels[i]:
            top1_correct += 1
        top5_match = "✓" if labels[i] in top5_idx[i] else "✗"
        print(f"  Sample {i+1:2d} | True: {labels[i]:3d} | "
              f"Pred Top-1: {top5_idx[i,0]:3d} ({top5_prob[i,0]*100:.1f}%) | Top-5 hit: {top5_match}")

    print(f"\nTop-1 on {NUM_SAMPLES} samples: {top1_correct}/{NUM_SAMPLES} ({top1_correct/NUM_SAMPLES*100:.1f}%)")


if __name__ == "__main__":
    main()
