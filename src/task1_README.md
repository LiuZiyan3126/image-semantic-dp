# Task 1: Oxford 102 Flowers — 图像分类

## 任务概述

对 **Oxford 102 Flowers** 数据集进行 102 类花卉分类，使用 **ResNet-34**（ImageNet 预训练）迁移学习微调。

| 项目 | 说明 |
|------|------|
| 数据集 | Oxford 102 Flowers（102 类，~8189 张图像） |
| 模型 | ResNet-34（timm，ImageNet 预训练） |
| 框架 | PyTorch 2.x + timm + torchvision |
| 目标 GPU | AutoDL RTX 4090 D（24 GB） |
| 最终 Top-1 | **93.69%** |
| 最终 Top-5 | **98.41%** |

---

## 文件结构

```
├── src/
│   ├── task1_classification.py   # 训练主脚本
│   ├── task1_eval_only.py        # 纯评估脚本（加载已训练模型）
│   └── task1_README.md           # 本文件
├── output/task1/
│   ├── best_model.pth            # 最佳模型权重（~256 MB）
│   ├── results.json              # 评测结果
│   └── confusion_matrix.png      # 归一化混淆矩阵
├── requirements.txt
└── CLAUDE.md
```

---

## 环境搭建

### 1. 本地 / 云服务器通用

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install timm>=0.9.0
pip install numpy matplotlib seaborn scikit-learn
```

或一键安装：

```bash
pip install -r requirements.txt
```

### 2. AutoDL 云服务器注意事项

AutoDL RTX 4090 D 通常已预装 PyTorch，只需补充安装：

```bash
pip install timm seaborn
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 数据集

**Oxford 102 Flowers** 包含 102 种花卉，共约 8189 张图像，每类 40~258 张不等。

脚本会自动下载数据集（约 329 MB）到 `--data_dir` 目录（默认 `./data/`）。

| 数据划分 | 原始用途 | 本脚本用途 | 样本数 |
|----------|----------|-------------|--------|
| `train` | 训练 | 训练 | 1020 |
| `val` | 验证 | 训练（合并） | 1020 |
| `test` | 测试 | 验证/测试 | 6149 |

> **说明：** 原始训练数据仅 1020 张（每类约 10 张）太少，脚本将 `train` + `val` 合并为训练集（2040 张），用 `test` 作为验证集（6149 张）。

---

## 使用方法

### 一、训练（云服务器上运行）

上传项目到云服务器后，在项目根目录执行：

```bash
cd /root/autodl-tmp/imageSemanticDp

python src/task1_classification.py
```

脚本默认行为：
- 自动下载 Oxford 102 Flowers 数据集到 `./data/`
- 使用 ResNet-34（ImageNet 预训练）
- 训练 60 个 epoch
- 每个 epoch 结束后在验证集上评估
- 仅保存最佳模型到 `./output/task1/best_model.pth`
- 训练结束后自动生成训练曲线图（Loss / Top-1 / Top-5）和混淆矩阵

### 二、纯评估（本地或云服务器均可）

如果你只想加载已训练好的模型做评估，运行：

```bash
python src/task1_eval_only.py
```

前提条件：
1. `./output/task1/best_model.pth` 存在
2. 数据集已下载到 `./data/`

脚本会输出 Top-1、Top-5 精度，并生成混淆矩阵和 per-class accuracy 分析。

---

## 所有命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据集根目录 |
| `--output_dir` | `./output/task1` | 输出目录（权重、图表、JSON） |
| `--img_size` | `224` | 输入图像尺寸 |
| `--batch_size` | `64` | 每 GPU 批量大小 |
| `--num_workers` | `4` | DataLoader 工作线程数 |
| `--model_name` | `resnet34` | timm 模型名称 |
| `--dropout` | `0.3` | 分类头 Dropout 率 |
| `--epochs` | `60` | 训练总轮数 |
| `--lr` | `1e-3` | 峰值学习率 |
| `--weight_decay` | `1e-4` | AdamW 权重衰减 |
| `--label_smoothing` | `0.1` | 标签平滑系数 |
| `--warmup_epochs` | `3` | 线性 warm-up 轮数 |
| `--lr_min` | `1e-6` | Cosine 退火最小学习率 |
| `--use_amp` | `True` | 启用自动混合精度（加速训练） |
| `--no_amp` | — | 关闭 AMP |

### 示例：尝试不同配置

```bash
# 用更小的 batch size（显存不足时）
python src/task1_classification.py --batch_size 32

# 训练 90 个 epoch
python src/task1_classification.py --epochs 90 --warmup_epochs 5

# 尝试不同模型
python src/task1_classification.py --model_name resnet18 --epochs 40

# 关闭 AMP（调试用）
python src/task1_classification.py --no_amp
```

---

## 训练配置详解

| 配置项 | 取值 | 说明 |
|--------|------|------|
| 模型 | ResNet-34 | timm 实现，ImageNet-1K 预训练 |
| 输入尺寸 | 224×224 | 标准 ResNet 输入 |
| 优化器 | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| 学习率策略 | WarmUp + Cosine Annealing | 前 3 epoch 线性 warm-up，之后 cosine 退火至 `1e-6` |
| 损失函数 | CrossEntropyLoss | `label_smoothing=0.1` |
| 混合精度 | AMP（GradScaler） | 加速训练，减少显存占用 |
| Batch Size | 64 | RTX 4090 D (24 GB) 可稳定运行 |
| 数据增强 | RandomResizedCrop + HorizontalFlip + Rotation + ColorJitter | 中等强度，防止小数据集过拟合 |
| 随机种子 | 42 | 确保可复现 |

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳模型 checkpoint，包含 `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_top1`, `args` |
| `results.json` | JSON 格式的最终评测结果（Top-1, Top-5, best epoch） |
| `confusion_matrix.png` | 102×102 归一化混淆矩阵热力图 |
| `training_curves.png` | 训练/验证 Loss 和 Accuracy 曲线（仅训练脚本生成） |

---

## 调用自己的模型做推理

```python
import torch
import timm
from PIL import Image
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 102
IMG_SIZE = 224

# 1. 构建模型并加载权重
model = timm.create_model("resnet34", pretrained=False, num_classes=CLASSES)
ckpt = torch.load("./output/task1/best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# 2. 预处理单张图片
transform = T.Compose([
    T.Resize(int(IMG_SIZE * 1.143)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("your_flower.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# 3. 推理
with torch.no_grad():
    outputs = model(input_tensor)
    top5_probs, top5_indices = outputs.topk(5, dim=1)
    top5_probs = torch.softmax(top5_probs, dim=1)

for i, (idx, prob) in enumerate(zip(top5_indices.squeeze().tolist(), top5_probs.squeeze().tolist())):
    print(f"Top-{i+1}: Class {idx:3d}  ({prob*100:.2f}%)")
```

---

## 常见问题

**Q: 显存不足（OOM）？**
减小 `--batch_size`，例如 `--batch_size 32` 或 `--batch_size 16`。

**Q: 数据集下载失败？**
手动下载：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
放到 `./data/flowers-102/` 目录下。

**Q: 如何在另一台机器上评估？**
只需复制 `best_model.pth` 和 `task1_eval_only.py` 到目标机器，数据集会由脚本自动下载。

**Q: train/val/test 划分为什么这样设计？**
原始 Flowers102 的 `train`（1020 张）和 `val`（1020 张）各自太小。合并后训练集 2040 张，用原始的 `test`（6149 张）作为验证集更稳定。这不是标准做法，但在该数据集上常见。

---

## 预训练模型来源

- ResNet-34 预训练权重：timm 库自动下载的 ImageNet-1K 预训练权重
- 论文：[Deep Residual Learning for Image Recognition (He et al., CVPR 2016)](https://arxiv.org/abs/1512.03385)
