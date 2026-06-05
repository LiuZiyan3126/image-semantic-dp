# Task 2: Oxford-IIIT Pet — 图像分割

## 任务概述

对 **Oxford-IIIT Pet** 数据集进行宠物区域（前景/背景）分割，使用 **DeepLabV3Plus + ResNet-50**（ImageNet 预训练）fine-tune。

| 项目 | 说明 |
|------|------|
| 数据集 | Oxford-IIIT Pet（37 类猫狗品种，7349 张图像，pixel-level trimap 标注） |
| 模型 | DeepLabV3Plus + ResNet-50 encoder（smp，ImageNet 预训练） |
| 框架 | PyTorch 2.x + segmentation-models-pytorch + albumentations |
| 目标 GPU | RTX 4090 D（25 GB），本地 CPU 也可推理 |
| 最终 mIoU | **0.9721** |
| 最终 Dice (FG) | **0.9817** |
| 最终 Pixel Acc | **0.9871** |

---

## 文件结构

```
├── src/
│   ├── task2_segmentation.py   # 训练主脚本
│   ├── task2_inference.py      # 推理脚本（单张/文件夹）
│   ├── task2_viz_only.py       # 纯可视化脚本（加载模型生成样本对比图）
│   └── task2_README.md         # 本文件
├── output/task2/
│   ├── best_model.pth          # 最佳模型权重（~106 MB）
│   ├── results.json            # 评测结果
│   ├── training_curves.png     # 训练曲线（Loss / mIoU / Dice & Pixel Acc）
│   └── segmentation_samples.png # 8 张样本分割对比图（原图 / GT / 预测）
```

---

## 环境搭建

### 1. 云服务器训练环境

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install segmentation-models-pytorch>=0.3.0
pip install albumentations>=1.3.0
pip install numpy matplotlib Pillow
```

### 2. 本地推理环境（无需 GPU）

```bash
pip install torch torchvision segmentation-models-pytorch albumentations numpy Pillow
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 数据集

**Oxford-IIIT Pet** 包含 37 种猫狗品种共 7349 张图像，每张图像带有 pixel-level trimap 分割标注：

| trimap 值 | 含义 | 二分类处理 |
|-----------|------|------------|
| 1 | 前景（宠物） | → 类别 1（正类） |
| 2 | 背景 | → 类别 0（负类） |
| 3 | 边界/不确定 | → 255（ignore，不计入 loss） |

### 数据划分

脚本使用 `torchvision.datasets.OxfordIIITPet` 自动下载数据集到 `./data/`，并将 `trainval`（3680 张）按 80/20 随机划分为训练集和验证集：

| 集合 | 样本数 |
|------|--------|
| Training | 2944 |
| Validation | 736 |

> 数据集约 800 MB，首次运行自动下载。

---

## 使用方法

### 一、训练（云服务器运行）

上传项目到云服务器后，在项目根目录执行：

```bash
cd /path/to/CV_Project
python src/task2_segmentation.py
```

脚本行为：
- 自动下载 Oxford-IIIT Pet 到 `./data/`
- 使用 DeepLabV3Plus + ResNet-50（ImageNet 预训练）
- 训练 80 个 epoch
- 每个 epoch 结束后在验证集上评估
- 仅保存 mIoU 最佳模型到 `./output/task2/best_model.pth`
- 训练结束后自动生成训练曲线和分割样本对比图

### 二、纯可视化（不重新训练，补充生成图）

如果训练已完成但可视化图缺了，可以单独跑：

```bash
python src/task2_viz_only.py
```

前提：`./output/task2/best_model.pth` 和 `./data/` 数据集已存在。

### 三、推理（本地或云端均可）

**单张图片：**

```bash
python src/task2_inference.py --image path/to/your_pet_photo.jpg
```

**整个文件夹：**

```bash
python src/task2_inference.py --dir path/to/images/folder/
```

**自定义输出目录：**

```bash
python src/task2_inference.py --image cat.jpg --output_dir ./my_results/
```

推理输出文件保存在 `./output/task2/inference/`：
- `*_mask.png` — 黑白分割 mask（白色=宠物 / 黑色=背景）
- `*_overlay.png` — 原图 + 绿色半透明叠加

---

## 所有命令行参数

### 训练参数 (`task2_segmentation.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据集根目录 |
| `--output_dir` | `./output/task2` | 输出目录（权重、图表、JSON） |
| `--img_size` | `384` | 输入图像尺寸（正方形） |
| `--batch_size` | `16` | 每 GPU 批量大小 |
| `--num_workers` | `4` | DataLoader 工作线程数 |
| `--model_name` | `DeepLabV3Plus` | smp 架构名称 |
| `--encoder_name` | `resnet50` | smp encoder（resnet34/resnet50/efficientnet-b0 等） |
| `--num_classes` | `2` | 类别数（2=二分类，3=trimap 三分类） |
| `--epochs` | `80` | 训练总轮数 |
| `--lr` | `1e-3` | 峰值学习率 |
| `--weight_decay` | `1e-4` | AdamW 权重衰减 |
| `--warmup_epochs` | `3` | 线性 warm-up 轮数 |
| `--lr_min` | `1e-6` | Cosine 退火最小学习率 |
| `--loss_type` | `ce` | 损失函数（ce / dice / combined） |
| `--use_amp` | `True` | 启用自动混合精度 |
| `--no_amp` | — | 关闭 AMP |

### 推理参数 (`task2_inference.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | — | 单张图片路径 |
| `--dir` | — | 批量处理文件夹（`--image` 和 `--dir` 二选一） |
| `--output_dir` | `./output/task2/inference` | 推理结果输出目录 |
| `--model_path` | `./output/task2/best_model.pth` | 模型权重路径 |
| `--img_size` | `384` | 输入尺寸 |
| `--save_overlay` | `True` | 是否保存叠加图 |

### 示例：尝试不同配置

```bash
# 换轻量 encoder（显存不足时）
python src/task2_segmentation.py --encoder_name resnet34 --batch_size 8

# 三分类 trimap 模式
python src/task2_segmentation.py --num_classes 3 --loss_type dice

# 关闭 AMP（调试用）
python src/task2_segmentation.py --no_amp
```

---

## 训练配置详解

| 配置项 | 取值 | 说明 |
|--------|------|------|
| 模型 | DeepLabV3Plus | smp 实现，Atrous Spatial Pyramid Pooling + 解码器 |
| Encoder | ResNet-50 | smp 实现，ImageNet-1K 预训练 |
| 输入尺寸 | 384×384 | |
| 优化器 | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| 学习率策略 | WarmUp + Cosine Annealing | 前 3 epoch 线性 warm-up，之后 cosine 退火至 `1e-6` |
| 损失函数 | CrossEntropyLoss | `ignore_index=255`（跳过边界像素） |
| 混合精度 | AMP（GradScaler） | 加速训练，减少显存占用 |
| Batch Size | 16 | RTX 4090 D (25 GB) 可稳定运行 |
| 数据增强 | RandomResizedCrop + HorizontalFlip + BrightnessContrast | 随机裁剪 + 水平翻转 + 亮度对比度 |
| 随机种子 | 42 | 确保可复现 |
| 总参数量 | 26,677,842 | 全部可训练 |

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳模型 checkpoint，包含 `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_miou`, `args` |
| `results.json` | JSON 格式的最终评测结果 |
| `training_curves.png` | 训练/验证 Loss、mIoU、Dice & Pixel Acc 三子图曲线 |
| `segmentation_samples.png` | 8 张验证集样本的三列对比图（原图 / Ground Truth / 预测） |

---

## 从云服务器下载结果到本地

```bash
# 替换为你的服务器地址和端口
scp -rP 端口 root@服务器地址:~/autodl-tmp/CV_Project/output/task2/ D:\lab\imageSemanticDp\output\task2\
```

下载后本地直接就能推理（不需要数据集），见上方"推理"一节。

---

## 调用模型做推理（Python 示例）

```python
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384

# 1. 构建模型并加载权重
model = smp.create_model(arch="DeepLabV3Plus", encoder_name="resnet50",
                          encoder_weights=None, in_channels=3, classes=2)
ckpt = torch.load("./output/task2/best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# 2. 预处理
image_pil = Image.open("your_pet.jpg").convert("RGB")
image_np = np.array(image_pil)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
inp = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)

# 3. 推理
with torch.no_grad():
    pred = model(inp).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    # pred: (384, 384), 0=背景 1=前景

# 4. 保存 mask（还原到原始尺寸）
mask_img = Image.fromarray(pred * 255)
mask_img = mask_img.resize(image_pil.size, Image.NEAREST)
mask_img.save("output_mask.png")
```

---

## 预训练模型来源

- ResNet-50 Encoder 预训练权重：ImageNet-1K，通过 smp 的 `encoder_weights="imagenet"` 自动加载
- DeepLabV3Plus 论文：[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (Chen et al., ECCV 2018)](https://arxiv.org/abs/1802.02611)
- segmentation-models-pytorch 库：[https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

---

## 常见问题

**Q: 显存不足（OOM）？**
减小 `--batch_size` 或换轻量 encoder：`--batch_size 8 --encoder_name resnet34`

**Q: 数据集下载失败？**
可能网络访问受限，手动下载：
[https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
放到 `./data/oxford-iiit-pet/` 目录下。

**Q: 本地推理需要下载数据集吗？**
不需要。`task2_inference.py` 只加载模型权重和你的输入图片。

**Q: 如何只生成可视化图不重新训练？**
```bash
python src/task2_viz_only.py
```

**Q: 二分类和三分类的模型能互换吗？**
不能。训练时 `--num_classes` 决定了模型输出通道数和 mask 处理逻辑，必须保持一致。

**Q: 模型文件太大（~106 MB），能压缩吗？**
可只保存 `model_state_dict`，去掉 optimizer state：
```python
torch.save({"model_state_dict": checkpoint["model_state_dict"],
            "best_miou": checkpoint["best_miou"],
            "epoch": checkpoint["epoch"]}, "best_model_slim.pth")
```
