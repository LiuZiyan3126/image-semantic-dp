# Task 3: PASCAL VOC — 目标检测

## 任务概述

对 **PASCAL VOC** 数据集进行 20 类目标检测，使用 **Faster R-CNN + ResNet-50-FPN**（COCO 预训练）fine-tune。

| 项目 | 说明 |
|------|------|
| 数据集 | PASCAL VOC 2007 + 2012（20 类，train ~16551 张，test ~4952 张） |
| 模型 | Faster R-CNN + ResNet-50-FPN（torchvision，COCO 预训练） |
| 框架 | PyTorch 2.x + torchvision |
| 目标 GPU | RTX 4090 D（24 GB），本地 CPU 也可推理 |
| 预期 mAP@0.5 | **78-82%** |
| 预期 mAP@0.5:0.95 | **50-55%** |

---

## 文件结构

```
├── src/
│   ├── task3_object_detection.py  # 训练主脚本
│   ├── task3_inference.py         # 推理脚本（单张/文件夹）
│   ├── task3_design.md            # 详细设计文档
│   └── task3_README.md            # 本文件
├── output/task3/
│   ├── best_model.pth             # 最佳模型权重（~165 MB）
│   ├── history.json               # 逐 epoch 训练历史
│   ├── results.json               # 评测结果（含 per-class AP）
│   ├── training_curves.png        # 训练曲线（Loss / mAP@0.5 / mAP@0.5:0.95）
│   └── detection_samples.png      # 8 张测试集检测可视化
```

---

## 环境搭建

### 1. 云服务器训练环境

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy matplotlib Pillow
```

### 2. 本地推理环境（无需 GPU）

```bash
pip install torch torchvision numpy matplotlib Pillow
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 数据集

**PASCAL VOC** 包含 20 个常见物体类别的标注：

| 类别 | 类别 | 类别 | 类别 |
|------|------|------|------|
| aeroplane | bicycle | bird | boat |
| bottle | bus | car | cat |
| chair | cow | diningtable | dog |
| horse | motorbike | person | pottedplant |
| sheep | sofa | train | tvmonitor |

### 数据划分

脚本使用 `torchvision.datasets.VOCDetection` 自动下载数据集到 `./data/`：

| 集合 | 数据来源 | 样本数 |
|------|----------|--------|
| Training | VOC2007 trainval + VOC2012 trainval | ~16551 |
| Test | VOC2007 test | ~4952 |

> 数据集约 2 GB（VOC2007 + VOC2012），首次运行自动下载。

### 数据增强

| 阶段 | 增强方式 |
|------|----------|
| 训练 | 随机水平翻转 (p=0.5) + 多尺度 resize (max side ∈ [480, 800]) |
| 验证/推理 | 固定 resize（max side = 800，保持宽高比） |
| 后处理 | NMS IoU 阈值 0.5 |

---

## 使用方法

### 一、训练（云服务器运行）

上传项目到云服务器后，在项目根目录执行：

```bash
cd /path/to/CV_Project
python src/task3_object_detection.py
```

脚本行为：
- 自动下载 PASCAL VOC 2007 + 2012 到 `./data/`
- 使用 Faster R-CNN + ResNet-50-FPN（COCO 预训练）
- 训练 30 个 epoch
- 每个 epoch 结束后在 VOC2007 test 上评估 mAP@0.5
- 仅保存 mAP 最佳模型到 `./output/task3/best_model.pth`
- 训练结束后自动生成训练曲线和检测样本可视化图

### 二、推理（本地或云端均可）

**单张图片：**

```bash
python src/task3_inference.py --image path/to/your_photo.jpg
```

**整个文件夹：**

```bash
python src/task3_inference.py --dir path/to/images/folder/
```

**调整置信度阈值：**

```bash
python src/task3_inference.py --image cat.jpg --conf_thresh 0.3
```

推理输出文件保存在 `./output/task3/inference/`：
- `*_detected.png` — 带检测框和类别标签的标注图像
- 终端输出检测到的所有对象及其置信度和坐标

---

## 所有命令行参数

### 训练参数 (`task3_object_detection.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据集根目录 |
| `--output_dir` | `./output/task3` | 输出目录（权重、图表、JSON） |
| `--img_max_size` | `800` | 图像长边最大像素（保持宽高比） |
| `--batch_size` | `4` | 每 GPU 批量大小（Faster R-CNN 显存消耗大） |
| `--num_workers` | `4` | DataLoader 工作线程数 |
| `--epochs` | `30` | 训练总轮数 |
| `--lr` | `1e-3` | 峰值学习率 |
| `--weight_decay` | `1e-4` | AdamW 权重衰减 |
| `--warmup_epochs` | `2` | 线性 warm-up 轮数 |
| `--lr_min` | `1e-6` | Cosine 退火最小学习率 |
| `--use_amp` | `True` | 启用自动混合精度（FP16） |
| `--no_amp` | — | 关闭 AMP（调试用） |

### 推理参数 (`task3_inference.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | — | 单张图片路径 |
| `--dir` | — | 批量处理文件夹（`--image` 和 `--dir` 二选一） |
| `--output_dir` | `./output/task3/inference` | 推理结果输出目录 |
| `--model_path` | `./output/task3/best_model.pth` | 模型权重路径 |
| `--img_max_size` | `800` | 输入图像长边最大像素 |
| `--conf_thresh` | `0.5` | 显示检测框的最低置信度阈值 |

### 示例：调整训练配置

```bash
# 显存不足时减小 batch size
python src/task3_object_detection.py --batch_size 2 --img_max_size 600

# 关闭 AMP（调试用）
python src/task3_object_detection.py --no_amp

# 增加训练轮数
python src/task3_object_detection.py --epochs 50
```

---

## 训练配置详解

| 配置项 | 取值 | 说明 |
|--------|------|------|
| 模型 | Faster R-CNN | torchvision 内置，two-stage detector |
| Backbone | ResNet-50-FPN | 特征金字塔，5 层（P2-P6） |
| 预训练 | COCO | 81 类 COCO 预训练权重 |
| 输入尺寸 | 可变（max=800） | 保持原图宽高比 |
| 优化器 | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| 学习率策略 | WarmUp + Cosine Annealing | 前 2 epoch 线性 warm-up，之后 cosine 退火至 `1e-6` |
| 损失函数 | RPN + R-CNN Head（内部） | cls: CrossEntropy, reg: Smooth L1 |
| 混合精度 | AMP（GradScaler） | 加速训练，减少显存占用 |
| Batch Size | 4 | RTX 4090 D (24 GB) 满载 |
| 数据增强 | RandomHorizontalFlip + 多尺度 resize | 随机翻转 + resize 到 [480, 800] |
| 随机种子 | 42 | 确保可复现 |
| 总参数量 | ~41.7M | 含 FPN |

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳模型 checkpoint，包含 `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_map50`, `args` |
| `history.json` | JSON 格式逐 epoch 训练历史（train_loss, val_map50, val_map5095, lr） |
| `results.json` | JSON 格式最终评测结果（含 20 类 per-class AP） |
| `training_curves.png` | 训练/验证 Loss、mAP@0.5、mAP@0.5:0.95 三子图曲线 |
| `detection_samples.png` | 8 张测试集样本的检测框可视化（绿色虚线=GT，红色实线=预测） |

---

## 从云服务器下载结果到本地

```bash
# 替换为你的服务器地址和端口
scp -rP 端口 root@服务器地址:~/autodl-tmp/CV_Project/output/task3/ D:\lab\imageSemanticDp\output\task3\
scp -rP 端口 root@服务器地址:~/autodl-tmp/CV_Project/output/task3/inference/ D:\lab\imageSemanticDp\output\task3\inference\
```

下载后本地直接就能推理（不需要数据集），见上方"推理"一节。

---

## 调用模型做推理（Python 示例）

```python
import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
MAX_SIZE = 800

# 1. 构建模型并加载权重
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
ckpt = torch.load("./output/task3/best_model.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# 2. 预处理
image_pil = Image.open("your_photo.jpg").convert("RGB")
w, h = image_pil.size
scale = MAX_SIZE / max(w, h)
if scale < 1.0:
    image_pil = F.resize(image_pil, (int(h * scale), int(w * scale)))
image_tensor = F.to_tensor(image_pil)
image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

# 3. 推理
with torch.no_grad():
    pred = model([image_tensor.to(DEVICE)])[0]

# 4. 解析结果
keep = pred["scores"] > 0.5
boxes = pred["boxes"][keep].cpu().numpy() / scale
scores = pred["scores"][keep].cpu().numpy()
labels = pred["labels"][keep].cpu().numpy()

for box, score, label in zip(boxes, scores, labels):
    cls_name = VOC_CLASSES[int(label) - 1]
    print(f"  {cls_name:>15s} | conf={score:.3f} | box={box}")
```

---

## 预训练模型来源

- Faster R-CNN + FPN 预训练权重：MS COCO，通过 torchvision `weights="DEFAULT"` 自动加载
- Faster R-CNN 论文：[Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015](https://arxiv.org/abs/1506.01497)
- FPN 论文：[Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017](https://arxiv.org/abs/1612.03144)
- torchvision 实现：[https://github.com/pytorch/vision](https://github.com/pytorch/vision)

---

## 常见问题

**Q: 显存不足（OOM）？**

Faster R-CNN 显存消耗较大，尝试：
```bash
python src/task3_object_detection.py --batch_size 2 --img_max_size 600
```

**Q: 数据集下载太慢或失败？**

可能是 VOC 官方服务器网络受限。可以手动下载：
- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (trainval)
- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (test)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (trainval)

解压后放入 `./data/VOCdevkit/` 目录：
```
./data/VOCdevkit/
├── VOC2007/
│   ├── JPEGImages/
│   ├── Annotations/
│   └── ImageSets/
└── VOC2012/
    ├── JPEGImages/
    ├── Annotations/
    └── ImageSets/
```

**Q: 本地推理需要下载数据集吗？**

不需要。`task3_inference.py` 只加载模型权重和你的输入图片。

**Q: 训练时间大概多久？**

RTX 4090 D 上 ~16551 张训练图像、30 epoch、batch_size=4，预计约 8-12 小时（每 epoch 约 15-20 分钟）。

**Q: 为什么训练这么慢？**

Two-stage detector 的特性——RPN 生成 proposals + R-CNN Head 逐 proposal 分类/回归，比 one-stage detector 慢但精度更高。想更快可考虑：
- 增大 batch_size（更多并行）
- 减少 `img_max_size`（减少 backbone FLOPs）
- 使用 one-stage 替代方案（如 YOLOv8）

**Q: 模型文件太大（~165 MB），能压缩吗？**

可只保存 `model_state_dict`，去掉 optimizer state：
```python
torch.save({"model_state_dict": checkpoint["model_state_dict"],
            "val_map50": checkpoint["val_map50"],
            "epoch": checkpoint["epoch"]}, "best_model_slim.pth")
```
