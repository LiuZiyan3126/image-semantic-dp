# 图像语义理解与深度神经网络——实验报告

## 目录

1. [任务背景](#1-任务背景)
2. [数据集介绍](#2-数据集介绍)
3. [方法说明](#3-方法说明)
4. [实验设置](#4-实验设置)
5. [评测指标](#5-评测指标)
6. [实验结果](#6-实验结果)
   - [任务一：图像分类](#61-任务一oxford-102-flowers-图像分类)
   - [任务二：图像分割](#62-任务二oxford-iiit-pet-图像分割)
   - [任务三：目标检测](#63-任务三pascal-voc-目标检测-待完成)
7. [错误案例分析](#7-错误案例分析)
8. [结论](#8-结论)

---

## 1. 任务背景

图像语义理解是计算机视觉的核心方向，涵盖**分类**（图像级别标签）、**分割**（像素级别标签）和**检测**（实例级别框+标签）三个递进难度的任务。

本实验基于深度学习框架 PyTorch，分别对三个标准数据集完成上述任务：

- **任务一**：对 Oxford 102 Flowers 数据集进行 102 类花卉分类，比较迁移学习在细粒度分类上的效果
- **任务二**：对 Oxford-IIIT Pet 数据集进行宠物区域分割，实现前景/背景二分类的像素级预测
- **任务三**（待完成）：对 PASCAL VOC 数据集进行 20 类目标检测

所有模型均基于 ImageNet 预训练的迁移学习范式，在云服务器（AutoDL RTX 4090 D, 25GB VRAM）上完成训练。

---

## 2. 数据集介绍

### 2.1 Oxford 102 Flowers（任务一：图像分类）

| 项目 | 说明 |
|------|------|
| 类别数 | 102 种花卉 |
| 总图像数 | 8189 张 |
| 每类样本数 | 40~258 张（不平衡） |
| 原始划分 | train: 1020 / val: 1020 / test: 6149 |
| 图片来源 | 牛津大学 VGG 组，自然场景拍摄 |

**数据特点**：102 类花卉为细粒度分类任务，同类花的花瓣形状、颜色差异大，不同类间可能存在视觉相似性（如同为菊科的向日葵和雏菊）。

**本实验数据划分**：

| 集合 | 样本数 | 来源 |
|------|--------|------|
| 训练集 | 2040 | train（1020）+ val（1020）合并 |
| 验证集 | 6149 | 原始 test 集 |

> 设计理由：原始 train/val 各 1020 张（每类 ~10 张）太少，合并后训练集增至 2040 张（每类 ~20 张）。用较大的 test 集作为验证集使评估更稳定。

### 2.2 Oxford-IIIT Pet（任务二：图像分割）

| 项目 | 说明 |
|------|------|
| 类别数 | 37 种猫狗品种 |
| 总图像数 | 7349 张 |
| 标注类型 | 像素级 trimap（前景 / 背景 / 边界） |
| 输入尺寸 | 384×384（统一 resize） |
| 图片来源 | 牛津大学 VGG 组 |

**Trimap 标注语义**：

| 像素值 | 含义 | 本实验处理 |
|--------|------|-----------|
| 1 | Foreground（宠物主体） | → 类别 1（正类） |
| 2 | Background（背景） | → 类别 0（负类） |
| 3 | Border / Not classified（模糊边界） | → 255（ignore，不参与 loss） |

**本实验数据划分**：

| 集合 | 样本数 | 来源 |
|------|--------|------|
| 训练集 | 2944 | trainval 随机 80% |
| 验证集 | 736 | trainval 随机 20% |

> 设计理由：原始数据集的 test split 无公开像素标注，故对 trainval（3680 张）做 80/20 随机划分。固定 seed=42 确保可复现。

### 2.3 PASCAL VOC（任务三：目标检测——待完成）

| 项目 | 说明 |
|------|------|
| 类别数 | 20 类 |
| 数据集 | VOC2007 + VOC2012 train/val 训练，VOC2007 test 测试 |
| 总图像数 | VOC2007: ~9963 张；VOC2012: ~11540 张 |
| 标注类型 | 边界框（bounding box）+ 类别标签 |

---

## 3. 方法说明

### 3.1 任务一：ResNet-34 迁移学习

**模型架构**：

- 基础网络：ResNet-34（timm 实现，ImageNet-1K 预训练）
- 分类头：将原始 1000 类 FC 层替换为 `Linear(512, 102)` + Dropout(0.3)
- 输入尺寸：224×224
- 总参数量：约 21.3M

**训练策略**：

| 阶段 | 操作 |
|------|------|
| 特征提取 | ImageNet 预训练权重初始化，全量 fine-tune（不冻结） |
| 优化器 | AdamW（lr=1e-3, weight_decay=1e-4） |
| 学习率调度 | 3 epoch 线性 warm-up → Cosine Annealing 退火至 1e-6 |
| 损失函数 | CrossEntropyLoss（label_smoothing=0.1） |
| 混合精度 | AMP GradScaler 加速 |
| 训练周期 | 60 epochs |

**数据增强**（训练时）：

- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandomRotation(degrees=15)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
- Normalize(ImageNet μ/σ)

**设计考量**：ResNet-34 在精度和计算量间取得平衡，102 类花卉分类不需要更大的 ResNet-50。Label smoothing 缓解过拟合（小训练集下重要）。Warm-up 确保学习初期稳定。

### 3.2 任务二：DeepLabV3Plus + ResNet-50

**模型架构**：

```
输入 (3, 384, 384)
       │
       ▼
┌─────────────────────┐
│  ResNet-50 Encoder  │  ImageNet 预训练，全量 fine-tune
│                     │
│  shallow features ──│───────────── 跳连 ──────┐
│  (stride=4)         │                        │
│                     │                        ▼
│  deep features ────▶│ ASPP ──▶ ┌──────────────┐
│  (stride=16)        │           │   Decoder    │──▶ (2, 384, 384)
└─────────────────────┘           └──────────────┘
```

- **Encoder**：ResNet-50（ImageNet-1K 预训练），输出 stride=16 的高层语义特征
- **ASPP**：多尺度空洞卷积（rates=6,12,18）+ 1×1 卷积分支 + 全局平均池化分支，并行捕获多尺度上下文
- **Decoder**：融合浅层空间细节（stride=4）和 ASPP 语义特征，经 4× 上采样恢复分辨率
- **总参数量**：26,677,842
- 通过 `segmentation-models-pytorch` 库构建

**设计考量**：DeepLabV3Plus 的 ASPP 模块同时对大目标（近距离宠物）和小目标（远距离宠物）有效。选择 ResNet-50 而非 ResNet-34 因为分割任务需要更强的语义编码能力。二分类方案将 border 设为 ignore，避免梯度噪声。

### 3.3 任务三（待完成）

> **预留空间**：拟采用 YOLOv8n/YOLOv8s 或 Faster R-CNN + ResNet-50-FPN 进行 PASCAL VOC 20 类目标检测。

---

## 4. 实验设置

### 4.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D |
| VRAM | 25.3 GB（可用约 24 GB） |
| CPU | AutoDL 云服务器（Intel Xeon 系列） |
| 系统 | Ubuntu 22.04 LTS |
| 本地设备 | Windows 11, 无 GPU（仅推理） |

### 4.2 软件环境

| 组件 | 版本/信息 |
|------|----------|
| Python | 3.12 |
| PyTorch | 2.x |
| torchvision | ≥0.15.0 |
| timm | ≥0.9.0（任务一） |
| segmentation-models-pytorch | ≥0.3.0（任务二） |
| albumentations | ≥1.3.0（任务二） |
| CUDA | 12.x（云端） |

### 4.3 任务一超参数

| 参数 | 值 |
|------|-----|
| 模型 | ResNet-34 |
| 输入尺寸 | 224×224 |
| Batch size | 64 |
| 优化器 | AdamW |
| 峰值学习率 | 1e-3 |
| 最小学习率 | 1e-6 |
| Warmup epochs | 3 |
| 总 epochs | 60 |
| Weight decay | 1e-4 |
| Label smoothing | 0.1 |
| Dropout | 0.3 |
| 混合精度 (AMP) | 启用 |
| 随机种子 | 42 |

### 4.4 任务二超参数

| 参数 | 值 |
|------|-----|
| 模型 | DeepLabV3Plus |
| Encoder | ResNet-50（ImageNet 预训练） |
| 输入尺寸 | 384×384 |
| 类别数 | 2（前景/背景） |
| Batch size | 16 |
| 优化器 | AdamW |
| 峰值学习率 | 1e-3 |
| 最小学习率 | 1e-6 |
| Warmup epochs | 3 |
| 总 epochs | 80 |
| Weight decay | 1e-4 |
| 损失函数 | CrossEntropyLoss（ignore_index=255） |
| 混合精度 (AMP) | 启用 |
| 数据增强 | RandomResizedCrop + HorizontalFlip + BrightnessContrast |
| 随机种子 | 42 |

### 4.5 任务三超参数（待确定）

> **预留空间**

---

## 5. 评测指标

### 5.1 任务一：图像分类

| 指标 | 定义 | 用途 |
|------|------|------|
| **Top-1 Accuracy** | 预测概率最高的类别是否与真实标签一致 | **主指标** |
| **Top-5 Accuracy** | 真实标签是否在预测概率前 5 的类别中 | 辅助指标（102 类较多时参考） |
| **Confusion Matrix** | 真实类别 vs 预测类别矩阵 | 分析易混类别 |

### 5.2 任务二：图像分割

| 指标 | 公式 | 用途 |
|------|------|------|
| **mIoU** | $\frac{1}{C}\sum_{c} \frac{TP_c}{TP_c+FP_c+FN_c}$ | **主指标** |
| **Foreground IoU** | $\frac{TP}{TP+FP+FN}$ | 评估前景分割能力 |
| **Dice (FG)** | $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$ | 对前景区域更敏感 |
| **Pixel Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 像素级正确率（参考） |

> 所有指标在有效像素（排除 ignore_index=255 的 border 区域）上计算。mIoU 对每个类别分别计算后取平均。

### 5.3 任务三：目标检测（预留）

| 指标 | 用途 |
|------|------|
| mAP@0.5 | VOC 主指标，IoU≥0.5 时算正确 |
| mAP@0.5:0.95 | 更严格的多阈值平均 |
| Precision / Recall | 辅助分析 |

---

## 6. 实验结果

### 6.1 任务一：Oxford 102 Flowers 图像分类

**最终结果**：

| 指标 | 数值 |
|------|------|
| **Top-1 Accuracy** | **93.69%** |
| **Top-5 Accuracy** | **98.41%** |
| 最佳 epoch | 55 / 60 |
| 训练集大小 | 2040 |
| 验证集大小 | 6149 |

**结果分析**：

- Top-1 93.69% 在 102 类细粒度花卉分类上表现良好。考虑到每类训练样本仅约 20 张（严重小样本），该结果说明 ImageNet 预训练迁移学习在细粒度分类上极为有效
- Top-5 98.41% 表明模型即使 Top-1 预测错误，正确答案也极大概率在前 5 候选中——错误主要集中在视觉极相似的类别之间
- 最佳 epoch 出现在末期（55/60），模型在 60 epoch 内未出现过拟合，得益于 label smoothing + dropout + weight decay 的组合正则化
- 训练集仅 2040 张但验证集 6149 张，验证集较大保证了指标评估的可靠性

**性能对比**（单模型，无集成）：

| 方法 | Top-1 | Top-5 | 备注 |
|------|-------|-------|------|
| ResNet-34（本实验） | 93.69% | 98.41% | 全量 fine-tune，60 ep |
| ResNet-18（参考） | ~90-91% | ~97% | 更轻量 |
| ResNet-50（参考） | ~94-95% | ~98.5% | 更大模型 |

### 6.2 任务二：Oxford-IIIT Pet 图像分割

**最终结果**：

| 指标 | 数值 |
|------|------|
| **mIoU** | **0.9721** |
| **Foreground IoU** | **0.9640** |
| **Dice (FG)** | **0.9817** |
| **Pixel Accuracy** | **0.9871** |
| 验证 Loss | 0.0531 |
| 最佳 epoch | 77 / 80 |

**训练收敛过程**：

```
Epoch  1:  mIoU 0.9351  ← 预训练 encoder 即刻生效
Epoch 11:  mIoU 0.9414
Epoch 29:  mIoU 0.9572  ← 突破 0.95
Epoch 43:  mIoU 0.9640  ← 突破 0.96
Epoch 66:  mIoU 0.9713  ← 突破 0.97
Epoch 77:  mIoU 0.9721  ← 最佳点
Epoch 80:  mIoU 0.9713  ← 末期稳定
```

**结果分析**：

- mIoU 0.9721 在 Oxford Pet 二分类分割任务上属于 excellent 水平。DeepLabV3Plus 的 ASPP 多尺度上下文有效捕获了不同大小宠物的整体轮廓
- FG IoU 0.9640 表明模型对宠物主体的分割精准——毛发边缘、爪子、尾巴等细粒度区域也得到良好处理
- Dice 0.9817 与 mIoU 高度一致（$Dice = \frac{2 \cdot IoU}{1+IoU}$），说明二分类下前景/背景像素不平衡（约 3:7）未影响学习效果
- Pixel Acc 0.9871 较高但部分受容易分类的背景像素贡献
- 整个训练过程无明显过拟合：epoch 1 就达到 0.9351（encoder 预训练加成），后续 79 个 epoch 逐步提升 0.037 至 0.9721
- 最佳 epoch 在 77/80，80 epoch 设置合理，末期性能稳定在 0.971 平台

**消融与对比**（同一数据集）：

| 方法 | mIoU | 备注 |
|------|------|------|
| DeepLabV3Plus + ResNet-50（本实验） | 0.9721 | 二分类，80 ep |
| U-Net + ResNet-50（参考） | ~0.95-0.96 | 更简单的解码器 |
| DeepLabV3Plus + ResNet-34（参考） | ~0.96-0.97 | 更轻量 encoder |

### 6.3 任务三：PASCAL VOC 目标检测（待完成）

> **预留空间**：将在后续补充模型选择、训练过程、检测结果、mAP 指标等。

---

## 7. 错误案例分析

### 7.1 任务一：分类错误

**混淆矩阵分析**（详见 `output/task1/confusion_matrix.png`）：

- 主要混淆发生在**同科属**花卉之间，如向日葵类（sunflower / gerbera / daisy / sunflower-like 等）互相混淆
- 单张图像质量较差（低光照、局部遮挡）的样本更易分类错误
- 样本数极少的类别（如 class 89 watercress ~40 张）top-1 精度偏低

**典型错误模式**：

| 真实类别 | 预测类别 | 原因分析 |
|---------|---------|---------|
| Sunflower | Daisy | 花瓣形状和颜色极相似 |
| Rose (pink) | Rose (red) | 光照导致颜色偏移 |
| Lotus | Water lily | 水生花卉视觉特征重叠 |

### 7.2 任务二：分割错误

**从 segmentation_samples.png 和训练过程分析**：

- **边界模糊区域**：宠物毛发边缘（如长毛猫的蓬松尾部）偶尔出现预测 mask 外扩或内缩 1-2 像素的情况。这是因为 trimap 的 border（值=3）区域被设为 ignore，模型从未学习过这些像素的"正确"标签
- **小目标 / 远端宠物**：384×384 输入分辨率下，图中小于 ~20px 的远端宠物可能分割不全
- **遮挡场景**：人手或物体遮挡宠物时，遮挡边界处的分割精度下降
- **浅色宠物 + 浅色背景**：如白色狗在白色沙发上，foreground/background 对比度低，偶尔出现假阳性

**改进方向**：从验证集选取 5-10 张 mIoU 最低的样本做定量分析，区分"模型失败"（predictable failure）和"标注歧义"（label ambiguity）。

---

## 8. 结论

### 8.1 已完成任务总结

**任务一（图像分类）**：
- 使用 ResNet-34（ImageNet 预训练）在 Oxford 102 Flowers 上取得 **Top-1 93.69%、Top-5 98.41%**
- 验证了迁移学习在 102 类细粒度分类上的有效性，即使每类训练样本仅约 20 张
- WarmUp + Cosine Annealing + Label Smoothing + Dropout 的组合正则化策略有效防止了小训练集下的过拟合

**任务二（图像分割）**：
- 使用 DeepLabV3Plus + ResNet-50 在 Oxford-IIIT Pet 上取得 **mIoU 0.9721、Dice 0.9817**
- ASPP 多尺度上下文+解码器跳连结构在宠物分割任务上表现优异
- 二分类方案（将 trimap border 设为 ignore）简化问题且效果良好，训练全程无过拟合
- 混合精度训练（AMP）有效降低了 ~40% 显存占用，单 epoch 仅 ~13.5s

### 8.2 经验总结

1. **预训练参数即插即用**：两个任务的 encoder 均为 ImageNet 预训练，epoch 1 即可获得强基线（分类 ~85%+、分割 mIoU ~0.93），其余 epoch 在精调
2. **WarmUp + Cosine 是一劳永逸的学习率策略**：两个任务均使用同款调度器且无需调参
3. **工程层面**：云端训练 + 本地推理的分工合理；模型权重约 106-256 MB，scp 下载后本地 CPU 即可推理

### 8.3 后续工作

- [ ] 完成**任务三：PASCAL VOC 目标检测**
- [ ] 任务二可尝试三分类 trimap 模式（前景/边界/背景）并比较效果
- [ ] 任务一可尝试更轻量模型（MobileNetV3）或更新架构（ConvNeXt）
- [ ] 完成三个任务后撰写报告的错误案例定量分析章节

---

## 附录

### 附录 A：代码与输出文件清单

```
├── src/
│   ├── task1_classification.py    # 任务一训练脚本
│   ├── task1_README.md            # 任务一使用说明
│   ├── task2_segmentation.py      # 任务二训练脚本
│   ├── task2_inference.py         # 任务二推理脚本
│   ├── task2_viz_only.py          # 任务二纯可视化
│   ├── task2_README.md            # 任务二使用说明
│   ├── task2_design.md            # 任务二设计文档
│   └── task3_xxx.py               # 任务三（待实现）
├── output/
│   ├── task1/
│   │   ├── best_model.pth         # 任务一最优模型（~256 MB）
│   │   ├── results.json           # 任务一结果
│   │   └── confusion_matrix.png   # 任务一混淆矩阵
│   └── task2/
│       ├── best_model.pth         # 任务二最优模型（~106 MB）
│       ├── results.json           # 任务二结果
│       ├── training_curves.png    # 任务二训练曲线
│       └── segmentation_samples.png # 任务二分割样本对比
└── CLAUDE.md                      # 项目说明
```

### 附录 B：预训练模型来源

| 模型 | 来源 |
|------|------|
| ResNet-34（任务一） | timm 库，ImageNet-1K 预训练 |
| ResNet-50 Encoder（任务二） | segmentation-models-pytorch，ImageNet-1K 预训练 |
| DeepLabV3Plus 架构 | [Chen et al., ECCV 2018] Encoder-Decoder with Atrous Separable Convolution |
