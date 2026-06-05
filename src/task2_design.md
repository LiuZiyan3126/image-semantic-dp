# Task 2 设计文档：Oxford-IIIT Pet 图像分割

## 1. 问题定义

### 1.1 任务目标

对 Oxford-IIIT Pet 数据集实现像素级图像分割，从输入图像中分离出**宠物区域（前景）**与**非宠物区域（背景）**。本质是一个二分类的 dense prediction 问题——为图像中的每个像素预测其类别标签。

### 1.2 输入与输出

| | 规格 |
|---|---|
| **输入** | RGB 图像，原始尺寸不一（约 200~500 px） |
| **输出** | 与输入同尺寸的二值 mask，0 = 背景，1 = 前景（宠物） |
| **中间张量** | 模型在 384×384 固定分辨率上运算 |

### 1.3 难点

- **边界模糊**：宠物毛发边缘与背景过渡自然，trimap 中有专门的"border"（值=3）标注不确定区域
- **姿态多样性**：猫狗坐、卧、站、跑等姿态变化大
- **品种差异**：37 个品种，短毛/长毛、立耳/垂耳视觉差异显著
- **遮挡**：部分图像中宠物被物体或人手遮挡

---

## 2. 数据集分析

### 2.1 数据来源

Oxford-IIIT Pet Dataset，由 Oxford Visual Geometry Group 发布，通过 `torchvision.datasets.OxfordIIITPet` 自动下载。

### 2.2 数据结构

```
oxford-iiit-pet/
├── images/          # 7349 张 RGB 图像（.jpg）
├── annotations/
│   ├── trimaps/     # 像素级标注（.png，3 值图）
│   ├── xmls/        # 目标框标注（未使用）
│   └── list.txt     # 图像列表 & 品种标签
```

### 2.3 Trimap 标注语义

| 像素值 | 语义 | 占比（约） |
|--------|------|-----------|
| 1 | Foreground（宠物身体区域） | ~30% |
| 2 | Background（背景） | ~60% |
| 3 | Border / Not classified（模糊边界） | ~10% |

### 2.4 标签重映射逻辑

本设计采用**二分类**方案。`SegmentationDataset.__getitem__` 中执行：

```python
# 二分类映射
原始值 1 (FG)      → 类别 1     # 正类
原始值 2 (BG)      → 类别 0     # 负类
原始值 3 (Border)  → 255        # ignore_index，loss 不参与计算
```

**设计理由**：trimap 三分类中 border 类极难学习（边界定义模糊），且实际场景更关心前景/背景的区分。将 border 设为 ignore 可避免模型学习模糊目标、减少噪声梯度。

### 2.5 数据划分

```
trainval (3680 images)
    │
    ├── Train: 2944 (80%)  ← 随机采样，固定 seed=42
    └── Val:   736  (20%)
```

- 原始数据仅提供 `trainval` 和 `test` 两个 split，`test` 无公开标注，故采用 **80/20 随机划分**
- 固定 `np.random.RandomState(42)` 确保每次运行划分一致、结果可复现

---

## 3. 模型架构

### 3.1 总体结构：DeepLabV3Plus + ResNet-50

```
输入 (3, 384, 384)
       │
       ▼
┌─────────────────────┐
│  ResNet-50 Encoder  │  ← ImageNet 预训练（冻结？否，全量 fine-tune）
│  (output stride=16) │
│                     │
│  low-level features │──── 跳连 ──────────────┐
│  (stride 4, 256ch)  │                        │
│                     │                        ▼
│  high-level features│──▶ ASPP ──▶ ┌──────────────┐
│  (stride 16, 2048ch)│              │   Decoder    │──▶ 输出 (2, 384, 384)
└─────────────────────┘              │ Conv+BN+ReLU │
                                     │ 2× Upsample  │
                                     └──────────────┘
```

### 3.2 关键模块

**Atrous Spatial Pyramid Pooling (ASPP)**

- 在 encoder 输出的高维特征图（stride=16）上并行施加多种 dilation rate 的空洞卷积（6, 12, 18）
- 配合 1×1 卷积和全局平均池化分支，捕获多尺度上下文
- 解决普通 CNN 感受野有限、无法建模整体对象轮廓的问题

**Decoder**

- 融合 encoder 浅层特征（stride=4，保留空间细节）与 ASPP 输出（stride=16，含语义信息）
- 浅层特征经 1×1 卷积降维后与 4× 上采样的 ASPP 输出 concat
- 两个 3×3 卷积 + 4× 双线性上采样恢复至输入分辨率

**为什么选 DeepLabV3Plus 而非 U-Net？**

| | U-Net | DeepLabV3Plus |
|---|---|---|
| 感受野 | 渐进式，依赖深度 | ASPP 多尺度并行捕获 |
| 边界精度 | 跳连直接拼接，细节保留好 | 解码器有额外卷积细化 |
| 预训练兼容性 | encoder 可预训练 | encoder 完全预训练友好 |
| 参数量 | 较小（~17M for ResNet-50） | 较大（~27M） |

DeepLabV3Plus 的 ASPP 多尺度上下文对同时存在大、小宠物的场景更有优势。参数量大但 RTX 4090 D (25GB) 足以承载。

### 3.3 模型规格

| 属性 | 值 |
|------|-----|
| 总参数量 | 26,677,842 |
| 可训练参数 | 26,677,842（100%） |
| Encoder | ResNet-50（ImageNet-1K 预训练） |
| Decoder + ASPP | 随机初始化 |
| 输入通道 | 3 (RGB) |
| 输出通道 | 2 (背景, 前景) |

---

## 4. 数据流水线

### 4.1 训练增强

```python
A.Compose([
    RandomResizedCrop(384, scale=(0.5, 1.0)),  # 随机裁剪+缩放
    HorizontalFlip(p=0.5),                       # 水平翻转
    RandomBrightnessContrast(brightness=0.2,      # 亮度对比度扰动
                             contrast=0.2, p=0.3),
    Normalize(ImageNet μ/σ),                     # 标准化
    ToTensorV2(),                                # HWC→CHW, uint8→float32
])
```

**设计考虑**：
- `RandomResizedCrop scale=(0.5, 1.0)`：下界 0.5 而非更激进的值（如 0.3），因为分割任务中宠物主体占比变化不宜过大
- `HorizontalFlip`：宠物左右对称，翻转不会改变语义
- 不包含 `ShiftScaleRotate` / `ElasticTransform`：宠物器官（眼、鼻）位置对分割很重要，仿射变形可能导致训练信号混乱
- 不包含颜色过强扰动：品种识别依赖毛色特征

### 4.2 验证/推理预处理

```python
A.Compose([
    Resize(384, 384),              # 固定尺寸重采样
    Normalize(ImageNet μ/σ),
    ToTensorV2(),
])
```

验证时不裁剪，仅等比例缩放至 384×384。推理时额外需要将输出 mask 双线性/最近邻 resize 回原始图像尺寸。

### 4.3 标准化参数

使用 ImageNet 统计量（`μ=[0.485, 0.456, 0.406]`, `σ=[0.229, 0.224, 0.225]`），与 encoder 预训练时的数据分布一致。

### 4.4 Mask 重采样方法

albumentations 默认对 mask 使用 **最近邻插值（NEAREST）**，确保 resize 后像素值仍为离散标签（0/1/255），不会引入浮点误差。

---

## 5. 训练策略

### 5.1 优化器

**AdamW**（解耦权重衰减的 Adam）

| 参数 | 值 |
|------|-----|
| `lr`（峰值） | 1e-3 |
| `weight_decay` | 1e-4 |
| `betas` | (0.9, 0.999)（默认） |

选 AdamW 而非 SGD 的原因：DeepLabV3Plus decoder 部分为随机初始化，Adam 的自适应学习率对新初始化层收敛更快。

### 5.2 学习率调度

**WarmUp + Cosine Annealing**：

```
lr(epoch) =
  │  (epoch+1)/3 * peak_lr                           epoch < 3 (warmup)
  │  lr_min + 0.5*(peak_lr - lr_min)*[1+cos(π*p)]    epoch ≥ 3 (cosine decay)
  │  其中 p = (epoch-3)/(total_epochs-3)
```

| 参数 | 值 |
|------|-----|
| `warmup_epochs` | 3 |
| `peak_lr` | 1e-3 |
| `lr_min` | 1e-6（衰减 1000×） |

**设计理由**：
- WarmUp：前 3 epoch 学习率从 0 线性增长到 1e-3，避免随机初始化的 decoder 梯度爆炸
- Cosine：平滑衰减至极小值，训练后期参数稳定在一个狭窄盆地区域，不跳出局部最优

### 5.3 混合精度训练（AMP）

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    loss = criterion(model(images), masks)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

- 前向/反向传播使用 FP16，减少 ~40% 显存占用和 ~30% 训练时间
- `GradScaler` 自动处理梯度下溢（scale→unscale 回 FP32）
- 仅在 CUDA 设备上启用

### 5.4 超参数总览

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| `img_size` | 384 | 平衡细节与显存（384²=147K 像素，512²=262K） |
| `batch_size` | 16 | RTX 4090 D (25GB) 满载 |
| `epochs` | 80 | 观测到 mIoU 在 70+ epoch 趋于平稳 |
| `lr` | 1e-3 | AdamW 常见峰值 |
| `warmup_epochs` | 3 | 约占总 epoch 的 3.75% |
| `weight_decay` | 1e-4 | 适度正则化 |
| `SEED` | 42 | 处处固定，保证可复现 |

---

## 6. 损失函数

### 6.1 默认：CrossEntropyLoss with Ignore Index

```python
nn.CrossEntropyLoss(ignore_index=255)
```

- 标准多分类交叉熵，PyTorch 内置数值稳定性
- `ignore_index=255`：border 区域（标签=255）不参与梯度计算
- 二分类输出为 (B, 2, H, W) logits，由 CrossEntropyLoss 内部做 softmax

### 6.2 可选方案

| loss_type | 实现 | 适用场景 |
|-----------|------|---------|
| `"ce"` | `CrossEntropyLoss(ignore=255)` | 默认，稳定 |
| `"dice"` | `smp.losses.DiceLoss(mode="multiclass")` | 前景/背景不平衡时 |
| `"combined"` | `CE + Dice` 等权相加 | 追求极致 Dice |

实验中默认 `ce` 已取得 0.9817 的 Dice 分数，说明 Oxford Pet 数据集的类别不平衡不影响学习效果（前景/背景像素比约 3:7，尚可接受）。

---

## 7. 评测指标

### 7.1 标识定义

| 符号 | 含义 |
|------|------|
| TP | 预测为前景 & 标注为前景 |
| FP | 预测为前景 & 标注为背景 |
| FN | 预测为背景 & 标注为前景 |
| TN | 预测为背景 & 标注为背景 |

### 7.2 指标计算

**mIoU（mean Intersection over Union）** — 主指标

$$IoU_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|} = \frac{TP_c}{TP_c + FP_c + FN_c}$$

$$mIoU = \frac{1}{C}\sum_{c=0}^{C-1} IoU_c$$

- 对每个类别分别计算 IoU 后取平均
- 二分类时 mIoU = (IoU_bg + IoU_fg) / 2

**Foreground IoU**

$$FG\ IoU = \frac{TP}{TP + FP + FN}$$

- 仅在报告中使用，强调模型对宠物主体的分割能力
- 注意：bg IoU 通常极高（大面积易分割），会拉高 mIoU

**Dice Coefficient（F1-score per pixel）**

$$Dice = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

- 与 IoU 正相关（$Dice = \frac{2 \cdot IoU}{1 + IoU}$）
- 对前景更敏感，常用于医学图像 / 分割比赛

**Pixel Accuracy**

$$PA = \frac{TP + TN}{TP + TN + FP + FN}$$

- 所有有效像素（排除 ignore）中的正确率
- 受背景像素占比影响大，仅作参考

### 7.3 实现细节

```python
valid = masks != 255                                    # 排除 border
for c in range(num_classes):
    pred_c = (preds == c) & valid
    target_c = (masks == c) & valid
    intersection[c] += (pred_c & target_c).sum()
    union[c] += (pred_c | target_c).sum()

ious = intersection / (union + 1e-6)                    # 防除零
miou = ious.mean().item()
```

- 逐 batch 累计 intersection/union，最后统一计算 IoU，而非逐 batch 平均（后者在 batch 间样本数不均衡时有偏）
- `1e-6` 平滑项防止某类别从未出现时除零

---

## 8. 代码模块设计

### 8.1 模块划分

| 模块 | 位置 | 职责 |
|------|------|------|
| 参数解析 | `parse_args()` | CLI 参数 → argparse.Namespace |
| Dataset | `SegmentationDataset` | 原始数据 → 标签映射 + 增强 → tensor |
| Transform | `get_transforms()` | albumentations Compose 工厂 |
| Data 组装 | `build_datasets()` | 数据集创建 + 划分 + DataLoader |
| Model | `build_model()` | smp.create_model 工厂 |
| Loss | `build_loss()` | 根据 loss_type 返回损失函数 |
| Train | `train_one_epoch()` | 单 epoch 训练（AMP 支持） |
| Validate | `validate()` | 单 epoch 评估，返回全部指标 |
| Plot | `plot_curves()` | 三子图训练曲线 |
| Plot | `plot_segmentation_samples()` | 8 样本对比可视化 |
| Main | `main()` | 编排流程 |

### 8.2 关键设计决策

**决策 1：自定义 Dataset 而非直接使用 `OxfordIIITPet`**

原因：需要在 `__getitem__` 中做 trimap→label 重映射、集成 albumentations 增强，且支持按索引子集划分。

**决策 2：Checkpoint 同时保存 optimizer state**

原因：支持中断恢复训练（`optimizer_state_dict` 包含 AdamW 的 momentum/variance），代价是 checkpoint 体积翻倍（~106 MB → 若去掉 optimizer 可压缩至 ~50 MB）。

**决策 3：`set_to_none=True` on zero_grad**

```python
optimizer.zero_grad(set_to_none=True)
```

将梯度 tensor 设为 `None` 而非填充零值，减少显存碎片化和内存带宽浪费。

**决策 4：`drop_last=True` 仅训练集**

验证集保留最后一个不完整 batch，避免丢弃评估样本。

### 8.3 可复现性保证

```python
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- `deterministic=True`：强制使用确定性算法（代价约 5-10% 速度）
- `benchmark=False`：禁止 cuDNN 自适应选择最优 kernel（防止因形状不同走不同 kernel 路径导致非确定性）

---

## 9. 实验结果

### 9.1 最终性能

| 指标 | 数值 | Epoch |
|------|------|-------|
| **mIoU** | **0.9721** | 77 |
| **Foreground IoU** | **0.9640** | 77 |
| **Dice (FG)** | **0.9817** | 77 |
| **Pixel Accuracy** | **0.9871** | 77 |
| **Val Loss** | **0.0531** | 77 |

### 9.2 训练收敛分析

```
Epoch  1:  mIoU 0.9351  ← 初始快速上升（encoder 预训练生效）
Epoch 11:  mIoU 0.9414  ← 稳步震荡上升
Epoch 29:  mIoU 0.9572  ← 突破 0.95
Epoch 43:  mIoU 0.9640  ← 突破 0.96
Epoch 66:  mIoU 0.9713  ← 突破 0.97
Epoch 77:  mIoU 0.9721  ← 最佳点
Epoch 80:  mIoU 0.9713  ← 末期稳定在 0.971 附近
```

- 整个训练过程 **无过拟合迹象**（val loss 与 train loss 同步下降）
- 最佳 epoch 在训练末期（77/80），说明 80 epoch 设置刚好
- mIoU 从 epoch 60 开始进入 ~0.97 平台期，后 20 epoch 波动 < 0.002

### 9.3 输出文件

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳 checkpoint（~106 MB） |
| `results.json` | JSON 格式实验结果 |
| `training_curves.png` | Loss / IoU / Dice+PixelAcc 三子图 |
| `segmentation_samples.png` | 8 张验证集样本对比 |

---

## 10. 推理设计

### 10.1 推理流程

```
输入图像 (任意尺寸)
    │
    ▼
Resize → (384, 384)         ← albumentations
    │
    ▼
Normalize (ImageNet μ/σ)
    │
    ▼
模型前向 → logits (1, 2, 384, 384)
    │
    ▼
argmax → mask (384, 384)    ← 0/1 二值图
    │
    ▼
Resize → 原始图像尺寸        ← NEAREST 插值
    │
    ▼
输出：mask.png + overlay.png
```

### 10.2 本地推理可行性

- 模型 26.7M 参数、输入 384×384，CPU 上一次前向传播约 1-3 秒
- 不依赖数据集、不依赖 GPU
- 支持单张 (`--image`) 或批量文件夹 (`--dir`)

---

## 11. 已知限制与改进方向

### 11.1 当前限制

1. **二分类简化**：将 border 设为 ignore 损失了三类分割的能力，无法直接输出 trimap
2. **固定输入尺寸**：384×384 对小目标（如远处的小猫）可能分辨率不足
3. **不考虑品种信息**：未利用 37 类品种标签，可做多任务学习提升分割精度
4. **数据增强保守**：未使用 CutMix/MixUp 等分割特定的混合增强

### 11.2 潜在改进

| 方向 | 方法 |
|------|------|
| 提升边界精度 | 加 boundary-aware loss（如 Hausdorff distance） |
| 多尺度推理 | TTA（Test-Time Augmentation）：多尺度 + flip 集成 |
| 更轻量化 | 换 MobileNetV3 encoder 用于移动端部署 |
| 三分类 | 训练 `num_classes=3` 版本，用 Dice + Focal Loss |
