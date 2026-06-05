# Task 3 设计文档：PASCAL VOC 目标检测

## 1. 问题定义

### 1.1 任务目标

对 PASCAL VOC 数据集实现目标检测——为输入图像中的每个目标物体预测**类别标签**和**包围框（Bounding Box）**。本质是一个多任务学习问题：同时解决"物体在哪"（定位回归）和"物体是什么"（分类）。

### 1.2 输入与输出

| | 规格 |
|---|---|
| **输入** | RGB 图像，原始尺寸不一（约 200×200 ~ 500×500 px） |
| **输出** | 一组检测框，每框含：类别标签 c ∈ {1..20}、置信度 s ∈ [0,1]、坐标 (x1, y1, x2, y2) |
| **训练时** | RPN 生成 region proposals，R-CNN head 做分类+回归 |

### 1.3 难点

- **多尺度目标**：同一场景中可能同时存在大目标（如人）和小目标（如鸟、瓶子）
- **密集场景**：部分图像目标密集排布，NMS 后处理对去重至关重要
- **类别不平衡**：`person` 类样本远超 `sheep`、`cow` 等类别
- **遮挡和截断**：VOC 标注中包含 `difficult` 和 `truncated` 标记

---

## 2. 数据集分析

### 2.1 数据来源

PASCAL Visual Object Classes (VOC) Challenge，通过 `torchvision.datasets.VOCDetection` 自动下载。

### 2.2 数据结构

```
VOCdevkit/
├── VOC2007/
│   ├── JPEGImages/       # 9963 张图像
│   ├── Annotations/      # XML 标注文件（同名）
│   ├── ImageSets/Main/   # train/val/test 划分文件
│   └── ...
├── VOC2012/
│   ├── JPEGImages/       # 11540 张图像
│   ├── Annotations/
│   └── ImageSets/Main/
```

### 2.3 类别定义（20 类）

| 类别 | 中文 | 类别 | 中文 |
|------|------|------|------|
| aeroplane | 飞机 | diningtable | 餐桌 |
| bicycle | 自行车 | dog | 狗 |
| bird | 鸟 | horse | 马 |
| boat | 船 | motorbike | 摩托车 |
| bottle | 瓶子 | person | 人 |
| bus | 公交车 | pottedplant | 盆栽 |
| car | 轿车 | sheep | 羊 |
| cat | 猫 | sofa | 沙发 |
| chair | 椅子 | train | 火车 |
| cow | 牛 | tvmonitor | 电视/显示器 |

### 2.4 数据划分

```
VOC2007 trainval (5011 images)  ──┐
                                   ├── Train: ~16551 images
VOC2012 trainval (11540 images) ──┘

VOC2007 test (4952 images) ──────── Val/Test: ~4952 images
```

- 训练集：VOC2007 trainval + VOC2012 trainval（~16551 张），业内标准做法
- 测试集：VOC2007 test（~4952 张），仅用于最终评估
- 标注中的 `difficult=1` 目标在训练时跳过（避免混淆模型），评估时也不计入

### 2.5 XML 标注解析

```xml
<annotation>
    <object>
        <name>dog</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>68</xmin><ymin>248</ymin>
            <xmax>196</xmax><ymax>479</ymax>
        </bndbox>
    </object>
    <!-- 可能有多个 object -->
</annotation>
```

- `voc_target_to_boxes()` 函数完成 XML → `tensor(boxes), tensor(labels)` 的转换
- 坐标格式为 `[x1, y1, x2, y2]`（左上/右下角，pixel 坐标），与 torchvision Faster R-CNN 一致
- Label 为 1-indexed（1-20），0 保留为 background

---

## 3. 模型架构

### 3.1 总体结构：Faster R-CNN + ResNet-50-FPN

```
输入图像 (3, H, W)
       │
       ▼
┌─────────────────────┐
│  ResNet-50 + FPN    │  ← ImageNet / COCO 预训练
│  多尺度特征金字塔    │
│  P2～P6 (stride 4~64)│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Region Proposal    │
│  Network (RPN)      │  ← 在每个特征层级生成 anchors
│  - 分类：obj/bg     │     预测 objectness + bbox 修正
│  - 回归：Δxywh      │
└──────┬──────────────┘
       │ proposals (N个候选区域)
       ▼
┌─────────────────────┐
│  RoI Align          │  ← 将 proposals 映射到特征图并池化至 7×7
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  R-CNN Head         │
│  - FC 分类器 → 21类 │  ← 替换 pretrained 分类头为 VOC 20+1 类
│  - FC 回归器 → 4Δ   │
└──────┬──────────────┘
       │
       ▼
    检测结果
    (boxes, labels, scores)
```

### 3.2 关键模块

**Feature Pyramid Network (FPN)**

- 在 ResNet 各 stage 输出上构建自上而下的特征金字塔（P2, P3, P4, P5, P6）
- 浅层 (P2) 分辨率高、感受野小 → 适合检测小目标（bottle, bird）
- 深层 (P6) 分辨率低、感受野大 → 适合检测大目标（bus, sofa）
- 每个层级独立预测，不同尺度的 anchor 分配到不同层级：
  - P2: 32², P3: 64², P4: 128², P5: 256², P6: 512²

**Region Proposal Network (RPN)**

- 滑动窗口式 anchor-based 提议生成
- 每个位置 k=3 种 aspect ratios (0.5, 1.0, 2.0) × 3 种 scales = 9 anchors
- 输出：objectness score（前景/背景 二分类）+ bbox delta（4 个坐标偏移）
- 训练时采样 256 个 anchors（正负比例 1:1）
- 推理时取 Top-N proposals 送入第二阶段

**RoI Align**

- 将任意大小的 proposal 区域精确映射到固定 7×7 输出
- 双线性插值 + 平均池化，避免 RoI Pooling 的量化误差
- 对检测精度（尤其小目标）有 ~1-2% mAP 提升

### 3.3 为什么选 Faster R-CNN

|  | Faster R-CNN | YOLOv5 | RetinaNet |
|---|---|---|---|
| 检测范式 | Two-stage | One-stage | One-stage |
| mAP@0.5 (VOC) | ~80-82% | ~78-82% | ~78-80% |
| 小目标检测 | 优秀 (two-stage) | 一般 | 较好 (FPN) |
| 训练稳定性 | 稳定 | 需大量数据增强 | 较好 |
| 代码依赖 | torchvision 内置 | 需克隆 ultralytics 仓库 | torchvision 内置 |
| 训练速度 | 较慢（~0.5s/iter） | 快（~0.1s/iter） | 中等 |
| 推理速度 | 较慢（~100ms） | 快（~10ms） | 中等（~50ms） |

选择 Faster R-CNN 的理由：
1. **自包含**：torchvision 内置，无需克隆外部仓库，与项目风格一致
2. **预训练成熟**：COCO 预训练权重可直接加载
3. **精度高**：two-stage 范式对小目标和密集场景更友好
4. **教学意义**：完整展示 RPN → RoI Align → R-CNN Head 的 two-stage 流水线

### 3.4 模型规格

| 属性 | 值 |
|------|-----|
| 总参数量 | ~41.7M |
| Backbone | ResNet-50（ImageNet-1K 预训练） |
| FPN | 5 层金字塔（P2-P6） |
| RPN anchor | 5 scales × 3 ratios = 15 per location |
| Box Predictor | FastRCNNPredictor: 1024 → 21 classes |
| 输入尺寸 | 可变尺寸（resize 至 max=800px，保持宽高比） |
| 输出 | 21 类（20 VOC + background） |

---

## 4. 数据流水线

### 4.1 预处理

```python
# 训练时
1. 随机水平翻转 (p=0.5)，同时翻转 boxes 坐标
2. 随机 resize：max side ∈ [480, 800]（多尺度训练）
3. Normalize(ImageNet μ/σ)
4. 返回 (image_tensor, {"boxes": FloatTensor, "labels": Int64Tensor})

# 验证/推理时
1. Resize：max side = 800（保持宽高比）
2. Normalize(ImageNet μ/σ)
```

### 4.2 数据增强设计

- **随机水平翻转**：物体左右对称，翻转不改变语义，同时镜像 boxes（x1' = w - x2, x2' = w - x1）
- **多尺度训练**：`max_size ∈ [480, 800]` 随机采样，模拟不同距离/焦距下的目标尺度变化
- **不包含 ColorJitter**：Faster R-CNN 训练对颜色扰动敏感（可能影响 RPN objectness 判断），且 COCO 预训练已有足够泛化能力
- **不包含随机裁剪**：可能裁掉标注目标（裁剪后 box 需要截断处理，增加复杂度）

### 4.3 Collate Function

```python
def collate_fn(batch):
    return tuple(zip(*batch))
```

- 不做 padding/stacking，直接返回 `(list[image_tensor], list[target_dict])`
- Faster R-CNN 原生支持 variable-size batch 输入
- 每个 image 在模型内部独立前向（FPN + RPN），仅在 GPU kernel 层面并行

---

## 5. 训练策略

### 5.1 优化器

**AdamW**（解耦权重衰减的 Adam）

| 参数 | 值 |
|------|-----|
| `lr`（峰值） | 1e-3 |
| `weight_decay` | 1e-4 |
| `betas` | (0.9, 0.999)（默认） |

将 backbone 的学习率设为全局 lr 的 0.1 倍是一种常见的 fine-tuning 策略，但本设计使用 **end-to-end 等学习率** 训练，因为：
1. FPN + RPN + R-CNN Head 结构复杂，分层 lr 容易导致某些模块欠拟合
2. 全局 warmup 已经给了 backbone 逐步适应的缓冲

### 5.2 学习率调度

**WarmUp + Cosine Annealing**：

```
lr(epoch) =
  │  (epoch+1)/2 * peak_lr                            epoch < 2 (warmup)
  │  lr_min + 0.5*(peak_lr - lr_min)*[1+cos(π*p)]     epoch ≥ 2 (cosine)
```

| 参数 | 值 |
|------|-----|
| `warmup_epochs` | 2（~7% of total） |
| `peak_lr` | 1e-3 |
| `lr_min` | 1e-6 |

### 5.3 混合精度训练（AMP）

- 前向/反向传播使用 FP16
- RPN 的 objectness logits 和 box regression 在 FP16 下数值稳定
- 预期加速 ~30%，显存节省 ~30-40%

### 5.4 超参数总览

| 参数 | 值 | 选择依据 |
|------|-----|---------|
| `img_max_size` | 800 | VOC 标准配置，与 Faster R-CNN 论文一致 |
| `batch_size` | 4 | RTX 4090 D (24GB) 满载含 FP32 master weights |
| `epochs` | 30 | 观测 mAP 在 20-25 epoch 趋于平台期 |
| `lr` | 1e-3 | AdamW 常见峰值 |
| `warmup_epochs` | 2 | 约 7% 总 epoch |
| `weight_decay` | 1e-4 | 适度正则，防止小数据集过拟合 |
| `SEED` | 42 | 处处固定 |

---

## 6. 损失函数

### 6.1 RPN Loss

$$L_{RPN} = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* \cdot L_{reg}(t_i, t_i^*)$$

- $L_{cls}$：二分类交叉熵（前景 vs 背景）
- $L_{reg}$：Smooth L1 loss（仅对正样本 anchor 计算）
- $\lambda = 1$（torchvision 默认）
- $N_{cls} = 256$（每 batch 采样 anchor 数），$N_{reg} = N_{pos}$

### 6.2 R-CNN Head Loss

$$L_{RCNN} = L_{cls}(c, c^*) + [c^* \geq 1] \cdot L_{reg}(t, t^*)$$

- $L_{cls}$：21 类交叉熵（含 background）
- $L_{reg}$：Smooth L1 loss（仅对非背景 proposal 计算）
- 训练时采样 512 proposals（正负比例 1:3）

### 6.3 总 Loss

$$L_{total} = L_{RPN} + L_{RCNN}$$

torchvision Faster R-CNN 内部自动计算并汇总，`model(images, targets)` 返回包含各子 loss 的 dict。

---

## 7. 评测指标

### 7.1 mAP@0.5（主指标）

**VOC 标准指标**：预测框与真实框 IoU ≥ 0.5 时判定为 True Positive。

$$Precision(t) = \frac{TP(t)}{TP(t) + FP(t)}, \quad Recall(t) = \frac{TP(t)}{\#GT}$$

按置信度降序排序所有预测，逐点计算 precision-recall 对，使用**全点插值法**（VOC 2010+ style）计算 AP：

$$AP = \sum_{k=0}^{K-1} (R_{k+1} - R_k) \cdot \max_{i \geq k+1} P_i$$

$$mAP = \frac{1}{20}\sum_{c=1}^{20} AP_c$$

### 7.2 mAP@0.5:0.95（辅助指标）

在 IoU = [0.5, 0.55, 0.60, ..., 0.95] 共 10 个阈值上分别计算 mAP，取平均。COCO 风格指标，更严格地衡量定位精度。

### 7.3 Precision / Recall

- **Precision**：检测出的目标中有多少是真正的目标（FP 控制）
- **Recall**：真实目标中有多少被检测出来（FN 控制）
- 通过调整 confidence threshold 可以在两者间 trade off

### 7.4 实现细节

- NMS 后处理使用 IoU 阈值 0.5（torchvision 默认）
- 每张图最多保留 100 个检测框
- 评估时 confidence threshold 设为 0.05（低阈值保证召回率，AP 计算会考虑完整 PR 曲线）

---

## 8. 代码模块设计

### 8.1 模块划分

| 模块 | 位置 | 职责 |
|------|------|------|
| 参数解析 | `parse_args()` | CLI 参数 → argparse.Namespace |
| Dataset | `VOCDataset` | VOCDetection wrapper：XML 解析 + 增强 + tensor |
| Collate | `collate_fn()` | 可变尺寸 batch 组装 |
| Model | `build_model()` | Faster R-CNN 工厂，替换分类头 |
| Train | `train_one_epoch()` | 单 epoch 训练（AMP + loss dict 汇总） |
| Eval | `evaluate()` | 单 epoch 推理，收集全部 preds/gts |
| mAP | `compute_map()` | VOC 风格 mAP@IoU 计算 |
| mAP | `compute_voc_ap()` | 单类 AP 计算（全点插值法） |
| Plot | `plot_curves()` | 训练曲线（loss + mAP） |
| Plot | `plot_detection_samples()` | 检测框可视化对比 |
| Main | `main()` | 编排流程 |

### 8.2 关键设计决策

**决策 1：使用 torchvision 内置 Faster R-CNN 而非 YOLO**

理由：自包含、预训练权重成熟、与项目风格一致（Task 1 用 timm、Task 2 用 smp、Task 3 用 torchvision）。代码完全在单个 Python 文件内，无需克隆外部仓库或配置复杂的 YOLO 训练 pipeline。

**决策 2：`collate_fn` 不做 padding**

理由：Faster R-CNN 原生支持 list of tensors 输入，不同尺寸的图像在 FPN 中固有地产生不同尺寸的特征图。若强行 padding 反而引入无用的 pad 区域计算。

**决策 3：手动实现 mAP 而非依赖 torchmetrics**

理由：torchmetrics MeanAveragePrecision API 在不同版本间变化频繁，且计算逻辑不透明。手动实现 VOC 2010+ 全点插值法保证可复现、可调试。

**决策 4：多尺度训练而非固定尺寸**

```python
random_max = np.random.randint(480, self.max_size + 1)  # [480, 800]
```

模拟检测中目标尺度变化，相当于一种数据增强，通常带来 ~1-2% mAP 提升。

### 8.3 可复现性保证

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

注意：`cudnn.deterministic=True` 对 Faster R-CNN 的 RPN NMS（top-k 选择）仍有微量非确定性，但整体浮动 < 0.1% mAP，在可接受范围内。

---

## 9. 预期结果

### 9.1 预期性能

| 指标 | 预期值 | 说明 |
|------|--------|------|
| mAP@0.5 | 78-82% | VOC 主流水平 |
| mAP@0.5:0.95 | 50-55% | COCO 风格，更严格 |

### 9.2 易错类别

| 类别 | 难度 | 原因 |
|------|------|------|
| bottle | 困难 | 小目标、透明、反光 |
| pottedplant | 困难 | 形状不规则、与背景混淆 |
| chair | 中等 | 视角多变、遮挡常见 |
| person | 容易 | 样本多、特征明显 |
| car | 容易 | 形状规整、样本多 |

### 9.3 输出文件

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 最佳 checkpoint（~165 MB，含 optimizer state） |
| `results.json` | JSON 格式实验结果（含 per-class AP） |
| `history.json` | 逐 epoch 训练历史 |
| `training_curves.png` | Loss / mAP@0.5 / mAP@0.5:0.95 三子图 |
| `detection_samples.png` | 8 张测试集预测可视化 |

---

## 10. 推理设计

### 10.1 推理流程

```
输入图像 (任意尺寸)
    │
    ▼
Resize → max_side=800（保持宽高比）
    │
    ▼
Normalize (ImageNet μ/σ)
    │
    ▼
模型前向 → {"boxes", "labels", "scores"}
    │
    ▼
Confidence filter (default > 0.5)
    │
    ▼
Scale boxes → 原始图像尺寸
    │
    ▼
输出：annotated_image.png + 终端检测列表
```

### 10.2 推理脚本功能

- 支持单张 (`--image`) 或批量文件夹 (`--dir`) 模式
- 可调节 confidence threshold (`--conf_thresh`)
- 20 种类别使用不同颜色区分
- 输出带框和标签的可视化图像

---

## 11. 已知限制与改进方向

### 11.1 当前限制

1. **推理速度**：Two-stage detector 推理慢于 one-stage（约 5-10 FPS on RTX 4090），不适合实时场景
2. **小目标召回**：虽然 FPN 已改善多尺度检测，但极小目标（< 16×16 px）仍容易漏检
3. **类别不平衡**：无针对性处理（如 class-aware sampling、focal loss），尾部类别 AP 可能偏低
4. **无 TTA**：测试时未使用多尺度/flip 增强推理

### 11.2 潜在改进

| 方向 | 方法 |
|------|------|
| 提升推理速度 | 换 YOLOv8n/s 或使用 TensorRT 部署 |
| 提升小目标 | 增加多尺度训练范围、使用更大的输入尺寸（max 1333） |
| 类别平衡 | Focal Loss、Class-aware NMS |
| 更高精度 | 换 ResNet-101 backbone、使用 Cascade R-CNN |
| TTA | 推理时多尺度 (600/800/1000) + flip 集成 |
| 模型轻量化 | 换 MobileNetV3 backbone + FPN |
