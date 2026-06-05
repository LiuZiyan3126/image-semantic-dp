# Task 3 交付清单与部署指南

## 1. 文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `src/task3_object_detection.py` | 26 KB | **训练主脚本** — Faster R-CNN + ResNet-50-FPN |
| `src/task3_inference.py` | 6.8 KB | **推理脚本** — 单张/批量图片检测+可视化 |
| `src/task3_design.md` | 16 KB | **设计文档** — 架构、损失函数、指标、模块设计 |
| `src/task3_README.md` | 11 KB | **使用说明** — 环境、参数、示例、FAQ |
| `doc/task3_report.md` | — | **实验报告** — 任务说明、推理效果、功能介绍 |

## 2. 技术选型

| 项目 | 选型与说明 |
|------|-----------|
| **模型** | Faster R-CNN + ResNet-50-FPN（torchvision 内置，COCO 预训练） |
| **数据集** | PASCAL VOC 2007+2012 trainval（训练，~16551 张），VOC2007 test（测试，~4952 张） |
| **主指标** | mAP@0.5（VOC 标准） |
| **辅助指标** | mAP@0.5:0.95、Per-class AP |
| **优化器** | AdamW，lr=1e-3，weight_decay=1e-4 |
| **调度器** | WarmUp (2 epoch) + Cosine Annealing → lr_min=1e-6 |
| **训练加速** | AMP 混合精度（FP16），GradScaler |
| **数据增强** | 随机水平翻转 + 多尺度 resize (480~800px) |
| **与 Task 1/2 一致** | CLI 参数化、相对路径、checkpoint 保存、自动可视化 |

## 3. 云服务器部署流程

### 3.1 第一步：创建 AutoDL 实例

1. 登录 AutoDL，选择 GPU 实例（推荐 RTX 4090 D，24 GB VRAM）
2. 镜像选择：**PyTorch 2.x + CUDA 12.x**（或任何预装 PyTorch ≥ 2.0 的镜像）
3. 创建完成后记下 **SSH 登录命令**（含 IP 和端口）

### 3.2 第二步：上传项目到服务器

**方式 A — SCP 上传（推荐）**：

在本地 PowerShell 终端执行：

```bash
# 替换 <SSH端口> 和 <服务器IP> 为实际值
scp -rP <SSH端口> D:\lab\imageSemanticDp\ root@<服务器IP>:~/autodl-tmp/
```

**方式 B — Git 克隆**（如果有远程仓库）：

在服务器上执行：

```bash
cd ~/autodl-tmp
git clone <你的仓库地址>
```

### 3.3 第三步：SSH 登录并安装依赖

```bash
# SSH 登录（使用 AutoDL 提供的命令）
ssh -p <SSH端口> root@<服务器IP>

# 进入项目目录
cd ~/autodl-tmp/CV_Project

# 安装额外依赖（PyTorch 通常已预装）
pip install numpy matplotlib Pillow

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 预期输出: True / NVIDIA GeForce RTX 4090 D
```

### 3.4 第四步：启动训练

```bash
# 使用 tmux 防止 SSH 断开导致训练中断
tmux new -s task3

# 启动训练（默认参数）
python src/task3_object_detection.py

# ---- tmux 操作 ----
# 按 Ctrl+B 然后按 D  →  分离会话（训练在后台继续）
# tmux attach -t task3 →  重新连接查看进度
# tmux ls              →  列出所有会话
```

训练过程中脚本自动执行：

1. 下载 PASCAL VOC 2007 + 2012 到 `./data/`（约 2 GB，首次需几分钟）
2. 加载 COCO 预训练权重（首次自动下载到 torch cache）
3. 训练 30 个 epoch，每个 epoch 输出 loss、mAP@0.5、mAP@0.5:0.95
4. mAP 最高时自动保存 checkpoint 到 `./output/task3/best_model.pth`
5. 训练结束后自动生成曲线图和检测样本图

### 3.5 第五步：监控训练

```bash
# 查看 GPU 使用率
nvidia-smi

# 查看输出目录
ls -lh ./output/task3/

# 查看最新日志（如果重定向了输出）
tail -f train.log
```

### 3.6 第六步：下载结果到本地

训练完成后，在本地 PowerShell 执行：

```bash
# 下载整个 task3 输出目录
scp -rP <端口> root@<IP>:~/autodl-tmp/CV_Project/output/task3/ D:\lab\imageSemanticDp\output\task3\
```

下载后的本地目录结构：

```
output/task3/
├── best_model.pth             # 最佳 checkpoint（~165 MB）
├── history.json               # 逐 epoch 训练历史
├── results.json               # 最终评测结果（含 per-class AP）
├── training_curves.png        # 训练曲线（Loss / mAP / mAP5095）
└── detection_samples.png      # 8 张测试集检测可视化
```

## 4. 本地推理

下载模型后，在本地（无需 GPU）即可推理：

```bash
# 单张图片
python src/task3_inference.py --image D:\your_photo.jpg

# 整个文件夹（默认置信度 0.5）
python src/task3_inference.py --dir D:\photos\

# 降低阈值以检测更多目标
python src/task3_inference.py --dir D:\photos\ --conf_thresh 0.3
```

推理输出保存到 `./output/task3/inference/`：`*_detected.png`（带框标注的图片）。

## 5. 训练参数速查

### 默认配置训练

```bash
python src/task3_object_detection.py
```

### 显存不足时

```bash
python src/task3_object_detection.py --batch_size 2 --img_max_size 600
```

### 追求更高精度

```bash
python src/task3_object_detection.py --epochs 50
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data` | 数据集根目录 |
| `--output_dir` | `./output/task3` | 输出目录 |
| `--img_max_size` | `800` | 图像长边最大像素 |
| `--batch_size` | `4` | 每 GPU 批量大小 |
| `--num_workers` | `4` | DataLoader 线程数 |
| `--epochs` | `30` | 训练轮数 |
| `--lr` | `1e-3` | 峰值学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--warmup_epochs` | `2` | 线性 warm-up 轮数 |
| `--lr_min` | `1e-6` | Cosine 退火最小 lr |
| `--use_amp` / `--no_amp` | True | 混合精度开关 |

## 6. 预计时间

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 数据集下载 | 5-10 分钟 | VOC2007+2012 约 2 GB |
| 模型权重下载 | 1 分钟 | COCO 预训练约 160 MB |
| 训练（30 epoch） | **8-12 小时** | RTX 4090 D，batch_size=4 |
| 结果下载 | 2-5 分钟 | checkpoint 约 165 MB |

## 7. 常见问题

**Q: 数据集下载失败？**

可能是 VOC 官网网络问题，手动下载后放入 `./data/VOCdevkit/` 目录：

- [VOC2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [VOC2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
- [VOC2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

目录结构应为 `./data/VOCdevkit/VOC2007/` 和 `./data/VOCdevkit/VOC2012/`。

**Q: OOM（显存不足）？**

Faster R-CNN 显存消耗大，减小 batch size 和图像尺寸：

```bash
python src/task3_object_detection.py --batch_size 2 --img_max_size 600
```

**Q: 训练中断了，能续训吗？**

当前版本不支持断点续训。但每个 epoch 结束后如果 mAP 更高就会覆盖保存 checkpoint，所以最坏情况只损失最后一个 epoch 的进度。如有需要，加载 `best_model.pth` 中的 `optimizer_state_dict` 可手动恢复。

**Q: 所有路径都是相对路径吗？**

是的。数据路径（`--data_dir ./data`）、输出路径（`--output_dir ./output/task3`）均使用相对路径，上传到服务器后无需修改任何参数即可直接运行。
