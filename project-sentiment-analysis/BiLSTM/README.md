# 电影评论情感分析

基于BiLSTM的电影评论情感分析项目，实现5分类情感预测（0-4，从极负面到极正面）。

**最终测试准确率**: **61.090%**

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无（CPU可运行） | CUDA兼容GPU（训练加速10-20倍） |
| 存储 | 2GB | 5GB+（含训练检查点） |

### 软件要求

- **Python** >= 3.12
- **PyTorch** >= 2.0.0
- **CUDA** >= 11.8（可选，GPU加速）
- **操作系统**: Linux / macOS / Windows

### Python依赖

```bash
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
filelock>=3.20.1
```

---

## 安装步骤

### 方法1: 使用 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 方法2: 使用 pip

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install torch>=2.0.0 pandas>=2.0.0 numpy>=1.24.0 \
            scikit-learn>=1.3.0 pyyaml>=6.0 filelock>=3.20.1
```

### 验证安装

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); \
            print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 项目结构

```
├── 核心代码（5个文件）
│   ├── model.py                # BiLSTM模型定义
│   ├── data_loader.py          # 数据加载和词汇表构建
│   ├── utils.py                # 工具函数（Focal Loss、检查点管理）
│   ├── train_full_data.py      # 全数据训练脚本
│   └── ensemble_inference.py   # 集成推理脚本
│
├── data/                       # 数据集
│   ├── train.tsv               # 156,060训练样本
│   ├── test.tsv                # 66,292测试样本
│   └── sampleSubmission.csv    # 提交格式示例
│
├── docs/                       # 文档
│   ├── EXPERIMENT_REPORT.md    # 实验报告
│   └── FINAL_SUMMARY.md        # 完整技术总结
│
├── pyproject.toml              # 项目配置和依赖
├── README.md                   # 本文件
└── CLAUDE.md                   # Claude Code工作指南
```

**注意**: 训练过程会自动生成 `checkpoints_full_data/` 目录存储模型检查点。

---

## 运行方式

### 1. 训练模型

```bash
# 本地训练（CPU，约2-4小时）
python3 train_full_data.py --max_epochs 10

# 本地训练（GPU，约15-30分钟）
python3 train_full_data.py --max_epochs 10 --batch_size 128
```

**主要参数**:
- `--max_epochs`: 训练轮数（默认10）
- `--batch_size`: 批次大小（CPU:64, GPU:128）
- `--learning_rate`: 学习率（默认0.001）
- `--output_dir`: 检查点保存目录（默认checkpoints_full_data）

训练完成后会在 `checkpoints_full_data/` 生成：
- `checkpoint_epoch_*.pt` - 各轮检查点
- `vocabulary.pkl` - 词汇表
- `training_history.json` - 训练历史

### 2. 生成预测

```bash
# 使用集成推理（8个epoch，最佳配置）
python3 ensemble_inference.py \
  --epochs "1,2,4,5,6,7,8,9" \
  --input_dir checkpoints_full_data \
  --output submission.csv
```

**主要参数**:
- `--epochs`: 用于集成的epoch列表（逗号分隔）
- `--input_dir`: 检查点目录
- `--output`: 输出文件名
- `--method`: 投票方式（majority/weighted，默认majority）

**预期结果**: `submission.csv` 准确率约 **61.090%**

---

## 模型架构

### BiLSTM + Attention + Focal Loss

```python
{
    'vocab_size': 18522,         # 词汇表大小
    'embedding_dim': 300,        # 词向量维度
    'hidden_dim': 256,           # LSTM隐藏层维度
    'num_layers': 3,             # 3层BiLSTM
    'dropout': 0.4,              # Dropout比率
    'use_attention': True,       # 单头注意力机制
    'use_focal_loss': True,      # Focal Loss（关键）
    'focal_gamma': 0.5,          # γ=0.5最优
}
```

### 集成策略

- **方法**: 简单多数投票
- **模型数**: 8个（Epoch 1,2,4,5,6,7,8,9）
- **排除**: Epoch 3（存在负效应）

---

## 数据说明

### 训练数据 (train.tsv)

| 字段 | 说明 |
|------|------|
| PhraseId | 短语ID |
| SentenceId | 句子ID |
| Phrase | 文本短语 |
| Sentiment | 情感标签（0-4） |

**情感标签**:
- 0: 极负面 (4.5%)
- 1: 负面 (17.5%)
- 2: 中性 (51.0%) - 占主导
- 3: 正面 (21.1%)
- 4: 极正面 (5.9%)

### 测试数据 (test.tsv)

包含 PhraseId, SentenceId, Phrase（无标签），用于生成预测。

---

## 核心技术

### 1. Focal Loss (γ=0.5)
针对类别不平衡问题，单次优化提升 +3.39%

```python
FL(pt) = -α(1-pt)^γ * log(pt)
```

### 2. 全数据训练
- 词汇表覆盖度 +22% (15,187 → 18,522词)
- 训练样本 +25% (124,848 → 156,060)

### 3. 集成学习
- 早期epoch泛化能力更强（"低训练准确率悖论"）
- 简单多数投票优于加权投票
- 排除Epoch 3后 +0.19%

---

## 快速FAQ

**Q: 训练中断了怎么办？**
A: 脚本支持断点恢复，重新运行相同命令即可。

**Q: 如何查看训练历史？**
A: 查看 `checkpoints_full_data/training_history.json`

---
