"""
BiLSTM 情感分析模型
包含词嵌入层、双向LSTM层和分类层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BiLSTMSentimentModel(nn.Module):
    """
    基于双向LSTM的情感分析模型

    架构:
        1. Embedding Layer: 将词索引转换为词向量
        2. BiLSTM Layers: 多层双向LSTM提取上下文特征
        3. Attention Layer (可选): 注意力机制
        4. Fully Connected Layers: 全连接层进行分类
        5. Output Layer: 5分类softmax
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.5,
        use_attention: bool = False,
        bidirectional: bool = True
    ):
        """
        初始化模型

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量（情感标签0-4，共5类）
            dropout: Dropout比率
            use_attention: 是否使用注意力机制
            bidirectional: 是否使用双向LSTM
        """
        super(BiLSTMSentimentModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 注意力层（可选）
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * self.num_directions)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        fc_input_dim = hidden_dim * self.num_directions

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len]
            lengths: 每个序列的实际长度，形状 [batch_size]

        Returns:
            输出张量，形状 [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # 如果提供了长度信息，使用pack_padded_sequence
        if lengths is not None:
            # 对长度进行排序（pack_padded_sequence要求）
            lengths_cpu = lengths.cpu()
            # 确保所有长度至少为1
            lengths_cpu = torch.clamp(lengths_cpu, min=1)
            sorted_lengths, sorted_idx = torch.sort(lengths_cpu, descending=True)
            _, unsorted_idx = torch.sort(sorted_idx)

            embedded = embedded[sorted_idx]

            # Pack序列
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed_embedded)

            # Unpack序列
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

            # 恢复原始顺序
            lstm_output = lstm_output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
        else:
            # 不使用pack_padded_sequence
            lstm_output, (hidden, cell) = self.lstm(embedded)

        # 使用注意力机制或最后一个时间步的隐藏状态
        if self.use_attention:
            # 使用注意力机制
            context_vector = self.attention(lstm_output, lengths)
            output = self.dropout(context_vector)
        else:
            # 使用最后一层的隐藏状态
            # hidden形状: [num_layers * num_directions, batch_size, hidden_dim]
            if self.bidirectional:
                # 拼接前向和后向的最后一层隐藏状态
                hidden_forward = hidden[-2, :, :]
                hidden_backward = hidden[-1, :, :]
                hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                hidden_concat = hidden[-1, :, :]

            output = self.dropout(hidden_concat)

        # 全连接层
        logits = self.fc(output)  # [batch_size, num_classes]

        return logits

    def init_embeddings(self, pretrained_embeddings: torch.Tensor):
        """
        使用预训练的词向量初始化嵌入层

        Args:
            pretrained_embeddings: 预训练词向量，形状 [vocab_size, embedding_dim]
        """
        self.embedding.weight.data.copy_(pretrained_embeddings)
        print('已加载预训练词向量')


class AttentionLayer(nn.Module):
    """简单注意力机制层（单头）"""

    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        计算注意力加权的上下文向量

        Args:
            lstm_output: LSTM输出，形状 [batch_size, seq_len, hidden_dim]
            lengths: 每个序列的实际长度，形状 [batch_size]

        Returns:
            上下文向量，形状 [batch_size, hidden_dim]
        """
        # 计算注意力分数
        attention_scores = self.attention_weights(lstm_output).squeeze(-1)
        # [batch_size, seq_len]

        # 如果提供了长度信息，对padding部分应用mask
        if lengths is not None:
            max_len = lstm_output.size(1)
            mask = torch.arange(max_len, device=lstm_output.device).expand(
                len(lengths), max_len
            ) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # 计算注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)
        # [batch_size, seq_len]

        # 加权求和
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), lstm_output
        ).squeeze(1)
        # [batch_size, hidden_dim]

        return context_vector


class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡和困难样本问题

    公式: FL(pt) = -α(1-pt)^γ * log(pt)

    参数:
        alpha: 类别权重 [num_classes]，用于处理类别不平衡
        gamma: 聚焦参数，通常设为2.0
               - gamma=0时退化为交叉熵损失
               - gamma>0时降低简单样本的权重，关注困难样本
        reduction: 'mean', 'sum' 或 'none'

    优势:
        1. 自动降低简单样本（已正确分类）的损失权重
        2. 集中优化困难样本（易混淆样本）
        3. 对类别不平衡问题更有效
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        初始化Focal Loss

        Args:
            alpha: 类别权重张量 [num_classes]，None表示不使用权重
            gamma: 聚焦参数（通常2.0）
            reduction: 损失聚合方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: 模型logits，形状 [batch_size, num_classes]
            targets: 目标标签，形状 [batch_size]

        Returns:
            损失值
        """
        # 计算交叉熵损失（不进行reduction）
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.alpha
        )

        # 计算预测概率 pt
        # pt = exp(-ce_loss)，即预测正确类别的概率
        pt = torch.exp(-ce_loss)

        # 计算Focal Loss
        # FL = (1-pt)^gamma * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def create_model(
    vocab_size: int,
    embedding_dim: int = 300,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.5,
    use_attention: bool = False,
    device: torch.device = torch.device('cpu')
) -> BiLSTMSentimentModel:
    """
    创建并初始化模型

    Args:
        vocab_size: 词汇表大小
        embedding_dim: 词嵌入维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        num_classes: 分类数量
        dropout: Dropout比率
        use_attention: 是否使用注意力机制
        device: 设备

    Returns:
        初始化的模型
    """
    model = BiLSTMSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        use_attention=use_attention
    )

    # 尝试将模型移到设备上，如果CUDA失败则回退到CPU
    try:
        model = model.to(device)
    except RuntimeError as e:
        if 'CUDA' in str(e) and device.type == 'cuda':
            print(f'\n模型加载到CUDA失败: {e}')
            print('尝试使用CPU设备...')
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise

    # 统计模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'模型总参数数量: {total_params:,}')
    print(f'可训练参数数量: {trainable_params:,}')

    return model
