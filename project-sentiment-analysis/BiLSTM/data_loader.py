"""
数据加载和预处理模块
负责加载 TSV 数据，构建词汇表，将文本转换为序列
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import Tuple, Dict, List
import re


class Vocabulary:
    """词汇表类，用于文本和索引之间的转换"""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()

    def build_vocab(self, texts: List[str]):
        """从文本列表构建词汇表"""
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)

        # 只保留出现次数 >= min_freq 的词
        idx = len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f'词汇表大小: {len(self.word2idx)}')

    def tokenize(self, text: str) -> List[str]:
        """
        改进的分词器
        - 处理常见的英语缩写（n't -> not等）
        - 保留情感标点符号（! ?）
        """
        # 转换为小写
        text = text.lower()

        # 处理常见缩写（保留否定词的情感信息）
        text = re.sub(r"n't", " not", text)      # don't -> do not
        text = re.sub(r"'re", " are", text)      # you're -> you are
        text = re.sub(r"'ve", " have", text)     # I've -> I have
        text = re.sub(r"'ll", " will", text)     # I'll -> I will
        text = re.sub(r"'m", " am", text)        # I'm -> I am
        text = re.sub(r"'d", " would", text)     # I'd -> I would
        text = re.sub(r"'s", " is", text)        # it's -> it is (简化处理)

        # 分词：保留字母、数字、感叹号、问号
        # 这样可以保留情感强度信息（如 "great!" vs "great"）
        words = re.findall(r'\b\w+\b|[!?]', text)

        return words

    def encode(self, text: str) -> List[int]:
        """将文本转换为索引列表"""
        words = self.tokenize(text)
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

    def __len__(self):
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """情感分析数据集类"""

    def __init__(self, phrases: List[str], labels: List[int], vocab: Vocabulary):
        self.phrases = phrases
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        label = self.labels[idx]

        # 将文本转换为索引序列
        indices = self.vocab.encode(phrase)

        # 确保至少有一个token（使用UNK）
        if len(indices) == 0:
            indices = [self.vocab.word2idx['<UNK>']]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """自定义批处理函数，对不同长度的序列进行填充"""
    sequences, labels = zip(*batch)

    # 填充序列到相同长度
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    # 计算每个序列的实际长度（用于pack_padded_sequence）
    # 确保所有长度至少为1
    lengths = torch.tensor([max(1, len(seq)) for seq in sequences], dtype=torch.long)

    return sequences_padded, labels, lengths


def load_data(data_path: str) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    加载训练数据

    Args:
        data_path: train.tsv 文件路径

    Returns:
        DataFrame, 短语列表, 标签列表
    """
    df = pd.read_csv(data_path, sep='\t')

    # 检查必要的列是否存在
    required_cols = ['Phrase', 'Sentiment']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'数据集缺少必要的列: {col}')

    phrases = df['Phrase'].astype(str).tolist()
    labels = df['Sentiment'].tolist()

    print(f'加载了 {len(phrases)} 条训练数据')
    print(f'情感标签分布:')
    print(df['Sentiment'].value_counts().sort_index())

    return df, phrases, labels


def load_test_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载测试数据

    Args:
        data_path: test.tsv 文件路径

    Returns:
        DataFrame, 短语列表
    """
    df = pd.read_csv(data_path, sep='\t')

    if 'Phrase' not in df.columns:
        raise ValueError('测试数据集缺少 Phrase 列')

    phrases = df['Phrase'].astype(str).tolist()

    print(f'加载了 {len(phrases)} 条测试数据')

    return df, phrases


def create_data_loaders(
    train_phrases: List[str],
    train_labels: List[int],
    val_phrases: List[str],
    val_labels: List[int],
    vocab: Vocabulary,
    batch_size: int = 64,
    num_workers: int = 0,
    use_weighted_sampler: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_phrases: 训练短语列表
        train_labels: 训练标签列表
        val_phrases: 验证短语列表
        val_labels: 验证标签列表
        vocab: 词汇表对象
        batch_size: 批处理大小
        num_workers: 数据加载的工作进程数
        use_weighted_sampler: 是否使用加权采样器（用于处理类别不平衡）

    Returns:
        训练数据加载器, 验证数据加载器
    """
    train_dataset = SentimentDataset(train_phrases, train_labels, vocab)
    val_dataset = SentimentDataset(val_phrases, val_labels, vocab)

    # 如果使用加权采样器
    if use_weighted_sampler:
        # 计算每个样本的权重（基于其类别的逆频率）
        label_counts = Counter(train_labels)
        total = len(train_labels)

        # 每个类别的权重 = 总样本数 / 该类别样本数
        class_weights = {label: total / count for label, count in label_counts.items()}

        # 为每个样本分配权重
        sample_weights = [class_weights[label] for label in train_labels]

        # 创建加权采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True  # 允许重复采样
        )

        print(f'\n使用加权采样器:')
        for label, count in sorted(label_counts.items()):
            print(f'  标签{label}: 样本数={count}, 权重={class_weights[label]:.4f}')

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # 使用sampler时不能使用shuffle
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False
        )
    else:
        # 标准的随机打乱
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader


def split_data(
    phrases: List[str],
    labels: List[int],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    将数据划分为训练集和验证集

    Args:
        phrases: 所有短语列表
        labels: 所有标签列表
        val_ratio: 验证集比例
        random_seed: 随机种子

    Returns:
        train_phrases, train_labels, val_phrases, val_labels
    """
    import random

    # 设置随机种子以保证可复现性
    random.seed(random_seed)

    # 创建索引并打乱
    indices = list(range(len(phrases)))
    random.shuffle(indices)

    # 计算划分点
    split_idx = int(len(indices) * (1 - val_ratio))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_phrases = [phrases[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_phrases = [phrases[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f'训练集大小: {len(train_phrases)}')
    print(f'验证集大小: {len(val_phrases)}')

    return train_phrases, train_labels, val_phrases, val_labels
