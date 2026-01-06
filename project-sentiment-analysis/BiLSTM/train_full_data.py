#!/usr/bin/env python3
"""
使用全部156K训练数据训练Phase 4a配置
不划分验证集，通过每轮测试集推理监控性能
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import (
    load_data, load_test_data, Vocabulary,
    SentimentDataset, collate_fn
)
from model import create_model, FocalLoss
from utils import (
    get_device, train_one_epoch, save_checkpoint,
    save_training_history
)


# Phase 4a最佳配置
PHASE4A_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'hidden_dim': 256,
    'num_layers': 3,
    'embedding_dim': 300,
    'dropout': 0.4,
    'use_attention': True,
    'weight_decay': 0.0001,
    'use_focal_loss': True,
    'focal_gamma': 0.5,
}


def create_train_loader(phrases, labels, vocab, batch_size=64):
    """创建训练数据加载器"""
    dataset = SentimentDataset(phrases, labels, vocab)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    return loader


def get_class_weights(train_labels, device):
    """计算类别权重"""
    label_counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.tensor([
        total / label_counts[i] for i in range(5)
    ], dtype=torch.float32).to(device)
    return weights


def run_inference_on_test(model, vocab, device, output_path):
    """对测试集进行推理"""
    print('\n  开始测试集推理...')

    # 加载测试数据
    test_df, test_phrases = load_test_data('data/test.tsv')

    # 编码
    test_encoded = []
    for phrase in test_phrases:
        test_encoded.append(vocab.encode(phrase))

    # 批量预测
    model.eval()
    predictions = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(test_encoded), batch_size):
            batch_data = test_encoded[i:i+batch_size]
            max_len = max(len(seq) for seq in batch_data)

            batch_padded = []
            batch_lengths = []
            for seq in batch_data:
                padded = seq + [0] * (max_len - len(seq))
                batch_padded.append(padded)
                batch_lengths.append(max(1, len(seq)))

            inputs = torch.LongTensor(batch_padded).to(device)
            lengths = torch.LongTensor(batch_lengths).to(device)

            outputs = model(inputs, lengths)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy().tolist())

    # 保存
    submission = pd.DataFrame({
        'PhraseId': test_df['PhraseId'],
        'Sentiment': predictions
    })
    submission.to_csv(output_path, index=False)

    # 打印分布
    sentiment_dist = submission['Sentiment'].value_counts().sort_index()
    print(f'  测试集预测已保存: {output_path}')
    print(f'  预测分布: ', dict(sentiment_dist))

    return predictions


def train_full_data(args):
    """主训练函数"""
    print('=' * 80)
    print('Phase 4a - 全量数据训练（156K样本）')
    print('=' * 80)

    # 设备
    device = get_device()
    print(f'\n使用设备: {device}')

    # 1. 加载全部训练数据
    print('\n1. 加载全部训练数据...')
    df, all_phrases, all_labels = load_data('data/train.tsv')
    print(f'总样本数: {len(all_phrases)}')

    # 打印类别分布
    label_counts = Counter(all_labels)
    print('\n类别分布:')
    for i in range(5):
        count = label_counts[i]
        pct = 100 * count / len(all_labels)
        print(f'  标签{i}: {count:6d} ({pct:.1f}%)')

    # 2. 构建词汇表（使用全部数据）
    print('\n2. 构建词汇表...')
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(all_phrases)
    print(f'词汇表大小: {len(vocab)}')

    # 保存词汇表
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / 'vocabulary.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'词汇表已保存: {vocab_path}')

    # 3. 创建训练数据加载器
    print('\n3. 创建训练数据加载器...')
    train_loader = create_train_loader(
        all_phrases, all_labels, vocab,
        batch_size=PHASE4A_CONFIG['batch_size']
    )
    print(f'训练批次数: {len(train_loader)}')

    # 4. 创建模型
    print('\n4. 创建模型...')
    model = create_model(
        vocab_size=len(vocab),
        embedding_dim=PHASE4A_CONFIG['embedding_dim'],
        hidden_dim=PHASE4A_CONFIG['hidden_dim'],
        num_layers=PHASE4A_CONFIG['num_layers'],
        dropout=PHASE4A_CONFIG['dropout'],
        use_attention=PHASE4A_CONFIG['use_attention'],
        device=device
    )

    # 获取实际设备
    actual_device = next(model.parameters()).device
    if actual_device != device:
        print(f'\n警告：模型已从 {device} 回退到 {actual_device}')
        device = actual_device

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数量: {param_count:,}')

    # 5. 损失函数和优化器
    print('\n5. 配置训练组件...')
    class_weights = get_class_weights(all_labels, device)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=PHASE4A_CONFIG['focal_gamma']
    )
    print(f'损失函数: Focal Loss (gamma={PHASE4A_CONFIG["focal_gamma"]})')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PHASE4A_CONFIG['learning_rate'],
        weight_decay=PHASE4A_CONFIG['weight_decay']
    )
    print(f'优化器: Adam (lr={PHASE4A_CONFIG["learning_rate"]})')

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # 6. 训练循环
    print('\n6. 开始训练...')
    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(1, args.max_epochs + 1):
        print(f'\n{"=" * 80}')
        print(f'Epoch {epoch}/{args.max_epochs}')
        print(f'{"=" * 80}')

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, print_freq=100
        )

        print(f'\nEpoch {epoch} 结果:')
        print(f'  训练Loss: {train_loss:.4f}')
        print(f'  训练Acc:  {train_acc:.2f}%')

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 保存检查点
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        save_checkpoint(
            model, optimizer, epoch, train_loss, 0.0, train_acc,
            PHASE4A_CONFIG, str(checkpoint_path), is_best=False
        )
        print(f'  检查点已保存: {checkpoint_path.name}')

        # 测试集推理
        submission_path = output_dir / f'submission_epoch_{epoch}.csv'
        run_inference_on_test(model, vocab, device, str(submission_path))

        # 更新学习率（基于训练准确率）
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  当前学习率: {current_lr:.6f}')

        # 过拟合检查
        if train_acc > 75.0:
            print('\n 警告：训练准确率超过75%，可能过拟合！')
            print('  建议选择较早轮次的模型（第6-8轮）')

        # 推荐检查
        if 62.0 <= train_acc <= 70.0:
            print(f' 训练准确率在推荐范围内（62-70%）')

    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    save_training_history(history, str(history_path))
    print(f'\n训练历史已保存: {history_path}')

    print('\n' + '=' * 80)
    print('训练完成！')
    print('=' * 80)
    print(f'\n所有结果已保存到: {args.output_dir}/')


def main():
    parser = argparse.ArgumentParser(
        description='使用全量数据（156K）训练Phase 4a配置'
    )
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='最大训练轮数（默认10）')
    parser.add_argument('--output_dir', type=str, default='checkpoints_full_data',
                        help='输出目录（默认checkpoints_full_data）')
    args = parser.parse_args()

    print('\n配置:')
    print(f'  最大轮数: {args.max_epochs}')
    print(f'  输出目录: {args.output_dir}')
    print(f'  Phase 4a配置: Focal Loss (gamma=0.5), BiLSTM (hidden=256, layers=3)')
    print()

    train_full_data(args)


if __name__ == '__main__':
    main()
