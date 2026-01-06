"""
工具函数模块
包含检查点保存/加载、设备管理、训练/评估等功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import time
from datetime import datetime


def get_device() -> torch.device:
    """
    自动选择可用的设备（优先CUDA > CPU）
    增加CUDA兼容性测试

    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        try:
            # 测试CUDA是否真正可用
            device = torch.device('cuda')
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
            print(f'使用 CUDA 设备: {torch.cuda.get_device_name(0)}')
            print(f'GPU 数量: {torch.cuda.device_count()}')
            print(f'当前 GPU: {torch.cuda.current_device()}')
            return device
        except RuntimeError as e:
            print(f'CUDA初始化失败: {e}')
            print('回退到CPU设备')
            return torch.device('cpu')
    else:
        print('CUDA 不可用，使用 CPU 设备')
        return torch.device('cpu')


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    config: Dict,
    checkpoint_path: str,
    is_best: bool = False
):
    """
    保存模型检查点

    Args:
        model: 模型对象
        optimizer: 优化器对象
        epoch: 当前训练轮数
        train_loss: 训练损失
        val_loss: 验证损失
        val_accuracy: 验证准确率
        config: 配置字典
        checkpoint_path: 检查点保存路径
        is_best: 是否为最佳模型
    """
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f'检查点已保存: {checkpoint_path}')

    if is_best:
        best_path = Path(checkpoint_path).parent / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f'最佳模型已保存: {best_path}')


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型对象
        optimizer: 优化器对象（可选）
        device: 设备

    Returns:
        包含检查点信息的字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f'检查点已加载: {checkpoint_path}')
    print(f'轮数: {checkpoint["epoch"]}, 验证准确率: {checkpoint["val_accuracy"]:.4f}')

    return checkpoint


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 100
) -> Tuple[float, float]:
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前轮数
        print_freq: 打印频率

    Returns:
        平均损失, 准确率
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (sequences, labels, lengths) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(sequences, lengths)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印进度
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {avg_loss:.4f} Acc: {accuracy:.2f}% Time: {elapsed:.1f}s')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        平均损失, 准确率
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels, lengths in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # 前向传播
            outputs = model(sequences, lengths)

            # 计算损失
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_training_history(history: Dict, save_path: str):
    """
    保存训练历史记录

    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f'训练历史已保存: {save_path}')


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    对数据进行预测

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备

    Returns:
        预测结果张量
    """
    model.eval()

    predictions = []

    with torch.no_grad():
        for sequences, _, lengths in data_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs, 1)

            predictions.append(predicted.cpu())

    predictions = torch.cat(predictions, dim=0)

    return predictions


