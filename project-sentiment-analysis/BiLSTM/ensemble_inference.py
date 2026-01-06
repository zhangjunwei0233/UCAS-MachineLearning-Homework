#!/usr/bin/env python3
"""
集成多个epoch的预测结果（投票法）
"""

import argparse
from collections import Counter
from pathlib import Path


def ensemble_predictions(submission_paths, output_path, method='majority'):
    """
    集成多个预测结果（多数投票）

    Args:
        submission_paths: 提交文件路径列表
        output_path: 输出文件路径
        method: 'majority' (多数投票)
    """
    print(f'集成方法: {method}')
    print(f'输入文件数: {len(submission_paths)}')

    # 读取所有预测
    all_predictions = {}  # {phrase_id: [pred1, pred2, ...]}

    for i, path in enumerate(submission_paths, 1):
        print(f'  读取文件 {i}/{len(submission_paths)}: {Path(path).name}')
        with open(path, 'r') as f:
            lines = f.readlines()[1:]  # 跳过标题
            for line in lines:
                phrase_id, sentiment = line.strip().split(',')
                phrase_id = int(phrase_id)
                sentiment = int(sentiment)

                if phrase_id not in all_predictions:
                    all_predictions[phrase_id] = []
                all_predictions[phrase_id].append(sentiment)

    # 集成预测
    print(f'\n开始集成 {len(all_predictions)} 个样本...')
    final_predictions = {}

    for phrase_id, preds in all_predictions.items():
        # 多数投票
        counter = Counter(preds)
        final_pred = counter.most_common(1)[0][0]
        final_predictions[phrase_id] = final_pred

    # 保存结果
    with open(output_path, 'w') as f:
        f.write('PhraseId,Sentiment\n')
        for phrase_id in sorted(final_predictions.keys()):
            f.write(f'{phrase_id},{final_predictions[phrase_id]}\n')

    print(f'\n集成结果已保存: {output_path}')

    # 打印分布
    sentiment_dist = Counter(final_predictions.values())
    total = len(final_predictions)
    print('\n预测分布:')
    for i in range(5):
        count = sentiment_dist[i]
        pct = 100 * count / total
        print(f'  Sentiment {i}: {count:6d} ({pct:.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='集成多个epoch的预测结果')
    parser.add_argument('--epochs', type=str, required=True,
                        help='要集成的epoch，逗号分隔，如 "7,8,9,10"')
    parser.add_argument('--input_dir', type=str, default='checkpoints_full_data',
                        help='输入目录（默认checkpoints_full_data）')
    parser.add_argument('--output', type=str, default='submission_ensemble.csv',
                        help='输出文件（默认submission_ensemble.csv）')
    parser.add_argument('--method', type=str, default='majority',
                        choices=['majority'],
                        help='集成方法：majority (多数投票)')
    args = parser.parse_args()

    # 解析epoch列表
    epochs = [int(e.strip()) for e in args.epochs.split(',')]
    print('=' * 80)
    print('集成预测')
    print('=' * 80)
    print(f'集成epochs: {epochs}')

    # 构建文件路径列表
    input_dir = Path(args.input_dir)
    submission_paths = []
    for epoch in epochs:
        path = input_dir / f'submission_epoch_{epoch}.csv'
        if not path.exists():
            print(f'错误：文件不存在: {path}')
            return
        submission_paths.append(str(path))

    # 集成
    ensemble_predictions(submission_paths, args.output, args.method)

    print('\n' + '=' * 80)
    print('完成！')
    print('=' * 80)


if __name__ == '__main__':
    main()
