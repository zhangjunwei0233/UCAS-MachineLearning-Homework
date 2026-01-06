"""
Hyperparameter search script for TextCNN model.
Performs grid search over specified hyperparameters and reports the best configuration.
"""

import argparse
import itertools
from typing import Dict, List, Sequence

import text_cnn


def make_args(base_args: argparse.Namespace, overrides: Dict) -> argparse.Namespace:
    """
    Create a fresh argparse.Namespace merging base args with overrides.

    Args:
        base_args: Base argument namespace with default values
        overrides: Dictionary of parameters to override

    Returns:
        New Namespace with merged parameters
    """
    params = vars(base_args).copy()
    params.update(overrides)
    return argparse.Namespace(**params)


def parse_cli() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameter search."""
    parser = argparse.ArgumentParser(description="Grid search for TextCNN hyperparameters.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for averaging performance.",
    )
    parser.add_argument("--glove_path", type=str, default="", help="Optional GloVe path shared across runs.")
    return parser.parse_args()


def main() -> None:
    """
    Main function for hyperparameter grid search.
    Iterates through all combinations of hyperparameters and trains models.
    Reports the best configuration based on validation accuracy.
    """
    cli_args = parse_cli()
    # Start from text_cnn defaults, then override per run.
    base_args = text_cnn.parse_args([])
    if cli_args.glove_path:
        base_args.glove_path = cli_args.glove_path

    # Search space per requirements.
    search_space = {
        "lr": [5e-4, 7e-4, 1e-3],
        "dropout": [0.5, 0.6],
        "max_len": [80, 100, 120],
        "num_filters": [200, 256],
        "embed_dim": [200, 300],
    }

    # Fixed hyperparameters that are not searched
    fixed_overrides = {
        "use_class_weights": True,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "epochs": 18,
        "batch_size": 128,
        "use_multichannel": True,
        "use_batchnorm": True,
        "use_double_conv": True,
        "kernel_sizes": "2,3,4,5,7",
        "proj_dim": 256,
        "label_smoothing": 0.1,
        "warmup_epochs": 2,
        "early_stop_patience": 6,
        "print_confusion": False,
        "max_vocab_size": 60000,
        "use_char_cnn": True,
        "char_max_len": 180,
        "char_embed_dim": 50,
        "char_num_filters": 50,
        "char_kernel_sizes": "3,4,5",
        "fc_weight_decay": 1e-3,
        "word_dropout": 0.05,
    }

    # Generate all combinations of hyperparameters
    keys: List[str] = list(search_space.keys())
    combos: Sequence = list(itertools.product(*[search_space[k] for k in keys]))

    best_acc = -1.0
    best_cfg = None

    print(f"Total configs: {len(combos)}, seeds per config: {len(cli_args.seeds)}")
    for idx, values in enumerate(combos, start=1):
        overrides = dict(zip(keys, values))
        overrides.update(fixed_overrides)
        overrides["save_path"] = f"checkpoints/search_run_{idx}.pt"

        # Train with multiple seeds and average results
        seed_accs = []
        for seed in cli_args.seeds:
            overrides["seed"] = seed
            args = make_args(base_args, overrides)

            print(f"\n=== Run {idx}/{len(combos)} | seed {seed} ===")
            print(overrides)
            val_acc = text_cnn.run_training(args)
            seed_accs.append(val_acc)

        avg_acc = sum(seed_accs) / len(seed_accs)
        print(f"Average acc over seeds: {avg_acc:.4f}")

        # Track best configuration
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_cfg = overrides.copy()

    print("\n=== Search complete ===")
    print(f"Best val_acc={best_acc:.4f}")
    print(f"Best config (last seed used): {best_cfg}")


if __name__ == "__main__":
    main()
