"""
Batch experiment runner for TextCNN model.
Runs multiple training configurations and logs results for comparison.
Useful for ablation studies and comparing different model architectures.
"""

import argparse
import csv
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Regular expressions to parse validation accuracy from training logs
BEST_ACC_RE = re.compile(r"Best validation accuracy:\s*([0-9]*\.?[0-9]+)")
VAL_ACC_RE = re.compile(r"Epoch\s+\d+\s+\|\s+.*val_acc=([0-9]*\.?[0-9]+)")

@dataclass(frozen=True)
class Experiment:
    """
    Represents a single experiment configuration.

    Attributes:
        name: Descriptive name for the experiment
        extra_args: Additional command-line arguments to pass to training script
    """
    name: str
    extra_args: Tuple[str, ...]

def parse_best_acc(stdout: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse best and last validation accuracy from training output.

    Args:
        stdout: Training script output as string

    Returns:
        Tuple of (best_val_acc, last_val_acc)
    """
    m = BEST_ACC_RE.findall(stdout)
    best = float(m[-1]) if m else None
    m2 = VAL_ACC_RE.findall(stdout)
    last_val = float(m2[-1]) if m2 else None
    return best, last_val

def remove_flag_and_value(args: List[str], flag: str) -> List[str]:
    """
    Remove a command-line flag and its value from argument list.

    Args:
        args: List of command-line arguments
        flag: Flag to remove (e.g., "--glove_path")

    Returns:
        New list with flag and its value removed
    """
    out: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            # remove flag + optional value
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
        else:
            out.append(args[i])
            i += 1
    return out

def remove_flag_only(args: List[str], flag: str) -> List[str]:
    """Remove a boolean flag from argument list (no value)."""
    return [x for x in args if x != flag]

def strip_glove_from_base(args: List[str]) -> List[str]:
    """Remove --glove_path and its associated --glove_dim from base args."""
    a = list(args)
    a = remove_flag_and_value(a, "--glove_path")
    a = remove_flag_and_value(a, "--glove_dim")
    return a

def exp_overrides_glove_to_empty(exp: Experiment) -> bool:
    """Detect if exp explicitly sets --glove_path '' (empty)."""
    xs = list(exp.extra_args)
    for i, x in enumerate(xs):
        if x == "--glove_path" and i + 1 < len(xs):
            return xs[i + 1] == ""
    return False

def run_one(
    python_bin: str,
    train_script: str,
    base_args: List[str],
    exp: Experiment,
    out_dir: Path,
    dry_run: bool,
) -> Dict[str, object]:
    """
    Run a single training experiment.

    Args:
        python_bin: Path to Python executable
        train_script: Path to training script (text_cnn.py)
        base_args: Base command-line arguments
        exp: Experiment configuration
        out_dir: Output directory for logs and checkpoints
        dry_run: If True, only print command without executing

    Returns:
        Dictionary with experiment results (name, status, accuracy, paths, etc.)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = out_dir / f"{exp.name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_path = exp_dir / "model.pt"
    log_path = exp_dir / "train.log"

    # Force unbuffered output: python -u
    cmd = [python_bin, "-u", train_script] + base_args + list(exp.extra_args) + ["--save_path", str(save_path)]

    print("\n" + "=" * 96)
    print(f"[RUN] {exp.name}")
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))
    print("LOG:", log_path)
    print("=" * 96 + "\n")

    if dry_run:
        return {
            "name": exp.name,
            "status": "DRY_RUN",
            "best_val_acc": None,
            "last_val_acc": None,
            "save_path": str(save_path),
            "log_path": str(log_path),
            "cmd": " ".join(shlex.quote(x) for x in cmd),
        }

    # Also set env var for unbuffered output (extra safety on Windows)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Live stream to terminal + write to log
    stdout_lines: List[str] = []
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")   # live to terminal
            f.write(line)         # to log file
            stdout_lines.append(line)
        proc.wait()

    stdout = "".join(stdout_lines)
    best_acc, last_val_acc = parse_best_acc(stdout)
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"

    print(f"\n[DONE] {exp.name} | status={status} | best_val_acc={best_acc} | last_val_acc={last_val_acc}")
    if proc.returncode != 0:
        print(f"[ERROR] See log: {log_path}")

    return {
        "name": exp.name,
        "status": status,
        "best_val_acc": best_acc,
        "last_val_acc": last_val_acc,
        "save_path": str(save_path),
        "log_path": str(log_path),
        "cmd": " ".join(shlex.quote(x) for x in cmd),
    }

def main() -> None:
    """
    Main function to run batch experiments.
    Defines experiment configurations, runs them sequentially, and generates a summary CSV.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="python", help="Python executable, e.g. python or .venv\\Scripts\\python.exe")
    ap.add_argument("--train_script", type=str, default="text_cnn.py")
    ap.add_argument("--out_dir", type=str, default="exp_runs_report")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--fast", action="store_true", help="Run fewer epochs for quick check (good for sanity)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base config (your strong setup) to keep comparisons fair.
    base_args: List[str] = [
        "--train_path", "data/train.tsv",

        "--glove_path", "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt",
        "--glove_dim", "300",
        "--embed_dim", "300",

        "--max_len", "110",
        "--max_vocab_size", "60000",
        "--min_freq", "1",

        "--batch_size", "128",
        "--epochs", "30",
        "--seed", "42",

        "--num_filters", "256",
        "--kernel_sizes", "2,3,4,5,7",
        "--proj_dim", "256",

        "--dropout", "0.28",
        "--word_dropout", "0.01",

        "--use_one_cycle",
        "--lr", "6e-4",
        "--max_lr", "1.8e-3",
        "--weight_decay", "1e-4",
        "--fc_weight_decay", "1e-4",
        "--grad_clip", "1.0",

        "--use_ema",
        "--ema_decay", "0.995",
        "--early_stop_patience", "10",
        "--print_confusion",
    ]

    if args.fast:
        # Reduce epochs for quick testing
        base_args = remove_flag_and_value(base_args, "--epochs")
        base_args = remove_flag_and_value(base_args, "--early_stop_patience")
        base_args += ["--epochs", "10", "--early_stop_patience", "4"]

    # 12 report-friendly runs for ablation study
    experiments: List[Experiment] = [
        Experiment("01_baseline_random_single", ("--glove_path", "")),
        Experiment("02_single_glove_init", tuple()),
        Experiment("03_multichannel_glove", ("--use_multichannel",)),
        Experiment("04_multi_plus_bn", ("--use_multichannel", "--use_batchnorm")),
        Experiment("05_multi_plus_doubleconv", ("--use_multichannel", "--use_double_conv")),
        Experiment("06_multi_double_bn", ("--use_multichannel", "--use_double_conv", "--use_batchnorm")),
        Experiment("07_multi_double_bn_charcnn", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "--use_char_cnn")),
        Experiment("08_kernel_345_on_best_structure", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "--kernel_sizes", "3,4,5")),
        Experiment("09_nf128_on_best_structure", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "--num_filters", "128")),
        Experiment("10_dropout05_on_best_structure", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "--dropout", "0.5")),
        Experiment("11_no_ema_on_best_structure", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "__NO_EMA__")),
        Experiment("12_ordinal_coral_on_best_structure", ("--use_multichannel", "--use_double_conv", "--use_batchnorm", "--use_ordinal", "--ordinal_pos_weight", "inv", "--ordinal_decode", "expected")),
    ]

    def base_for_exp(exp: Experiment) -> List[str]:
        """Prepare base arguments for a specific experiment."""
        b = list(base_args)

        # If exp sets glove_path="" remove glove from base to avoid duplicates
        if exp_overrides_glove_to_empty(exp):
            b = strip_glove_from_base(b)

        # Handle turning EMA off (base includes --use_ema)
        if "__NO_EMA__" in exp.extra_args:
            b = remove_flag_only(b, "--use_ema")
            b = remove_flag_and_value(b, "--ema_decay")

        return b

    def exp_args(exp: Experiment) -> Tuple[str, ...]:
        """Clean experiment arguments by removing placeholders."""
        # strip placeholder
        return tuple(x for x in exp.extra_args if x != "__NO_EMA__")

    # Run all experiments
    results: List[Dict[str, object]] = []
    for exp in experiments:
        b = base_for_exp(exp)
        exp2 = Experiment(exp.name, exp_args(exp))
        res = run_one(args.python, args.train_script, b, exp2, out_dir, args.dry_run)
        results.append(res)

    # Write summary CSV
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "status", "best_val_acc", "last_val_acc", "save_path", "log_path", "cmd"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "=" * 96)
    print("All experiments done.")
    print("Summary CSV:", csv_path)
    print("=" * 96)

    # Display ranking of experiments by validation accuracy
    ok = [r for r in results if r["best_val_acc"] is not None]
    ok_sorted = sorted(ok, key=lambda x: x["best_val_acc"], reverse=True)

    print("\nRanking (top):")
    for r in ok_sorted[:12]:
        print(f"- {r['name']}: best_val_acc={r['best_val_acc']} | log={r['log_path']}")

if __name__ == "__main__":
    main()
