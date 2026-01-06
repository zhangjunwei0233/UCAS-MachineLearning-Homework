"""
Batch prediction script for multiple trained models.
Automatically finds all experiment checkpoints and generates predictions for each.
Useful for generating submissions from all models trained by run_experiments.py.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from text_cnn import TextCNN, regex_tokenize, encode_tokens, encode_chars


# -------------------------
# Dataset for test inference
# -------------------------
class TestDataset(Dataset):
    """
    PyTorch Dataset for test data (without labels).

    Args:
        df: DataFrame with 'Phrase' column
        vocab: Word vocabulary dictionary
        max_len: Maximum word sequence length
        use_char: Whether to use character-level features
        char_vocab: Character vocabulary dictionary (if use_char=True)
        max_char_len: Maximum character sequence length
    """
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        max_len: int,
        use_char: bool = False,
        char_vocab: Optional[Dict[str, int]] = None,
        max_char_len: int = 0,
    ):
        self.phrases = df["Phrase"].tolist()
        self.vocab = vocab
        self.max_len = int(max_len)
        self.use_char = bool(use_char)
        self.char_vocab = char_vocab
        self.max_char_len = int(max_char_len)

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single test sample.

        Returns:
            If use_char=False: (word_tensor,)
            If use_char=True: (word_tensor, char_tensor)
        """
        tokens = regex_tokenize(self.phrases[idx])
        encoded = encode_tokens(tokens, self.vocab, self.max_len)
        word_tensor = torch.tensor(encoded, dtype=torch.long)

        if self.use_char and self.char_vocab is not None and self.max_char_len > 0:
            char_encoded = encode_chars(self.phrases[idx], self.char_vocab, self.max_char_len)
            char_tensor = torch.tensor(char_encoded, dtype=torch.long)
            return word_tensor, char_tensor

        return (word_tensor,)


# -------------------------
# Load model from checkpoint
# -------------------------
def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device
) -> Tuple[TextCNN, Dict[str, int], Optional[Dict[str, int]], dict]:
    """
    Load a trained TextCNN model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on (CPU or CUDA)

    Returns:
        Tuple of (model, vocab, char_vocab, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {}) or {}
    vocab = ckpt["vocab"]
    char_vocab = ckpt.get("char_vocab", None)
    num_classes = int(ckpt.get("num_classes", config.get("num_classes", 5)))

    use_multichannel = bool(config.get("use_multichannel", False))
    embed_dim = int(config.get("embed_dim", 256))

    # If multichannel=True, TextCNN ctor requires static_embedding;
    # provide a dummy matrix; real weights will be loaded from state_dict.
    static_init = None
    if use_multichannel:
        static_init = torch.zeros(len(vocab), embed_dim)

    kernel_sizes = [int(k) for k in str(config.get("kernel_sizes", "2,3,4,5,7")).split(",") if k.strip()]
    char_kernel_sizes = [int(k) for k in str(config.get("char_kernel_sizes", "3,4,5")).split(",") if k.strip()]

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=int(config.get("num_filters", 256)),
        dropout=float(config.get("dropout", 0.5)),
        multichannel=use_multichannel,
        static_embedding=static_init,
        use_batchnorm=bool(config.get("use_batchnorm", False)),
        use_double_conv=bool(config.get("use_double_conv", False)),
        proj_dim=int(config.get("proj_dim", 0)),
        use_char=bool(config.get("use_char_cnn", False)),
        char_vocab_size=len(char_vocab) if char_vocab is not None else 0,
        char_embed_dim=int(config.get("char_embed_dim", 50)),
        char_kernel_sizes=char_kernel_sizes,
        char_num_filters=int(config.get("char_num_filters", 50)),
    ).to(device)

    state = ckpt["model_state_dict"]
    # strict=False to tolerate older keys if any
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, vocab, char_vocab, config


# -------------------------
# Predict & write submission
# -------------------------
def predict_one(
    checkpoint_path: str,
    test_path: str,
    output_path: str,
    batch_size: int,
) -> None:
    """
    Generate predictions for a single model checkpoint.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_path: Path to test TSV file
        output_path: Path to save predictions CSV
        batch_size: Batch size for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREDICT] ckpt={checkpoint_path}")
    print(f"Using device: {device}")

    # Load model and vocabularies
    model, vocab, char_vocab, config = load_checkpoint_model(checkpoint_path, device)

    use_char = bool(config.get("use_char_cnn", False)) and (char_vocab is not None)
    max_len = int(config.get("max_len", 110))
    max_char_len = int(config.get("char_max_len", 200))

    # Load test data
    df_test = pd.read_csv(test_path, sep="\t")
    if "Phrase" not in df_test.columns or "PhraseId" not in df_test.columns:
        raise ValueError("Test file must have columns: PhraseId, Phrase.")

    dataset = TestDataset(
        df=df_test,
        vocab=vocab,
        max_len=max_len,
        use_char=use_char,
        char_vocab=char_vocab,
        max_char_len=max_char_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Generate predictions
    all_preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                words, chars = batch
                chars = chars.to(device)
            else:
                (words,) = batch
                chars = None

            words = words.to(device)
            logits = model(words, chars)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())

    # Save predictions to CSV
    out = pd.DataFrame({"PhraseId": df_test["PhraseId"], "Sentiment": all_preds})
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}\n")


# -------------------------
# Batch: pick latest run per exp name
# -------------------------
# Regular expression to parse experiment directory names (format: name_YYYYMMDD_HHMMSS)
RUN_DIR_RE = re.compile(r"^(?P<name>.+)_(?P<ts>\d{8}_\d{6})$")


def collect_latest_checkpoints(
    runs_dir: str,
    exclude_prefixes: Tuple[str, ...] = ("12_ordinal",),
) -> List[Tuple[str, Path]]:
    """
    Collect the latest checkpoint for each experiment.
    Returns list of (exp_name, model_pt_path), choosing the latest timestamp run for each exp_name.

    Args:
        runs_dir: Directory containing experiment run folders
        exclude_prefixes: Tuple of experiment name prefixes to exclude

    Returns:
        List of (experiment_name, checkpoint_path) tuples
    """
    root = Path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    best: Dict[str, Tuple[str, Path]] = {}  # exp_name -> (ts, model_path)
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = RUN_DIR_RE.match(p.name)
        if not m:
            continue

        exp_name = m.group("name")
        ts = m.group("ts")

        # Skip experiments with excluded prefixes
        if any(exp_name.startswith(pref) for pref in exclude_prefixes):
            continue

        model_path = p / "model.pt"
        if not model_path.exists():
            continue

        # Keep the latest timestamp for each experiment
        if exp_name not in best or ts > best[exp_name][0]:
            best[exp_name] = (ts, model_path)

    # sort by exp_name for stable order
    items = sorted(((k, v[1]) for k, v in best.items()), key=lambda x: x[0])
    return items


def sanitize_filename(s: str) -> str:
    """Replace non-alphanumeric characters with underscores for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)


def main():
    """
    Main function for batch prediction.
    Finds all experiment checkpoints and generates predictions for each.
    """
    ap = argparse.ArgumentParser(description="Batch predict test.tsv for all experiment checkpoints (non-ordinal).")
    ap.add_argument("--runs_dir", type=str, default="exp_runs_report", help="Directory containing experiment run folders.")
    ap.add_argument("--test_path", type=str, default="data/test.tsv", help="Path to test.tsv")
    ap.add_argument("--out_dir", type=str, default="submissions", help="Output directory for submission CSVs.")
    ap.add_argument("--batch_size", type=int, default=256, help="Inference batch size.")
    ap.add_argument("--exclude_prefix", type=str, default="12_ordinal", help="Exclude experiments whose name starts with this prefix. Comma-separated allowed.")
    args = ap.parse_args()

    # Parse excluded prefixes
    exclude_prefixes = tuple([x.strip() for x in args.exclude_prefix.split(",") if x.strip()])
    ckpts = collect_latest_checkpoints(args.runs_dir, exclude_prefixes=exclude_prefixes)

    if not ckpts:
        print("No checkpoints found. Check runs_dir and folder naming.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Found {len(ckpts)} experiments (latest checkpoint each):")
    for name, path in ckpts:
        print(f" - {name}: {path}")

    # Generate predictions for each checkpoint
    for exp_name, ckpt_path in ckpts:
        out_name = f"submission_{sanitize_filename(exp_name)}.csv"
        out_path = str(Path(args.out_dir) / out_name)
        predict_one(
            checkpoint_path=str(ckpt_path),
            test_path=args.test_path,
            output_path=out_path,
            batch_size=args.batch_size,
        )

    print("All submissions generated.")


if __name__ == "__main__":
    main()
