"""
Prediction script for trained TextCNN model.
Loads a trained model checkpoint and generates predictions on test data.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from text_cnn import (
    TextCNN,
    regex_tokenize,
    encode_tokens,
    encode_chars,
)


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


def load_checkpoint_model(
    checkpoint_path: str, device: torch.device
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
    config = ckpt.get("config", {})
    vocab = ckpt["vocab"]
    char_vocab = ckpt.get("char_vocab", None)
    num_classes = int(ckpt.get("num_classes", config.get("num_classes", 5)))

    use_multichannel = bool(config.get("use_multichannel", False))
    embed_dim = int(config.get("embed_dim", 256))

    # Provide a dummy static matrix if multichannel to satisfy ctor;
    # weights will be loaded from state_dict.
    static_init = None
    if use_multichannel:
        static_init = torch.zeros(len(vocab), embed_dim)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=num_classes,
        kernel_sizes=[int(k) for k in str(config.get("kernel_sizes", "2,3,4,5,7")).split(",") if k.strip()],
        num_filters=int(config.get("num_filters", 200)),
        dropout=float(config.get("dropout", 0.5)),
        multichannel=use_multichannel,
        static_embedding=static_init,
        use_batchnorm=bool(config.get("use_batchnorm", False)),
        use_double_conv=bool(config.get("use_double_conv", False)),
        proj_dim=int(config.get("proj_dim", 0)),
        use_char=bool(config.get("use_char_cnn", False)),
        char_vocab_size=len(char_vocab) if char_vocab is not None else 0,
        char_embed_dim=int(config.get("char_embed_dim", 50)),
        char_kernel_sizes=[int(k) for k in str(config.get("char_kernel_sizes", "3,4,5")).split(",") if k.strip()],
        char_num_filters=int(config.get("char_num_filters", 50)),
    ).to(device)

    state = ckpt["model_state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, vocab, char_vocab, config


def predict(
    checkpoint_path: str,
    test_path: str,
    output_path: str,
    batch_size: int,
) -> None:
    """
    Generate predictions on test data using a trained model.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_path: Path to test TSV file
        output_path: Path to save predictions CSV
        batch_size: Batch size for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and vocabularies from checkpoint
    model, vocab, char_vocab, config = load_checkpoint_model(checkpoint_path, device)
    use_char = bool(config.get("use_char_cnn", False)) and char_vocab is not None

    max_len = int(config.get("max_len", 100))
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
    print(f"Saved predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained TextCNN on test set and produce submission CSV.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--test_path", type=str, default="data/test.tsv", help="Path to test TSV.")
    parser.add_argument("--output_path", type=str, default="submission.csv", help="Path to save predictions CSV.")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        checkpoint_path=args.checkpoint,
        test_path=args.test_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
