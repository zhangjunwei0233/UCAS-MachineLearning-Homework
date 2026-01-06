"""
TextCNN model for sentiment analysis on movie reviews (SST-5 dataset).
Implements a CNN-based text classifier with multiple advanced features:
- Multi-channel embeddings (Kim 2014)
- Character-level CNN
- Label smoothing and class weighting
- EMA (Exponential Moving Average) for model weights
- Warmup + Cosine annealing learning rate schedule
"""

import argparse
import math
import os
import random
import re
import copy
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Special tokens for vocabulary
PAD_TOKEN = "<pad>"  # Padding token for sequences
UNK_TOKEN = "<unk>"  # Unknown token for out-of-vocabulary words
NUM_TOKEN = "<num>"  # Token to replace all numeric values
CHAR_PAD = "<c_pad>"  # Padding token for character sequences
CHAR_UNK = "<c_unk>"  # Unknown token for characters


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, PyTorch CPU and GPU."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regex_tokenize(text: str) -> List[str]:
    """
    Regex-based tokenizer that keeps English contractions and normalizes numbers.
    - Lowercases text.
    - Replaces digit sequences with NUM_TOKEN.
    - Extracts words with optional apostrophes (e.g., don't, isn't).
    """
    text = str(text).lower()
    text = re.sub(r"\d+(\.\d+)?", f" {NUM_TOKEN} ", text)
    pattern = r"[a-z]+(?:'[a-z]+)?|<num>"
    return re.findall(pattern, text)


def build_char_vocab(texts: Iterable[str], max_size: int = 128) -> Dict[str, int]:
    """
    Build character-level vocabulary from texts.

    Args:
        texts: Iterable of text strings
        max_size: Maximum vocabulary size (default: 128)

    Returns:
        Dictionary mapping characters to integer IDs
    """
    counter: Counter = Counter()
    for line in texts:
        for ch in str(line):
            counter[ch] += 1
    vocab = {CHAR_PAD: 0, CHAR_UNK: 1}
    for ch, _ in counter.most_common():
        if len(vocab) >= max_size:
            break
        vocab[ch] = len(vocab)
    return vocab


def build_vocab(texts: Iterable[str], min_freq: int, max_size: int) -> Dict[str, int]:
    """
    Build word-level vocabulary from texts.

    Args:
        texts: Iterable of text strings
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size

    Returns:
        Dictionary mapping tokens to integer IDs
    """
    counter: Counter = Counter()
    for line in texts:
        counter.update(regex_tokenize(line))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, NUM_TOKEN: 2}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    """
    Encode a list of tokens to integer IDs with padding/truncation.

    Args:
        tokens: List of token strings
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length

    Returns:
        List of integer IDs (padded or truncated to max_len)
    """
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Tokenize and encode a text string to integer IDs."""
    return encode_tokens(regex_tokenize(text), vocab, max_len)


def encode_chars(text: str, char_vocab: Dict[str, int], max_char_len: int) -> List[int]:
    """
    Encode text at character level with padding/truncation.

    Args:
        text: Input text string
        char_vocab: Character vocabulary dictionary
        max_char_len: Maximum character sequence length

    Returns:
        List of character IDs (padded or truncated to max_char_len)
    """
    text = str(text)
    ids = [char_vocab.get(ch, char_vocab[CHAR_UNK]) for ch in text[:max_char_len]]
    if len(ids) < max_char_len:
        ids = ids + [char_vocab[CHAR_PAD]] * (max_char_len - len(ids))
    return ids


# -------------------------
# Optional label smoothing (CE)
# -------------------------
class LabelSmoothingLoss(nn.Module):
    """
    Fallback label smoothing (CE) if torch CE doesn't support label_smoothing.
    """
    def __init__(self, smoothing: float, num_classes: int, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.num_classes = int(num_classes)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)  # [B, C]
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        per_sample = -(true_dist * log_probs).sum(dim=1)  # [B]
        if self.weight is not None:
            sample_weights = self.weight[targets]         # [B]
            per_sample = per_sample * sample_weights
        return per_sample.mean()


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        max_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate after warmup
        min_lr: Minimum learning rate at the end of cosine decay
    """
    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)

    def step(self, epoch: int) -> None:
        """Update learning rate based on current epoch."""
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            # Linear warmup phase
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Maintains a shadow copy of the model with exponentially decayed weights.

    Args:
        model: The model to track
        decay: EMA decay factor (default: 0.999)
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    def update(self, model: nn.Module) -> None:
        """Update EMA weights: ema = decay * ema + (1 - decay) * model."""
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                ema_v.copy_(ema_v * self.decay + msd[k] * (1.0 - self.decay))


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis phrases.
    Supports both word-level and character-level encoding.

    Args:
        df: DataFrame with 'Phrase' and 'Sentiment' columns
        vocab: Word vocabulary dictionary
        max_len: Maximum word sequence length
        use_char: Whether to use character-level features
        char_vocab: Character vocabulary dictionary (if use_char=True)
        max_char_len: Maximum character sequence length
        word_dropout: Probability of dropping words during training
        is_train: Whether this is training data (enables word dropout)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        max_len: int,
        use_char: bool = False,
        char_vocab: Optional[Dict[str, int]] = None,
        max_char_len: int = 0,
        word_dropout: float = 0.0,
        is_train: bool = False,
    ):
        self.phrases = df["Phrase"].tolist()
        self.labels = df["Sentiment"].tolist()
        self.vocab = vocab
        self.max_len = int(max_len)
        self.use_char = bool(use_char)
        self.char_vocab = char_vocab
        self.max_char_len = int(max_char_len)
        self.word_dropout = float(word_dropout)
        self.is_train = bool(is_train)

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Returns:
            If use_char=False: (word_tensor, label_tensor)
            If use_char=True: (word_tensor, char_tensor, label_tensor)
        """
        tokens = regex_tokenize(self.phrases[idx])
        # Apply word dropout during training
        if self.is_train and self.word_dropout > 0:
            tokens = [t for t in tokens if random.random() > self.word_dropout] or tokens[:1]
        encoded = encode_tokens(tokens, self.vocab, self.max_len)
        word_tensor = torch.tensor(encoded, dtype=torch.long)
        label_tensor = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.use_char and self.char_vocab is not None and self.max_char_len > 0:
            char_encoded = encode_chars(self.phrases[idx], self.char_vocab, self.max_char_len)
            char_tensor = torch.tensor(char_encoded, dtype=torch.long)
            return word_tensor, char_tensor, label_tensor
        return word_tensor, label_tensor


class TextCNN(nn.Module):
    """
    Standard multi-class TextCNN (CE classification).
    - Supports: multichannel (Kim), double conv, batchnorm, projection, char-CNN channel.
    - Output: logits [B, num_classes]
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        kernel_sizes: List[int],
        num_filters: int,
        dropout: float,
        multichannel: bool = False,
        static_embedding: Optional[torch.Tensor] = None,
        use_batchnorm: bool = False,
        use_double_conv: bool = False,
        proj_dim: int = 0,
        use_char: bool = False,
        char_vocab_size: int = 0,
        char_embed_dim: int = 0,
        char_kernel_sizes: Optional[List[int]] = None,
        char_num_filters: int = 0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        channels = 2 if multichannel else 1
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_static: Optional[nn.Embedding] = None

        if static_embedding is not None:
            if static_embedding.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"static_embedding shape {tuple(static_embedding.shape)} != {(vocab_size, embed_dim)}"
                )
            self.embedding.weight.data.copy_(static_embedding)

        if multichannel:
            if static_embedding is None:
                raise ValueError(
                    "use_multichannel=True requires static_embedding (e.g., GloVe). "
                    "Disable --use_multichannel if not using GloVe."
                )
            self.embedding_static = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.embedding_static.weight.data.copy_(static_embedding)
            self.embedding_static.weight.requires_grad = False
            self.embedding.weight.data.copy_(static_embedding)

        in_channels = embed_dim * channels
        self.use_double_conv = bool(use_double_conv)
        if self.use_double_conv:
            self.convs1 = nn.ModuleList(
                [nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
            )
            self.convs2 = nn.ModuleList(
                [nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=1) for _ in kernel_sizes]
            )
            self.convs = None
        else:
            self.convs = nn.ModuleList(
                [nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
            )
            self.convs1 = None
            self.convs2 = None

        self.use_batchnorm = bool(use_batchnorm)
        self.bn_word = nn.BatchNorm1d(num_filters * len(kernel_sizes)) if self.use_batchnorm else None
        self.dropout = nn.Dropout(float(dropout))

        # Char CNN
        self.use_char = bool(use_char) and char_vocab_size > 0 and char_embed_dim > 0 and char_num_filters > 0
        if self.use_char:
            ck = char_kernel_sizes if char_kernel_sizes is not None else [3, 4, 5]
            self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
            self.char_convs = nn.ModuleList(
                [nn.Conv1d(in_channels=char_embed_dim, out_channels=char_num_filters, kernel_size=k) for k in ck]
            )
            self.char_bn = nn.BatchNorm1d(char_num_filters * len(ck)) if self.use_batchnorm else None
            self.char_kernel_sizes = ck
            self.char_num_filters = int(char_num_filters)
        else:
            self.char_embedding = None
            self.char_convs = None
            self.char_bn = None
            self.char_kernel_sizes = []
            self.char_num_filters = 0

        feature_dim = num_filters * len(kernel_sizes)
        if self.use_char:
            feature_dim += self.char_num_filters * len(self.char_kernel_sizes)

        self.proj = nn.Linear(feature_dim, proj_dim) if proj_dim > 0 else None
        final_dim = proj_dim if proj_dim > 0 else feature_dim

        self.fc = nn.Linear(final_dim, self.num_classes)

    def forward(self, inputs: torch.Tensor, char_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        embed_trainable = self.embedding(inputs)  # [B, L, D]
        if self.embedding_static is not None:
            embed_static = self.embedding_static(inputs)
            embedded = torch.cat([embed_trainable, embed_static], dim=2)  # [B, L, 2D]
        else:
            embedded = embed_trainable

        embedded = embedded.transpose(1, 2)  # [B, D*(channels), L]

        conv_outputs: List[torch.Tensor] = []
        if self.use_double_conv:
            assert self.convs1 is not None and self.convs2 is not None
            for conv1, conv2 in zip(self.convs1, self.convs2):
                x = torch.relu(conv1(embedded))
                x = torch.relu(conv2(x))
                pooled = torch.max(x, dim=2).values
                conv_outputs.append(pooled)
        else:
            assert self.convs is not None
            for conv in self.convs:
                x = torch.relu(conv(embedded))
                pooled = torch.max(x, dim=2).values
                conv_outputs.append(pooled)

        features = torch.cat(conv_outputs, dim=1)
        if self.use_batchnorm and self.bn_word is not None:
            features = self.bn_word(features)

        if self.use_char and char_inputs is not None and self.char_embedding is not None and self.char_convs is not None:
            char_embed = self.char_embedding(char_inputs)  # [B, CL, CD]
            char_embed = char_embed.transpose(1, 2)        # [B, CD, CL]
            char_outs: List[torch.Tensor] = []
            for conv in self.char_convs:
                cx = torch.relu(conv(char_embed))
                cpooled = torch.max(cx, dim=2).values
                char_outs.append(cpooled)
            char_features = torch.cat(char_outs, dim=1)
            if self.use_batchnorm and self.char_bn is not None:
                char_features = self.char_bn(char_features)
            features = torch.cat([features, char_features], dim=1)

        if self.proj is not None:
            features = torch.relu(self.proj(features))

        features = self.dropout(features)
        return self.fc(features)  # [B, K]


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets with stratification by sentiment.

    Args:
        df: Input DataFrame with 'Sentiment' column
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    rng = random.Random(seed)
    val_parts, train_parts = [], []
    for _, group in df.groupby("Sentiment"):
        indices = list(group.index)
        rng.shuffle(indices)
        split = int(len(indices) * val_ratio)
        val_idx = indices[:split]
        train_idx = indices[split:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[train_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def stratified_kfold_indices(df: pd.DataFrame, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """
    Generate stratified k-fold cross-validation indices.

    Args:
        df: Input DataFrame with 'Sentiment' column
        k: Number of folds
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for label, group in df.groupby("Sentiment"):
        idx = list(group.index)
        rng.shuffle(idx)
        label_to_indices[int(label)] = idx

    # Distribute samples across folds in round-robin fashion
    folds: List[List[int]] = [[] for _ in range(k)]
    for _, indices in label_to_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    all_indices = set(df.index.tolist())
    fold_pairs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = list(all_indices - set(val_idx))
        fold_pairs.append((train_idx, val_idx))
    return fold_pairs


def load_data(train_path: str) -> pd.DataFrame:
    """
    Load training data from TSV file.

    Args:
        train_path: Path to training TSV file

    Returns:
        DataFrame with 'Phrase' and 'Sentiment' columns

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    df = pd.read_csv(train_path, sep="\t")
    if "Phrase" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("Expected columns: 'Phrase' and 'Sentiment'.")
    return df


def load_glove_embeddings(glove_path: str, vocab: Dict[str, int], embed_dim: int) -> Optional[torch.Tensor]:
    """
    Load pre-trained GloVe embeddings for vocabulary.

    Args:
        glove_path: Path to GloVe text file
        vocab: Vocabulary dictionary
        embed_dim: Embedding dimension

    Returns:
        Embedding matrix tensor of shape (vocab_size, embed_dim), or None if no path provided

    Raises:
        FileNotFoundError: If GloVe file doesn't exist
    """
    if not glove_path:
        return None
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    embed_dim = int(embed_dim)
    # Random init for OOV; keep PAD as zeros.
    embedding_matrix = torch.empty(len(vocab), embed_dim).uniform_(-0.25, 0.25)
    embedding_matrix[vocab[PAD_TOKEN]] = 0.0

    needed = set(vocab.keys())
    found = 0
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word not in needed:
                continue
            try:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            except ValueError:
                continue
            embedding_matrix[vocab[word]] = vec
            found += 1
    print(f"Loaded GloVe vectors: found {found}/{len(vocab)} tokens")
    return embedding_matrix


def apply_suffix_to_path(path: str, suffix: str) -> str:
    """
    Add a suffix to a file path before the extension.
    Example: apply_suffix_to_path("model.pt", "_fold1") -> "model_fold1.pt"
    """
    if not suffix:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: Dict[str, int],
    args: argparse.Namespace,
    char_vocab: Optional[Dict[str, int]] = None,
):
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        vocab: Word vocabulary
        args: Command-line arguments containing hyperparameters
        char_vocab: Character vocabulary (optional)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    use_char = args.use_char_cnn and char_vocab is not None
    train_dataset = PhraseDataset(
        train_df,
        vocab,
        args.max_len,
        use_char=use_char,
        char_vocab=char_vocab,
        max_char_len=args.char_max_len,
        word_dropout=args.word_dropout,
        is_train=True,
    )
    val_dataset = PhraseDataset(
        val_df,
        vocab,
        args.max_len,
        use_char=use_char,
        char_vocab=char_vocab,
        max_char_len=args.char_max_len,
        word_dropout=0.0,
        is_train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def make_ce_criterion(
    num_classes: int, class_weights: Optional[torch.Tensor], label_smoothing: float, device: torch.device
) -> nn.Module:
    weight = class_weights.to(device) if class_weights is not None else None
    if label_smoothing and label_smoothing > 0:
        try:
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(label_smoothing))
        except TypeError:
            return LabelSmoothingLoss(smoothing=float(label_smoothing), num_classes=int(num_classes), weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: Optional[float],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_per_step: bool = False,
    ema: Optional["ModelEMA"] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        if len(batch) == 3:
            batch_inputs, batch_chars, batch_labels = batch
            batch_chars = batch_chars.to(device)
        else:
            batch_inputs, batch_labels = batch
            batch_chars = None

        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_inputs, batch_chars)
        loss = criterion(logits, batch_labels)

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)
        if scheduler_per_step and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    compute_confusion: bool = False,
) -> Tuple[float, float, Optional[List[List[int]]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    confusion = None
    if compute_confusion:
        confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                batch_inputs, batch_chars, batch_labels = batch
                batch_chars = batch_chars.to(device)
            else:
                batch_inputs, batch_labels = batch
                batch_chars = None

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_inputs, batch_chars)
            loss = criterion(logits, batch_labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * batch_inputs.size(0)
            correct += (preds == batch_labels).sum().item()

            if confusion is not None:
                for p, t in zip(preds.tolist(), batch_labels.tolist()):
                    confusion[int(t)][int(p)] += 1

    avg_loss = total_loss / len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)
    return avg_loss, acc, confusion


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextCNN for Sentiment Analysis on Movie Reviews (Kaggle/SST-5).")
    parser.add_argument("--train_path", type=str, default="data/train.tsv", help="Path to training TSV file.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of training data used for validation.")
    parser.add_argument("--max_vocab_size", type=int, default=60000, help="Maximum vocabulary size.")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency for vocabulary.")

    parser.add_argument("--max_len", type=int, default=110, help="Maximum tokens per phrase.")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_filters", type=int, default=256, help="Number of filters per kernel size.")
    parser.add_argument("--kernel_sizes", type=str, default="2,3,4,5,7", help="Comma-separated kernel sizes.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument("--max_lr", type=float, default=3e-3, help="Max LR for OneCycle (if enabled).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (base).")
    parser.add_argument("--fc_weight_decay", type=float, default=1e-3, help="Weight decay for classifier head.")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights (CE only).")
    parser.add_argument("--class_weight_sqrt", action="store_true", help="Use sqrt inverse-freq weights (CE only).")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing (CE only).")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs before cosine decay.")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--use_one_cycle", action="store_true", help="Use OneCycleLR scheduler instead of warmup-cosine.")
    parser.add_argument("--early_stop_patience", type=int, default=8, help="Early stop patience.")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping max norm (<=0 disables).")
    parser.add_argument("--word_dropout", type=float, default=0.05, help="Word-level dropout in training.")
    parser.add_argument("--use_ema", action="store_true", help="Track EMA of weights for evaluation.")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA decay factor.")

    parser.add_argument("--use_multichannel", action="store_true", help="Use dual-channel embeddings (requires GloVe).")
    parser.add_argument("--glove_path", type=str, default="", help="Path to GloVe vectors (txt).")
    parser.add_argument("--glove_dim", type=int, default=300, help="Dimension of provided GloVe vectors.")

    parser.add_argument("--use_batchnorm", action="store_true", help="Apply BatchNorm after conv pooling.")
    parser.add_argument("--use_double_conv", action="store_true", help="Two-layer conv block with 1x1 conv.")
    parser.add_argument("--proj_dim", type=int, default=0, help="Optional projection dim after concat (0 disables).")

    parser.add_argument("--use_char_cnn", action="store_true", help="Add character-level CNN channel.")
    parser.add_argument("--char_max_len", type=int, default=200, help="Max characters per phrase.")
    parser.add_argument("--char_embed_dim", type=int, default=50, help="Character embedding dimension.")
    parser.add_argument("--char_num_filters", type=int, default=50, help="Char filters per kernel size.")
    parser.add_argument("--char_kernel_sizes", type=str, default="3,4,5", help="Comma-separated char kernel sizes.")

    parser.add_argument("--k_folds", type=int, default=1, help="Use k-fold CV if >1.")
    parser.add_argument("--print_confusion", action="store_true", help="Print confusion matrix.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_path", type=str, default="checkpoints/text_cnn.pt", help="Where to save model.")
    return parser.parse_args(argv)


def train_one_split(args: argparse.Namespace, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_suffix: str = "") -> float:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = build_vocab(train_df["Phrase"].tolist(), args.min_freq, args.max_vocab_size)
    print(f"Vocab size: {len(vocab)}")

    char_vocab = None
    if args.use_char_cnn:
        char_vocab = build_char_vocab(train_df["Phrase"].tolist(), max_size=128)
        print(f"Char vocab size: {len(char_vocab)}")

    glove_matrix = None
    if args.glove_path:
        glove_matrix = load_glove_embeddings(args.glove_path, vocab, args.glove_dim)
        if glove_matrix is not None and glove_matrix.shape[1] != args.embed_dim:
            print(f"GloVe dim {glove_matrix.shape[1]} != embed_dim {args.embed_dim}, ignoring glove.")
            glove_matrix = None
    if args.use_multichannel:
        if glove_matrix is None:
            print("use_multichannel requested but no static embedding available; disabling multichannel.")
            args.use_multichannel = False

    num_classes = 5

    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",") if k.strip()]
    char_kernel_sizes = [int(k) for k in args.char_kernel_sizes.split(",") if k.strip()]

    train_loader, val_loader = make_dataloaders(train_df, val_df, vocab, args, char_vocab=char_vocab)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=args.num_filters,
        dropout=args.dropout,
        multichannel=args.use_multichannel,
        static_embedding=glove_matrix,
        use_batchnorm=args.use_batchnorm,
        use_double_conv=args.use_double_conv,
        proj_dim=args.proj_dim,
        use_char=args.use_char_cnn,
        char_vocab_size=len(char_vocab) if char_vocab is not None else 0,
        char_embed_dim=args.char_embed_dim,
        char_kernel_sizes=char_kernel_sizes,
        char_num_filters=args.char_num_filters,
    ).to(device)

    # Optimizer param groups (separate decay for classifier head)
    decay_base, decay_head, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(param)
        elif name == "fc.weight":
            decay_head.append(param)
        else:
            decay_base.append(param)

    groups = []
    if decay_base:
        groups.append({"params": decay_base, "weight_decay": args.weight_decay})
    if decay_head:
        groups.append({"params": decay_head, "weight_decay": args.fc_weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(groups, lr=args.lr)

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    # Criterion (always CE)
    class_weights = None
    if args.use_class_weights:
        class_counts = train_df["Sentiment"].value_counts().to_dict()
        max_count = max(class_counts.values())
        if args.class_weight_sqrt:
            class_weights = torch.tensor(
                [math.sqrt(max_count / class_counts.get(i, 1)) for i in range(num_classes)],
                dtype=torch.float
            )
        else:
            class_weights = torch.tensor(
                [max_count / class_counts.get(i, 1) for i in range(num_classes)],
                dtype=torch.float
            )
        print(f"Using class weights: {class_weights.tolist()}")
    criterion = make_ce_criterion(num_classes, class_weights, args.label_smoothing, device)

    scheduler = None
    scheduler_per_step = False
    if args.use_one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            steps_per_epoch=max(1, len(train_loader)),
            epochs=args.epochs,
            pct_start=0.3,
            anneal_strategy="cos",
        )
        scheduler_per_step = True
    else:
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            max_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
        )
        scheduler_per_step = False

    grad_clip = None if args.grad_clip <= 0 else args.grad_clip

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    save_path = apply_suffix_to_path(args.save_path, fold_suffix)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if not scheduler_per_step and scheduler is not None:
            scheduler.step(epoch)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
            scheduler=scheduler if scheduler_per_step else None,
            scheduler_per_step=scheduler_per_step,
            ema=ema,
        )

        eval_model = ema.ema if ema is not None else model

        val_loss, val_acc, _ = evaluate(
            model=eval_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            compute_confusion=False,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            src_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            best_state = src_state
            torch.save(
                {
                    "model_state_dict": src_state,
                    "vocab": vocab,
                    "char_vocab": char_vocab,
                    "config": vars(args),
                    "num_classes": num_classes,
                },
                save_path,
            )
            print(f"Saved improved model to {save_path}")
            no_improve = 0
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.print_confusion:
            _, _, confusion = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                compute_confusion=True,
            )
            if confusion is not None:
                print("Validation confusion matrix (rows=gold, cols=pred):")
                for row in confusion:
                    print(row)

    return best_val_acc


def run_training(args: argparse.Namespace) -> float:
    df = load_data(args.train_path).reset_index(drop=True)

    if args.k_folds <= 1:
        train_df, val_df = stratified_split(df, args.val_ratio, args.seed)
        return train_one_split(args, train_df, val_df)

    print(f"Running {args.k_folds}-fold cross validation")
    folds = stratified_kfold_indices(df, args.k_folds, args.seed)
    accs: List[float] = []
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_id}/{args.k_folds} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        acc = train_one_split(args, train_df, val_df, fold_suffix=f"_fold{fold_id}")
        accs.append(acc)
    avg_acc = sum(accs) / len(accs)
    print(f"\nAverage val accuracy over {args.k_folds} folds: {avg_acc:.4f}")
    return avg_acc


def main() -> None:
    args = parse_args()
    best_val_acc = run_training(args)
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
