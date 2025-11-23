from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SHOW_FIGURES = "agg" not in plt.get_backend().lower()
TUNE_EPOCHS = 2
FINAL_EPOCHS = 6
BATCH_SIZE = 128
VAL_SIZE = 5_000


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)


@dataclass
class TrialResult:
    lr: float
    val_accuracy: float
    val_loss: float


@dataclass
class OptimizerResult:
    name: str
    best_lr: float
    trials: List[TrialResult]
    history: TrainingHistory
    test_loss: float
    test_accuracy: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    batch_size: int = BATCH_SIZE, val_size: int = VAL_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(__file__).parent / "data"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator)

    pin_memory = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=2, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_optimizer(name: str, params: Sequence[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgdm":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_one_configuration(
    opt_name: str,
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> Tuple[nn.Module, TrainingHistory]:
    set_seed(42)
    model = SimpleCNN().to(device)
    model.apply(init_weights)
    optimizer = build_optimizer(opt_name, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory()
    for _ in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accuracies.append(train_acc)
        history.val_accuracies.append(val_acc)

    return model, history


def plot_curves(name: str, history: TrainingHistory, output_path: Path) -> None:
    epochs = np.arange(1, len(history.train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history.train_losses, label="Train")
    axes[0].plot(epochs, history.val_losses, label="Validation")
    axes[0].set_title(f"{name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_accuracies, label="Train")
    axes[1].plot(epochs, history.val_accuracies, label="Validation")
    axes[1].set_title(f"{name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def save_results_table(results: List[OptimizerResult], output_path: Path) -> None:
    lines = [
        "| Optimizer | Best LR | Test Accuracy | Test Loss |",
        "| --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.name} | {res.best_lr:.4g} | {res.test_accuracy * 100:.2f}% | {res.test_loss:.4f} |"
        )
    output_path.write_text("\n".join(lines))


def format_trials(trials: List[TrialResult]) -> str:
    parts = []
    for trial in trials:
        parts.append(
            f"lr={trial.lr:.4g} (val acc={trial.val_accuracy * 100:.2f}%, val loss={trial.val_loss:.4f})"
        )
    return "; ".join(parts)


def main() -> None:
    set_seed(123)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders()

    lr_candidates: Dict[str, List[float]] = {
        "SGD": [0.1, 0.05, 0.01],
        "SGDM": [0.1, 0.05, 0.01],
        "Adagrad": [0.1, 0.05, 0.01],
        "RMSprop": [0.01, 0.005, 0.001],
        "Adam": [0.001, 0.0005, 0.0002],
    }

    output_dir = Path(__file__).parent
    results: List[OptimizerResult] = []
    criterion = nn.CrossEntropyLoss()

    for opt_name, candidates in lr_candidates.items():
        print(f"\n=== {opt_name} tuning ===")
        trials: List[TrialResult] = []
        best_lr = candidates[0]
        best_val_acc = -float("inf")

        for lr in candidates:
            _, history = train_one_configuration(
                opt_name, lr, train_loader, val_loader, device, epochs=TUNE_EPOCHS
            )
            val_acc = max(history.val_accuracies)
            val_loss = min(history.val_losses)
            trials.append(TrialResult(lr=lr, val_accuracy=val_acc, val_loss=val_loss))
            print(f"lr={lr:.4g}: val acc={val_acc * 100:.2f}%, val loss={val_loss:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr

        print(f"Selected learning rate for {opt_name}: {best_lr:.4g}")

        final_model, history = train_one_configuration(
            opt_name, best_lr, train_loader, val_loader, device, epochs=FINAL_EPOCHS
        )
        test_loss, test_acc = run_epoch(final_model, test_loader, criterion, device)
        print(
            f"{opt_name}: test acc={test_acc * 100:.2f}%, test loss={test_loss:.4f} | LR trials: {format_trials(trials)}"
        )

        plot_curves(opt_name, history, output_dir / f"{opt_name.lower()}_curves.png")
        results.append(
            OptimizerResult(
                name=opt_name,
                best_lr=best_lr,
                trials=trials,
                history=history,
                test_loss=test_loss,
                test_accuracy=test_acc,
            )
        )

    save_results_table(results, output_dir / "results_table.md")
    print("\nFinished. Results table saved to results_table.md")


if __name__ == "__main__":
    main()
