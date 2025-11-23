from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SHOW_FIGURES = "agg" not in plt.get_backend().lower()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingHistory:
    train_losses: List[float]
    val_losses: List[float]


@dataclass
class ExperimentConfig:
    name: str
    loss_type: str  # "hinge" or "logistic"
    lambda_reg: float
    lr: float
    epochs: int
    batch_size: int


def generate_binary_gaussians(
    n_samples: int = 400,
    class_distance: float = 3.0,
    noise_std: float = 0.8,
    label_flip_fraction: float = 0.05,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two Gaussian blobs with optional label noise."""
    assert n_samples % 2 == 0, "n_samples should be even."
    set_seed(seed)
    half = n_samples // 2

    mean_a = torch.tensor([-class_distance / 2, -1.0])
    mean_b = torch.tensor([class_distance / 2, 1.0])

    cov = noise_std * torch.eye(2)
    dist_a = torch.distributions.MultivariateNormal(mean_a, covariance_matrix=cov)
    dist_b = torch.distributions.MultivariateNormal(mean_b, covariance_matrix=cov)

    class_a = dist_a.sample((half,))
    class_b = dist_b.sample((half,))

    X = torch.cat([class_a, class_b], dim=0)
    y = torch.cat([torch.zeros(half), torch.ones(half)], dim=0)

    # Add some label noise to illustrate role of regularization
    n_flip = max(1, int(n_samples * label_flip_fraction))
    flip_idx = torch.randperm(n_samples)[:n_flip]
    y[flip_idx] = 1 - y[flip_idx]

    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def train_val_split(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.75,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    set_seed(seed)
    n = X.shape[0]
    n_train = int(n * train_ratio)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class LinearBinaryClassifier(nn.Module):
    """Simple linear layer for binary classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)


def hinge_loss(
    outputs: torch.Tensor, y: torch.Tensor, model: nn.Module, lambda_reg: float
) -> torch.Tensor:
    """Compute hinge loss with L2 penalty on weights."""
    signed_targets = y.float() * 2 - 1  # convert from {0,1} to {-1,+1}
    scores = outputs.squeeze()
    margins = 1 - signed_targets * scores
    hinge = torch.clamp(margins, min=0)
    loss = hinge.mean()
    reg = 0.5 * lambda_reg * torch.sum(model.linear.weight.pow(2))
    return loss + reg


def logistic_loss(
    outputs: torch.Tensor, y: torch.Tensor, model: nn.Module, lambda_reg: float
) -> torch.Tensor:
    logits = outputs.squeeze()
    y_float = y.float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_float)
    reg = 0.5 * lambda_reg * torch.sum(model.linear.weight.pow(2))
    return loss + reg


LOSS_FUNCS: Dict[str, Callable[[torch.Tensor, torch.Tensor, nn.Module, float], torch.Tensor]] = {
    "hinge": hinge_loss,
    "logistic": logistic_loss,
}


def get_predictions(outputs: torch.Tensor, loss_type: str) -> torch.Tensor:
    logits = outputs.squeeze()
    if loss_type == "logistic":
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).long()
    return (logits >= 0).long()


def train_model(
    model: LinearBinaryClassifier,
    loss_type: str,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lambda_reg: float,
    epochs: int,
    batch_size: int,
) -> TrainingHistory:
    dataset = TensorDataset(X_train, y_train)
    history = TrainingHistory(train_losses=[], val_losses=[])

    loss_fn = LOSS_FUNCS[loss_type]

    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        epoch_losses: List[float] = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y, model, lambda_reg)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val, model, lambda_reg).item()

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

    return history


def evaluate_model(
    model: LinearBinaryClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_type: str,
    lambda_reg: float,
) -> Dict[str, float]:
    loss_fn = LOSS_FUNCS[loss_type]
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = loss_fn(outputs, y, model, lambda_reg).item()
        preds = get_predictions(outputs, loss_type)
        accuracy = float((preds == y.long()).float().mean().item())

    return {"loss": loss, "accuracy": accuracy}


def plot_loss_curves(
    histories: Sequence[Tuple[str, TrainingHistory]],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, history in histories:
        ax.plot(history.train_losses, label=f"{label} - train")
        ax.plot(history.val_losses, linestyle="--", label=f"{label} - val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_binary_decision_boundaries(
    models: Sequence[Tuple[str, LinearBinaryClassifier, str]],
    X: torch.Tensor,
    y: torch.Tensor,
    output_path: Path,
) -> None:
    x_min, x_max = X[:, 0].min().item() - 1.0, X[:, 0].max().item() + 1.0
    y_min, y_max = X[:, 1].min().item() - 1.0, X[:, 1].max().item() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    fig, ax = plt.subplots(figsize=(6, 5))
    colors_by_class = {0: "tab:blue", 1: "tab:red"}
    for cls, color in colors_by_class.items():
        mask = (y.numpy() == cls)
        ax.scatter(
            X[mask, 0].numpy(),
            X[mask, 1].numpy(),
            color=color,
            label=f"Class {cls}",
            s=30,
            edgecolors="k",
            alpha=0.75,
        )

    colors = ["tab:green", "tab:purple", "tab:orange"]
    custom_lines: List[Line2D] = []

    for (label, model, loss_type), color in zip(models, colors):
        model.eval()
        with torch.no_grad():
            outputs = model(grid).reshape(xx.shape)
        contour = ax.contour(
            xx,
            yy,
            outputs.numpy(),
            levels=[0],
            colors=[color],
            linewidths=2,
        )
        custom_lines.append(Line2D([0], [0], color=color, lw=2, label=label))

    ax.set_title("Decision Boundaries: SVM vs Logistic Regression")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    labels.extend([line.get_label() for line in custom_lines])
    ax.legend(handles=handles, labels=labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def run_binary_experiment(output_dir: Path) -> None:
    print("Generating binary classification dataset for SVM vs Logistic Regression...")
    X, y = generate_binary_gaussians(
        n_samples=500,
        class_distance=3.0,
        noise_std=0.9,
        label_flip_fraction=0.08,
        seed=123,
    )
    X_train, y_train, X_val, y_val = train_val_split(X, y, train_ratio=0.8, seed=21)
    configs = [
        ExperimentConfig(
            name="Linear SVM (hinge)",
            loss_type="hinge",
            lambda_reg=0.05,
            lr=0.1,
            epochs=200,
            batch_size=64,
        ),
        ExperimentConfig(
            name="Logistic Regression",
            loss_type="logistic",
            lambda_reg=0.01,
            lr=0.1,
            epochs=200,
            batch_size=64,
        ),
    ]

    histories: List[Tuple[str, TrainingHistory]] = []
    decision_models: List[Tuple[str, LinearBinaryClassifier, str]] = []

    for config in configs:
        print(f"\nTraining {config.name} ...")
        model = LinearBinaryClassifier(input_dim=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        history = train_model(
            model,
            config.loss_type,
            optimizer,
            X_train,
            y_train,
            X_val,
            y_val,
            config.lambda_reg,
            config.epochs,
            config.batch_size,
        )
        metrics_val = evaluate_model(
            model, X_val, y_val, config.loss_type, config.lambda_reg
        )
        print(
            f"→ Final val loss: {metrics_val['loss']:.4f}, accuracy: {metrics_val['accuracy']*100:.2f}%"
        )
        histories.append((config.name, history))
        decision_models.append((config.name, model, config.loss_type))

    plot_loss_curves(
        histories,
        output_dir / "binary_loss_curves.png",
        title="Training vs Validation Loss (Binary)",
    )
    plot_binary_decision_boundaries(
        decision_models,
        X_train,
        y_train,
        output_dir / "binary_decision_boundaries.png",
    )


def main() -> None:
    set_seed(42)
    output_dir = Path(__file__).parent
    run_binary_experiment(output_dir)
    print(f"\nDone! Plots and outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
