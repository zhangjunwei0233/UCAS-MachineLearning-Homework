from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SHOW_FIGURES = "agg" not in plt.get_backend().lower()
TRAINING_MODES: List[Tuple[str, str]] = [
    ("gd", "Full-Batch Gradient Descent"),
    ("sgd", "Stochastic Gradient Descent"),
]
TRAINING_MODE_LABELS: Dict[str, str] = dict(TRAINING_MODES)


@dataclass
class TaskConfig:
    name: str
    task_type: str
    input_dim: int
    num_classes: int
    data_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor, Dict]]
    epochs: int
    lr: float
    batch_size: int
    data_kwargs: Dict | None = None


@dataclass
class TrainingHistory:
    optimizer_type: str
    losses: List[float]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class UnifiedLinearModel(nn.Module):
    """Single linear model that can act as regressor or classifier."""

    def __init__(self, input_dim: int, task_type: str, num_classes: int = 1):
        super().__init__()
        if task_type not in {"regression", "binary", "multiclass"}:
            raise ValueError(f"Unknown task type: {task_type}")

        self.task_type = task_type
        output_dim = 1 if task_type in {"regression", "binary"} else num_classes
        self.linear = nn.Linear(input_dim, output_dim)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(X)

        if self.task_type == "regression":
            target = y.float()
            return self.mse_loss(logits, target)
        if self.task_type == "binary":
            target = y.float()
            if target.ndim == 1:
                target = target.unsqueeze(1)
            return self.bce_loss(logits, target)

        target = y.squeeze(-1).long()
        return self.ce_loss(logits, target)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.forward(X)
        if self.task_type == "regression":
            return logits
        if self.task_type == "binary":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def predict_classes(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_type == "regression":
            raise ValueError("Regression model does not support class predictions.")

        if self.task_type == "binary":
            return (self.predict(X) >= 0.5).long()
        return torch.argmax(self.forward(X), dim=1, keepdim=True)


def train_model(
    model: UnifiedLinearModel,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer_type: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> TrainingHistory:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: List[float] = []

    if optimizer_type == "gd":
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = model.loss(X, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return TrainingHistory(optimizer_type, losses)

    if optimizer_type == "sgd":
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            epoch_losses = []
            for batch_X, batch_y in loader:
                model.train()
                optimizer.zero_grad()
                loss = model.loss(batch_X, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(float(np.mean(epoch_losses)))

        return TrainingHistory(optimizer_type, losses)

    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def evaluate_model(
    model: UnifiedLinearModel, X: torch.Tensor, y: torch.Tensor, task_type: str
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        if task_type == "regression":
            preds = model.predict(X).squeeze().cpu().numpy()
            targets = y.squeeze().cpu().numpy()
            mse = float(np.mean((preds - targets) ** 2))
            mae = float(np.mean(np.abs(preds - targets)))
            denom = np.sum((targets - targets.mean()) ** 2)
            r2 = 1.0 - float(np.sum((targets - preds) ** 2) / (denom + 1e-8))
            return {"MSE": mse, "MAE": mae, "R2": r2}

        if task_type == "binary":
            probs = model.predict(X).squeeze().cpu().numpy()
            targets = y.squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            accuracy = float(np.mean(preds == targets))
            return {"Accuracy": accuracy}

        probs = model.predict(X)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        targets = y.squeeze().cpu().numpy()
        accuracy = float(np.mean(preds == targets))
        return {"Accuracy": accuracy}


def generate_regression_data(
    n_samples: int = 200, noise_std: float = 0.8, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    rng = torch.Generator().manual_seed(seed)
    X = torch.linspace(-3.5, 3.5, n_samples).unsqueeze(1)
    true_w = torch.tensor([2.5])
    true_b = -1.0
    noise = torch.randn((n_samples, 1), generator=rng) * noise_std
    y = X * true_w + true_b + noise
    return X.float(), y.float(), {"true_w": float(true_w.item()), "true_b": true_b}


def generate_binary_classification_data(
    n_samples: int = 400, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    assert n_samples % 2 == 0, "n_samples must be even for binary dataset."
    rng = torch.Generator().manual_seed(seed)
    n_per_class = n_samples // 2

    mean_a = torch.tensor([-2.0, -1.5])
    mean_b = torch.tensor([2.0, 2.5])

    class_a = torch.randn((n_per_class, 2), generator=rng) * 0.8 + mean_a
    class_b = torch.randn((n_per_class, 2), generator=rng) * 0.8 + mean_b

    X = torch.cat([class_a, class_b], dim=0)
    y = torch.cat(
        [torch.zeros((n_per_class, 1)), torch.ones((n_per_class, 1))], dim=0
    )

    perm = torch.randperm(n_samples, generator=rng)
    X = X[perm]
    y = y[perm]

    return X.float(), y.float(), {"means": (mean_a.tolist(), mean_b.tolist())}


def generate_multiclass_classification_data(
    n_classes: int = 3, n_samples_per_class: int = 120, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    rng = torch.Generator().manual_seed(seed)
    centers = [
        torch.tensor([-2.5, -2.0]),
        torch.tensor([2.5, -1.0]),
        torch.tensor([0.0, 2.5]),
    ]

    features = []
    labels = []
    for idx in range(n_classes):
        center = centers[idx % len(centers)]
        spread = 0.7 + 0.1 * idx
        points = torch.randn(
            (n_samples_per_class, 2), generator=rng
        ) * spread + center
        features.append(points)
        labels.append(torch.full((n_samples_per_class, 1), idx, dtype=torch.long))

    X = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    perm = torch.randperm(X.size(0), generator=rng)
    X = X[perm]
    y = y[perm]

    return X.float(), y, {"n_classes": n_classes, "centers": [c.tolist() for c in centers]}


def plot_loss_curves(
    task_name: str,
    histories: List[TrainingHistory],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for history in histories:
        ax.plot(history.losses, label=TRAINING_MODE_LABELS[history.optimizer_type])
    ax.set_title(f"{task_name}: Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_regression_results(
    X: torch.Tensor,
    y: torch.Tensor,
    models: List[Tuple[str, UnifiedLinearModel]],
    metadata: Dict,
    output_path: Path,
) -> None:
    x_np = X.squeeze().cpu().numpy()
    y_np = y.squeeze().cpu().numpy()
    x_line = np.linspace(x_np.min() - 0.5, x_np.max() + 0.5, 300).reshape(-1, 1)
    x_tensor = torch.from_numpy(x_line).float()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_np, y_np, color="gray", alpha=0.6, label="Data")

    for key, model in models:
        model.eval()
        with torch.no_grad():
            preds = model.predict(x_tensor).squeeze().cpu().numpy()
        ax.plot(
            x_line,
            preds,
            linewidth=2,
            label=f"{TRAINING_MODE_LABELS[key]} Prediction",
        )

    if "true_w" in metadata and "true_b" in metadata:
        w = metadata["true_w"]
        b = metadata["true_b"]
        ax.plot(
            x_line,
            w * x_line + b,
            "k--",
            alpha=0.7,
            label="True Function",
        )

    ax.set_title("Linear Regression Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def _create_meshgrid(
    X: torch.Tensor, padding: float = 1.0, steps: int = 250
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    x_min = X[:, 0].min().item() - padding
    x_max = X[:, 0].max().item() + padding
    y_min = X[:, 1].min().item() - padding
    y_max = X[:, 1].max().item() + padding

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float()
    return xx, yy, grid_tensor


def plot_binary_decision_boundaries(
    X: torch.Tensor,
    y: torch.Tensor,
    models: List[Tuple[str, UnifiedLinearModel]],
    output_path: Path,
) -> None:
    xx, yy, grid_tensor = _create_meshgrid(X)
    n_cols = len(models)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 4), squeeze=False, constrained_layout=True
    )

    for ax, (key, model) in zip(axes[0], models):
        model.eval()
        with torch.no_grad():
            probs = model.predict(grid_tensor).reshape(xx.shape).cpu().numpy()

        contour = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.8)
        ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.5)
        scatter = ax.scatter(
            X[:, 0].cpu(),
            X[:, 1].cpu(),
            c=y.squeeze().cpu(),
            cmap="bwr",
            edgecolors="k",
            s=35,
        )
        ax.set_title(TRAINING_MODE_LABELS[key])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    fig.colorbar(contour, ax=axes.ravel().tolist(), label="P(y=1)")
    fig.suptitle("Binary Classification Decision Boundaries", y=1.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_multiclass_decision_boundaries(
    X: torch.Tensor,
    y: torch.Tensor,
    models: List[Tuple[str, UnifiedLinearModel]],
    output_path: Path,
) -> None:
    xx, yy, grid_tensor = _create_meshgrid(X)
    n_cols = len(models)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 4), squeeze=False, constrained_layout=True
    )

    for ax, (key, model) in zip(axes[0], models):
        model.eval()
        with torch.no_grad():
            logits = model.forward(grid_tensor)
            preds = torch.argmax(logits, dim=1).reshape(xx.shape).cpu().numpy()

        ax.contourf(xx, yy, preds, levels=np.arange(preds.max() + 2) - 0.5, cmap="tab10", alpha=0.75)
        ax.scatter(
            X[:, 0].cpu(),
            X[:, 1].cpu(),
            c=y.squeeze().cpu(),
            cmap="tab10",
            edgecolors="k",
            s=35,
        )
        ax.set_title(TRAINING_MODE_LABELS[key])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Multiclass Softmax Decision Regions", y=1.02)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items())


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def run_task(config: TaskConfig, output_dir: Path) -> None:
    print("\n" + "=" * 80)
    print(f"Running task: {config.name}")
    print("=" * 80)

    data_kwargs = config.data_kwargs or {}
    X, y, metadata = config.data_fn(**data_kwargs)
    print(f"Dataset generated with shape: X={tuple(X.shape)}, y={tuple(y.shape)}")

    results: List[Tuple[str, UnifiedLinearModel, TrainingHistory, Dict[str, float]]] = []

    for key, _ in TRAINING_MODES:
        model = UnifiedLinearModel(
            input_dim=config.input_dim,
            task_type=config.task_type,
            num_classes=config.num_classes,
        )
        history = train_model(
            model,
            X,
            y,
            optimizer_type=key,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
        )
        metrics = evaluate_model(model, X, y, config.task_type)
        results.append((key, model, history, metrics))
        print(
            f"- {TRAINING_MODE_LABELS[key]} finished. Last loss: {history.losses[-1]:.4f}. "
            f"Metrics: {format_metrics(metrics)}"
        )

    histories = [history for _, _, history, _ in results]
    slug = slugify(config.name)
    plot_loss_curves(
        config.name,
        histories,
        output_dir / f"{slug}_loss_curve.png",
    )

    models_for_plot = [(key, model) for key, model, _, _ in results]
    if config.task_type == "regression":
        plot_regression_results(
            X,
            y,
            models_for_plot,
            metadata,
            output_dir / f"{slug}_fit.png",
        )
    elif config.task_type == "binary":
        plot_binary_decision_boundaries(
            X,
            y,
            models_for_plot,
            output_dir / f"{slug}_decision_boundary.png",
        )
    else:
        plot_multiclass_decision_boundaries(
            X,
            y,
            models_for_plot,
            output_dir / f"{slug}_decision_boundary.png",
        )


def main() -> None:
    set_seed(123)
    output_dir = Path(__file__).parent

    tasks = [
        TaskConfig(
            name="Linear Regression",
            task_type="regression",
            input_dim=1,
            num_classes=1,
            data_fn=generate_regression_data,
            epochs=300,
            lr=0.05,
            batch_size=32,
            data_kwargs={"n_samples": 200, "noise_std": 0.7, "seed": 123},
        ),
        TaskConfig(
            name="Binary Classification",
            task_type="binary",
            input_dim=2,
            num_classes=1,
            data_fn=generate_binary_classification_data,
            epochs=250,
            lr=0.1,
            batch_size=32,
            data_kwargs={"n_samples": 400, "seed": 456},
        ),
        TaskConfig(
            name="Multiclass Classification",
            task_type="multiclass",
            input_dim=2,
            num_classes=3,
            data_fn=generate_multiclass_classification_data,
            epochs=250,
            lr=0.1,
            batch_size=32,
            data_kwargs={"n_classes": 3, "n_samples_per_class": 120, "seed": 789},
        ),
    ]

    for task in tasks:
        run_task(task, output_dir)

    print("\nAll tasks finished. Plots saved to:", output_dir)


if __name__ == "__main__":
    main()
