from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SHOW_FIGURES = "agg" not in plt.get_backend().lower()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_two_moons(
    n_samples: int = 400, noise: float = 0.2, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    assert n_samples % 2 == 0, "n_samples must be even."
    half = n_samples // 2

    theta = torch.linspace(0, np.pi, half)
    x_inner = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    x_outer = torch.stack(
        [1 - torch.cos(theta) + 0.5, -torch.sin(theta) - 0.2], dim=1
    )

    X = torch.cat([x_inner, x_outer], dim=0)
    noise_term = noise * torch.randn_like(X)
    X = X + noise_term

    y = torch.cat([torch.zeros(half), torch.ones(half)], dim=0).long()

    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(output_dim))
        self.activation = torch.tanh

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z1 = X @ self.W1 + self.b1
        h = self.activation(z1)
        z2 = h @ self.W2 + self.b2
        return z2


@dataclass
class TrainingHistory:
    losses: List[float]
    accuracies: List[float]


def train_network(
    model: TwoLayerNet,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.05,
    batch_size: int = 64,
) -> TrainingHistory:
    dataset = TensorDataset(X, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory(losses=[], accuracies=[])

    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        epoch_losses = []
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        history.losses.append(float(np.mean(epoch_losses)))
        history.accuracies.append(correct / total)

    return history


def plot_loss_and_accuracy(history: TrainingHistory, output_path: Path) -> None:
    epochs = np.arange(1, len(history.losses) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, history.losses, label="Loss", color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history.accuracies, label="Accuracy", color="tab:blue")
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_decision_boundary(
    model: TwoLayerNet,
    X: torch.Tensor,
    y: torch.Tensor,
    output_path: Path,
) -> None:
    x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
    y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        logits = model(grid)
        preds = torch.argmax(logits, dim=1).numpy()

    preds = preds.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.3)

    colors = np.where(y.numpy() == 1, "tab:blue", "tab:red")
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=colors, edgecolors="k", s=30)
    ax.set_title("Two-Layer NN Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def main() -> None:
    set_seed(123)
    output_dir = Path(__file__).parent

    X, y = generate_two_moons(n_samples=500, noise=0.25, seed=123)
    model = TwoLayerNet(input_dim=2, hidden_dim=16, output_dim=2)
    history = train_network(model, X, y, epochs=1500, lr=0.05, batch_size=64)

    plot_loss_and_accuracy(history, output_dir / "loss_accuracy.png")
    plot_decision_boundary(model, X, y, output_dir / "decision_boundary.png")

    final_logits = model(X)
    final_preds = torch.argmax(final_logits, dim=1)
    final_acc = (final_preds == y).float().mean().item()
    print(f"Final training accuracy: {final_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

