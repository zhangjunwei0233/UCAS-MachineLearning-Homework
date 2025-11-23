from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SHOW_FIGURES = "agg" not in plt.get_backend().lower()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_checkerboard_data(
    n_samples: int = 400,
    noise: float = 0.4,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate an XOR-like dataset requiring multiple stumps to separate."""
    set_seed(seed)
    assert n_samples % 4 == 0, "n_samples must be divisible by 4."
    per_quad = n_samples // 4

    centers = [
        (torch.tensor([1.5, 1.5]), 1),
        (torch.tensor([-1.5, -1.5]), 1),
        (torch.tensor([-1.5, 1.5]), -1),
        (torch.tensor([1.5, -1.5]), -1),
    ]

    samples = []
    labels = []
    for center, label in centers:
        cov = noise * torch.eye(2)
        dist = torch.distributions.MultivariateNormal(center, cov)
        samples.append(dist.sample((per_quad,)))
        labels.append(torch.full((per_quad,), label, dtype=torch.float32))

    X = torch.cat(samples, dim=0)
    y = torch.cat(labels, dim=0)

    flip_indices = torch.randperm(n_samples)[: int(0.05 * n_samples)]
    y[flip_indices] *= -1

    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


def generate_two_moons(
    n_samples: int = 400,
    noise: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a noisy two-moons dataset."""
    set_seed(seed)
    assert n_samples % 2 == 0, "n_samples must be even."
    half = n_samples // 2
    angles = torch.linspace(0, math.pi, half)

    upper = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    lower = torch.stack(
        [1 - torch.cos(angles) + 0.3, -torch.sin(angles) - 0.3], dim=1
    )

    upper += noise * torch.randn_like(upper)
    lower += noise * torch.randn_like(lower)

    X = torch.cat([upper, lower], dim=0)
    y = torch.cat([torch.ones(half), -torch.ones(half)], dim=0)

    flip_idx = torch.randperm(n_samples)[: int(0.03 * n_samples)]
    y[flip_idx] *= -1

    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


@dataclass
class DecisionStump:
    feature_idx: int = 0
    threshold: float = 0.0
    polarity: int = 1

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        feature_values = X[:, self.feature_idx]
        predictions = torch.ones_like(feature_values)
        predictions[feature_values >= self.threshold] = -1.0
        return predictions * self.polarity

    @staticmethod
    def fit_stump(
        X: torch.Tensor, y: torch.Tensor, sample_weights: torch.Tensor
    ) -> "DecisionStump":
        n_samples, n_features = X.shape
        best_error = float("inf")
        best_stump = DecisionStump()

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_vals = torch.unique(feature_values)
            if unique_vals.numel() == 1:
                thresholds = unique_vals
            else:
                midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
                thresholds = torch.cat(
                    [unique_vals[:1] - 1e-3, midpoints, unique_vals[-1:] + 1e-3]
                )

            for threshold in thresholds:
                base_predictions = torch.ones(n_samples, dtype=torch.float32)
                base_predictions[X[:, feature_idx] >= threshold] = -1

                for polarity in [1, -1]:
                    preds = base_predictions * polarity
                    misclassified = preds != y
                    error = torch.sum(sample_weights[misclassified]).item()

                    if error < best_error:
                        best_error = error
                        best_stump = DecisionStump(
                            feature_idx=feature_idx,
                            threshold=float(threshold.item()),
                            polarity=polarity,
                        )

        return best_stump


def feature_map(X: torch.Tensor) -> torch.Tensor:
    """Feature map used by weighted linear weak learners."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    features = torch.stack([x1, x2], dim=1)
    return features


@dataclass
class LinearWeakLearner:
    weight: torch.Tensor
    bias: torch.Tensor

    def predict_scores(self, X: torch.Tensor) -> torch.Tensor:
        features = feature_map(X)
        scores = features @ self.weight.t() + self.bias
        return scores.squeeze()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        scores = self.predict_scores(X)
        preds = torch.sign(scores)
        preds[preds == 0] = 1
        return preds

    @staticmethod
    def fit_weighted(
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weights: torch.Tensor,
        steps: int = 20,
        lr: float = 0.1,
    ) -> "LinearWeakLearner":
        features = feature_map(X)
        model = nn.Linear(features.shape[1], 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        targets = (y > 0).float()

        for _ in range(steps):
            logits = model(features).squeeze()
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            weighted_loss = torch.sum(loss * sample_weights)
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

        weight = model.weight.detach().clone()
        bias = model.bias.detach().clone()
        return LinearWeakLearner(weight=weight, bias=bias)


class AdaBoost:
    def __init__(
        self,
        n_estimators: int = 30,
        weak_learner: str = "stump",
        linear_steps: int = 20,
        linear_lr: float = 0.1,
    ):
        if weak_learner not in {"stump", "linear"}:
            raise ValueError("weak_learner must be 'stump' or 'linear'")
        self.n_estimators = n_estimators
        self.weak_learner = weak_learner
        self.linear_steps = linear_steps
        self.linear_lr = linear_lr
        self.learners: List[object] = []
        self.alphas: List[float] = []
        self.training_error_history: List[float] = []
        self.exp_loss_history: List[float] = []
        self.weight_history: List[torch.Tensor] = []

    def _train_weak_learner(
        self, X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor
    ):
        if self.weak_learner == "stump":
            return DecisionStump.fit_stump(X, y, weights)
        return LinearWeakLearner.fit_weighted(
            X, y, weights, steps=self.linear_steps, lr=self.linear_lr
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        n_samples = X.shape[0]
        weights = torch.full((n_samples,), 1.0 / n_samples)
        self.weight_history = [weights.clone()]

        aggregated_scores = torch.zeros(n_samples)

        for _ in range(self.n_estimators):
            learner = self._train_weak_learner(X, y, weights)
            predictions = learner.predict(X)

            misclassified = predictions != y
            epsilon = torch.sum(weights[misclassified])
            epsilon = torch.clamp(epsilon, min=1e-10, max=1 - 1e-10)
            alpha = 0.5 * torch.log((1 - epsilon) / epsilon)

            weights = weights * torch.exp(-alpha * y * predictions)
            weights = weights / weights.sum()

            aggregated_scores += alpha * predictions
            ensemble_preds = torch.sign(aggregated_scores)
            ensemble_preds[ensemble_preds == 0] = 1

            train_error = (ensemble_preds != y).float().mean().item()
            exp_loss = torch.mean(torch.exp(-y * aggregated_scores)).item()

            self.learners.append(learner)
            self.alphas.append(float(alpha))
            self.training_error_history.append(train_error)
            self.exp_loss_history.append(exp_loss)
            self.weight_history.append(weights.clone())

    def predict_scores(self, X: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        upto = len(self.learners) if upto is None else min(upto, len(self.learners))
        if upto == 0:
            return torch.zeros(X.shape[0])

        scores = torch.zeros(X.shape[0])
        for stump, alpha in zip(self.learners[:upto], self.alphas[:upto]):
            scores += alpha * stump.predict(X)
        return scores

    def predict(self, X: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        scores = self.predict_scores(X, upto)
        preds = torch.sign(scores)
        preds[preds == 0] = 1
        return preds


def plot_decision_boundary(
    ax: plt.Axes,
    model: AdaBoost,
    X: torch.Tensor,
    y: torch.Tensor,
    upto: int | None,
    title: str,
) -> None:
    x_min, x_max = X[:, 0].min().item() - 1.2, X[:, 0].max().item() + 1.2
    y_min, y_max = X[:, 1].min().item() - 1.2, X[:, 1].max().item() + 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        scores = model.predict_scores(grid, upto=upto)
        preds = torch.sign(scores).reshape(xx.shape).numpy()

    ax.contourf(xx, yy, preds, levels=[-np.inf, 0, np.inf], alpha=0.2, colors=["#FFBBBB", "#BBCCFF"])
    ax.contour(xx, yy, preds, levels=[0], colors="k", linewidths=1)

    colors = np.where(y.numpy() == 1, "tab:blue", "tab:red")
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=colors, s=25, edgecolors="k", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.2)


def plot_round_boundaries(
    model: AdaBoost,
    X: torch.Tensor,
    y: torch.Tensor,
    rounds: Sequence[int],
    output_path: Path,
) -> None:
    cols = 5
    rows = math.ceil(len(rounds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes)

    for idx, round_num in enumerate(rounds):
        r = idx // cols
        c = idx % cols
        plot_decision_boundary(
            axes[r, c], model, X, y, upto=round_num, title=f"Round {round_num}"
        )

    for idx in range(len(rounds), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle("AdaBoost Decision Boundary Progression", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_weight_distributions(
    model: AdaBoost,
    X: torch.Tensor,
    y: torch.Tensor,
    rounds: Sequence[int],
    output_path: Path,
) -> None:
    cols = len(rounds)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 3.5))
    axes = np.atleast_1d(axes)

    for ax, round_num in zip(axes, rounds):
        round_num = min(round_num, len(model.weight_history) - 1)
        weights = model.weight_history[round_num]
        sizes = (weights / weights.max()).numpy() * 300 + 20
        colors = np.where(y.numpy() == 1, "tab:blue", "tab:red")
        ax.scatter(
            X[:, 0].numpy(),
            X[:, 1].numpy(),
            s=sizes,
            c=colors,
            alpha=0.7,
            edgecolors="k",
        )
        ax.set_title(f"Weights D_{round_num+1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Sample Weight Evolution", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_metrics(model: AdaBoost, output_path: Path) -> None:
    rounds = np.arange(1, len(model.training_error_history) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(rounds, model.training_error_history, label="Training error", color="tab:blue")
    ax1.set_xlabel("Boosting round")
    ax1.set_ylabel("Training error", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(rounds, model.exp_loss_history, label="Exponential loss", color="tab:red")
    ax2.set_ylabel("Exponential loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    ax1.legend(loc="upper right")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_single_vs_boosted(
    model: AdaBoost,
    X: torch.Tensor,
    y: torch.Tensor,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_decision_boundary(
        axes[0], model, X, y, upto=1, title="Single Decision Stump"
    )
    plot_decision_boundary(
        axes[1], model, X, y, upto=len(model.learners), title="AdaBoost (Final)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def evaluate_ensembles(model: AdaBoost, X: torch.Tensor, y: torch.Tensor, rounds: Sequence[int]) -> List[Tuple[int, float]]:
    results = []
    for round_num in rounds:
        preds = model.predict(X, upto=round_num)
        accuracy = (preds == y).float().mean().item()
        results.append((round_num, accuracy))
    return results


def main() -> None:
    set_seed(123)
    output_dir = Path(__file__).parent

    X, y = generate_two_moons(n_samples=480, noise=0.25, seed=123)
    boosting_rounds = 50
    model = AdaBoost(
        n_estimators=boosting_rounds,
        weak_learner="linear",
        linear_steps=25,
        linear_lr=0.2,
    )
    model.fit(X, y)

    comparison_rounds = [1, 10, 30, boosting_rounds]
    accuracies = evaluate_ensembles(model, X, y, comparison_rounds)

    print("Training accuracy comparison:")
    for round_num, acc in accuracies:
        label = "Single Stump" if round_num == 1 else f"AdaBoost (T={round_num})"
        print(f"- {label:>18}: {acc*100:.2f}%")

    rounds_to_visualize = list(range(1, min(10, boosting_rounds) + 1))
    plot_round_boundaries(
        model,
        X,
        y,
        rounds=rounds_to_visualize,
        output_path=output_dir / "decision_boundaries_progression.png",
    )

    weight_rounds = [0, 4, 9, min(19, boosting_rounds - 1), boosting_rounds - 1]
    plot_weight_distributions(
        model,
        X,
        y,
        rounds=weight_rounds,
        output_path=output_dir / "weight_distributions.png",
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_decision_boundary(
        ax=ax,
        model=model,
        X=X,
        y=y,
        upto=len(model.learners),
        title="Final AdaBoost Decision Boundary",
    )
    fig.tight_layout()
    final_path = output_dir / "final_decision_boundary.png"
    fig.savefig(final_path, dpi=300, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    plot_metrics(model, output_dir / "training_metrics.png")
    plot_single_vs_boosted(model, X, y, output_dir / "single_vs_boosted.png")

    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
