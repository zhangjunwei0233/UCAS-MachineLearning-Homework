import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def create_polynomial_features(X, degree, include_bias=True):
    """
    Create polynomial features for 1D input.

    Args:
        X: 1D array of input values
        degree: polynomial degree
        include_bias: whether to include bias term (X^0 = 1)

    Returns:
        2D array with polynomial features [1, X, X^2, ..., X^degree]
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n_samples = X.shape[0]

    # Create polynomial features
    features = []
    start_degree = 0 if include_bias else 1

    for d in range(start_degree, degree + 1):
        if d == 0:
            features.append(np.ones((n_samples, 1)))
        else:
            features.append(X**d)

    return np.concatenate(features, axis=1)


class StandardScaler:
    """Simple standardization: (X - mean) / std"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X * self.std_ + self.mean_


def generate_synthetic_data(n_samples=40, noise_std=0.3, seed=42):
    """
    Generate synthetic data with a clear quadratic pattern for demonstrating
    under-fitting and over-fitting.

    True function: y = -0.5x² + 0.8x + 0.1 + noise
    This creates a nice parabolic curve that's perfect for polynomial regression demo.
    """
    np.random.seed(seed)

    # Generate X values uniformly distributed
    X = np.linspace(-2, 3, n_samples)

    # True quadratic function with interesting coefficients
    y_true = -0.5 * X**2 + 0.8 * X + 0.1

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    print(f"Generated {n_samples} synthetic data points")
    print(f"True function: y = -0.5x² + 0.8x + 0.1 + noise(σ={noise_std})")
    print(f"X range: [{X.min():.1f}, {X.max():.1f}]")
    print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y


def load_and_preprocess_data(filename=None):
    """
    Generate synthetic polynomial data for clear demonstration.
    filename parameter kept for compatibility but ignored.
    """
    return generate_synthetic_data()


def train_polynomial_model(X, y, degree, epochs=1000, lr=0.01):
    """
    Train a polynomial regression model of given degree.
    Returns: trained model, training losses, predictions
    """
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree, include_bias=True)

    # Standardize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_poly)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)

    # Create model
    model = nn.Linear(X_scaled.shape[1], 1)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor).squeeze()
        loss = criterion(predictions, y_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Degree {degree}, Epoch {epoch}, Loss: {loss.item():.6f}")

    # Generate predictions for plotting
    X_plot = np.linspace(X.min(), X.max(), 100)
    X_plot_poly = create_polynomial_features(X_plot, degree, include_bias=True)
    X_plot_scaled = scaler_X.transform(X_plot_poly)
    X_plot_tensor = torch.FloatTensor(X_plot_scaled)

    with torch.no_grad():
        y_plot_scaled = model(X_plot_tensor).squeeze().numpy()
        y_plot = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).flatten()

    return model, losses, X_plot, y_plot, scaler_X, scaler_y


def calculate_metrics(X, y, model, degree, scaler_X, scaler_y):
    """Calculate R² and MSE for model evaluation"""
    X_poly = create_polynomial_features(X, degree, include_bias=True)
    X_scaled = scaler_X.transform(X_poly)
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).squeeze().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Calculate MSE
    mse = np.mean((y - y_pred) ** 2)

    return r2, mse


def main():
    # Generate synthetic data
    print("Generating synthetic polynomial data for demonstration...")
    X, y = load_and_preprocess_data()
    print()

    degrees = [
        1,
        2,
        100,
    ]  # Under-fitting (linear), optimal (quadratic), over-fitting (high-order)

    # Train models with different degrees
    models = {}
    results = {}

    print("\nTraining polynomial regression models...")
    for degree in degrees:
        print(f"\n--- Training degree {degree} polynomial ---")
        model, losses, X_plot, y_plot, scaler_X, scaler_y = train_polynomial_model(
            X, y, degree, epochs=1000, lr=0.01
        )

        # Calculate metrics
        r2, mse = calculate_metrics(X, y, model, degree, scaler_X, scaler_y)

        models[degree] = model
        results[degree] = {
            "losses": losses,
            "X_plot": X_plot,
            "y_plot": y_plot,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "r2": r2,
            "mse": mse,
        }

        print(f"Final R²: {r2:.4f}, MSE: {mse:.2e}")

    # Create visualizations
    create_comparison_plots(X, y, degrees, results, models)

    # Educational analysis
    analyze_results(degrees, results)


def create_comparison_plots(X, y, degrees, results, models):
    """Create comprehensive plots comparing different polynomial degrees"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Data and model fits
    ax1 = axes[0]
    ax1.scatter(X, y, alpha=0.6, color="black", s=30, label="Data")

    # Plot true function for reference
    X_true = np.linspace(X.min(), X.max(), 100)
    y_true = -0.5 * X_true**2 + 0.8 * X_true + 0.1
    ax1.plot(X_true, y_true, "k--", alpha=0.5, linewidth=1, label="True function")

    colors = ["red", "blue", "green", "orange", "purple"]
    for i, degree in enumerate(degrees):
        result = results[degree]
        ax1.plot(
            result["X_plot"],
            result["y_plot"],
            color=colors[i],
            linewidth=2,
            label=f"Degree {degree} (R²={result['r2']:.3f})",
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Polynomial Regression: Under-fitting vs Over-fitting")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training loss curves
    ax2 = axes[1]
    for i, degree in enumerate(degrees):
        result = results[degree]
        ax2.plot(result["losses"], color=colors[i], label=f"Degree {degree}")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("polynomial_regression_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_results(degrees, results):
    """Provide educational analysis of the results"""
    print("\n" + "=" * 60)
    print("EDUCATIONAL ANALYSIS: Under-fitting vs Over-fitting")
    print("=" * 60)

    print("\n📊 Model Performance Summary:")
    print("-" * 40)
    for degree in degrees:
        result = results[degree]
        print(f"Degree {degree:2d}: R² = {result['r2']:.4f}, MSE = {result['mse']:.2e}")

    print("\n🎯 Key Observations:")
    print("-" * 40)

    # Find best performing model
    best_degree = max(degrees, key=lambda d: results[d]["r2"])
    worst_degree = min(degrees, key=lambda d: results[d]["r2"])

    print(f"• Best R² score: Degree {best_degree} ({results[best_degree]['r2']:.4f})")
    print(
        f"• Lowest R² score: Degree {worst_degree} ({results[worst_degree]['r2']:.4f})"
    )

    if 1 in degrees:
        print(
            f"• Linear model (degree 1): {'Under-fitting' if results[1]['r2'] < 0.7 else 'Reasonable fit'}"
        )

    high_degree = max(degrees)
    if high_degree >= 8:
        print(
            f"• High-degree model (degree {high_degree}): Risk of over-fitting to training data"
        )

    print("\n🔍 What This Demonstrates:")
    print("-" * 40)
    print("• Degree 1 (Linear): Under-fits the quadratic pattern - too simple!")
    print("• Degree 2 (Quadratic): Optimal fit - matches true function!")
    print(
        f"• Degree {max(degrees)} (High-order): Over-fits - memorizes noise, complex curve!"
    )
    print("• Key insight: Model complexity should match data complexity")
    print("• Real-world advice: Use cross-validation to select optimal degree")

    print("\n📐 Expected Pattern:")
    print("-" * 40)
    print("• True function: y = -0.5x² + 0.8x + 0.1 + noise")
    print("• Degree 2 should have highest R² score - matches true function")
    print("• Higher degrees may overfit to training noise")
    print("• Bias-variance tradeoff: complexity vs generalization")


if __name__ == "__main__":
    main()
