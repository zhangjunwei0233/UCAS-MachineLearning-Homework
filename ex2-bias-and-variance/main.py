import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def generate_data(n_samples=1000, noise_level=0.1, seed=42):
    """Generate synthetic data following y = sin(2πx) + noise"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    x = np.random.uniform(0, 1, n_samples)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_level, n_samples)

    return x, y


def create_polynomial_features(x, degree):
    """Create polynomial features for given degree"""
    x = x.reshape(-1, 1)
    features = []
    for i in range(1, degree + 1):
        features.append(x**i)
    return np.concatenate(features, axis=1)


class PolynomialRegression(nn.Module):
    """PyTorch polynomial regression model"""

    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(degree, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(X_train, y_train, X_test, y_test, degree, max_iterations=1000, lr=0.01):
    """Train polynomial regression model and return training history"""
    model = PolynomialRegression(degree)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_features = create_polynomial_features(X_train, degree)
    X_test_features = create_polynomial_features(X_test, degree)

    X_train_tensor = torch.FloatTensor(X_train_features)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_features)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    train_errors = []
    test_errors = []

    for iteration in range(max_iterations):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor)

        train_errors.append(train_loss.item())
        test_errors.append(test_loss.item())

    return train_errors, test_errors, model


def experiment_1_error_vs_iterations(
    X_train, y_train, X_test, y_test, train_size=100, degree=5, max_iterations=1000
):
    """Experiment 1: Test error vs iterations with fixed training size and degree"""
    # Subsample training data
    indices = np.random.choice(len(X_train), size=train_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]

    train_errors, test_errors, _ = train_model(
        X_train_sub, y_train_sub, X_test, y_test, degree, max_iterations
    )

    return train_errors, test_errors


def experiment_2_error_vs_training_size(
    X_train,
    y_train,
    X_test,
    y_test,
    degree=5,
    iterations=5000,
    train_sizes=[20, 50, 100, 200, 400],
):
    """Experiment 2: Test error vs training dataset size with fixed iterations and degree"""
    train_errors = []
    test_errors = []

    for train_size in train_sizes:
        indices = np.random.choice(len(X_train), size=train_size, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]

        train_error_history, test_error_history, _ = train_model(
            X_train_sub, y_train_sub, X_test, y_test, degree, iterations
        )
        # Take final test error
        test_errors.append(test_error_history[-1])
        train_errors.append(train_error_history[-1])

    return train_sizes, train_errors, test_errors


def experiment_3_error_vs_degree(
    X_train,
    y_train,
    X_test,
    y_test,
    train_size=30,
    iterations=500,
    degrees=[1, 5, 10, 15, 20, 25, 30, 35, 40],
):
    """Experiment 3: Test error vs polynomial degree with fixed iterations and training size"""
    # Subsample training data
    indices = np.random.choice(len(X_train), size=train_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]

    train_errors = []
    test_errors = []

    for degree in degrees:
        train_error_history, test_error_history, _ = train_model(
            X_train_sub, y_train_sub, X_test, y_test, degree, iterations
        )
        # Take final test error
        test_errors.append(test_error_history[-1])
        train_errors.append(train_error_history[-1])

    return degrees, train_errors, test_errors


def plot_experiments():
    """Run all three experiments and create plots"""
    # Generate data
    X, y = generate_data(n_samples=1000, noise_level=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Experiment 1: Error vs Iterations
    print("Running Experiment 1: Test error vs iterations...")
    train_errors, test_errors = experiment_1_error_vs_iterations(
        X_train, y_train, X_test, y_test
    )

    axes[0].plot(train_errors, label="Training Error", alpha=0.8)
    axes[0].plot(test_errors, label="Test Error", alpha=0.8)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].set_title("Test Error vs Iterations\n(Training Size=100, Degree=5)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Experiment 2: Error vs Training Size
    print("Running Experiment 2: Test error vs training size...")
    train_sizes, train_errors_2, test_errors_2 = experiment_2_error_vs_training_size(
        X_train, y_train, X_test, y_test
    )

    axes[1].plot(train_sizes, train_errors_2, "o-", label="Training Error")
    axes[1].plot(train_sizes, test_errors_2, "o-", label="Test Error")
    axes[1].set_xlabel("Training Dataset Size")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].set_title("Test Error vs Training Size\n(Iterations=5000, Degree=5)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Experiment 3: Error vs Polynomial Degree
    print("Running Experiment 3: Test error vs polynomial degree...")
    degrees, train_errors_3, test_errors_3 = experiment_3_error_vs_degree(
        X_train,
        y_train,
        X_test,
        y_test,
        iterations=5000,
    )

    axes[2].plot(degrees, train_errors_3, "s-", label="Training Error")
    axes[2].plot(degrees, test_errors_3, "s-", label="Test Error", color="orange")
    axes[2].set_xlabel("Polynomial Degree")
    axes[2].set_ylabel("Mean Squared Error")
    axes[2].set_title(
        "Test Error vs Polynomial Degree\n(Training Size=30, Iterations=5000)"
    )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bias_variance_experiments.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("All experiments completed! Results saved as 'bias_variance_experiments.png'")


def main():
    """Main function to run bias-variance demonstration"""
    print("Bias-Variance Demonstration")
    print("=" * 40)
    print("Generating synthetic data: y = sin(2πx) + noise")
    print("Running three experiments to demonstrate bias-variance tradeoff...")
    print()

    plot_experiments()


if __name__ == "__main__":
    main()
