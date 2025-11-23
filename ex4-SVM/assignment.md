# **Assignment 5: Training Linear SVM with Hinge Loss using PyTorch**

## 1. Overview

In this assignment, you will implement a **linear Support Vector Machine (SVM)** classifier using **hinge loss with L2 regularization**, and train it with **gradient-based method (SGD)** in **PyTorch**.

You must:

1. **Construct your own synthetic datasets** (no external datasets).
3. Implement **hinge loss + L2 regularization** and train an SVM via **stochastic gradient descent**.
4. **Compare SVM (hinge loss)** with **logistic regression (logistic loss)** on the same data.

---

## 2. Tasks to Complete

You will focus on **classification** tasks. At minimum, you must complete:

1. **Binary classification with linear SVM (required)**
2. **Multiclass classification using SVM (One-vs-Rest or another reduction, recommended but optional if time is limited)**

For both, you must:

* Use **synthetic data** you generate.
* Implement and compare:

  * **Linear SVM with hinge loss + L2 regularization**
  * **Logistic regression with logistic loss**

---

### 2.1 Binary Classification with Linear SVM (Required)

#### Data

* Construct a 2D or higher-dimensional synthetic dataset with two classes ( y \in {-1, +1} ) or ( {0, 1} ) (you may internally map to ({-1,+1})).
* You should generate data that is:

  * **Roughly linearly separable**, and
  * Optionally corrupted with **a few noisy or mislabeled points** to see the effect of the regularization parameter.

Example (you can design your own):

* Two Gaussian blobs with different means, plus a small number of outliers.

#### Model: Linear SVM

Use a linear decision function:
[
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
]

Prediction:

* Label by (\operatorname{sign}(f(\mathbf{x}))) or threshold at 0.

#### Loss: Hinge Loss + L2 Regularization (ERM form)

Use the empirical risk minimization form shown in the slides:

[
\min_{\mathbf{w}, b} \quad
\frac{1}{n}\sum_{i=1}^n \big[1 - y^{(i)} (\mathbf{w}^\top \mathbf{x}^{(i)} + b)\big]_+
;+; \lambda ,|\mathbf{w}|^2
]

where:

* ([z]_+ = \max(0, z)) is the hinge function.
* (y^{(i)} \in {-1, +1}).
* (\lambda > 0) is a regularization coefficient (related to (C) in soft-margin SVM).

You must implement this loss **in PyTorch**, either by:

* writing a custom loss function using tensor operations, or
* writing your own `nn.Module` that outputs the loss.

#### Optimization: SGD

**Stochastic / Mini-batch Gradient Descent (SGD)**

   * Use either:

     * Pure SGD (batch size = 1), or
     * Mini-batch SGD (batch size > 1 but < n).

You can use:

* PyTorch’s **automatic differentiation** (recommended), and
* Either your own update loop or PyTorch optimizers (e.g., `torch.optim.SGD` with `weight_decay` disabled if you already include the (\lambda|\mathbf{w}|^2) term in the loss).

---

### 2.2 Logistic Regression Baseline (Required)

To compare **hinge loss vs logistic loss** as suggested by the slides , you must also train **logistic regression** on the **same binary classification dataset**:

* Model:
  [
  p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)
  ]
* Loss: binary cross-entropy (logistic loss) + optional L2 regularization.
* Optimization: SGD, same scheme as SVM.

You may **reuse your Exercise 4 implementation** with minimal changes.

You will then **compare**:

* Decision boundaries
* Training/validation loss curves
* Classification accuracy & generalization behavior
* Effect of regularization

---

### 2.3 Multiclass SVM (Recommended, Optional)

If time permits, extend your SVM implementation to **multiclass classification**:

* Data: synthetically generate at least 3 classes, such as 3 Gaussian clusters in 2D.
* Approach: choose one of:

  * **One-vs-Rest (OvR)**: train one SVM per class vs all others.
  * **One-vs-One (OvO)**: train SVM between each pair of classes, then use voting.

For each method, you should:

* Train using hinge loss + L2 regularization (per binary sub-problem).
* At inference, combine the binary SVM outputs to get the predicted class.
* Optionally compare to a **Softmax logistic regression** baseline.

---

## 3. Unified Program Requirements

### 3.1 Unified Data Interface

Represent data as PyTorch tensors:

```python
X: shape (n_samples, n_features)
y: shape (n_samples,)   # labels (either -1/+1 or 0/1 or 0..K-1)
```

### 3.2 Unified Model Abstraction

Define a common interface (classes or functions) so that SVM and logistic regression are interchangeable in the same training loop. For example:

```python
class LinearClassifier(nn.Module):
    def forward(self, X): ...
    # returns scores or logits

def svm_hinge_loss(outputs, y, lambda_reg, model): ...
def logistic_loss(outputs, y, lambda_reg, model): ...
```

Or define model classes with:

* `forward(X)`
* `compute_loss(X, y)`
* `parameters()` (inherited from `nn.Module`)

### 3.3 Unified Training Loop

Implement a **generic training loop** that can be used for:

* SVM vs logistic regression,
* Binary vs multiclass tasks.

For example, a function:

```python
def train(model, loss_fn, optimizer, X_train, y_train, 
          batch_size, num_epochs, X_val=None, y_val=None):
    ...
```

This loop should:

1. Shuffle data as needed
2. Create batches
3. For each epoch:

   * Forward pass
   * Compute loss
   * Backprop (`loss.backward()` in PyTorch)
   * Update parameters (`optimizer.step()`, then `optimizer.zero_grad()`)
4. Record and return loss values for plotting
5. Optionally evaluate on a validation set

---

## 4. Synthetic Data Requirements (Mandatory)

You **must construct all datasets yourself**. Do not load any external datasets.

Recommended patterns:

### Binary Classification

* 2D: two Gaussian blobs with different means/covariance.
* Add a few overlapping points or label noise to illustrate the role of the regularization parameter (\lambda) (or equivalently (C) in soft margin SVM).

### Multiclass (if implemented)

* Three or more Gaussian blobs in 2D.
* Optionally nonlinearly separable patterns (e.g., “circles” or “moons”) to motivate kernels conceptually—though you are not required to implement kernel SVM here.

Your code should make it easy to:

* change number of samples
* change distance between class centers
* control amount of noise

---

## 5. Visualization Requirements

For **binary SVM vs logistic regression**, at minimum:

1. **Loss curves**

   * Plot **training loss vs epoch** (or vs iteration) for:

     * SVM (hinge loss + L2), SGD
     * Logistic regression, SGD (you can choose a subset if plots get too crowded).
2. **Decision boundary plot (2D case)**

   * Show data points colored by class.
   * Overlay decision boundaries for:

     * SVM
     * Logistic regression
   * Optionally also show **support vectors** for SVM (e.g., mark points with non-zero hinge loss or margin close to 1).

For **multiclass (if implemented)**:

* Plot 2D scatter with distinct colors per class.
* Overlay decision regions (or at least decision boundaries) for your multiclass SVM.

Use PyTorch for computations and any plotting library (e.g., matplotlib) for visualization.