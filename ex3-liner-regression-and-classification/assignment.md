
# **Assignment 4: Unified Linear Models for Regression, Binary Classification, and Multiclass Classification (using PyTorch)**

## **1. Overview**

The goal of this assignment is to implement a **unified training framework** for linear models across three different machine learning tasks:

1. **Regression**
2. **Binary classification**
3. **Multiclass classification**

You must implement all models **using PyTorch**, and you must **construct your own synthetic datasets** for each task.

The purpose of this assignment is to help you understand:

* How different prediction tasks can be solved with linear models
* How to design unified code structures across tasks
* How loss functions differ across regression and classification
* How to use **Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)** in practice
* How PyTorch tensors and autograd can simplify model training

---

## **2. Tasks to Complete**

Your program must successfully train a linear model for the following three tasks:

---

### **Task 1 — Regression**

* **Data:**
  Construct synthetic data following a linear function:
  [
  y = \mathbf{w}^\top \mathbf{x} + \epsilon \quad (\epsilon\sim\text{Gaussian noise})
  ]

* **Model:**
  Linear regression model
  [
  f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
  ]

* **Loss:** Mean Squared Error (MSE)
  [
  L = \frac12\sum_i (f(x_i) - y_i)^2
  ]

* **Optimization:** GD and SGD (implemented using PyTorch)

---

### **Task 2 — Binary Classification**

* **Data:**
  Construct two clusters of points in 2D or higher-dimensional space and assign labels ( y \in {0,1} ).
  You may generate linearly separable or non-separable data.

* **Model:** Logistic Regression
  [
  p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}),\quad
  \sigma(z)=\frac{1}{1+e^{-z}}
  ]

* **Loss:** Logistic loss (binary cross-entropy)
  [
  L = -\sum_i \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]
  ]

* **Optimization:** GD and SGD using PyTorch

---

### **Task 3 — Multiclass Classification**

* **Data:**
  Construct at least **3 clusters** of points.
  Label them ( y \in {1,\ldots,K} ) with ( K>2 ).

* **Model:** Softmax Linear Classifier
  [
  p(y=k|\mathbf{x}) =
  \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}
  {\sum_j \exp(\mathbf{w}_j^\top \mathbf{x})}
  ]

* **Loss:** Cross-entropy
  [
  L = -\sum_i \log p(y_i|\mathbf{x}_i)
  ]

* **Optimization:** GD and SGD, implemented in PyTorch

---

## **3. Unified Program Requirements**

Your implementation must follow a unified structure.
Regardless of the task, your code must contain:

### **(1) Unified data representation**

```
X: n × d tensor
y: labels (n × 1 tensor or one-hot encoding)
```

### **(2) Unified model interface**

Your code must include functions such as:

```python
model.forward(X)
model.loss(X, y)
model.parameters()
```

You may implement the models by:

* writing custom `nn.Module` classes, or
* implementing manual forward and loss functions using tensors

### **(3) Unified training pipeline**

Each task must use the following workflow:

1. Generate synthetic data
2. Initialize model parameters
3. Select optimizer type (GD or SGD)
4. For each epoch:

   * forward
   * compute loss
   * compute gradients
   * update parameters
5. Plot and analyze the loss curve
6. Evaluate and visualize results

---

## **5. Visualization Requirements**

For each task, you must provide:

### **1. Loss curve**

* x-axis: training step or epoch
* y-axis: loss
* Separate curves for GD and SGD (or overlay)

### **2. (Optional but recommended) Decision boundary**

* Regression: fitted line or surface
* Binary classification: scatter plot + boundary
* Multiclass: color-coded points + partition boundaries
