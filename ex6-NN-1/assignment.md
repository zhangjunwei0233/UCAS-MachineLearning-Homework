# **Assignment 6: Training a Two-Layer Neural Network with Gradient Descent (Using PyTorch + Synthetic Data)**

## **1. Overview**

In this assignment, you will implement and train a **two-layer feedforward neural network (a 1-hidden-layer MLP)** using **gradient descent**. You must:

1. **Construct synthetic training data**
2. Implement a neural network with:

   * One hidden layer
   * Nonlinear activation (e.g., ReLU, Sigmoid, or Tanh)
   * A linear output layer
3. Implement **forward propagation**
4. Implement **backpropagation manually OR using PyTorch autograd**
5. Train the network using **gradient descent or stochastic gradient descent**
6. Visualize the decision boundary or regression fit
7. Evaluate the network’s learning behavior over training

This assignment corresponds to **Exercise 6** in the neural networks lecture. 

---

## **2. Goals of the Assignment**

By completing this assignment, you will:

* Understand how a multi-layer perceptron computes forward and backward passes
* Understand loss functions, activation functions, and parameter updates
* Familiarize yourself with PyTorch’s tensor operations and automatic differentiation
* Relate training to the backpropagation algorithm explained in the slides (pages 26–34)
* Develop intuition for how nonlinear functions + hidden layers allow a network to represent non-linearly separable problems (e.g., XOR)

---

## **3. Tasks to Complete**

Your work must include all of the following components.

---

## **Task 1 — Construct Your Own Synthetic Dataset**

You must generate **your own dataset** (no external datasets).
You may choose either classification or regression:

### Option A: **Binary Classification (Recommended)**

Create a dataset that is *not linearly separable* to illustrate the benefit of hidden layers.
Examples:

* XOR dataset (matching slides pp. 11–12)
* Two-moons shape
* Two intertwined spirals
* Ring vs disk

### Option B: **Regression**

Example targets:

* A nonlinear function such as
  [
  y = \sin(3x) + 0.3 \epsilon
  ]
* A 2D nonlinear surface

Your dataset must be generated using PyTorch or NumPy.

---

## **Task 2 — Implement a Two-Layer Neural Network**

The network architecture must follow:

[
h = \sigma(W_1 x + b_1), \qquad
\hat{y} = W_2 h + b_2
]

Where:

* (W_1), (b_1) are hidden layer weights
* (W_2), (b_2) are output layer weights
* (\sigma(\cdot)) is a nonlinear activation function

  * You must choose at least one: **ReLU**, **Sigmoid**, **Tanh**, or **LeakyReLU**
  * Slides recommend these activations (pp. 24–25)

The network must be implemented in PyTorch, using:

* Either a custom class using `torch.nn.Module`
* OR purely functional code using tensors

---

## **Task 3 — Implement Forward Propagation**

Your code must compute:

1. Linear transformation (z_1 = W_1 x + b_1)
2. Nonlinear activation (h = \sigma(z_1))
3. Output (z_2 = W_2 h + b_2)
4. Depending on task:

### For Classification

Apply either:

* Sigmoid → Binary Cross Entropy (if binary classification), or
* Softmax → Cross Entropy (if multiclass)

### For Regression

Use:

* Mean Squared Error (MSE)

Slides explicitly discuss loss choices and output unit types (pp. 22–24). 

---

## **Task 4 — Implement Backpropagation + Gradient Descent**

You may choose **either** method:

### Option A — **Manual Backpropagation**

Using the chain rule as shown in the slides (pp. 26–33).

You must implement:

* Gradients of output layer
* Gradients of hidden layer
* Parameter updates
  [
  W \leftarrow W - \eta \cdot \nabla W
  ]

### Option B — **PyTorch Autograd**

You may let PyTorch compute all gradients automatically, but you must:

* Call `.backward()` explicitly
* Manually step parameters (or use `torch.optim.SGD`)

### Both options must implement gradient descent or mini-batch SGD.

---

## **Task 5 — Train the Network**

You must train the network for multiple epochs and track:

* Training loss over time
* (If classification) training accuracy
* (Optional) validation metrics

You must demonstrate that the network learns a **nonlinear function** that a linear model cannot learn (e.g., XOR).

This connects directly to the slides’ demonstration of XOR modeling (pp. 18–21).

---

## **Task 6 — Visualization Requirements**

You must produce at least:

### (1) Loss curve over epochs

* Shows gradient descent convergence

### (2) Classification boundary (if classification)

* 2D color plot of decision regions
* Overlaid scatter of training samples

### (3) Regression curve/surface (if regression)

* Show fitted function vs ground truth

### Optional (but encouraged)

* Hidden layer activations
* Comparison with a linear model (showing failure on XOR)

---

## **4. Implementation Constraints**

Your implementation must satisfy:

* Use **PyTorch** for:

  * Data
  * Tensors
  * Gradient computation (manual or autograd)
* Use **synthetic data only**
* Implement **your own network**, not `torch.nn.Sequential` shortcuts
  (Sequential allowed only if you show you understand the underlying operations.)
* Use **gradient descent** (not Adam, RMSProp, etc.)