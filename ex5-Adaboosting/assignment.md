# **Assignment 5: Implementing AdaBoost from Scratch Using PyTorch (with Synthetic Data)**

## **1. Overview**

In this assignment, you will implement **the AdaBoost (Adaptive Boosting) algorithm** from scratch using **PyTorch**.

You must:

1. **Construct your own synthetic dataset(s)**
2. Implement the full AdaBoost algorithm
3. Implement a weak learner (base classifier)
4. Implement weighted training and weight updates
5. Visualize the boosting process and final classifier
6. Compare AdaBoost performance to a single weak learner

This assignment corresponds to **Exercise 5** in the course slides on Boosting algorithms .

---

## **2. Goal of the Assignment**

The purpose of this assignment is to help you understand:

* How AdaBoost converts **weak learners** (slightly better than random guessing) into a **strong classifier**
* How to update example weights across boosting rounds
* How the weak classifier weights (αₜ) are derived
* How boosting iteratively focuses on misclassified samples
* How ensemble methods improve classification accuracy in practice

---

## **3. Tasks to Complete**

You must finish **all** of the following tasks.

---

### **Task 1 — Create a Synthetic Binary Classification Dataset**

Construct your own 2D dataset suitable for visualization. Requirements:

* Two classes with labels **y ∈ {−1, +1}**
* Dataset must not be perfectly separable; include some overlap or mild noise
* Example recommendations:

  * Two Gaussian clusters with overlapping boundaries
  * Concentric circles or a moon-shaped dataset

---

### **Task 2 — Implement a Weak Learner**

AdaBoost requires weak learners that are only slightly better than random.
For this assignment, you must implement one of the following weak learners:

#### Option A (Recommended): **Decision Stump (1-D threshold classifier)**

A decision stump has the form:

[
h(x) =
\begin{cases}
+1, & x_j < v \
-1, & x_j \ge v
\end{cases}
]

Where:

* You may choose **one feature dimension j**
* You search over possible thresholds **v**
* You choose the stump that minimizes the **weighted classification error** under distribution (D_t)

This matches the example in the slides (e.g., threshold classifier in Page 23–27) .

#### Option B: Linear classifier trained with weighted loss

You may implement a small weighted linear model using PyTorch, trained for only 1–3 gradient steps to keep it "weak".

---

### **Task 3 — Implement the Full AdaBoost Algorithm**

Your implementation must follow the algorithm given on slide 22 (AdaBoost pseudocode) .

Specifically:

1. **Initialize weights**
   [
   D_1(i) = \frac{1}{n}
   ]

2. For t = 1 … T boosting rounds:

   1. Train weak learner (h_t) that minimizes the **weighted** error
   2. Compute weighted error
      [
      \epsilon_t = \sum_{i=1}^n D_t(i), \mathbf{1}[h_t(x_i) \ne y_i]
      ]
   3. Compute classifier weight
      [
      \alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}
      ]
   4. Update sample weights
      [
      D_{t+1}(i) =
      \frac{
      D_t(i)\exp(-\alpha_t y_i h_t(x_i))
      }{Z_t}
      ]
   5. Normalize (D_{t+1})

3. Output final classifier
   [
   H(x) = \operatorname{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
   ]

You must implement this mathematically, not using scikit-learn.

---

### **Task 4 — Visualize the AdaBoost Process**

You must produce the following plots:

1. **Decision boundary after each boosting round**

   * For at least T = 10
   * Shows how the classifier becomes stronger over time

2. **Weight distribution visualization**

   * Plot sample weights (D_t) as point sizes on a scatter plot
   * Show how the algorithm emphasizes misclassified points

3. **Final decision boundary**

4. **Training error vs boosting round**

5. **(Optional) Exponential loss vs boosting round**
   [
   L_{\exp}(H) = \sum_{i=1}^n e^{-y_i H(x_i)}
   ]

---

### **Task 6 — Compare Against a Single Weak Learner**

Evaluate:

* Training accuracy of:

  * A single weak learner (e.g., one decision stump)
  * AdaBoost with T=10, 30, 50
* Visual comparison:

  * Single stump boundary vs final AdaBoost boundary

---

## **4. Implementation Requirements**

Your code must include:

### **(1) Weighted training of weak learners**

Stumps must choose thresholds by minimizing weighted classification error.

### **(2) Correct computation of αₜ**

As defined on slide 20:

[
\alpha_t = \frac12 \ln \frac{1 - \epsilon_t}{\epsilon_t}
]


### **(3) Correct update of sample weights**

Using the formula from slide 21:

[
D_{t+1}(i) \propto D_t(i)\exp(-\alpha_t y_i h_t(x_i))
]


---