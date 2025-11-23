# **Assignment 7: Train a Convolutional Neural Network (CNN) on MNIST Using Five Optimization Algorithms (PyTorch)**

## **1. Overview**

In this assignment, you will design and train a **simple convolutional neural network (CNN)** on the **MNIST handwritten digit dataset**.
Your CNN must include:

* **At least one convolution layer**
* **At least one pooling layer**
* **At least one fully connected (dense) layer**

You must train this CNN using **five different optimization algorithms**:

1. SGD
2. SGD with Momentum (SGDM)
3. AdaGrad
4. RMSProp
5. Adam

These requirements come directly from the lecture instructions (slide 52). 

Your goal is to **train the network**, **tune learning rates using a validation set**, and **record & compare performance** across optimizers.

---

## **2. Objectives**

By completing this assignment, you will:

* Understand how convolution, pooling, and fully connected layers are combined into a CNN
* Gain experience training CNNs with different optimizers
* Observe the impact of optimization algorithms on convergence speed and generalization
* Learn to tune hyperparameters (primarily learning rate) using a validation set
* Compare training/validation/test loss and accuracy across multiple training runs

---

## **3. Dataset Requirements**

Use **MNIST**, with the following splits:

* **Training set**
* **Validation set** (you must create this by splitting off part of the training data)
* **Test set**

You may use `torchvision.datasets.MNIST` for loading images and labels.

---

## **4. CNN Architecture Requirements**

Your CNN must be **as simple as possible**, but must include:

1. **Convolution layer(s)**

   * At least one convolution kernel
   * Nonlinearity (ReLU recommended)

2. **Pooling layer(s)**

   * Max-pooling is recommended
   * Kernel size = 2 is acceptable

3. **Fully connected (dense) layer(s)**

   * Output layer must produce **10 logits** (digits 0–9)
   * Softmax cross-entropy loss

### Example Minimal Architecture (Allowed)

```
Conv2d(1, 8, kernel_size=3, padding=1)
ReLU
MaxPool2d(kernel_size=2)

Conv2d(8, 16, kernel_size=3, padding=1)
ReLU
MaxPool2d(kernel_size=2)

Flatten

Linear(16*7*7, 64)
ReLU
Linear(64, 10)
```

You are free to choose your own simple design as long as it contains:

* Convolution
* Pooling
* Fully connected layer

Slides 10–41 provide intuition about how convolution, pooling and CNN architecture work. 

---

## **5. Optimization Algorithms to Implement**

Your training script must support **all five**:

### (1) **SGD**

Plain stochastic gradient descent.

### (2) **SGD with Momentum (SGDM)**

Using PyTorch’s `momentum=` option.

### (3) **AdaGrad**

### (4) **RMSProp**

### (5) **Adam**

These methods correspond to the optimizers discussed on slides 6–8. 

You may use:

```python
torch.optim.SGD
torch.optim.Adagrad
torch.optim.RMSprop
torch.optim.Adam
```

---

## **6. Hyperparameter Tuning**

For each optimizer:

* You must **tune the learning rate** using a **validation set**
* You must document:

  * Learning rates tried
  * Validation performance
  * Final chosen learning rate

Early stopping is optional but allowed (related to slide 3 on early stopping) .

---

## **7. Training Procedure**

For **each optimizer**, run:

1. Initialize the CNN weights
2. Select candidate learning rates
3. For each learning rate:

   * Train the CNN
   * Compute validation loss/accuracy every epoch
4. Choose the best learning rate
5. Retrain or continue training using that learning rate
6. Evaluate on test set
7. Record the results

You must record:

* **Training loss/accuracy per epoch**
* **Validation loss/accuracy per epoch**
* **Final test loss/accuracy**

---

## **8. Required Visualizations**

For each optimizer, plot:

### **1. Training loss curve**

### **2. Validation loss curve**

### **3. Training accuracy curve**

### **4. Validation accuracy curve**

And provide:

### **5. Final test accuracy table**

Example table:

| Optimizer | Best LR | Final Test Accuracy | Notes                |
| --------- | ------- | ------------------- | -------------------- |
| SGD       | 0.1     | 97.8%               | Slow but stable      |
| SGDM      | 0.05    | 98.6%               | Faster convergence   |
| AdaGrad   | 0.01    | 97.3%               | LR shrinks quickly   |
| RMSProp   | 0.001   | 98.8%               | Very smooth training |
| Adam      | 0.001   | 99.1%               | Fastest convergence  |
