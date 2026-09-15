# Backpropagation From Scratch — Simple Neural Network

## Architecture

আমরা একটি simple neural network ব্যবহার করছি:

```text
2 Inputs → 2 Hidden Neurons → 1 Output

x1 ──┐
     ├──→ h1 ──┐
x2 ──┘         │
               ├──→ Output → Loss
x1 ──┐         │
     ├──→ h2 ──┘
x2 ──┘
```

Architecture:

```text
2 → 2 → 1
```

- Input neurons = 2
- Hidden layers = 1
- Hidden neurons = 2
- Hidden activation = ReLU
- Output neurons = 1
- Output activation = Sigmoid
- Loss function = Mean Squared Error (MSE)

---

# 1. Input Data

```python
import numpy as np

X = np.array([
    [1.0],
    [2.0]
])

y = np.array([[1.0]])
```

Here:

```text
x1 = 1
x2 = 2

Actual output y = 1
```

---

# 2. Initialize Weights and Biases

## Input → Hidden Layer

```python
W1 = np.array([
    [0.2, 0.3],
    [0.4, 0.5]
])

b1 = np.array([
    [0.1],
    [0.1]
])
```

Individual connections:

```text
x1 → h1 = w11 = 0.2
x2 → h1 = w12 = 0.3

x1 → h2 = w21 = 0.4
x2 → h2 = w22 = 0.5
```

So:

```text
       x1    x2

W1 = [0.2   0.3]  ← h1
     [0.4   0.5]  ← h2
```

---

## Hidden Layer → Output

```python
W2 = np.array([
    [0.6, 0.7]
])

b2 = np.array([
    [0.1]
])
```

Connections:

```text
h1 → Output = 0.6
h2 → Output = 0.7
```

---

# 3. Forward Propagation

Forward propagation goes:

```text
X
↓
W1, b1
↓
z1
↓
ReLU
↓
h1
↓
W2, b2
↓
z2
↓
Sigmoid
↓
y_pred
↓
Loss
```

## Hidden Layer

First calculate `z1`:

```python
z1 = W1 @ X + b1
```

Then apply ReLU:

```python
h1 = np.maximum(0, z1)
```

`z` means:

> Value before activation.

`h` means:

> Value after activation.

---

## Output Layer

```python
z2 = W2 @ h1 + b2
```

Apply Sigmoid:

```python
y_pred = 1 / (1 + np.exp(-z2))
```

Now `y_pred` is the model prediction.

---

# 4. Loss Function

We use Mean Squared Error:

```python
loss = (y_pred - y) ** 2
```

MSE measures the difference between:

```text
Prediction
    vs
Actual Value
```

---

# 5. What is Backpropagation?

Forward propagation creates the prediction.

Backpropagation finds:

> Which weights and biases were responsible for the error, and how much?

Backward direction:

```text
Loss
 ↑
Output
 ↑
Hidden Layer
 ↑
Input
```

For our network:

```text
Loss
 ↓
dz2
 ↓
dW2, db2
 ↓
dh1
 ↓
dz1
 ↓
dW1, db1
```

---

# 6. Meaning of d

During backpropagation, `d` means gradient.

For example:

```text
dW = gradient of Weight
db = gradient of Bias
dh = gradient of hidden output
dz = gradient of z
```

Easy meaning:

> Gradient tells us how much a value affects the final loss.

---

# 7. Output Layer Backpropagation

We start from the output because it is closest to the Loss.

Forward was:

```text
z2
 ↓
Sigmoid
 ↓
y_pred
 ↓
Loss
```

Backward goes in the opposite direction.

## Calculate dz2

Because we used:

- MSE Loss
- Sigmoid activation

we calculate:

```python
dz2 = 2 * (y_pred - y) * y_pred * (1 - y_pred)
```

Here:

```text
2 * (y_pred - y)
```

is the MSE derivative.

And:

```text
y_pred * (1 - y_pred)
```

is the Sigmoid derivative.

Therefore:

```text
dz2 = Output error gradient
```

---

# 8. Calculate dW2

Forward:

```text
h1 ── W2 ──→ z2
```

We need to know how much each weight in `W2` contributed to the loss.

Formula:

```python
dW2 = dz2 @ h1.T
```

Easy shortcut:

```text
dW = current error × previous layer output
```

For example, if:

```text
dz2 = -0.1

h1 = [
    0.8
    1.5
]
```

then:

```text
h1.T = [0.8  1.5]
```

So:

```text
dW2 = -0.1 × [0.8  1.5]

     = [-0.08  -0.15]
```

Meaning:

```text
W2  = [ 0.6    0.7  ]
         ↓      ↓
dW2 = [-0.08  -0.15]
```

Each weight gets its own gradient.

---

# 9. Calculate db2

Bias gradient:

```python
db2 = dz2
```

So now we have:

```text
dz2
 ├──→ dW2
 └──→ db2
```

We have calculated the gradients for the output layer.

---

# 10. Send Error Back to Hidden Layer

Now the output layer is done.

But `h1` was responsible for producing the output too.

So we send the error backward:

```python
dh1 = W2.T @ dz2
```

`dh1` means:

> How much each hidden neuron contributed to the final loss.

Our hidden layer contains two neurons.

Therefore `dh1` contains two gradients.

```text
Output Error
     ↓
    W2
     ↓
    dh1
   /   \
 h1     h2
error  error
```

---

# 11. Backward Through ReLU

During forward propagation:

```text
z1
 ↓
ReLU
 ↓
h1
```

Therefore during backward propagation:

```text
dh1
 ↓
ReLU derivative
 ↓
dz1
```

Code:

```python
dz1 = dh1 * (z1 > 0)
```

ReLU derivative works like:

```text
If z > 0:
    gradient passes → 1

If z <= 0:
    gradient stops → 0
```

Example:

```text
z1  = [0.8, -0.5]

dh1 = [-0.2, -0.4]
```

ReLU derivative:

```text
z1 > 0

= [1, 0]
```

Therefore:

```text
dz1 = dh1 × [1, 0]

    = [-0.2, 0]
```

The second neuron does not pass the gradient because its `z` was negative.

---

# 12. Calculate dW1

Now we know `dz1`.

We need the gradients of the first layer weights.

Formula:

```python
dW1 = dz1 @ X.T
```

Again, same rule:

```text
dW = current error × previous layer output
```

For the first hidden layer, the previous values are the original inputs `X`.

Therefore:

```text
dz1 @ X.T
```

gives us the gradient for every weight inside `W1`.

---

# 13. Calculate db1

```python
db1 = dz1
```

Now backpropagation is complete.

We have:

```text
dW1
db1

dW2
db2
```

---

# 14. Complete Backpropagation Code

```python
# =========================
# OUTPUT LAYER
# =========================

# Output error gradient
dz2 = 2 * (y_pred - y) * y_pred * (1 - y_pred)

# Output weight gradients
dW2 = dz2 @ h1.T

# Output bias gradient
db2 = dz2


# =========================
# HIDDEN LAYER
# =========================

# Send error back to hidden layer
dh1 = W2.T @ dz2

# Backward through ReLU
dz1 = dh1 * (z1 > 0)

# Hidden layer weight gradients
dW1 = dz1 @ X.T

# Hidden layer bias gradients
db1 = dz1
```

---

# 15. Backpropagation Common Pattern

The most important pattern is:

```text
OUTPUT:

dz
↓
dW, db


HIDDEN LAYER:

dh
↓
dz
↓
dW, db
```

For a hidden layer:

```python
dh = W_next.T @ dz_next

dz = dh * activation_derivative

dW = dz @ previous_output.T

db = dz
```

For ReLU:

```python
dz = dh * (z > 0)
```

---

# 16. Gradient Descent

Backpropagation DOES NOT update the weights.

Backpropagation only calculates:

```text
dW
db
```

Gradient Descent uses those gradients to update the parameters.

Formula:

```text
New Weight = Old Weight - Learning Rate × Gradient
```

Code:

```python
lr = 0.1

W1 = W1 - lr * dW1
b1 = b1 - lr * db1

W2 = W2 - lr * dW2
b2 = b2 - lr * db2
```

---

# 17. Complete Training Flow

The complete process is:

```text
        FORWARD
           ↓
        Prediction
           ↓
          Loss
           ↓
       BACKWARD
           ↓
     dW1, db1, dW2, db2
           ↓
    GRADIENT DESCENT
           ↓
     Update W and b
           ↓
        FORWARD AGAIN
           ↓
        New Loss
           ↓
          Repeat
```

This repeated process is called **training**.

---

# 18. Full Code

```python
import numpy as np


# =========================
# DATA
# =========================

X = np.array([
    [1.0],
    [2.0]
])

y = np.array([[1.0]])


# =========================
# PARAMETERS
# =========================

W1 = np.array([
    [0.2, 0.3],
    [0.4, 0.5]
])

b1 = np.array([
    [0.1],
    [0.1]
])

W2 = np.array([
    [0.6, 0.7]
])

b2 = np.array([
    [0.1]
])


# =========================
# FORWARD
# =========================

# Hidden layer
z1 = W1 @ X + b1
h1 = np.maximum(0, z1)

# Output layer
z2 = W2 @ h1 + b2

# Sigmoid
y_pred = 1 / (1 + np.exp(-z2))

# MSE
loss = (y_pred - y) ** 2

print("Prediction:", y_pred)
print("Loss:", loss)


# =========================
# BACKWARD
# =========================

# Output error
dz2 = 2 * (y_pred - y) * y_pred * (1 - y_pred)

# Output weight and bias gradients
dW2 = dz2 @ h1.T
db2 = dz2

# Send error back to hidden layer
dh1 = W2.T @ dz2

# ReLU backward
dz1 = dh1 * (z1 > 0)

# Hidden layer weight and bias gradients
dW1 = dz1 @ X.T
db1 = dz1


print("\ndW2:")
print(dW2)

print("\ndb2:")
print(db2)

print("\ndW1:")
print(dW1)

print("\ndb1:")
print(db1)


# =========================
# GRADIENT DESCENT
# =========================

lr = 0.1

W1 = W1 - lr * dW1
b1 = b1 - lr * db1

W2 = W2 - lr * dW2
b2 = b2 - lr * db2


print("\nUpdated W1:")
print(W1)

print("\nUpdated W2:")
print(W2)

print("\nUpdated b1:")
print(b1)

print("\nUpdated b2:")
print(b2)
```

---

# Quick Revision

Remember these:

```text
z  = value before activation

h  = output after activation

dz = gradient of z

dh = gradient of hidden output

dW = gradient of weight

db = gradient of bias
```

### Backpropagation Shortcut

```text
Output:
dz → dW, db

Hidden:
dh → dz → dW, db
```

### Gradient Descent Shortcut

```python
W = W - lr * dW
b = b - lr * db
```

### Final Concept

```text
Forward Propagation
= Make prediction

Loss
= Measure error

Backpropagation
= Find gradients

Gradient Descent
= Update weights and biases

Repeat
= Train the neural network
```
