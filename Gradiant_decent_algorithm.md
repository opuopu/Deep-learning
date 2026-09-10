# Gradient Descent From Scratch — Step-by-Step Regression Example

In this project, we will understand **Gradient Descent from scratch** using a simple real-world regression problem.

We will learn:

- What Gradient Descent does
- Train/Test Split
- Prediction
- Error
- Loss
- Weight and Bias
- Gradient calculation
- Weight and Bias update
- Epochs
- Stopping condition
- Testing the model
- MAE, MSE, RMSE and R²
- How this connects to Deep Learning

---

# 1. Problem

Suppose we want to predict a student's **marks based on study hours**.

Our dataset:

| Study Hours | Marks |
|---:|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |
| 6 | 13 |
| 7 | 15 |
| 8 | 17 |
| 9 | 19 |
| 10 | 21 |

The actual relationship in this dataset is:

```text
Marks = 2 × Study Hours + 1
```

Therefore, the ideal parameters are:

```text
Weight = 2
Bias = 1
```

But our Machine Learning model does **not know these values**.

Our goal is to let the model discover the correct `weight` and `bias` using **Gradient Descent**.

---

# 2. Model Formula

For simple Linear Regression:

```text
y_pred = weight × x + bias
```

Or:

```text
ŷ = wx + b
```

Where:

```text
x = input
ŷ = predicted output
w = weight
b = bias
```

### Weight

Weight controls how strongly the input affects the prediction.

Example:

```text
weight = 2
```

means increasing `x` by 1 increases the prediction by approximately 2.

### Bias

Bias is the starting/base value of the model.

Example:

```text
y = 2x + 1
```

Here:

```text
weight = 2
bias = 1
```

---

# 3. Import Libraries

```python
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# 4. Create Dataset

```python
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
], dtype=float)


y = np.array([
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21
], dtype=float)
```

Here:

```text
X = Features / Inputs
y = Target / Correct Answers
```

---

# 5. Train Test Split

We should not train and test our model using exactly the same data.

So we split the dataset:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Meaning:

```text
Full Dataset
     |
     ↓
Train Test Split
     |
     ├───────────────┐
     ↓               ↓
Training Data     Testing Data
    80%               20%
```

### Training Data

```text
X_train = Training Questions / Inputs
y_train = Training Answers
```

The model learns from:

```python
X_train
y_train
```

### Testing Data

```text
X_test = New unseen inputs
y_test = Actual answers
```

The model does NOT use the test data for learning.

---

# 6. Create Prediction Function

Our model follows:

```text
prediction = weight × input + bias
```

Create a reusable function:

```python
def predict(X, weight, bias):

    y_pred = weight * X.flatten() + bias

    return y_pred
```

Example:

```python
predictions = predict(
    X_train,
    weight=1,
    bias=0
)

print(predictions)
```

---

# 7. Error

For each prediction:

```text
Error = Actual - Prediction
```

Example:

```text
Actual = 10
Prediction = 8

Error = 10 - 8
      = 2
```

Error tells us how wrong **one prediction** is.

---

# 8. Loss

Error and Loss are not exactly the same.

```text
Error
=
How wrong one prediction is
```

```text
Loss
=
How wrong the model is overall
```

We will use **Mean Squared Error (MSE)**.

Formula:

```text
MSE = Average of (Actual - Prediction)²
```

Reusable function:

```python
def calculate_loss(y_true, y_pred):

    errors = y_true - y_pred

    mse = np.mean(errors ** 2)

    return mse
```

Example:

```text
Errors = [2, 3]

Squared Errors:

2² = 4
3² = 9

MSE = (4 + 9) / 2

MSE = 6.5
```

Our Gradient Descent algorithm wants to make this loss smaller.

---

# 9. What is a Gradient?

Gradient tells us:

> Which direction should we change the parameters to reduce the loss?

We have two parameters:

```text
weight
bias
```

Therefore, we need:

```text
gradient_weight
gradient_bias
```

---

# 10. Weight Gradient

For MSE, the weight gradient is:

```text
gradient_w = -(2/n) × Σ[x × (actual - prediction)]
```

Where:

```text
n = number of training samples

x = input

actual - prediction = error
```

---

# 11. Bias Gradient

Bias gradient:

```text
gradient_b = -(2/n) × Σ(actual - prediction)
```

---

# 12. Reusable Gradient Function

```python
def calculate_gradients(X, y_true, y_pred):

    n = len(X)

    errors = y_true - y_pred

    gradient_w = -(2 / n) * np.sum(
        X.flatten() * errors
    )

    gradient_b = -(2 / n) * np.sum(
        errors
    )

    return gradient_w, gradient_b
```

This function calculates how we should change:

```text
weight
bias
```

---

# 13. Learning Rate

We don't want to change the weight and bias by a huge amount at once.

So we use a **Learning Rate**.

Example:

```python
learning_rate = 0.01
```

Learning Rate controls the step size.

```text
Very Large Learning Rate
        ↓
May overshoot the minimum


Very Small Learning Rate
        ↓
Training becomes very slow


Good Learning Rate
        ↓
Gradually reaches minimum loss
```

---

# 14. Update Weight and Bias

Gradient Descent update rule:

```text
new_weight
=
old_weight - learning_rate × gradient_w
```

And:

```text
new_bias
=
old_bias - learning_rate × gradient_b
```

Create reusable function:

```python
def update_parameters(
    weight,
    bias,
    gradient_w,
    gradient_b,
    learning_rate
):

    weight = weight - learning_rate * gradient_w

    bias = bias - learning_rate * gradient_b

    return weight, bias
```

---

# 15. One Complete Gradient Descent Iteration

One iteration works like this:

```text
Current Weight & Bias
        ↓
Prediction
        ↓
Calculate Error
        ↓
Calculate Loss
        ↓
Calculate Gradients
        ↓
Update Weight & Bias
        ↓
Better Weight & Bias
```

Then we repeat the same process.

---

# 16. What is an Epoch?

One complete pass through the entire training dataset is called an:

```text
Epoch
```

For example:

```text
Epoch 1
Prediction → Loss → Gradient → Update

Epoch 2
Prediction → Loss → Gradient → Update

Epoch 3
Prediction → Loss → Gradient → Update

...

Epoch 1000
```

As training continues, ideally:

```text
Loss ↓
Loss ↓
Loss ↓
```

---

# 17. Stopping Condition

We don't want Gradient Descent to run forever.

There are two common ways to stop.

### Maximum Epochs

Example:

```python
epochs = 5000
```

This means:

```text
Run maximum 5000 iterations.
```

### Tolerance / Early Stopping

Suppose:

```text
Old Loss = 0.03001
New Loss = 0.03000
```

The improvement is extremely small.

We can stop training.

Example:

```python
tolerance = 0.000001
```

Check:

```python
if abs(previous_loss - loss) < tolerance:
    break
```

Meaning:

```text
If loss is no longer improving significantly
→ Stop Training
```

---

# 18. Build Complete Training Function

Now combine everything.

```python
def train_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=5000,
    tolerance=0.000001
):

    # Initial parameters
    weight = 0.0
    bias = 0.0

    # Initially we don't have any previous loss
    previous_loss = float("inf")

    for epoch in range(epochs):

        # -------------------------
        # Step 1: Prediction
        # -------------------------

        y_pred = predict(
            X_train,
            weight,
            bias
        )


        # -------------------------
        # Step 2: Calculate Loss
        # -------------------------

        loss = calculate_loss(
            y_train,
            y_pred
        )


        # -------------------------
        # Step 3: Calculate Gradient
        # -------------------------

        gradient_w, gradient_b = calculate_gradients(
            X_train,
            y_train,
            y_pred
        )


        # -------------------------
        # Step 4: Update Parameters
        # -------------------------

        weight, bias = update_parameters(
            weight,
            bias,
            gradient_w,
            gradient_b,
            learning_rate
        )


        # -------------------------
        # Step 5: Show Progress
        # -------------------------

        if epoch % 100 == 0:

            print(
                f"Epoch: {epoch}, "
                f"Loss: {loss:.6f}, "
                f"Weight: {weight:.4f}, "
                f"Bias: {bias:.4f}"
            )


        # -------------------------
        # Step 6: Early Stopping
        # -------------------------

        if abs(previous_loss - loss) < tolerance:

            print(
                f"\nTraining stopped at epoch: {epoch}"
            )

            break


        previous_loss = loss


    return weight, bias
```

---

# 19. Train Our Model

Now call the training function:

```python
weight, bias = train_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=5000
)
```

Print final parameters:

```python
print("Final Weight:", weight)
print("Final Bias:", bias)
```

Because our original dataset follows:

```text
y = 2x + 1
```

we expect the model to learn values close to:

```text
Weight ≈ 2

Bias ≈ 1
```

---

# 20. Test the Model

Training is complete.

Now we should check whether our model works on data it did not use for training.

Make predictions:

```python
y_pred_test = predict(
    X_test,
    weight,
    bias
)
```

Print actual values:

```python
print("Actual Values:")
print(y_test)
```

Print predicted values:

```python
print("Predicted Values:")
print(y_pred_test)
```

We are comparing:

```text
y_test
   ↓
Actual Answer


y_pred_test
   ↓
Model Prediction
```

---

# 21. Evaluate Model Performance

Now calculate:

```text
MAE
MSE
RMSE
R²
```

---

## MAE — Mean Absolute Error

MAE tells us:

> On average, how much is our prediction wrong?

```python
mae = mean_absolute_error(
    y_test,
    y_pred_test
)
```

Smaller is better.

Perfect:

```text
MAE = 0
```

---

## MSE — Mean Squared Error

MSE squares the errors.

```python
mse = mean_squared_error(
    y_test,
    y_pred_test
)
```

Large mistakes receive a bigger penalty.

Smaller is better.

Perfect:

```text
MSE = 0
```

---

## RMSE — Root Mean Squared Error

RMSE is:

```text
RMSE = √MSE
```

Code:

```python
rmse = np.sqrt(mse)
```

It brings the error back to the original unit.

Smaller is better.

Perfect:

```text
RMSE = 0
```

---

## R² Score

R² tells us how well the model explains the variation in the target.

```python
r2 = r2_score(
    y_test,
    y_pred_test
)
```

Perfect:

```text
R² = 1
```

Important:

```text
R² is NOT accuracy.
```

It is a regression evaluation metric.

---

# 22. Print Evaluation Results

```python
print("\nModel Evaluation")

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2)
```

---

# 23. Perfect Model

A theoretically perfect regression model would have:

```text
MAE  = 0

MSE  = 0

RMSE = 0

R²   = 1
```

However, real-world datasets contain noise, so perfect results are usually not expected.

---

# 24. Complete Code

```python
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. DATASET
# ==========================================

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
], dtype=float)


y = np.array([
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21
], dtype=float)


# ==========================================
# 2. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict(X, weight, bias):

    y_pred = weight * X.flatten() + bias

    return y_pred


# ==========================================
# 4. LOSS FUNCTION
# ==========================================

def calculate_loss(y_true, y_pred):

    errors = y_true - y_pred

    mse = np.mean(errors ** 2)

    return mse


# ==========================================
# 5. GRADIENT FUNCTION
# ==========================================

def calculate_gradients(X, y_true, y_pred):

    n = len(X)

    errors = y_true - y_pred

    gradient_w = -(2 / n) * np.sum(
        X.flatten() * errors
    )

    gradient_b = -(2 / n) * np.sum(
        errors
    )

    return gradient_w, gradient_b


# ==========================================
# 6. UPDATE PARAMETERS
# ==========================================

def update_parameters(
    weight,
    bias,
    gradient_w,
    gradient_b,
    learning_rate
):

    weight = weight - learning_rate * gradient_w

    bias = bias - learning_rate * gradient_b

    return weight, bias


# ==========================================
# 7. TRAINING FUNCTION
# ==========================================

def train_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=5000,
    tolerance=0.000001
):

    # Initial weight and bias
    weight = 0.0
    bias = 0.0

    previous_loss = float("inf")


    for epoch in range(epochs):


        # ----------------------------------
        # Prediction
        # ----------------------------------

        y_pred = predict(
            X_train,
            weight,
            bias
        )


        # ----------------------------------
        # Loss
        # ----------------------------------

        loss = calculate_loss(
            y_train,
            y_pred
        )


        # ----------------------------------
        # Gradient
        # ----------------------------------

        gradient_w, gradient_b = calculate_gradients(
            X_train,
            y_train,
            y_pred
        )


        # ----------------------------------
        # Update Weight & Bias
        # ----------------------------------

        weight, bias = update_parameters(
            weight,
            bias,
            gradient_w,
            gradient_b,
            learning_rate
        )


        # ----------------------------------
        # Print Training Progress
        # ----------------------------------

        if epoch % 100 == 0:

            print(
                f"Epoch: {epoch}, "
                f"Loss: {loss:.6f}, "
                f"Weight: {weight:.4f}, "
                f"Bias: {bias:.4f}"
            )


        # ----------------------------------
        # Early Stopping
        # ----------------------------------

        if abs(previous_loss - loss) < tolerance:

            print(
                f"\nTraining stopped at epoch: {epoch}"
            )

            break


        previous_loss = loss


    return weight, bias


# ==========================================
# 8. TRAIN MODEL
# ==========================================

weight, bias = train_gradient_descent(
    X_train,
    y_train,
    learning_rate=0.01,
    epochs=5000
)


print("\nFinal Weight:", weight)

print("Final Bias:", bias)


# ==========================================
# 9. TEST MODEL
# ==========================================

y_pred_test = predict(
    X_test,
    weight,
    bias
)


print("\nActual Values:")

print(y_test)


print("\nPredicted Values:")

print(y_pred_test)


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred_test
)


mse = mean_squared_error(
    y_test,
    y_pred_test
)


rmse = np.sqrt(mse)


r2 = r2_score(
    y_test,
    y_pred_test
)


# ==========================================
# 11. PRINT RESULTS
# ==========================================

print("\nModel Evaluation")

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)

print("R2 Score:", r2)
```

---

# 25. Complete Gradient Descent Flow

```text
Dataset
   ↓
Separate X and y
   ↓
Train Test Split
   ↓
X_train + y_train
   ↓
Initialize Weight & Bias
   ↓
Prediction
   ↓
Calculate Error
   ↓
Calculate Loss
   ↓
Calculate Gradients
   ↓
Update Weight & Bias
   ↓
Repeat
   ↓
Loss Stops Improving
   ↓
Training Stops
   ↓
Use X_test
   ↓
Generate Predictions
   ↓
Compare with y_test
   ↓
MAE / MSE / RMSE / R²
```

---

# 26. Most Important Concept

Gradient Descent can be summarized as:

```text
Prediction
    ↓
Error
    ↓
Loss
    ↓
Gradient
    ↓
Update Weight & Bias
    ↓
Better Prediction
    ↓
Repeat
```

The main goal is:

```text
MINIMIZE LOSS
```

by finding better:

```text
Weight
Bias
```

---

# 27. Connection With Deep Learning

The exact same fundamental idea is used in Deep Learning.

Our simple regression model has:

```text
Input
 ↓
Weight + Bias
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Update
```

A Neural Network has:

```text
Input Layer
      ↓
Hidden Layer
      ↓
Hidden Layer
      ↓
Output Layer
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimizer
      ↓
Update Weights & Biases
```

The main difference is scale.

Our example may have:

```text
1 Weight
1 Bias
```

A Deep Learning model may have:

```text
Thousands
Millions
or
Billions of Parameters
```

But the core learning idea remains:

```text
Make Prediction
      ↓
Measure Loss
      ↓
Find Gradients
      ↓
Update Parameters
      ↓
Make Better Prediction
```

---

# 28. Gradient Descent vs Deep Learning Terminology

| Our Example | Deep Learning |
|---|---|
| Prediction | Forward Propagation |
| MSE | Loss Function |
| Calculate Gradient | Backpropagation |
| Weight/Bias Update | Optimizer Step |
| Learning Rate | Learning Rate |
| Repeating Training | Epochs |
| Weight + Bias | Model Parameters |

---

# 29. Final Summary

Remember this:

```text
X_train
=
Training Inputs


y_train
=
Training Correct Answers


Weight
=
How strongly input affects prediction


Bias
=
Starting/base value


Prediction
=
Model's answer


Error
=
Actual - Prediction


Loss
=
Overall model error


Gradient
=
Direction and rate of loss change


Learning Rate
=
How large each update step should be


Epoch
=
One complete training cycle


X_test
=
Unseen inputs


y_test
=
Actual answers for test data
```

And finally:

```text
GRADIENT DESCENT

Prediction
    ↓
Loss
    ↓
Gradient
    ↓
Update Weight & Bias
    ↓
Repeat
    ↓
Minimum Loss
```

This is one of the fundamental ideas behind Machine Learning and Deep Learning.
