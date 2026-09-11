# 5.6 Binary Cross Entropy (BCE)

## BCE কী?

Binary Cross Entropy হলো একটি **Loss Function**।

এটি Binary Classification-এ model-এর prediction কতটা ভুল হয়েছে তা measure করে।

Example:

    Spam     = 1
    Not Spam = 0

---

## Formula

    BCE = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

যেখানে:

    y = Actual Value (0 অথবা 1)
    ŷ = Predicted Probability

---

## সহজ Example

ধরো:

    Actual = 1

### Good Prediction

    Prediction = 0.9
    Loss ≈ 0.105 ✅

### Bad Prediction

    Prediction = 0.1
    Loss ≈ 2.303 ❌

অর্থাৎ:

    Prediction Actual-এর কাছে → Loss কম
    Prediction Actual থেকে দূরে → Loss বেশি

---

## Shortcut

Actual `1` হলে:

    Loss = -log(Prediction)

Actual `0` হলে:

    Loss = -log(1 - Prediction)

---

## Deep Learning Flow

    Input
      ↓
    wx + b
      ↓
    Sigmoid
      ↓
    Prediction
      ↓
    Binary Cross Entropy
      ↓
    Loss
      ↓
    Gradient Descent
      ↓
    Weight & Bias Update

---

## Memory Trick

> **Sigmoid → Probability দেয়**
>
> **BCE → Prediction কতটা ভুল সেটা বলে**
>
> **Gradient Descent → Loss কমাতে Weight & Bias update করে**

### One-Line Definition

**Binary Cross Entropy = Binary Classification-এর prediction error measure করার Loss Function।**
