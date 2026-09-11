# 5.3 Sigmoid Activation Function

## 1. Sigmoid কী?

Sigmoid হলো একটি **Activation Function**।

এর main কাজ:

> যেকোনো input number-কে **0 থেকে 1-এর মধ্যে** convert করা।

Example:

| Input (z) | Sigmoid Output |
|---:|---:|
| -10 | ~0.000 |
| -5 | ~0.007 |
| -2 | ~0.119 |
| 0 | **0.500** |
| 2 | ~0.881 |
| 5 | ~0.993 |
| 10 | ~1.000 |

### সহজ Pattern

- অনেক Negative Number → Output প্রায় `0`
- `0` → Output exactly `0.5`
- অনেক Positive Number → Output প্রায় `1`

---

## 2. Sigmoid কেন দরকার?

ধরো আমরা একটি Neural Network দিয়ে predict করছি:

**Email Spam কি না?**

আমাদের final output এমন হলে বুঝতে সুবিধা:

- `0` → Not Spam
- `1` → Spam

কিন্তু neuron প্রথমে এমন calculation করতে পারে:

    z = wx + b

এবং result আসতে পারে:

    z = 2.5

`2.5` সরাসরি probability না।

Sigmoid এটাকে 0 থেকে 1-এর মধ্যে নিয়ে আসবে:

    2.5 → Sigmoid → 0.924

অর্থাৎ প্রায়:

    92.4% probability

---

## 3. Sigmoid Formula

Formula:

    sigmoid(z) = 1 / (1 + e^(-z))

এখানে:

- `z` = neuron-এর calculated value
- `e` ≈ 2.718
- `sigmoid(z)` = final value between 0 and 1

Neural Network-এ সাধারণত:

    z = wx + b

তারপর:

    output = sigmoid(z)

---

# 4. Step-by-Step Example

ধরি:

    z = 2

Sigmoid formula:

    sigmoid(z) = 1 / (1 + e^(-z))

তাহলে:

    sigmoid(2) = 1 / (1 + e^(-2))

আমরা জানি:

    e^(-2) ≈ 0.135

তাই:

    sigmoid(2)
    = 1 / (1 + 0.135)
    = 1 / 1.135
    ≈ 0.881

Final:

    sigmoid(2) ≈ 0.881

অর্থাৎ:

    2 → Sigmoid → 0.881

---

# 5. z = 0 হলে কী হবে?

Formula:

    sigmoid(0) = 1 / (1 + e^0)

আমরা জানি:

    e^0 = 1

তাই:

    sigmoid(0)
    = 1 / (1 + 1)
    = 1 / 2
    = 0.5

### Important

    Sigmoid(0) = 0.5

এটা মনে রাখা খুব useful।

---

# 6. Neural Network-এর Full Example

ধরি আমাদের দুইটা input আছে:

    x1 = 2
    x2 = 3

Weights:

    w1 = 0.5
    w2 = 0.7

Bias:

    b = -1

Neuron প্রথমে calculate করবে:

    z = (x1 × w1) + (x2 × w2) + b

Values বসাই:

    z = (2 × 0.5) + (3 × 0.7) - 1

    z = 1 + 2.1 - 1

    z = 2.1

এখন Sigmoid apply করবো:

    output = sigmoid(2.1)

Formula:

    sigmoid(2.1) = 1 / (1 + e^(-2.1))

Result:

    sigmoid(2.1) ≈ 0.891

Final Output:

    0.891

Binary classification-এর ক্ষেত্রে এটাকে প্রায়:

    89.1% probability

হিসেবে interpret করা যায়।

---

# 7. Full Flow

Neural Network-এর basic flow:

    Inputs
       ↓
    Weight + Bias
       ↓
    z = wx + b
       ↓
    Sigmoid(z)
       ↓
    0 থেকে 1-এর মধ্যে Output

আমাদের example:

    x1 = 2
    x2 = 3
       ↓
    z = 2.1
       ↓
    Sigmoid(2.1)
       ↓
    0.891

---

# 8. Sigmoid-এর Graph

Sigmoid দেখতে S-shaped:

    1.0 |                         ______
        |                     ___/
        |                  __/
    0.5 |---------------●---------------
        |             __/
        |          __/
    0.0 |_________/
           -5     0      5

Important:

    z = 0 → Sigmoid = 0.5

Positive দিকে গেলে:

    Output → 1

Negative দিকে গেলে:

    Output → 0

---

# 9. Shortcut মনে রাখার জন্য

## Formula

    z = wx + b

তারপর:

    sigmoid(z) = 1 / (1 + e^(-z))

## Pattern

    Negative → কাছাকাছি 0
    Zero     → 0.5
    Positive → কাছাকাছি 1

## One-Line Definition

> Sigmoid Activation Function neuron-এর `wx + b` result-কে
> `0 থেকে 1` range-এর মধ্যে convert করে।

---

# 10. Super Easy Memory Trick

মনে রাখবো:

    Input
      ↓
    wx + b
      ↓
    Sigmoid
      ↓
    0 ————— 0.5 ————— 1
    NO      Unsure      YES

Example:

    z = -5  → 0.007
    z =  0  → 0.500
    z =  5  → 0.993

So:

**Large Negative → 0**

**Zero → 0.5**

**Large Positive → 1**
