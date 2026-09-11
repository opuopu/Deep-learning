# 5.5 Softmax Activation Function

## 1. Softmax কী?

Softmax হলো একটি **Activation Function**, যেটা সাধারণত
**Multi-Class Classification-এর Output Layer**-এ ব্যবহার করা হয়।

এর main কাজ:

> অনেকগুলো class-এর raw score-কে probability-তে convert করা,
> যেখানে সব probability যোগ করলে 1 বা 100% হয়।

---

# 2. সহজ Example

ধরো Neural Network একটা ছবি দেখে predict করবে:

    Cat
    Dog
    Horse

Network প্রথমে raw score দিল:

    Cat   → 2
    Dog   → 3
    Horse → 1

এগুলো এখনো probability না।

Softmax apply করার পরে:

    Cat   → 24.5%
    Dog   → 66.5%
    Horse →  9.0%
            -----
    Total → 100%

এখানে Dog-এর probability সবচেয়ে বেশি।

তাই:

    Prediction = Dog

---

# 3. Softmax Formula

Formula:

    Softmax(zi) = e^zi / Σ(e^zj)

সহজভাবে:

    প্রতিটা score-এর e^score বের করো
                ↓
    সবগুলো যোগ করো (Σ)
                ↓
    প্রত্যেকটাকে Total দিয়ে ভাগ করো
                ↓
    Probability

---

# 4. Step-by-Step Calculation

আমাদের raw scores:

    Cat   = 2
    Dog   = 3
    Horse = 1

## Step 1: e^score বের করি

আমরা জানি:

    e ≈ 2.718

তাই:

    Cat:
    e^2 ≈ 7.39

    Dog:
    e^3 ≈ 20.09

    Horse:
    e^1 ≈ 2.72

এখন:

    Cat   →  7.39
    Dog   → 20.09
    Horse →  2.72

---

## Step 2: সবগুলো যোগ করি

এখানে Σ মানে সবগুলো যোগ করা।

    Total = 7.39 + 20.09 + 2.72

    Total = 30.20

---

## Step 3: প্রত্যেকটাকে Total দিয়ে ভাগ করি

### Cat

    7.39 / 30.20
    = 0.245
    = 24.5%

### Dog

    20.09 / 30.20
    = 0.665
    = 66.5%

### Horse

    2.72 / 30.20
    = 0.090
    = 9%

---

# 5. Final Result

    Cat   → 24.5%
    Dog   → 66.5%  ← Highest
    Horse →  9.0%
            -----
    Total → 100%

তাই Model predict করবে:

    Prediction = Dog

---

# 6. Full Softmax Flow

    Neural Network
          ↓
      Raw Scores
          ↓
    [2, 3, 1]
          ↓
       Softmax
          ↓
    [0.245, 0.665, 0.090]
          ↓
    [24.5%, 66.5%, 9%]
          ↓
     Highest = Dog

---

# 7. Softmax কোথায় ব্যবহার হয়?

Softmax সাধারণত ব্যবহার হয় যখন:

> অনেকগুলো class আছে, কিন্তু final answer হবে একটি class।

Example:

### Animal Classification

    Cat
    Dog
    Horse

### Digit Classification

    0
    1
    2
    3
    ...
    9

### Fruit Classification

    Apple
    Banana
    Mango
    Orange

এগুলো সব:

    Multi-Class Classification

তাই সাধারণত:

    Output Layer → Softmax

---

# 8. Sigmoid vs Softmax

## Sigmoid

যখন Binary Classification:

    Spam / Not Spam

    Fraud / Not Fraud

    Pass / Fail

সাধারণত:

    Output Layer → Sigmoid

---

## Softmax

যখন Multi-Class Classification:

    Cat / Dog / Horse

    Apple / Mango / Banana

    Car / Bus / Bike

সাধারণত:

    Output Layer → Softmax

---

# 9. ReLU + Sigmoid + Softmax

সহজভাবে মনে রাখি:

    Hidden Layer
         ↓
       ReLU

    Binary Classification
    Yes / No
         ↓
      Sigmoid

    Multi-Class Classification
    Cat / Dog / Horse
         ↓
      Softmax

---

# 10. Memory Trick

Softmax মনে রাখার সবচেয়ে সহজ উপায়:

    Multiple Classes
          ↓
      Raw Scores
          ↓
       Softmax
          ↓
    Probabilities
          ↓
    Total = 100%
          ↓
    Highest = Prediction

Example:

    Cat   → 2
    Dog   → 3
    Horse → 1

          ↓ Softmax

    Cat   → 24.5%
    Dog   → 66.5%
    Horse →  9.0%

          ↓

    Prediction = Dog

---

# One-Line Definition

> **Softmax অনেকগুলো class-এর raw score-কে probability-তে
> convert করে, যেখানে সব probability-এর total 100% হয় এবং
> highest probability-র class সাধারণত final prediction হয়।**

---

# Super Short Revision

    Softmax = Multi-Class Classification

    Input  = Raw Scores

    Output = Probabilities

    Total Probability = 100%

    Highest Probability = Predicted Class
