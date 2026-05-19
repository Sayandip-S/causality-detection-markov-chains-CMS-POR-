# Markov Chain ML Predictor Demo

This repository contains a small proof-of-concept for learning an early-warning predictor from simulated Markov-chain paths.

The goal is to connect two ideas:

1. **Formal stochastic model idea:**  
   In a Markov chain, we can ask whether visiting a candidate state `C` raises the probability of later reaching an error/effect state `E`.

2. **Machine learning idea:**  
   We can simulate many paths, take only the first few states as a prefix, and train a classifier to predict whether the full path will eventually reach `E`.

---

## 1. Core Research Idea

The current project starts from a simple setting:

```text
Discrete-Time Markov Chain (DTMC) + Reachability Effect
```

The effect is:

```text
E = eventually reaching an error/bad state
```

The candidate cause or warning state is:

```text
C = a state that may increase the probability of reaching E
```

The probability-raising condition is:

```text
P(E | C) > P(E)
```

This means:

> The probability of reaching the error state after visiting `C` is higher than the general probability of reaching the error state.

In the toy experiment, the code estimates both:

```text
P(E)
P(E | C)
```

and checks whether `C` behaves like a probability-raising warning state.

---

## 2. States in the Toy Markov Chain

The demo uses a small Markov chain with six states:

| State | Meaning |
|---|---|
| `0` | Initial state |
| `1` | Normal intermediate state |
| `2` | Normal intermediate state |
| `3` | Candidate cause / warning state `C` |
| `4` | Error/effect state `E` |
| `5` | Safe absorbing state |

The important idea is:

```text
If the path visits C, reaching E becomes much more likely.
```

---

## 3. Transition Model

The transition model is defined as:

```python
transitions = {
    0: [(1, 0.5), (2, 0.5)],
    1: [(3, 0.7), (5, 0.3)],
    2: [(3, 0.2), (5, 0.8)],
    3: [(4, 0.8), (5, 0.2)],
    4: [(4, 1.0)],
    5: [(5, 1.0)]
}
```

This means:

```text
From state 0:
    go to state 1 with probability 0.5
    go to state 2 with probability 0.5

From state 1:
    go to C with probability 0.7
    go to SAFE with probability 0.3

From state 2:
    go to C with probability 0.2
    go to SAFE with probability 0.8

From C:
    go to E with probability 0.8
    go to SAFE with probability 0.2

From E:
    stay in E forever

From SAFE:
    stay in SAFE forever
```

States `4` and `5` are absorbing states.

That means once the path reaches one of them, the simulation can stop.

---

## 4. What Is a Path?

A path is one simulated execution of the Markov chain.

Example paths:

```text
[0, 1, 3, 4]  -> reaches error E
[0, 1, 3, 5]  -> visits C, but becomes safe
[0, 1, 5]     -> safe
[0, 2, 3, 4]  -> reaches error E
[0, 2, 3, 5]  -> visits C, but becomes safe
[0, 2, 5]     -> safe
```

Because the Markov chain is probabilistic, every generated path is sampled randomly according to the transition probabilities.

---

## 5. Dataset Creation

The function `create_dataset(num_paths=5000, prefix_length=3)` creates the machine learning dataset.

The important line is:

```python
X_train, y_train = create_dataset(num_paths=5000, prefix_length=3)
```

This means:

> Simulate the Markov chain 5000 independent times.

It does **not** mean that all 5000 paths are unique.

Because the toy Markov chain is small, the same path can appear many times. This is normal and expected. Repetition reflects the transition probabilities of the Markov chain.

For each simulated path:

1. Generate one full path.
2. Check whether the full path reaches the error state `E`.
3. Take only the first few states as a prefix.
4. Use the prefix as the ML input.
5. Use the error/no-error outcome as the ML label.

---

## 6. Example of One Training Example

Suppose the generated full path is:

```text
[0, 1, 3, 4]
```

The full path reaches error state `4`, so the label is:

```text
label = 1
```

If `prefix_length = 3`, the input prefix is:

```text
[0, 1, 3]
```

So the ML example is:

```text
X = [0, 1, 3]
y = 1
```

Another example:

```text
Full path: [0, 2, 5]
Prefix:    [0, 2, 5]
Label:     0
```

because this path reaches safe state `5` and does not reach error state `4`.

---

## 7. What Are `X` and `y`?

The dataset has two parts:

```text
X = inputs/features
y = labels/answers
```

Example:

```text
X                         y

[0, 1, 3]                 1
[0, 2, 5]                 0
[0, 1, 5]                 0
[0, 2, 3]                 1
[0, 1, 3]                 1
...
```

The ML model learns:

```text
prefix -> reaches E or does not reach E
```

So the learning problem is:

> Given only the first few states of a path, predict whether the full path will eventually reach the error state.

---

## 8. Why Use Prefixes?

The model is not given the full path.

It only sees an early prefix, such as:

```text
[0, 1, 3]
```

This makes the task an early-warning prediction task.

The idea is:

> Can the model warn us early that the error state is likely to be reached later?

This is connected to the idea of using ML as a monitor.

---

## 9. Padding Prefixes

ML models usually need fixed-size inputs.

But paths may have different lengths.

Example:

```text
[0, 1, 3, 4]
[0, 2, 5]
[0, 1, 5]
```

If a path is shorter than the required prefix length, the code pads it with `-1`.

Example:

```text
[0, 1] with prefix_length = 3 becomes [0, 1, -1]
```

Here, `-1` means:

```text
no state / padding value
```

---

## 10. Model Used: Random Forest Classifier

The demo uses:

```python
RandomForestClassifier
```

A Random Forest is a collection of many decision trees.

A decision tree learns rules such as:

```text
If the prefix contains state 3, predict error.
If the prefix contains state 5, predict safe.
If the path starts 0 -> 1, error is more likely than 0 -> 2.
```

A Random Forest trains many such trees and combines their predictions.

It is useful here because:

- it is simple to train;
- it works well for small tabular datasets;
- it is a strong baseline;
- it is easier to explain than a deep neural network.

---

## 11. Training Phase

The training phase is:

```text
Markov chain
    ↓
simulate 5000 paths
    ↓
take first 3 states as prefix
    ↓
label each path:
        1 = reaches E
        0 = does not reach E
    ↓
train Random Forest classifier
```

The model learns from the training examples.

For example, it may learn that:

```text
prefix contains C -> error is likely
prefix reaches SAFE -> error is unlikely
```

---

## 12. Testing Phase

The testing phase is separate from training.

The code creates new test data:

```python
X_test, y_test = create_dataset(num_paths=1000, prefix_length=3)
```

This means:

> Simulate 1000 new paths that the model did not see during training.

The model receives only:

```text
X_test
```

and predicts:

```text
y_pred
```

Then the predicted labels are compared with the true labels:

```text
y_pred vs y_test
```

This evaluates whether the model learned useful patterns or just memorized the training examples.

---

## 13. What Exactly Is Being Tested?

The test asks:

> Given only the first few states of a newly simulated path, can the trained model correctly predict whether the complete path reaches the error state?

So the test does not check whether the Markov chain itself works.

It checks whether the ML classifier works as an early-warning predictor.

---

## 14. Evaluation Metrics

The output contains standard machine learning classification metrics:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

These are normal ML metrics.

The stochastic part is the data-generation process, because the paths come from a probabilistic Markov chain.

So the distinction is:

```text
Data source: stochastic Markov chain
ML evaluation: accuracy, precision, recall, F1-score
Formal stochastic check: P(E), P(E | C)
```

---

## 15. Confusion Matrix

The confusion matrix compares the model prediction with the true label.

There are four cases:

| Case | Meaning |
|---|---|
| True Positive (TP) | The path reaches error `E`, and the model predicts error |
| True Negative (TN) | The path is safe, and the model predicts safe |
| False Positive (FP) | The path is safe, but the model predicts error |
| False Negative (FN) | The path reaches error `E`, but the model predicts safe |

In this project, false negatives are especially important because they mean:

> The monitor missed an error-reaching path.

---

## 16. Example Result

One run produced:

```text
Accuracy : 0.897
Precision: 0.775
Recall   : 1.000
F1-score : 0.873

Confusion matrix:
[[543 103]
 [  0 354]]
```

This confusion matrix means:

| Actual / Predicted | Predicted Safe | Predicted Error |
|---|---:|---:|
| Actually Safe | 543 | 103 |
| Actually Error | 0 | 354 |

So:

```text
TN = 543
FP = 103
FN = 0
TP = 354
```

Interpretation:

- `543` safe paths were correctly predicted as safe.
- `103` safe paths were wrongly predicted as error. These are false alarms.
- `0` error paths were missed.
- `354` error paths were correctly predicted as error.

The important point is:

```text
FN = 0
```

This means the model did not miss any error-reaching path in this test run.

---

## 17. Accuracy

Accuracy measures the fraction of all predictions that were correct.

Formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Using the example result:

```text
Accuracy = (354 + 543) / (354 + 543 + 103 + 0)
         = 897 / 1000
         = 0.897
```

Interpretation:

> The model classified 89.7% of the test paths correctly.

---

## 18. Precision

Precision answers:

> When the model predicts error, how often is it correct?

Formula:

```text
Precision = TP / (TP + FP)
```

Using the example result:

```text
Precision = 354 / (354 + 103)
          = 354 / 457
          ≈ 0.775
```

Interpretation:

> When the model raised an error warning, it was correct about 77.5% of the time.

Lower precision means more false alarms.

---

## 19. Recall

Recall answers:

> Out of all paths that really reached the error state, how many did the model detect?

Formula:

```text
Recall = TP / (TP + FN)
```

Using the example result:

```text
Recall = 354 / (354 + 0)
       = 1.000
```

Interpretation:

> The model detected all actual error-reaching paths in the test set.

For a warning system, high recall is important because missing an error can be more serious than raising a false alarm.

---

## 20. F1-Score

F1-score combines precision and recall into one value.

Formula:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Using the example result:

```text
F1 = 2 * 0.775 * 1.000 / (0.775 + 1.000)
   ≈ 0.873
```

Interpretation:

> The model has a strong balance between precision and recall.

---

## 21. ML Metrics vs Stochastic/Formal Metrics

The metrics:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

are standard ML evaluation metrics.

They evaluate:

> How well does the trained classifier predict the correct label?

The stochastic/formal metrics are:

```text
P(E)
P(E | C)
```

They evaluate:

> Does visiting candidate state C increase the probability of reaching the error state E?

So the current prototype has two evaluation parts:

| Evaluation | Purpose |
|---|---|
| ML metrics | Evaluate the trained predictor |
| Probability-raising check | Evaluate whether `C` is a probability-raising warning state |

---

## 22. Probability-Raising Check

The output also contains:

```text
P(E)     = 0.359
P(E | C) = 0.803
Number of paths with C: 4475
```

This means:

```text
P(E) = probability that a random path reaches the error state
P(E | C) = probability that a path reaches the error state given that it visited C
```

The result says:

```text
General probability of reaching E       ≈ 35.9%
Probability of reaching E after C       ≈ 80.3%
```

Since:

```text
P(E | C) > P(E)
```

state `C` strongly raises the probability of the effect `E`.

Therefore, in this toy Markov chain, `C` behaves like a probability-raising candidate cause or warning state.

---

## 23. Why the Number of Paths with C Is Around 4500

The code estimates probabilities using simulation.

In the toy Markov chain:

```text
P(C) = P(0 -> 1 -> C) + P(0 -> 2 -> C)
```

Using the transition probabilities:

```text
P(C) = (0.5 * 0.7) + (0.5 * 0.2)
     = 0.35 + 0.10
     = 0.45
```

So among 10000 simulated paths, we expect around:

```text
10000 * 0.45 = 4500
```

paths to visit `C`.

The observed result:

```text
Number of paths with C: 4475
```

is very close to the theoretical expectation.

---

## 24. Why the Same Output Appears Every Run

The code uses:

```python
np.random.seed(42)
```

This fixes the random number generator.

Because of this, every run produces the same random sequence and therefore the same output.

This is useful for reproducibility.

If this line is removed or commented out:

```python
# np.random.seed(42)
```

then the output will change slightly from run to run.

However, the values should remain close because many paths are simulated.

---

## 25. Theoretical Probability Check

The theoretical probability of visiting `C` is:

```text
P(C) = 0.45
```

The probability of reaching `E` after visiting `C` is:

```text
P(E | C) = 0.8
```

Therefore:

```text
P(E) = P(C) * P(E | C)
     = 0.45 * 0.8
     = 0.36
```

The simulation result:

```text
P(E) ≈ 0.359
```

matches the theoretical value very closely.

This confirms that the simulation behaves as expected.

---

## 26. How to Explain This Demo in a Meeting

A concise explanation:

> I implemented a toy Markov-chain example. The chain has an initial state, a candidate warning state `C`, an error/effect state `E`, and a safe absorbing state. I simulate many paths from the chain. For each path, I take only the first few states as a prefix and label the full path by whether it eventually reaches `E`. Then I train a Random Forest classifier to predict from the prefix whether the path will reach `E`.

Then:

> The classifier is evaluated on new simulated paths. In the current result, it achieves around 89.7% accuracy and 1.0 recall, meaning it did not miss any actual error-reaching paths. It has some false alarms, shown by the precision of 0.775.

Then:

> Separately, I estimate the probability-raising condition. The general probability of reaching `E` is about 0.359, while the probability of reaching `E` after visiting `C` is about 0.803. Since `P(E | C) > P(E)`, the candidate state `C` clearly raises the probability of the effect.

---

## 27. Possible Next Steps

Possible extensions:

1. Test different prefix lengths.
2. Compare multiple ML algorithms:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Neural Network
   - LSTM/Transformer for longer state sequences
3. Move from one manually selected state `C` to automatically discovered candidate causes.
4. Learn sets of states instead of a single state.
5. Learn temporal patterns or sequences of states.
6. Add repeated runs with different random seeds.
7. Report mean and standard deviation of metrics.
8. Add confidence intervals for probability estimates.
9. Compare simulation-based estimates with exact model-checking probabilities.
10. Later extend from Markov chains to MDPs, where policies and actions matter.

---

## 28. Short Summary

This demo shows:

```text
Markov chain -> simulated paths -> prefixes -> ML classifier -> prediction metrics
```

and also:

```text
candidate C -> estimate P(E) and P(E | C) -> probability-raising check
```

So the prototype connects the formal idea of probability raising with a machine learning predictor trained from simulated traces.
