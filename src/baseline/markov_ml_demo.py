import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility
#np.random.seed(42)

# States:
# 0 = initial
# 1, 2 = normal states
# 3 = C, possible cause / warning state
# 4 = E, error/effect state
# 5 = SAFE, safe absorbing state

transitions = {
    0: [(1, 0.5), (2, 0.5)],
    1: [(3, 0.7), (5, 0.3)],   # from state 1, high chance to visit C
    2: [(3, 0.2), (5, 0.8)],   # from state 2, low chance to visit C
    3: [(4, 0.8), (5, 0.2)],   # after C, high chance to reach error E
    4: [(4, 1.0)],             # error is absorbing
    5: [(5, 1.0)]              # safe is absorbing
}

ERROR_STATE = 4
CAUSE_STATE = 3
SAFE_STATE = 5


def sample_next_state(current_state):
    next_states, probabilities = zip(*transitions[current_state])
    return np.random.choice(next_states, p=probabilities)


def generate_path(max_steps=20):
    path = [0]

    for _ in range(max_steps):
        current = path[-1]

        # Stop once we hit absorbing state
        if current in [ERROR_STATE, SAFE_STATE]:
            break

        next_state = sample_next_state(current)
        path.append(next_state)

    return path


def reaches_error(path):
    return int(ERROR_STATE in path)


def pad_prefix(prefix, prefix_length, pad_value=-1):
    """
    Make all prefixes the same length.
    Example:
    [0, 1, 3] with prefix_length 5 becomes [0, 1, 3, -1, -1]
    """
    if len(prefix) >= prefix_length:
        return prefix[:prefix_length]
    return prefix + [pad_value] * (prefix_length - len(prefix))


def create_dataset(num_paths=5000, prefix_length=3):
    X = []
    y = []

    for _ in range(num_paths):
        path = generate_path()
        label = reaches_error(path)

        # Use only the first k states as early information
        prefix = pad_prefix(path, prefix_length)

        X.append(prefix)
        y.append(label)

    return np.array(X), np.array(y)


# Create train/test data
prefix_length = 3

X_train, y_train = create_dataset(num_paths=5000, prefix_length=prefix_length)
X_test, y_test = create_dataset(num_paths=1000, prefix_length=prefix_length)

# Train ML model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Evaluation results")
print("------------------")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
print()

print()
# Detailed confusion matrix explanation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print()
print("Detailed Confusion Matrix")
print("-------------------------")
print(f"True Negatives  (TN): {tn}  → Actually safe, predicted safe")
print(f"False Positives (FP): {fp}  → Actually safe, predicted error / false alarm")
print(f"False Negatives (FN): {fn}  → Actually error, predicted safe / missed error")
print(f"True Positives  (TP): {tp}  → Actually error, predicted error")
print()

print("Confusion Matrix Table")
print("----------------------")
print("                  Predicted Safe   Predicted Error")
print(f"Actually Safe         {tn:<15} {fp}")
print(f"Actually Error        {fn:<15} {tp}")


# Manual metric calculations
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_manual = (
    2 * precision_manual * recall_manual / (precision_manual + recall_manual)
    if (precision_manual + recall_manual) > 0
    else 0
)

print()
print("Manual Metric Explanation")
print("-------------------------")
print(f"Accuracy  = (TP + TN) / Total = ({tp} + {tn}) / {tp + tn + fp + fn} = {accuracy_manual:.3f}")
print(f"Precision = TP / (TP + FP)    = {tp} / ({tp} + {fp}) = {precision_manual:.3f}")
print(f"Recall    = TP / (TP + FN)    = {tp} / ({tp} + {fn}) = {recall_manual:.3f}")
print(f"F1-score  = 2 * Precision * Recall / (Precision + Recall) = {f1_manual:.3f}")
# Estimate probability raising manually
num_samples = 10000
paths = [generate_path() for _ in range(num_samples)]

p_error = np.mean([ERROR_STATE in path for path in paths])

paths_with_cause = [path for path in paths if CAUSE_STATE in path]
p_error_given_cause = np.mean([ERROR_STATE in path for path in paths_with_cause])

print()
print("Probability raising check")
print("-------------------------")
print(f"P(E)     = {p_error:.3f}")
print(f"P(E | C) = {p_error_given_cause:.3f}")
print(f"Number of paths with C: {len(paths_with_cause)}")