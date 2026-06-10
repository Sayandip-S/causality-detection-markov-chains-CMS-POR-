import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility
np.random.seed(42)

# Bigger Markov chain
#
# States:
# 0  = initial state
# 1,2,3,4,5,6,7,8 = intermediate system states
# 9  = ERROR absorbing state
# 10 = SAFE absorbing state
#
# Important:
# We are NOT manually marking the cause state C.
# The code will search for candidate cause states automatically.

transitions = {
    0: [(1, 0.4), (2, 0.4), (3, 0.2)],

    # Branch 1
    1: [(4, 0.6), (5, 0.4)],

    # Branch 2
    2: [(5, 0.5), (6, 0.5)],

    # Branch 3
    3: [(6, 0.3), (10, 0.7)],

    # Intermediate states
    4: [(7, 0.7), (10, 0.3)],
    5: [(7, 0.4), (8, 0.3), (10, 0.3)],
    6: [(8, 0.6), (10, 0.4)],

    # Suspicious/risky states, but we do NOT tell the algorithm this
    7: [(9, 0.75), (10, 0.25)],
    8: [(9, 0.55), (10, 0.45)],

    # Absorbing states
    9: [(9, 1.0)],       # ERROR
    10: [(10, 1.0)]      # SAFE
}

ERROR_STATE = 9
SAFE_STATE = 10
INITIAL_STATE = 0


def sample_next_state(current_state):
    """
    Sample the next state according to transition probabilities.
    """
    next_states, probabilities = zip(*transitions[current_state])
    return np.random.choice(next_states, p=probabilities)


def generate_path(max_steps=30):
    """
    Generate one random path through the Markov chain.
    The path starts at state 0 and stops when it reaches ERROR or SAFE.
    """
    path = [INITIAL_STATE]

    for _ in range(max_steps):
        current = path[-1]

        if current in [ERROR_STATE, SAFE_STATE]:
            break

        next_state = sample_next_state(current)
        path.append(next_state)

    return path


def reaches_error(path):
    """
    Label:
    1 = path reaches ERROR
    0 = path reaches SAFE / does not reach ERROR
    """
    return int(ERROR_STATE in path)


def pad_prefix(prefix, prefix_length, pad_value=-1):
    """
    Make path prefixes fixed length for machine learning.
    """
    if len(prefix) >= prefix_length:
        return prefix[:prefix_length]

    return prefix + [pad_value] * (prefix_length - len(prefix))


def create_dataset(num_paths=5000, prefix_length=4):
    """
    Create supervised ML dataset:
    X = early prefix of path
    y = whether full path reaches error
    """
    X = []
    y = []

    for _ in range(num_paths):
        path = generate_path()
        label = reaches_error(path)

        prefix = pad_prefix(path, prefix_length)

        X.append(prefix)
        y.append(label)

    return np.array(X), np.array(y)


def estimate_probability_raising(paths, error_state, exclude_states=None):
    """
    Automatically search for candidate cause states.

    For every state s:
        compute P(E | s)
        compare with P(E)
        raise = P(E | s) - P(E)

    A state is a stronger candidate cause if:
        P(E | s) is high
        and P(E | s) - P(E) is positive and large.
    """
    if exclude_states is None:
        exclude_states = set()

    # Overall probability of reaching error
    p_error = np.mean([error_state in path for path in paths])

    # Collect all states that appear in generated paths
    all_states = sorted(set(state for path in paths for state in path))

    results = []

    for state in all_states:
        if state in exclude_states:
            continue

        paths_with_state = [path for path in paths if state in path]

        if len(paths_with_state) == 0:
            continue

        p_error_given_state = np.mean([
            error_state in path for path in paths_with_state
        ])

        probability_raise = p_error_given_state - p_error

        results.append({
            "state": state,
            "p_error_given_state": p_error_given_state,
            "p_error": p_error,
            "raise": probability_raise,
            "count": len(paths_with_state)
        })

    # Sort by strongest probability raising
    results = sorted(results, key=lambda x: x["raise"], reverse=True)

    return results


# ------------------------------------------------------------
# 1. Generate train/test data
# ------------------------------------------------------------

prefix_length = 4

X_train, y_train = create_dataset(num_paths=5000, prefix_length=prefix_length)
X_test, y_test = create_dataset(num_paths=1000, prefix_length=prefix_length)


# ------------------------------------------------------------
# 2. Train Random Forest for early error prediction
# ------------------------------------------------------------

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=6
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# ------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------

print("Evaluation results")
print("------------------")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
print()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Detailed Confusion Matrix")
print("-------------------------")
print(f"True Negatives  (TN): {tn}  -> Actually safe, predicted safe")
print(f"False Positives (FP): {fp}  -> Actually safe, predicted error / false alarm")
print(f"False Negatives (FN): {fn}  -> Actually error, predicted safe / missed error")
print(f"True Positives  (TP): {tp}  -> Actually error, predicted error")
print()

print("Confusion Matrix Table")
print("----------------------")
print("                  Predicted Safe   Predicted Error")
print(f"Actually Safe         {tn:<15} {fp}")
print(f"Actually Error        {fn:<15} {tp}")
print()


# ------------------------------------------------------------
# 4. Automatic cause-state search
# ------------------------------------------------------------

num_samples = 10000
paths = [generate_path() for _ in range(num_samples)]

cause_results = estimate_probability_raising(
    paths=paths,
    error_state=ERROR_STATE,
    exclude_states={INITIAL_STATE, ERROR_STATE, SAFE_STATE}
)

print("Automatic Candidate Cause Search")
print("--------------------------------")
print("We do NOT manually mark C.")
print("For each state s, we compute P(E | s) and compare it with P(E).")
print()

print("State   Count   P(E | state)   P(E)     Raise")
print("------------------------------------------------")

for r in cause_results:
    print(
        f"{r['state']:<7}"
        f"{r['count']:<8}"
        f"{r['p_error_given_state']:<15.3f}"
        f"{r['p_error']:<9.3f}"
        f"{r['raise']:.3f}"
    )

print()
best = cause_results[0]

print("Strongest candidate cause")
print("-------------------------")
print(f"State {best['state']} is the strongest candidate cause.")
print(
    f"Reason: P(E | state {best['state']}) = {best['p_error_given_state']:.3f}, "
    f"while P(E) = {best['p_error']:.3f}."
)
print(
    f"So visiting state {best['state']} raises the probability of error by "
    f"{best['raise']:.3f}."
)