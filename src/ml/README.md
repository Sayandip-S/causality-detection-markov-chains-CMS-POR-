# ML Pipeline README

## 1. Project Context

The overall research goal is to combine probabilistic model checking with machine learning. In particular, the project aims to identify states, sets of states, or monitor-like conditions in Markov chains that can predict whether a future error, failure, or other target event will occur.

Rather than manually defining transition matrices or transition dictionaries in Python, the current implementation uses PRISM models loaded through StormPy. This makes it possible to work with established probabilistic-model benchmarks and to compare machine-learning results with exact model-checking results.

The intended workflow is:

```text
PRISM model
    → StormPy model construction
    → exact probability computation
    → trace generation
    → prefix dataset creation
    → machine learning
    → candidate state extraction
    → exact verification with Storm
```

## 2. What Was Done Before the ML Step

The first experiment used a small toy discrete-time Markov chain (DTMC) written in PRISM. It contained an initial state, normal states, a candidate or warning state, an error state, and a safe state.

StormPy successfully loaded this model, preserved its state labels and PRISM state valuations, and computed the exact probability of eventually reaching the error state:

```text
P(eventually error) = 0.4
```

This validated the basic StormPy integration before moving to a larger benchmark.

The next model was the official Bounded Retransmission Protocol (BRP) benchmark. BRP models the transmission of chunks over unreliable communication channels with a bounded number of retransmissions. The current configuration uses:

```text
N = 32
MAX = 1
```

- `N` is the number of chunks to transmit.
- `MAX` is the maximum number of retransmissions allowed.

For this configuration, Storm constructed a DTMC with:

- 804 reachable states
- 1,027 transitions

## 3. BRP Target Definitions

Different target labels were tested to investigate both final failures and intermediate warning conditions.

### Final error target

The first BRP target was the sender error state:

```prism
label "target" = s=5;
```

The successful terminal condition was:

```prism
label "success" = s=0 & srep=3;
```

This means that the sender returned to its idle state and reported successful completion.

For the final error target `s=5`, Storm computed:

```text
Exact target probability ≈ 0.0280295783
```

The results from 10,000 generated traces were:

- Target traces: 282
- Success traces: 9,718
- Truncated traces: 0
- Empirical target probability: 0.0282

The empirical result is consistent with the exact Storm probability. This dataset is imbalanced, but it is usable for an initial ML experiment because it contains 282 positive target examples.

### Retransmission warning target

An intermediate target was then tested:

```prism
label "target" = s=3;
```

Here, `s=3` is the sender retransmission state. It was selected because:

- `s=5` is the final sender error and is relatively rare.
- `s=3` is an intermediate warning or recovery state.
- It indicates that a message or acknowledgement timeout occurred and the protocol had to retransmit.
- It is more frequent and therefore useful for testing the ML pipeline quickly.

For the retransmission target `s=3`, Storm computed:

```text
Exact target probability ≈ 0.620195
```

Using 1,000 generated traces, the empirical results were:

- Target traces: 624
- Success traces: 376
- Truncated traces: 0
- Empirical target probability: 0.624
- Absolute estimation error: 0.003805

This agrees well with the exact probability and produces a substantially more balanced dataset.

## 4. Trace Generation

Traces are generated directly from the DTMC constructed by Storm, not from a manually written Python transition dictionary.

For each trace, the generator:

1. Starts at the model's initial state.
2. Reads the outgoing transitions and their probabilities from the Storm model.
3. Randomly samples the next state according to those probabilities.
4. Continues until the target label, the success label, or the maximum number of steps is reached.

The output categories mean:

- **Target trace:** a trace that reaches the selected target condition.
- **Success trace:** a trace that reaches the successful terminal condition without first reaching the target.
- **Truncated trace:** a trace that reaches neither target nor success before `max_steps`.
- **Empirical target probability:** the number of target traces divided by the total number of generated traces.
- **Exact target probability:** the probability computed by exact Storm model checking.
- **Absolute estimation error:** the absolute difference between the empirical and exact target probabilities.

All experiments so far produced zero truncated traces, indicating that `max_steps=500` was sufficient for the current BRP configuration.

## 5. Prefix-Based ML Idea

The intended ML task is not to classify a complete trace after its outcome is already known. Instead, the task is:

> Given only an observed prefix of a trace, predict whether the complete trace will eventually reach the target.

- A **full trace** is the complete generated execution.
- A **prefix** is only the first part of that trace and is used as ML input.
- The **label** is the final outcome of the full trace.

For example:

```text
Full trace:
0 → 10 → 24 → 37 → target

Prefix with prefix_fraction = 0.5:
0 → 10

Label:
target = 1
```

Varying the observed amount can help investigate how early the target becomes predictable. The current fractional-prefix implementation is an exploratory baseline: because it calculates the prefix length from the completed trace length, it can leak outcome-related length information. A stronger follow-up experiment should use an outcome-independent observation rule, such as a fixed number of initial transitions or a fixed protocol checkpoint.

## 6. First Dataset: Summary Prefix Features

The first ML dataset used simple summary features calculated from each prefix:

- `first_state`
- `last_state`
- `prefix_length`
- `unique_states`
- `total_state_visits`
- `min_state_id`
- `max_state_id`
- `target`

This provided a quick baseline, but it is not fully aligned with the main research goal because it does not tell the model exactly which states occurred in the prefix.

For the final error target `s=5`, the 10,000-trace dataset had the following distribution:

- `target = 0`: 9,718
- `target = 1`: 282

Three baseline classifiers were trained:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest performed best on the summary-feature dataset, with approximately:

- Accuracy: 0.9990
- Precision: 1.0000
- Recall: 0.9643
- F1-score: 0.9818

Although these results were strong, features such as `last_state` and `prefix_length` may be highly correlated with the final outcome. In particular, the relative-prefix construction can expose information about the completed trace length. The results must therefore be treated as preliminary rather than as final evidence of generalizable prediction.

## 7. Improved Dataset: Visited-State Prefix Features

A more meaningful representation was then created in which each feature indicates whether a particular Storm state occurred in the prefix. Example columns include:

- `visited_state_0`
- `visited_state_1`
- `visited_state_2`
- `...`
- `prefix_length`
- `last_state`
- `target`

This representation is more closely aligned with the research goal because the ML model can learn which states or sets of states are predictive of the target.

For example, if a prefix visits:

```text
0 → 12 → 25
```

then:

```text
visited_state_0  = 1
visited_state_12 = 1
visited_state_25 = 1
```

Most other `visited_state_*` features are `0`. This representation also enables later extraction of candidate predictive states.

For the final error target `s=5`, the visited-state dataset using 50% prefixes had:

- `target = 0`: 9,718
- `target = 1`: 282

The stratified training and test distributions were:

| Split | `target = 0` | `target = 1` |
|---|---:|---:|
| Training | 7,774 | 226 |
| Testing | 1,944 | 56 |

### Logistic Regression

- Accuracy: 0.9865
- Precision: 0.6933
- Recall: 0.9286
- F1-score: 0.7939
- Confusion matrix:

```text
[[1921, 23],
 [   4, 52]]
```

Logistic Regression detected 52 of the 56 target traces but produced 23 false positives.

### Decision Tree

- Accuracy: 0.9915
- Precision: 0.7910
- Recall: 0.9464
- F1-score: 0.8618
- Confusion matrix:

```text
[[1930, 14],
 [   3, 53]]
```

The Decision Tree detected 53 of the 56 target traces and produced 14 false positives.

### Random Forest

- Accuracy: 0.9985
- Precision: 1.0000
- Recall: 0.9464
- F1-score: 0.9725
- Confusion matrix:

```text
[[1944,  0],
 [   3, 53]]
```

Random Forest detected 53 of the 56 target traces and produced no false positives. It was the best model in this run.

Logistic Regression produced a convergence warning on the visited-state dataset, likely because of the large number of binary features and the class imbalance. Potential improvements include increasing the iteration limit, scaling suitable features, or testing a different solver.

As with the summary dataset, the visited-state results remain preliminary because a relative 50% prefix can indirectly reveal the completed trace length. The visited-state representation improves feature meaning, but it does not by itself remove this leakage.

## 8. Why These ML Models Were Chosen

### Logistic Regression

Logistic Regression is a simple linear classifier that estimates the probability that a trace prefix belongs to the target class.

It was selected because it is:

- Fast
- A simple baseline
- Relatively interpretable through its coefficients
- Useful for determining whether a linear relationship exists

### Decision Tree

A Decision Tree is a rule-based model that repeatedly divides the data using conditions on input features.

It was selected because it is:

- Interpretable
- Able to produce rule-like explanations
- Suitable for investigating predictive states or conditions

### Random Forest

A Random Forest combines the predictions of many decision trees.

It was selected because it:

- Is generally more robust than a single tree
- Handles nonlinear feature interactions
- Can provide feature-importance scores
- Can help identify candidate predictive states

These models were chosen before considering more complex techniques because the first objective was to validate the complete pipeline from model construction to ML evaluation.

## 9. Pros and Cons of the Current Approach

### Pros

- The pipeline works from PRISM model construction through StormPy, trace generation, dataset creation, and ML training.
- It no longer depends on manually written Python transition dictionaries.
- Exact Storm probabilities can be compared with empirical simulation probabilities.
- The visited-state representation supports the search for predictive states.
- Random Forest gives strong initial performance for the final error target.
- The retransmission target `s=3` provides a more balanced intermediate target for further experiments.

### Cons and limitations

- The final error target `s=5` remains imbalanced.
- The visited-state representation ignores state order, so prefixes `A → B` and `B → A` can look identical.
- A relative 50% prefix can reveal completed-trace length and make prediction artificially easy.
- Shorter fractions such as 20% and 10% should be explored, but fixed-transition or fixed-checkpoint prefixes are needed to eliminate the underlying look-ahead issue.
- The models predict outcomes but do not yet automatically verify candidate states with Storm.
- Sampled data can contain repeated traces or repeated prefixes, which must be considered when constructing train/test splits.
- Feature-importance extraction and exact probability-raising verification remain future work.
- Storm state IDs are model-specific; important states must be mapped back to their PRISM valuations for interpretation.

## 10. Current Conclusions

- The StormPy integration works.
- The BRP benchmark can be loaded, model checked, and sampled.
- The final error target `s=5` produces a usable but imbalanced dataset.
- A visited-state prefix representation is more aligned with the project goal than simple summary statistics.
- Random Forest has performed best so far for the final error target.
- The retransmission target `s=3` is promising because it is more frequent and represents an intermediate warning condition.
- Current ML scores are exploratory and must be re-evaluated with outcome-independent observation prefixes before making strong predictive claims.
- The next stage is to compare observation points, extract candidate states, and verify those candidates exactly with Storm.

## 11. Next Steps

1. Run ML training for the retransmission target `s=3`.
2. Compare the final error target `s=5` with the retransmission target `s=3`.
3. Compare prefix fractions of 50%, 20%, and 10% as exploratory experiments.
4. Add fixed-transition or fixed-protocol-checkpoint observations to remove completed-trace-length leakage.
5. Extract feature importances from Random Forest.
6. Identify candidate predictive states.
7. Map important Storm state IDs back to PRISM valuations.
8. Use Storm to verify probability raising:

   ```text
   P(target | candidate visited) > P(target)
   ```

9. Try other BRP properties where useful.
10. Later, consider order-aware features such as transition counts or sequence models.


