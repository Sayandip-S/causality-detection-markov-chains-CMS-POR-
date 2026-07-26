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

The visited-state dataset script supports two mutually exclusive observation modes. A fractional prefix uses a specified fraction of each completed trace, preserving the earlier exploratory behavior but making the observation length depend on the completed trace length. A fixed transition window observes the same number of transitions from the start of every retained trace: `--prefix-length k` selects the initial state plus the next `k` states, for `k + 1` observed state IDs. Traces with `k` or fewer transitions are excluded so that the observed window cannot contain or coincide with the terminal state.

Fixed transition windows are preferred for the primary prediction experiment because their observation length does not depend on the completed trace length. For example:

```bash
python -m src.ml.create_brp_visited_state_dataset \
    --input data/raw/brp_stress_tuned_traces_10000.csv \
    --output data/processed/brp_stress_tuned_visited_state_dataset_k10.csv \
    --prefix-length 10
```

Run the Logistic Regression, Decision Tree, and Random Forest baselines on the fixed `k5`, `k10`, `k20`, and `k50` datasets with:

```bash
python -m scripts.run_brp_fixed_window_baselines
```

The runner writes per-window JSON files (`k5.json`, `k10.json`, `k20.json`, and `k50.json`) and `combined_metrics.csv` under:

```text
results/metrics/brp_fixed_windows/
```

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

## Progress Update — 2026-07-09

### 1. Motivation After the Last Meeting

After the last meeting, the focus was to move beyond the initial BRP final-error experiment and investigate whether changing BRP model parameters and target definitions could produce better datasets for ML-based predictor discovery. The main issue was class imbalance.

The earlier experiment used final sender error as the target:

```prism
label "target" = s=5;
```

Here, `s=5` is the sender error state. With the previous BRP configuration, this event was rare:

- Exact Storm probability: approximately `0.0280295783`
- Generated traces: `10,000`
- Target traces: `282`
- Success traces: `9,718`
- Truncated traces: `0`
- Positive rate: `2.82%`

This dataset was usable, but it was strongly imbalanced toward success. The next step was therefore to create additional experimental settings that provide a more informative class distribution while keeping the interpretation of each target explicit.

### 2. Retransmission Target Experiment

I tested an intermediate target:

```prism
label "target" = s=3;
```

Here, `s=3` is the sender retransmission state. This target was chosen because:

- `s=5` represents final sender error and is relatively rare.
- `s=3` represents retransmission.
- Retransmission means the protocol has experienced message or acknowledgement loss and must retry.
- It is an intermediate warning or recovery state that may occur before final failure.
- Its higher frequency can provide a more balanced learning target.

This experiment used:

- Model: `models/prism/brp/brp_retransmit_target.pm`
- Property: `models/properties/brp/brp_target.pctl`

The property was:

```prism
P=? [ F "target" ]
```

The result for `1,000` traces was:

- Exact Storm target probability: `0.620195`
- Target traces: `624`
- Success traces: `376`
- Truncated traces: `0`
- Empirical target probability: `0.624000`
- Absolute estimation error: `0.003805`

This result showed that the retransmission target is much more balanced than the final-error target.

However, this is no longer a final-failure prediction task. It asks whether the sender will enter retransmission, rather than whether the protocol will end in sender error. It is therefore useful as a warning-event prediction experiment, but it must be clearly distinguished from final-error prediction.

### 3. BRP Stress / High-Loss Error Target Experiment

I then investigated a stress-test/high-loss BRP variant in which the target remains final sender error:

```prism
label "target" = s=5;
```

The goal was to preserve the final-error prediction task while changing channel reliability so that final error occurs more often. In BRP:

- `channelK` is the data/frame channel.
- `channelL` is the acknowledgement channel.
- Increasing loss probabilities in these channels makes communication less reliable and final sender error more likely.

The first high-loss setting was:

```prism
[aF] (k=0) -> 0.70 : (k'=1) + 0.30 : (k'=2);
[aA] (l=0) -> 0.80 : (l'=1) + 0.20 : (l'=2);
```

This produced too many final-error traces:

- States: `1,221`
- Transitions: `1,603`
- Exact Storm target probability: `0.9420998011`
- Generated traces: `10,000`
- Target traces: `9,435`
- Success traces: `565`
- Truncated traces: `0`
- Empirical target probability: `0.9435000000`
- Positive rate: `94.35%`

This confirmed that changing channel reliability had the intended effect, but the first setting overshot: the resulting dataset was strongly imbalanced toward the target class.

I therefore tuned the stress setting to a more moderate version:

```prism
[aF] (k=0) -> 0.85 : (k'=1) + 0.15 : (k'=2);
[aA] (l=0) -> 0.90 : (l'=1) + 0.10 : (l'=2);
```

This corresponds to:

- Data/frame loss probability: `15%`
- Acknowledgement loss probability: `10%`

The tuned experimental model was saved as:

```text
models/prism/brp/brp_stress_error_target.pm
```

For `10,000` generated traces, the tuned stress model produced:

- Number of traces: `10,000`
- Random seed: `42`
- Maximum steps per trace: `500`
- Target label: `target`
- Target traces: `3,429`
- Success traces: `6,571`
- Truncated traces: `0`
- Average transitions per trace: `180.0112`
- Exact Storm target probability: `0.341645`
- Empirical target probability: `0.342900`
- Absolute estimation error: `0.001255`

The class balance was:

- Positive target traces: `3,429` (`34.29%`)
- Negative success traces: `6,571` (`65.71%`)
- Truncated traces: `0` (`0.00%`)

This is a useful balance for baseline ML training because the target is neither too rare nor too frequent. This model remains an experimental high-loss BRP variant and should not be presented as the original benchmark configuration.

### 4. Generic Trace Generator Output Improvement

I noticed that the generic trace generator:

```text
src/storm/generate_traces.py
```

did not print the same helpful class-balance summary as the BRP-specific generator:

```text
src/storm/generate_brp_traces.py
```

The generic script is needed for experiments such as `brp_stress_error_target.pm` because it supports custom model, property, and label arguments. I improved its reporting so that it prints:

- Positive target traces
- Negative terminal traces, including negative success traces for BRP
- Truncated traces
- Class percentages
- Useful imbalance notices

For the tuned stress dataset, the improved output was:

```text
Class balance
-------------
Positive target traces: 3429 (34.29%)
Negative success traces: 6571 (65.71%)
Truncated traces: 0 (0.00%)

Notice: the dataset has a reasonably balanced target/success split for baseline ML training.
```

This makes the suitability of generated traces easier to evaluate before creating an ML dataset.

### 5. Dataset Inspection and Leakage Check

Before training on the tuned stress dataset, I inspected the raw traces because of a possible data-leakage issue. If a trace were very short, a 50% prefix might accidentally include its terminal target or success state. In that case, the classifier would see the answer rather than predict a future outcome.

The raw trace dataset was:

```text
data/raw/brp_stress_tuned_traces_10000.csv
```

It contained `10,000` rows and `7` columns:

- `trace_id`
- `state_ids`
- `valuations`
- `terminal_label`
- `reached_target`
- `reached_monitor`
- `number_of_transitions`

Terminal-label counts were:

- `success`: `6,571`
- `target`: `3,429`

Reached-target counts were:

- `0`: `6,571`
- `1`: `3,429`

The transition-count summary was:

| Statistic | Transitions |
| --- | ---: |
| Count | 10,000 |
| Mean | 180.01120 |
| Standard deviation | 63.82639 |
| Minimum | 8 |
| 25% | 156 |
| Median | 212 |
| 75% | 220 |
| Maximum | 254 |

The shortest traces had `8` transitions and were target traces. One example shortest trace was:

```text
0|1|3|5|8|11|16|21|28
```

I then checked 50% prefixes for direct terminal-state leakage:

- Total traces: `10,000`
- Very short traces with no more than 3 states: `0`
- Prefixes containing the terminal state: `0`
- Leakage rate: `0.0000%`

For the 50% prefix setting, there was no direct terminal-state leakage: the model did not directly see the final target or success state.

This does not prove that the task is completely free from easy patterns, nor does it make the task impossible. Some early states or patterns along short target paths may still be genuinely predictive. The check establishes only that the terminal state itself was not included in the observed prefixes.

### 6. Visited-State Dataset Creation

I created a visited-state prefix dataset from the tuned stress raw traces using the following conceptual configuration:

```text
input: data/raw/brp_stress_tuned_traces_10000.csv
output: data/processed/brp_stress_tuned_visited_state_dataset_50.csv
prefix_fraction: 0.5
```

The resulting dataset had:

- Rows: `10,000`
- Visited-state features: `552`
- `target = 0`: `6,571`
- `target = 1`: `3,429`
- Positive rate: `0.3429`

Example feature columns are:

```text
visited_state_0
visited_state_1
visited_state_2
...
```

The exact highest suffix depends on which Storm state IDs appear in the prefixes; there are approximately `552` such feature columns in this dataset.

Each `visited_state_*` feature records whether the corresponding state appeared anywhere in the observed prefix. The columns are sorted by state ID for consistency, but this does not mean that a trace visits states in numerical order. The raw `state_ids` column preserves actual execution order.

For example, these prefixes have the same visited-state representation:

```text
0 -> 10 -> 25 -> 14
0 -> 14 -> 25 -> 10
```

They contain the same set of states even though their execution orders differ. This loss of order is a limitation, but the representation remains useful for the current research goal of identifying candidate predictive states or state sets.

### 7. Ablation Study: Checking Whether `last_state` Caused High Accuracy

Initial training performance was high, so I tested whether it resulted from leakage through the `last_state` feature. The original feature set included:

- `visited_state_*` features
- `prefix_length`
- `last_state`

The concern was that the last observed prefix state might make prediction too easy if it were already close to target or success. I therefore compared three feature configurations:

1. `all_features`: `visited_state_*`, `prefix_length`, and `last_state`
2. `no_last_state`: `visited_state_*` and `prefix_length`
3. `visited_states_only`: only `visited_state_*`

#### All features

| Model | Accuracy | Precision | Recall | F1-score | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.9630 | 0.9888 | 0.9023 | 0.9436 | `[[1307, 7], [67, 619]]` |
| Decision Tree | 0.9650 | 0.9828 | 0.9140 | 0.9471 | `[[1303, 11], [59, 627]]` |
| Random Forest | 0.9610 | 0.9967 | 0.8892 | 0.9399 | `[[1312, 2], [76, 610]]` |

#### Without `last_state`

| Model | Accuracy | Precision | Recall | F1-score | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.9630 | 0.9873 | 0.9038 | 0.9437 | `[[1306, 8], [66, 620]]` |
| Decision Tree | 0.9635 | 0.9797 | 0.9125 | 0.9449 | `[[1301, 13], [60, 626]]` |
| Random Forest | 0.9630 | 0.9984 | 0.8936 | 0.9431 | `[[1313, 1], [73, 613]]` |

#### Visited-state features only

| Model | Accuracy | Precision | Recall | F1-score | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.9630 | 0.9873 | 0.9038 | 0.9437 | `[[1306, 8], [66, 620]]` |
| Decision Tree | 0.9450 | 0.9486 | 0.8878 | 0.9172 | `[[1281, 33], [77, 609]]` |
| Random Forest | 0.9580 | 0.9934 | 0.8834 | 0.9352 | `[[1310, 4], [80, 606]]` |

Removing `last_state` did not significantly reduce performance. Even with only visited-state features, Logistic Regression and Random Forest remained strong. This suggests that the predictive signal is not mainly caused by `last_state` leakage; the set of states visited in the prefix already contains useful information about whether the target will be reached.

This is evidence of predictive structure, not a formal causality result.

### 8. Current Interpretation

The tuned stress-target experiment is useful because:

- The target remains final sender error, `s=5`.
- The target probability is approximately `34%`, which is suitable for baseline ML.
- The empirical simulation probability closely matches the exact Storm probability.
- The 50% prefixes contain no direct terminal-state leakage.
- The ablation study suggests that the models are not relying only on `last_state`.
- The visited-state representation aligns with the project goal of identifying candidate predictive states.

Important limitations remain:

- A 50% prefix may still be highly informative even without containing the terminal state.
- The visited-state representation ignores execution order.
- Models may exploit state-set patterns that occur close to failure.
- The current results demonstrate prediction, not formal causality.
- Candidate states still need to be extracted and verified with Storm.
- The modified channel probabilities make this an experimental/high-loss BRP variant, not the original benchmark setting.

No claim is being made yet that the ML models have identified formal causes.

### 9. What I Plan to Do Next

1. Generate 20% and 10% prefix datasets for the tuned stress model.
2. Compare performance across 50%, 20%, and 10% prefixes.
3. Add a training-script option to exclude `last_state` and `prefix_length`, replacing the temporary ad hoc ablation code.
4. Extract feature importances or coefficients from the trained models.
5. Identify important `visited_state_*` features as candidate predictive states.
6. Map important Storm state IDs back to PRISM valuations.
7. Use Storm to test whether the candidates raise the probability of reaching the target.
8. Compare three experimental settings:
   - Rare final-error target
   - Retransmission warning target
   - Tuned stress final-error target
9. Later, consider transition-count features or order-aware representations because visited-state features ignore order.

---

## BRP Fixed-Window Candidate-State Experiment — 16 July 2026

### 1. Goal

This experiment connects the full analysis pipeline:

```text
PRISM/Storm model checking
    → trace generation
    → fixed-window ML prediction
    → candidate-state ranking
    → mapping Storm IDs to PRISM valuations
    → exact state-based Storm verification
```

Machine learning is used to prioritize preliminary candidate predictor states from sampled trace prefixes. Storm then performs the exact state-based check on those candidates. The ML score is therefore a discovery heuristic, while the Storm result is the model-derived probability of eventually reaching the target from a specified current state.

### 2. Model configuration

The experiment uses the BRP benchmark with the final sender-error target:

```prism
label "target" = s=5;
```

The actual tuned model configuration is:

| Item | Value |
| --- | --- |
| Number of chunks, \(N\) | 32 |
| Maximum retransmissions, \(\mathit{MAX}\) | 2 |
| Frame/data channel delivery probability | 0.85 |
| Frame/data channel loss probability | 0.15 |
| Acknowledgement channel delivery probability | 0.90 |
| Acknowledgement channel loss probability | 0.10 |
| Reachable Storm states | 1,221 |
| Storm transitions | 1,603 |
| Exact initial target probability | 0.341644584515 |

This is a tuned, higher-loss experimental BRP configuration rather than the original benchmark parameterization.

### 3. Raw trace dataset

The raw dataset contains 10,000 traces generated from the tuned model:

| Outcome | Traces | Fraction |
| --- | ---: | ---: |
| Target | 3,429 | 0.3429 |
| Success | 6,571 | 0.6571 |
| Total | 10,000 | 1.0000 |

The empirical target probability is `0.3429`, compared with the exact Storm probability `0.341644584515`.

Trace lengths below are numbers of transitions:

| Outcome | Minimum | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Target | 8 | 106 | 108.5733 | 250 |
| Success | 194 | 216 | 217.2902 | 254 |

Target traces can terminate much earlier than successful protocol executions. This difference is important because an observation rule based on completed trace length can indirectly expose information about the eventual outcome.

### 4. Why fixed observation windows were introduced

The earlier fractional-prefix experiment observed a fraction of each completed trace. Completed trace length is not available at prediction time, and the raw data shows substantially different target and success length distributions. Consequently, a fractional prefix length can reveal outcome-related length information even when the terminal state itself is absent.

Fixed windows instead observe exactly \(k\) transitions from the start of each trace. A trace that terminates at or before \(k\) is excluded, ensuring that the observed window neither contains nor coincides with its terminal target or success state.

| Window | Retained traces | Excluded target | Excluded success | Retained positive rate | Visited-state features |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10,000 | 0 | 0 | 0.3429 | 12 |
| 10 | 9,956 | 44 | 0 | 0.3400 | 39 |
| 20 | 9,723 | 277 | 0 | 0.3242 | 94 |
| 50 | 9,177 | 823 | 0 | 0.2840 | 251 |

Each retained row contains \(k+1\) observed state IDs because the initial state is included alongside the \(k\) transitions.

### 5. Fixed-window baseline results

All models use only ordered `visited_state_*` indicator columns. The split is stratified with `test_size=0.2` and `random_seed=42`.

| Window | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | TP | Features |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | Logistic Regression | 0.6305 | 0.4120 | 0.1808 | 0.2513 | 0.5245 | 562 | 124 | 12 |
| 5 | Decision Tree | 0.6305 | 0.4120 | 0.1808 | 0.2513 | 0.5245 | 562 | 124 | 12 |
| 5 | Random Forest | 0.6305 | 0.4120 | 0.1808 | 0.2513 | 0.5245 | 562 | 124 | 12 |
| 10 | Logistic Regression | 0.5979 | 0.3869 | 0.3131 | 0.3461 | 0.5276 | 465 | 212 | 39 |
| 10 | Decision Tree | 0.5979 | 0.3869 | 0.3131 | 0.3461 | 0.5277 | 465 | 212 | 39 |
| 10 | Random Forest | 0.5979 | 0.3869 | 0.3131 | 0.3461 | 0.5277 | 465 | 212 | 39 |
| 20 | Logistic Regression | 0.6314 | 0.3930 | 0.2504 | 0.3059 | 0.5397 | 473 | 158 | 94 |
| 20 | Decision Tree | 0.6540 | 0.4019 | 0.1363 | 0.2036 | 0.5147 | 545 | 86 | 94 |
| 20 | Random Forest | 0.6211 | 0.3848 | 0.2805 | 0.3245 | 0.5283 | 454 | 177 | 94 |
| 50 | Logistic Regression | 0.5969 | 0.3166 | 0.3628 | 0.3381 | 0.5383 | 332 | 189 | 251 |
| 50 | Decision Tree | 0.6716 | 0.3277 | 0.1497 | 0.2055 | 0.5173 | 443 | 78 | 251 |
| 50 | Random Forest | 0.6247 | 0.3213 | 0.2898 | 0.3047 | 0.5246 | 370 | 151 | 251 |

ROC-AUC is approximately 0.52–0.54, so early prediction from visited-state indicators alone is weak. Accuracy is also influenced by class imbalance and should not be read as strong discrimination. The earlier high fractional-prefix scores should therefore be interpreted cautiously because their observation length depended on completed trace length. This is a useful methodological result rather than a failed experiment: the fixed-window design exposes how much weaker the genuinely early signal is.

### 6. Selected k=20 candidate-state analysis

The k=20 window was selected for preliminary candidate extraction because Logistic Regression achieved the strongest ROC-AUC (`0.5397`), while k=50 was close (`0.5383`). The k=5 and k=10 datasets contained only 3 and 11 distinct visited-state patterns, respectively. In contrast, k=50 excluded 823 target traces, substantially more than the 277 excluded at k=20.

Candidate ranking combines three signals:

1. **Positive Logistic Regression coefficient:** direction and strength toward target/error prediction.
2. **Random Forest feature importance:** predictive usefulness, without indicating direction.
3. **Positive empirical probability difference:** \(P(\text{target}\mid\text{state visited in the first 20 transitions}) - P(\text{target}\mid\text{state not visited in the first 20 transitions})\).

The three non-negative signals are min-max normalized and averaged. The result is multiplied by the support-reliability weight

```text
visited_count / (visited_count + minimum_support)
```

with `minimum_support=50`. The combined score is a ranking heuristic, not an error probability. Empirical statistics use the complete retained k=20 dataset, including rows used during model training and testing, so candidate discovery is exploratory rather than an unbiased performance evaluation.

### 7. Top candidate states

| ML rank | Storm state ID | Combined score | LR coefficient | RF importance | Visits | P(target \| visited) | P(target \| not visited) | Empirical difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89 | 0.6041 | 0.1629 | 0.068688 | 176 | 0.5398 | 0.3202 | 0.2196 |
| 2 | 82 | 0.5675 | 0.1629 | 0.059003 | 176 | 0.5398 | 0.3202 | 0.2196 |
| 3 | 101 | 0.5539 | 0.4980 | 0.053027 | 83 | 0.5181 | 0.3225 | 0.1956 |
| 4 | 69 | 0.3401 | 0.3113 | 0.011687 | 250 | 0.4160 | 0.3218 | 0.0942 |
| 5 | 83 | 0.3226 | 0.3534 | 0.007011 | 217 | 0.4055 | 0.3223 | 0.0832 |
| 6 | 102 | 0.2659 | -0.0571 | 0.022411 | 111 | 0.5045 | 0.3221 | 0.1824 |
| 7 | 99 | 0.2509 | 0.3246 | 0.017147 | 85 | 0.3882 | 0.3236 | 0.0646 |
| 8 | 95 | 0.2509 | -0.0571 | 0.017931 | 111 | 0.5045 | 0.3221 | 0.1824 |
| 9 | 93 | 0.2110 | 0.2334 | 0.008462 | 485 | 0.3464 | 0.3230 | 0.0234 |
| 10 | 44 | 0.2012 | 0.3306 | 0.008346 | 166 | 0.3012 | 0.3246 | -0.0234 |

The ranking can retain a candidate with one negative component because the other components may still provide positive evidence. For example, state 102 has a negative Logistic Regression coefficient but positive Random Forest and empirical signals.

### 8. Storm ID to PRISM valuation mapping

Storm internal state IDs are indices in the constructed sparse model; they are not values of any PRISM variable. Trace generation, candidate mapping, and exact verification all use the same `load_prism_model` construction path, including state valuations and labels.

| Rank | State ID | s | i | nrtr | r | rrep | k | l |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 89 | 3 | 3 | 1 | 4 | 2 | 0 | 0 |
| 2 | 82 | 2 | 3 | 1 | 4 | 2 | 2 | 0 |
| 3 | 101 | 2 | 3 | 1 | 4 | 2 | 0 | 2 |
| 4 | 69 | 3 | 2 | 1 | 4 | 2 | 0 | 0 |
| 5 | 83 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| 6 | 102 | 2 | 3 | 2 | 2 | 2 | 0 | 0 |
| 7 | 99 | 2 | 3 | 1 | 4 | 2 | 2 | 0 |
| 8 | 95 | 2 | 3 | 2 | 4 | 2 | 1 | 0 |
| 9 | 93 | 3 | 3 | 0 | 4 | 2 | 0 | 0 |
| 10 | 44 | 2 | 2 | 1 | 4 | 1 | 2 | 0 |

In the PRISM model, `s` is the sender control state, `i` is the current chunk index, and `nrtr` is the retransmission count. The variables `r` and `rrep` are the receiver control and report states. Variables `k` and `l` are the frame/data and acknowledgement channel states. Their numeric meanings are documented by comments in the model.

No special labels are directly attached to these intermediate candidate states, which is expected: the model labels identify the final `target` and `success` conditions rather than every intermediate protocol state.

### 9. Exact Storm state-based verification

The exact baseline is:

```text
P(target from initial state 0) = 0.341644584515
```

The verification output is sorted by exact probability difference, while retaining the original ML rank:

| ML rank | State ID | Valuation summary | Empirical difference | Exact P(target) from state | Exact difference | Raises probability |
| ---: | ---: | --- | ---: | ---: | ---: | :---: |
| 4 | 69 | `s=3, i=2, nrtr=1, r=4, rrep=2, k=0, l=0` | 0.0942 | 0.483027 | 0.141382 | Yes |
| 1 | 89 | `s=3, i=3, nrtr=1, r=4, rrep=2, k=0, l=0` | 0.2196 | 0.476229 | 0.134585 | Yes |
| 2 | 82 | `s=2, i=3, nrtr=1, r=4, rrep=2, k=2, l=0` | 0.2196 | 0.476229 | 0.134585 | Yes |
| 3 | 101 | `s=2, i=3, nrtr=1, r=4, rrep=2, k=0, l=2` | 0.1956 | 0.476229 | 0.134585 | Yes |
| 5 | 83 | `s=2, i=2, nrtr=2, r=2, rrep=2, k=0, l=0` | 0.0832 | 0.391796 | 0.050152 | Yes |
| 16 | 56 | `s=2, i=2, nrtr=1, r=3, rrep=2, k=0, l=0` | 0.0165 | 0.340099 | -0.001546 | No |
| 20 | 22 | `s=1, i=2, nrtr=0, r=4, rrep=1, k=0, l=0` | -0.0161 | 0.332988 | -0.008656 | No |
| 15 | 100 | `s=2, i=3, nrtr=1, r=4, rrep=2, k=0, l=1` | -0.0372 | 0.315332 | -0.026312 | No |

Seventeen of the top 20 candidates raise the exact future target probability above the initial-state baseline. States 69, 44, 51, 61, and 63 all have an exact target probability of approximately `0.483027`. States 89, 82, 101, and 99 all have an exact probability of approximately `0.476229`.

The ML and exact Storm rankings are not identical. ML learns from sampled prefixes containing correlated state indicators, whereas Storm evaluates future behavior from one current state in the mathematical model. Several distinct sparse-model states can also have probability-equivalent future behavior. Storm computes the model probability from the current state exactly, subject to the numerical precision of the implementation.

### 10. Interpretation and limitation

The current exact state-based verification computes:

```text
P(eventually target | currently in candidate state)
```

It does **not** yet compute:

```text
P(eventually target | candidate was visited earlier)
```

The stricter historical or path-conditioned quantity requires a monitored/product model with a persistent `seen_candidate` condition, or an equivalent conditional path construction. The present results identify preliminary candidate predictor states and probability-raising candidates; they do not establish that these states are formal causes.

### 11. Reproduction commands

Generate the k=20 fixed-window dataset:

```bash
python -m src.ml.create_brp_visited_state_dataset \
    --input data/raw/brp_stress_tuned_traces_10000.csv \
    --output data/processed/brp_stress_tuned_visited_state_dataset_k20.csv \
    --prefix-length 20
```

Run all fixed-window baselines:

```bash
python -m scripts.run_brp_fixed_window_baselines
```

Train and save the k=20 models and schema:

```bash
python -m src.ml.train_brp_baselines \
    --dataset data/processed/brp_stress_tuned_visited_state_dataset_k20.csv \
    --feature-set visited_states_only \
    --test-size 0.2 \
    --random-seed 42 \
    --metrics-output results/metrics/brp_fixed_windows/k20.json \
    --model-output-dir results/models/brp_fixed_windows/k20 \
    --observation-window 20
```

Extract ranked candidates:

```bash
python -m scripts.extract_brp_candidate_states \
    --dataset data/processed/brp_stress_tuned_visited_state_dataset_k20.csv \
    --model-dir results/models/brp_fixed_windows/k20 \
    --output results/candidate_states/brp_k20_candidate_states.csv \
    --top-k 20 \
    --minimum-support 50
```

Map Storm IDs to PRISM valuations:

```bash
python -m scripts.map_brp_candidate_states \
    --model models/prism/brp/brp_stress_error_target.pm \
    --property models/properties/brp/brp_target.pctl \
    --candidates results/candidate_states/brp_k20_candidate_states.csv \
    --output results/candidate_states/brp_k20_candidate_state_valuations.csv \
    --top-k 20
```

Verify candidates with Storm:

```bash
python -m scripts.verify_brp_candidate_states \
    --model models/prism/brp/brp_stress_error_target.pm \
    --property models/properties/brp/brp_target.pctl \
    --candidates results/candidate_states/brp_k20_candidate_state_valuations.csv \
    --output results/candidate_states/brp_k20_candidate_storm_verification.csv \
    --top-k 20
```

### 12. Artifact links

The compact CSV/JSON experiment records and feature schema below should be
deliberately committed for reproducibility after regeneration from a clean
source commit. Raw and processed datasets, serialized joblib models, logs,
predictions, caches, and other bulky generated files should remain untracked.

- [Combined fixed-window metrics](../../results/metrics/brp_fixed_windows/combined_metrics.csv)
- [k=5 metrics](../../results/metrics/brp_fixed_windows/k5.json)
- [k=10 metrics](../../results/metrics/brp_fixed_windows/k10.json)
- [k=20 metrics](../../results/metrics/brp_fixed_windows/k20.json)
- [k=50 metrics](../../results/metrics/brp_fixed_windows/k50.json)
- [k=20 model metadata](../../results/models/brp_fixed_windows/k20/metadata.json)
- [k=20 feature schema](../../results/models/brp_fixed_windows/k20/feature_columns.json)
- [Ranked k=20 candidates](../../results/candidate_states/brp_k20_candidate_states.csv)
- [Candidate-ranking metadata](../../results/candidate_states/brp_k20_candidate_states.metadata.json)
- [Candidate PRISM valuations](../../results/candidate_states/brp_k20_candidate_state_valuations.csv)
- [Candidate-mapping metadata](../../results/candidate_states/brp_k20_candidate_state_valuations.metadata.json)
- [Exact Storm verification](../../results/candidate_states/brp_k20_candidate_storm_verification.csv)
- [Exact-verification metadata](../../results/candidate_states/brp_k20_candidate_storm_verification.metadata.json)
- [Tuned BRP PRISM model](../../models/prism/brp/brp_stress_error_target.pm)
- [Target property](../../models/properties/brp/brp_target.pctl)

### 13. Current status and next steps

Completed:

- [x] Tuned BRP target configuration
- [x] Exact and empirical target probability comparison
- [x] Fixed-window datasets
- [x] Reproducible baseline metrics
- [x] Saved k=20 models and feature schema
- [x] Candidate-state extraction
- [x] Storm-ID-to-valuation mapping
- [x] Exact state-based Storm verification

Pending:

- [ ] Independently generated seed-123 validation dataset
- [ ] Candidate-ranking stability across seeds
- [ ] Richer fixed-window features
- [ ] Strict path-conditioned probability raising
- [ ] Final experiment cleanup and documentation
