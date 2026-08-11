# BRP k=20 Sample-Size Stability: Full Five-Seed Run

## Scope and provenance

This is the full research run of the k=20 common-cohort experiment. Reduced
training sizes 500, 1,000, 2,500, and 5,000 use sampling seeds 42, 123, 456,
789, and 2026. The 7,341-row full training pool is fitted once without an
artificial sampling seed. Every run uses the same 1,836-trace test set.

Source provenance was captured before outputs were written. The source commit
is `f4d4f7af1c2109b6af3678822a1e919b82b62a86` on
`feat/brp-sample-size-stability`. `source_working_tree_dirty` is correctly
recorded as `false`; the run was generated from clean committed source, and
generated systematic-result paths are excluded from the source-cleanliness
check. The full run contains 63 model evaluations and 21 candidate rankings.

## Reconciliation of the earlier 17/20 and current 15/20 results

The difference is caused by candidate-discovery population, not an
implementation inconsistency.

- The earlier 17/20 ranking used the operational k20 dataset with 9,723 traces
  satisfying `number_of_transitions > 20`. Its candidate statistics used the
  complete processed dataset and its models came from the operational k20
  pipeline.
- The current 15/20 reference uses only the fixed 7,341-row training pool from
  the 9,177-trace common cohort satisfying `number_of_transitions > 50`. The
  fixed 1,836-row test set does not contribute candidate statistics.

The lists share 13 of 20 states. Their exact initial baseline is identical at
`0.3416445845150787`; exact probabilities and Boolean results for every shared
state match exactly. The legacy 88-state ranking is also reproduced numerically
by the shared ranking function. Seven population-dependent candidate
substitutions account for the change: the earlier list contains three
non-raising states and the current list contains five.

## A. Prediction performance

| Rows | Model | Precision mean ± SD | Recall mean ± SD | F1 mean ± SD | ROC-AUC mean ± SD |
|---:|---|---:|---:|---:|---:|
| 500 | Logistic Regression | 0.281 ± 0.004 | 0.462 ± 0.210 | 0.336 ± 0.062 | 0.497 ± 0.010 |
| 500 | Decision Tree | 0.285 ± 0.008 | 0.474 ± 0.205 | 0.343 ± 0.061 | 0.501 ± 0.014 |
| 500 | Random Forest | 0.283 ± 0.006 | 0.435 ± 0.197 | 0.330 ± 0.062 | 0.496 ± 0.013 |
| 1,000 | Logistic Regression | 0.287 ± 0.033 | 0.328 ± 0.172 | 0.288 ± 0.062 | 0.492 ± 0.021 |
| 1,000 | Decision Tree | 0.305 ± 0.040 | 0.284 ± 0.164 | 0.260 ± 0.115 | 0.498 ± 0.008 |
| 1,000 | Random Forest | 0.295 ± 0.024 | 0.366 ± 0.159 | 0.313 ± 0.055 | 0.504 ± 0.018 |
| 2,500 | Logistic Regression | 0.288 ± 0.019 | 0.279 ± 0.030 | 0.283 ± 0.022 | 0.495 ± 0.018 |
| 2,500 | Decision Tree | 0.283 ± 0.016 | 0.331 ± 0.218 | 0.284 ± 0.078 | 0.492 ± 0.017 |
| 2,500 | Random Forest | 0.286 ± 0.017 | 0.260 ± 0.038 | 0.271 ± 0.025 | 0.494 ± 0.017 |
| 5,000 | Logistic Regression | 0.286 ± 0.007 | 0.250 ± 0.049 | 0.264 ± 0.028 | 0.496 ± 0.011 |
| 5,000 | Decision Tree | 0.292 ± 0.011 | 0.225 ± 0.048 | 0.251 ± 0.032 | 0.500 ± 0.006 |
| 5,000 | Random Forest | 0.293 ± 0.009 | 0.259 ± 0.035 | 0.274 ± 0.022 | 0.502 ± 0.008 |
| 7,341 | Logistic Regression | 0.302 | 0.280 | 0.291 | 0.509 |
| 7,341 | Decision Tree | 0.307 | 0.169 | 0.218 | 0.505 |
| 7,341 | Random Forest | 0.305 | 0.248 | 0.273 | 0.506 |

ROC-AUC remains approximately 0.5 throughout. Relative agreement with the
full-data metrics must therefore not be interpreted as strong predictive
discrimination.

## B. Variance across five seeds

At 500 rows, F1 standard deviations are 0.062 for Logistic Regression, 0.061
for Decision Tree, and 0.062 for Random Forest, so none passes the selected
0.05 variability criterion. At 1,000 rows, Decision Tree remains particularly
variable with F1 SD 0.115. At 2,500 rows, Logistic Regression and Random Forest
fall below 0.05, while Decision Tree remains at 0.078. All three fall below
0.05 at 5,000 rows.

Under prediction-only operational checks, the first passing size remains 2,500
for Logistic Regression and Random Forest and 5,000 for Decision Tree.

## C. Candidate-ranking stability

| Rows | Top-10 overlap mean ± SD | Top-20 overlap mean ± SD | Shared-state Spearman mean ± SD | Mean absolute displacement |
|---:|---:|---:|---:|---:|
| 500 | 0.160 ± 0.089 | 0.310 ± 0.055 | 0.084 ± 0.559 | 6.21 |
| 1,000 | 0.280 ± 0.130 | 0.460 ± 0.096 | 0.255 ± 0.132 | 5.79 |
| 2,500 | 0.360 ± 0.089 | 0.560 ± 0.065 | 0.327 ± 0.144 | 5.84 |
| 5,000 | 0.460 ± 0.195 | 0.720 ± 0.091 | 0.284 ± 0.343 | 5.77 |
| 7,341 | 1.000 | 1.000 | 1.000 | 0.00 |

Top-20 overlap passes the selected 0.60 criterion at 5,000 rows, but top-10
overlap remains well below 0.70. Ranking order among shared states also remains
unstable.

## D. Exact probability-raising quality

| Rows | Passing states mean ± SD (of 20) | Mean exact difference ± SD | Mean exact reachability ± SD |
|---:|---:|---:|---:|
| 500 | 8.60 ± 2.30 | 0.0130 ± 0.0116 | 0.2536 ± 0.0629 |
| 1,000 | 10.80 ± 2.39 | 0.0250 ± 0.0110 | 0.2261 ± 0.0670 |
| 2,500 | 11.60 ± 2.30 | 0.0284 ± 0.0187 | 0.2145 ± 0.0392 |
| 5,000 | 13.20 ± 0.84 | 0.0382 ± 0.0122 | 0.1900 ± 0.0162 |
| 7,341 | 15.00 | 0.0590 | 0.1333 |

Only the full reference reaches the chosen 15/20 threshold. The exact cache
was reused without modification: all 88 required states were present, so the
full run performed zero new state verifications and did not rebuild the Storm
model.

## E. Ranking-method comparison

Mean exact probability-raising counts:

| Rows | Combined | Empirical only | LR only | RF only | Frequency only | Random mean ± SD |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 8.60 | 12.60 | 9.40 | 8.00 | 0.00 | 9.55 ± 2.07 |
| 1,000 | 10.80 | 12.00 | 11.60 | 9.80 | 0.00 | 10.13 ± 1.99 |
| 2,500 | 11.60 | 15.00 | 11.00 | 9.60 | 0.00 | 10.28 ± 2.01 |
| 5,000 | 13.20 | 16.40 | 12.60 | 11.60 | 0.00 | 10.15 ± 1.89 |
| 7,341 | 15.00 | 16.00 | 14.00 | 12.00 | 0.00 | 10.39 ± 2.06 |

There are 2,100 deterministic random sets: 100 for each of the 21 ranking
conditions. The random 2.5th–97.5th percentile pass-count intervals are 6–13
at 500 rows, 6–14 at 1,000 and 2,500 rows, 7–14 at 5,000 rows, and 7–14 at full
training.

At 500 rows the combined ranking is slightly worse than the random mean. It
exceeds the random mean from 1,000 rows onward, with a clearer gap at 5,000 and
7,341 rows. Empirical-difference-only ranking has the highest mean exact pass
count at every size. Frequency-only consistently selects highly reachable
states but none raises exact target probability.

## F. Runtime

The runner completed in 14.93 seconds. Mean full-reference training times were
1.142 seconds for Logistic Regression, 0.073 seconds for Decision Tree, and
0.351 seconds for Random Forest.

## G. Reliability criteria

No reduced sample size passes all analyst-selected operational criteria. At
5,000 rows all three models pass the prediction-relative criteria, top-20
overlap, and exact-pass variability, but the candidate ranking fails top-10
overlap and the requirement for a mean of at least 15 exact passing states.
Only the full 7,341-row reference passes every criterion.

These criteria are exploratory operational choices, not universal statistical
laws.

## H. Differences from the quick run

- The 500-row mean F1 values fall from roughly 0.37–0.38 to 0.33–0.34, and all
  three F1 SDs now exceed 0.05.
- At 5,000 rows, top-10 overlap falls from 0.567 to 0.460 and top-20 overlap
  falls from 0.767 to 0.720.
- The 5,000-row exact pass mean falls from 13.67 to 13.20; its SD remains below
  the selected threshold, increasing from 0.58 to 0.84.
- The full deterministic reference is unchanged: the same predictive metrics,
  ranking, and 15/20 exact pass result are reproduced.
- The full run adds the required 2,100 random ranking sets.

## I. Conclusions

The main conclusion does not change and is strengthened by the extra seeds.
Relative predictive stability appears earlier than candidate-ranking stability,
but absolute discrimination remains weak. Five thousand rows are sufficient
for the selected prediction-variance checks and top-20 overlap, yet not for
stable top-10 identity or the target exact pass count. Under the combined
criteria, no reduced-data condition is reliable; only the complete 7,341-row
training pool passes.

The new random baseline adds an important qualification: at 500 rows the
combined heuristic is not better than random on exact pass count. At larger
sizes it improves over random, but empirical-difference-only ranking remains
better on this outcome. The combined score is a ranking heuristic, not a
probability or causal score, and exact state-based probability raising does not
establish historical path-conditioned causality.

## J. Artifacts

- [Prediction runs](prediction_per_run.csv)
- [Prediction aggregation](prediction_aggregated.csv)
- [Run metadata](prediction_metadata.json)
- [Candidate rankings](candidate_rankings_per_run.csv)
- [Candidate stability](candidate_stability_per_run.csv)
- [Candidate stability aggregation](candidate_stability_aggregated.csv)
- [Exact cache](exact_candidate_cache.csv)
- [Exact quality](exact_candidate_quality_per_run.csv)
- [Exact quality aggregation](exact_candidate_quality_aggregated.csv)
- [Reliability assessment](reliability_assessment.csv)
- [Deterministic ranking comparison](ranking_method_comparison.csv)
- [Random baseline distribution](random_baseline_distribution.csv)
- [Plots](plots/)
