# BRP k=20 Training-Sample-Size and Candidate-Ranking Stability

## 1. Research question

This experiment asks when reduced training data makes BRP predictive
performance, variability across samples, candidate rankings, exact candidate
quality, or runtime unreliable, and whether the predictive threshold differs
across Logistic Regression, Decision Tree, and Random Forest.

This report records the required **quick meeting run** with sampling seeds 42,
123, and 456. It is preliminary: the configured five-seed full research run
has deliberately not been launched yet.

## 2. Experimental design

Only the fixed-window k=20 common-cohort dataset is used. It contains 9,177
traces and 88 binary `visited_state_*` features. One stratified split produces
a fixed pool of 7,341 training traces and a fixed test set of 1,836 traces.
Reduced training samples contain 500, 1,000, 2,500, or 5,000 rows; the 7,341-row
condition uses the complete training pool once without an artificial sampling
seed.

Every reduced sample is stratified within the fixed training pool. For a given
size and seed, Logistic Regression, Decision Tree, and Random Forest receive
exactly the same sampled trace IDs. All models have deterministic random state
42. Candidate statistics are constructed only from the sampled training rows.

Candidate eligibility uses:

```text
minimum support = max(5, ceil(0.005 × training sample size))
```

The resulting thresholds are 5, 5, 13, 25, and 37 rows for the five sample
sizes. The combined ranking uses the shared project implementation: normalized
positive Logistic Regression coefficient, Random Forest importance, normalized
positive empirical target-probability difference, and a support-reliability
weight. Its combined score is a ranking heuristic, not a probability or a
causal score.

## 3. Why the test set is fixed

Changing both the training sample and the test set would mix training-data
effects with test-composition noise. Every evaluation therefore reuses the
same 1,836 test traces. The reconstructed ordered hashes exactly match the
existing common-cohort provenance:

- training pool: `59c7b8869b0aab341fb7695b01157bbc731c8047ee271abc2e4105312e2122aa`
- test set: `80061fa2fc440c6a1b63de1014115d9c54823b174baabd3a73ff9d2051b31d42`

No sampled training trace occurs in the test set, and test labels are never
used for candidate construction.

## 4. Training sample sizes and seeds

The quick run evaluates seeds 42, 123, and 456 at each reduced size, plus one
seedless full-pool reference. This gives 39 predictive model evaluations and
13 combined candidate rankings. The full configuration adds seeds 789 and
2026, for 63 model evaluations and 21 rankings.

## 5. Prediction performance

Mean values below use three samples at reduced sizes and one fit at full size.

| Training rows | Model | F1 mean ± SD | ROC-AUC mean ± SD |
|---:|---|---:|---:|
| 500 | Logistic Regression | 0.375 ± 0.045 | 0.498 ± 0.012 |
| 500 | Decision Tree | 0.380 ± 0.043 | 0.501 ± 0.020 |
| 500 | Random Forest | 0.368 ± 0.048 | 0.499 ± 0.018 |
| 1,000 | Logistic Regression | 0.316 ± 0.065 | 0.493 ± 0.019 |
| 1,000 | Decision Tree | 0.245 ± 0.159 | 0.497 ± 0.007 |
| 1,000 | Random Forest | 0.338 ± 0.056 | 0.506 ± 0.016 |
| 2,500 | Logistic Regression | 0.274 ± 0.026 | 0.494 ± 0.020 |
| 2,500 | Decision Tree | 0.255 ± 0.056 | 0.494 ± 0.019 |
| 2,500 | Random Forest | 0.267 ± 0.033 | 0.492 ± 0.020 |
| 5,000 | Logistic Regression | 0.269 ± 0.036 | 0.499 ± 0.013 |
| 5,000 | Decision Tree | 0.250 ± 0.045 | 0.500 ± 0.008 |
| 5,000 | Random Forest | 0.289 ± 0.005 | 0.506 ± 0.008 |
| 7,341 | Logistic Regression | 0.291 | 0.509 |
| 7,341 | Decision Tree | 0.218 | 0.505 |
| 7,341 | Random Forest | 0.273 | 0.506 |

The F1 values are not monotonic with sample size. At 500 rows the balanced
models predict positives much more often, increasing recall and F1 even though
ROC-AUC remains near 0.5. This is not evidence that 500 rows are better. Across
all conditions, ROC-AUC is approximately chance, so closeness to the full-data
metric can mean closeness to a weak reference rather than adequate predictive
discrimination.

Reducing training data is not identical to underfitting. Smaller samples
increase estimator variance and can omit learnable patterns, while underfitting
describes insufficient model flexibility or optimization relative to the
available signal.

## 6. Variance across samples

The Decision Tree shows the largest quick-run F1 standard deviation, 0.159 at
1,000 rows, consistent with the hypothesis that a single tree can be unstable
under sampling changes. Its F1 SD remains 0.056 at 2,500 and falls to 0.045 at
5,000. Logistic Regression falls below the selected 0.05 SD threshold at 500,
2,500, and 5,000 rows, though its non-monotonic means caution against treating
one variance criterion in isolation. Random Forest reaches an F1 SD of 0.005 at
5,000 rows, consistent with variance reduction through ensembling, but it still
does not produce meaningful ROC-AUC discrimination.

The hypotheses that Logistic Regression might stabilize earlier because of
lower capacity, that Decision Trees may be unstable, and that Random Forests
reduce tree variance but still need sufficient samples are therefore only
partially supported. The five-seed run is needed for a firmer variance estimate.

## 7. Candidate-ranking stability

| Training rows | Top-10 overlap mean ± SD | Top-20 overlap mean ± SD | Shared-state Spearman mean ± SD |
|---:|---:|---:|---:|
| 500 | 0.167 ± 0.115 | 0.283 ± 0.029 | 0.333 ± 0.499 |
| 1,000 | 0.200 ± 0.100 | 0.433 ± 0.076 | 0.222 ± 0.162 |
| 2,500 | 0.400 ± 0.100 | 0.583 ± 0.029 | 0.393 ± 0.121 |
| 5,000 | 0.567 ± 0.153 | 0.767 ± 0.076 | 0.505 ± 0.203 |
| 7,341 | 1.000 | 1.000 | 1.000 |

Ranking stability improves with sample size, but the top of the list remains
unstable. At 5,000 rows, top-20 overlap passes the selected 0.60 criterion,
while top-10 overlap does not reach 0.70. Mean absolute displacement among
shared candidates is still 4.76 ranks.

## 8. Exact Storm verification quality

The existing 20 exact rows seeded a cache. The experiment built the 1,221-state,
1,603-transition Storm model once, computed the target result vector once, and
verified 68 previously uncached eligible state IDs. The exact initial target
baseline remained `0.3416445845150787`.

| Training rows | Exact pass count mean ± SD (of 20) | Mean exact probability difference ± SD | Mean exact candidate reachability |
|---:|---:|---:|---:|
| 500 | 9.00 ± 3.00 | 0.0143 ± 0.0128 | 0.2341 |
| 1,000 | 11.33 ± 3.06 | 0.0237 ± 0.0120 | 0.2041 |
| 2,500 | 11.33 ± 3.06 | 0.0272 ± 0.0252 | 0.2210 |
| 5,000 | 13.67 ± 0.58 | 0.0427 ± 0.0071 | 0.1895 |
| 7,341 | 15.00 | 0.0590 | 0.1333 |

Exact candidate quality improves overall with more data. The selected criterion
of at least 15 exact probability-raising states is reached only by the full
training reference. At 500 rows, an average of nine top-20 states have positive
empirical differences but negative exact differences; this mismatch falls to
three at 5,000 and remains three at full training.

These are state-based quantities, `P_candidate(F target)`, compared with the
initial-state baseline. They do not establish historical path-conditioned
causality or identify the effect of having visited a candidate along a path.

## 9. Model-specific data requirements

Considering only the three predictive operational checks—F1 within 0.05 of the
full reference, ROC-AUC within 0.02, and F1 SD below 0.05—the first passing
quick-run sizes are:

- Logistic Regression: 2,500 rows.
- Random Forest: 2,500 rows.
- Decision Tree: 5,000 rows.

These thresholds describe similarity to weak full-data predictive performance,
not useful absolute predictive quality. Candidate requirements cannot be
assigned to the Decision Tree because the combined candidate ranking uses
Logistic Regression and Random Forest. With all prediction and candidate
criteria combined, only the 7,341-row reference passes.

## 10. Comparison with simple ranking baselines

Mean exact probability-raising counts show that the combined heuristic does not
uniformly dominate simpler rankings:

| Training rows | Combined | Empirical difference only | Frequency/support only | LR coefficient only | RF importance only |
|---:|---:|---:|---:|---:|---:|
| 500 | 9.00 | 12.67 | 0.00 | 10.33 | 7.67 |
| 1,000 | 11.33 | 13.00 | 0.00 | 12.33 | 10.00 |
| 2,500 | 11.33 | 15.00 | 0.00 | 11.00 | 10.00 |
| 5,000 | 13.67 | 16.00 | 0.00 | 13.00 | 12.67 |
| 7,341 | 15.00 | 16.00 | 0.00 | 14.00 | 12.00 |

Empirical-difference-only ranking gives the largest mean pass count in this
quick run, while frequency-only consistently selects highly reachable states
that do not raise exact target probability. The latter result illustrates why
candidate reachability and probability raising are distinct quality axes. The
quick run intentionally omits random candidate sets; the full mode is configured
for 100 deterministic random sets per condition.

## 11. Runtime

The complete quick runner took 11.66 seconds on this machine; plotting took
approximately four additional seconds. Mean full-pool training time was 2.112
seconds for Logistic Regression, 0.063 seconds for Decision Tree, and 0.407
seconds for Random Forest. Sampling and feature preparation are separately
recorded in the per-run artifact.

A simple linear extrapolation from 13 to 21 ranking conditions suggests roughly
19 seconds for the runner on the same machine, plus plotting. This is only a
local estimate; system load and Storm/cache state can change runtime.

## 12. Reliability thresholds

The exploratory operational criteria are:

- F1 within 0.05 of full-data F1;
- ROC-AUC within 0.02 of full-data ROC-AUC;
- top-10 overlap at least 0.70;
- top-20 overlap at least 0.60;
- at least 15 of 20 candidates raise exact target probability;
- F1 SD below 0.05;
- exact-pass-count SD below 2.0.

These thresholds are analyst-selected operational criteria, not universal
statistical laws. No reduced size passes all seven criteria in the quick run.
At 5,000 rows all models pass the three prediction checks, and the candidate
ranking passes top-20 overlap and exact-count variability, but fails top-10
overlap and the mean exact pass-count requirement.

## 13. Main findings

1. Absolute prediction discrimination is weak at every size; ROC-AUC stays near
   0.5, so predictive-reference thresholds alone are insufficient.
2. Prediction variability suggests Logistic Regression and Random Forest meet
   the selected relative criteria earlier than the Decision Tree, but the means
   are non-monotonic and only three reduced-sample seeds are available.
3. Candidate rankings are much more data-sensitive than headline prediction
   metrics. Top-10 stability remains below the chosen threshold at 5,000 rows.
4. Exact probability-raising quality rises from 9.0/20 at 500 rows to 13.67/20
   at 5,000, compared with 15/20 at full training.
5. Under the combined operational assessment, this quick run supports no
   reduced-data reliability cutoff; only the full 7,341-row reference passes.
6. Empirical-difference-only ranking outperforms the combined ranking on exact
   pass count here, so the combined heuristic's added value is not established.

## 14. Limitations

- Reduced conditions have three rather than five seeds; standard deviations are
  preliminary and the full condition has no sampling-variance estimate.
- The full-data condition is one deterministic reference, intentionally not five
  duplicate datasets or repeated model-randomness fits.
- Reliability thresholds are operational choices and are sensitive to a weak
  full-data predictive reference.
- Candidate features encode state presence within a fixed prefix, not order,
  multiplicity, or historical causal effects.
- Empirical support is bounded and cohort-conditioned, while exact candidate
  reachability is unbounded and unconditional; they are not identical events.
- Exact probability raising from a state is necessary for the stated quality
  check but is not sufficient evidence of causality.
- Random ranking baselines await the full research run.

## 15. Reproduction commands

Quick meeting run (refuses to overwrite existing artifacts):

```bash
.venv/bin/python scripts/run_brp_sample_size_experiment.py \
  --config experiments/brp_k20_sample_size_stability.json \
  --quick \
  --output-root results/systematic/brp_stress_error/sample_size

MPLCONFIGDIR=/tmp/brp-matplotlib-config \
.venv/bin/python scripts/plot_brp_sample_size_experiment.py \
  --input-root results/systematic/brp_stress_error/sample_size \
  --output-directory results/systematic/brp_stress_error/sample_size/plots
```

Full research mode uses all configured seeds and 100 random sets per condition:

```bash
.venv/bin/python scripts/run_brp_sample_size_experiment.py \
  --config experiments/brp_k20_sample_size_stability.json \
  --output-root <new-empty-output-directory>
```

Use a new empty output directory because the runner never overwrites experiment
artifacts. The full run has not been executed as part of this report.

## 16. Artifact links

- [Experiment configuration](../experiments/brp_k20_sample_size_stability.json)
- [Prediction runs](../results/systematic/brp_stress_error/sample_size/prediction_per_run.csv)
- [Aggregated prediction metrics](../results/systematic/brp_stress_error/sample_size/prediction_aggregated.csv)
- [Candidate rankings](../results/systematic/brp_stress_error/sample_size/candidate_rankings_per_run.csv)
- [Candidate stability](../results/systematic/brp_stress_error/sample_size/candidate_stability_aggregated.csv)
- [Exact candidate quality](../results/systematic/brp_stress_error/sample_size/exact_candidate_quality_aggregated.csv)
- [Ranking-method comparison](../results/systematic/brp_stress_error/sample_size/ranking_method_comparison.csv)
- [Reliability assessment](../results/systematic/brp_stress_error/sample_size/reliability_assessment.csv)
- [Run metadata](../results/systematic/brp_stress_error/sample_size/prediction_metadata.json)
- [Plots](../results/systematic/brp_stress_error/sample_size/plots/)

## 17. Meeting summary

The quick experiment indicates that sample-size adequacy depends on the outcome
being protected. Relative predictive metrics can look stable by 2,500–5,000
rows even though absolute ROC-AUC is near chance. Candidate identity and exact
probability-raising quality require more data: 5,000 rows still misses the
top-10-overlap and exact-pass-count criteria. On the selected criteria, the only
reliable condition is the full 7,341-row training pool. This should be treated
as a quick-run finding pending the two additional seeds and random baselines.
