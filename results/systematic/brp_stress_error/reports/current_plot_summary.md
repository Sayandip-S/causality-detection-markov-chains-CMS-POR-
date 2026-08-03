# Current BRP experiment plot summary

All common-cohort windows use the same **9,177 traces**, constant class balance, and identical train/test membership. Logistic Regression has its best common-cohort F1 at k=50 and ROC-AUC at k=50; Decision Tree has its best F1 at k=10; Random Forest has its best F1 at k=10 and best ROC-AUC at k=50. The patterns are non-monotonic and ROC-AUC remains near 0.5.

At k=50 the common and operational populations are identical; the maximum reported F1/ROC-AUC difference is 0. Models use visited-state presence and do not preserve full sequence order.

## `common_cohort_f1_by_window.png`

**Sources:** `results/systematic/brp_stress_error/metrics/common_cohort_per_model.csv`, `results/systematic/brp_stress_error/metrics/common_cohort_manifest.json`

**Question:** How does F1 change as more prefix transitions are observed on one fixed cohort?

**Main numerical observation:** Logistic Regression peaks at k=50 (F1=0.338); Decision Tree peaks at k=10 (F1=0.328); Random Forest peaks at k=10 (F1=0.317).

**Limitation:** The common cohort fixes composition and membership, but finite-sample variation and model-fitting choices still affect the curves.

**Presentation sentence:** Across the same 9,177 traces and identical split IDs, longer prefixes do not improve F1 monotonically.

## `common_cohort_roc_auc_by_window.png`

**Sources:** `results/systematic/brp_stress_error/metrics/common_cohort_per_model.csv`, `results/systematic/brp_stress_error/metrics/common_cohort_manifest.json`

**Question:** Does ranking discrimination improve with a longer fixed-cohort prefix?

**Main numerical observation:** Logistic Regression peaks at k=50 (ROC-AUC=0.538); Random Forest peaks at k=50 (ROC-AUC=0.525), while every score remains near 0.5.

**Limitation:** Near-chance ROC-AUC indicates weak discrimination, not proof of no signal.

**Presentation sentence:** ROC-AUC is close to chance at every window; Logistic Regression and Random Forest are best at k=50.

## `operational_vs_common_f1.png`

**Sources:** `results/systematic/brp_stress_error/metrics/operational_vs_common_cohort.csv`

**Question:** How much of the operational F1 pattern changes after fixing cohort membership?

**Main numerical observation:** Operational and common-cohort F1 differ at k=5, 10, and 20, but coincide at k=50.

**Limitation:** Cohort and split changes are not the only possible source of differences; training stochasticity and feature availability also matter.

**Presentation sentence:** The cohort-controlled comparison exposes a clearly non-monotonic F1 response.

## `operational_vs_common_roc_auc.png`

**Sources:** `results/systematic/brp_stress_error/metrics/operational_vs_common_cohort.csv`

**Question:** How does cohort control change the observed ROC-AUC pattern?

**Main numerical observation:** All operational and common-cohort ROC-AUC values remain close to the 0.5 chance line.

**Limitation:** Small ROC-AUC differences should not be over-interpreted without uncertainty estimates.

**Presentation sentence:** Cohort control changes details, but does not turn the visited-state models into strong discriminators.

## `operational_retained_traces_by_window.png`

**Sources:** `results/metrics/brp_fixed_windows/k5.json`, `results/metrics/brp_fixed_windows/k10.json`, `results/metrics/brp_fixed_windows/k20.json`, `results/metrics/brp_fixed_windows/k50.json`

**Question:** How does the operational survival filter change the analysed population?

**Main numerical observation:** Retained traces fall from 10,000 at k=5 to 9,177 at k=50; target traces fall from 3,429 to 2,606, while success traces remain 6,571.

**Limitation:** Counts describe retained datasets and do not measure predictive performance.

**Presentation sentence:** Longer operational windows selectively remove early target-ending traces.

## `operational_target_rate_by_window.png`

**Sources:** `results/metrics/brp_fixed_windows/k5.json`, `results/metrics/brp_fixed_windows/k10.json`, `results/metrics/brp_fixed_windows/k20.json`, `results/metrics/brp_fixed_windows/k50.json`

**Question:** How does operational retention alter class balance?

**Main numerical observation:** The target rate declines from 0.343 at k=5 to 0.284 at k=50.

**Limitation:** The rate change is induced by cohort retention and is not a model effect.

**Presentation sentence:** Operational comparisons mix added prefix information with a changing class balance.

## `exact_candidate_reachability_vs_target_risk.png`

**Sources:** `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv`, `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.metadata.json`

**Question:** Which candidates combine frequent exact reachability with high future target risk?

**Main numerical observation:** State 22 is most reachable (0.765); state 69 has the highest target risk (0.483). 17 candidates lie above and 3 below the baseline.

**Limitation:** Quadrants are descriptive: upper-right means comparatively reachable and high-risk, upper-left rare and high-risk, lower-right reachable and lower-risk, and lower-left rare and lower-risk. No quadrant establishes causality.

**Presentation sentence:** Reachability and future risk are distinct axes; a common state need not raise target risk.

## `exact_probability_increase_by_candidate.png`

**Sources:** `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv`

**Question:** Which candidate states raise exact future target probability above the initial baseline?

**Main numerical observation:** 17 of 20 candidates raise probability; the largest increase is 0.141 at state 69.

**Limitation:** State-based probability raising relative to baseline is not by itself causal.

**Presentation sentence:** The exact comparison identifies 17 raising and 3 non-raising candidates.

## `empirical_support_vs_exact_reachability.png`

**Sources:** `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv`, `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.metadata.json`

**Question:** How do bounded k20 empirical support and unbounded exact reachability differ?

**Main numerical observation:** State 100 has the largest absolute descriptive gap (0.106).

**Limitation:** Empirical k20 support uses 9,723 retained traces with number_of_transitions > 20 and observes only the first 20 transitions; exact reachability is unbounded and unconditional. They are different events, so the gap is not pure estimation error.

**Presentation sentence:** Distance from the identity line mixes horizon, conditioning, sampling, and model-data effects.

## `exact_candidate_reachability_by_candidate.png`

**Sources:** `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv`

**Question:** How reachable is each selected candidate from the model's initial state?

**Main numerical observation:** Exact reachability ranges from 0.012 to 0.765; state 22 is highest.

**Limitation:** Values apply to this model/build and are not empirical visitation frequencies.

**Presentation sentence:** The selected candidates span more than an order of magnitude in exact reachability.

## `risk_weighted_coverage_by_candidate.png`

**Sources:** `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.csv`, `results/systematic/brp_stress_error/reachability/candidate_exact_reachability.metadata.json`

**Question:** Which candidates score highest on the reachability-times-risk heuristic?

**Main numerical observation:** State 22 has the largest risk-weighted coverage (0.255).

**Limitation:** risk_weighted_coverage is only the descriptive product of exact reachability and target probability from the state; it is not causal, path-conditioned, or a formal probability-raising measure.

**Presentation sentence:** Use the heuristic for descriptive prioritisation only, not causal attribution.
