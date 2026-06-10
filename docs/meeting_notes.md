# Meeting Notes

## Meeting 1: May 7, 2025

### Key Points
- Causality defined via probability raising: P(E|C) > P(E)
- Goal: Use ML to identify causes for error states in Markov chains
- Approach: Simulate paths, train classifier on prefixes to predict error

### Action Items
1. [Student] Read provided papers on causality in Markov chains
2. [Student] Implement baseline model (Random Forest on prefixes)
3. [Student] Prepare questions for next meeting
4. [Jakob] Send paper references

### Next Meeting
- Date: Tuesday, May 19th, 2025
- Time: 1:00 PM
- Focus: Discuss initial findings and technical questions


## Meeting 3: June 2, 2026

### Main Research Goal

* The overall objective is to identify a set of states, or more generally a monitor, that can predict whether an execution will eventually reach an error state.
* Exact computation of good causal state sets is computationally expensive because the number of possible state combinations grows exponentially.
* Machine learning may be used to identify promising candidate states or monitors without exhaustively checking every possible state set.
* A highly accurate predictor may still be useful even if it does not represent an exact formal cause.

### Discussion of the Current Prototype

* The current implementation generates traces from a manually defined Markov chain.
* A Random Forest classifier is trained on path prefixes to predict whether an error state will eventually be reached.
* The prototype demonstrates that execution traces can be converted into a supervised learning dataset.
* The current model is a useful baseline, but the manually written Markov chain should later be replaced or extended using established probabilistic models.

### Predictor as a Set of States

* The desired predictor may be represented as a set of states whose occurrence indicates a high future risk of reaching an error.
* A predictor does not necessarily need to be a single state.
* Single-state predictors may have high precision but low recall.
* Combining several predictive states may produce a more useful monitor.
* The predictor could also depend on the order or position at which states are observed in a trace.

### Important Experimental Parameters

The following parameters should be varied and evaluated:

* number of generated traces;
* maximum trace length;
* prefix length or learning depth;
* type of machine-learning model;
* model-specific hyperparameters;
* representation of the states and paths;
* prediction quality measures such as accuracy, precision, recall and F1-score.

### Storm and StormPy

* Storm can be used to work with the complete Markov-chain model and compute probabilities precisely.
* StormPy provides a Python interface to Storm.
* Instead of defining every transition manually in Python, probabilistic models can be loaded from model files and processed through StormPy.
* The next technical step is to load a small probabilistic model through StormPy and inspect its states, transitions and labels.
* Technical support may be requested from Calvin if difficulties arise with the StormPy interface.

### Relationship Between Machine Learning and Exact Verification

* Machine learning should generate a hypothesis, such as a candidate set of predictive states.
* Storm can then verify the selected candidate precisely on the original Markov chain.
* Storm does not need to search through all possible state combinations.
* It only needs to evaluate the candidates proposed by the learning method.
* The exact probabilities computed by Storm can be compared with the probabilities estimated from simulated traces.
* This can determine whether the candidate is probability-raising and how strongly it increases the probability of reaching the error state.

### Planned Machine-Learning Experiments

* Keep Random Forest as an initial baseline.
* Add simpler and more interpretable models, such as:

  * Logistic Regression;
  * Decision Tree;
  * Random Forest.
* Compare the models using the same training and test traces.
* Investigate whether model coefficients, tree rules or feature importance can identify candidate states.
* More complex sequential models may be considered later if the order of states becomes important.

### Immediate Action Items

1. Save and document the current manually defined Markov-chain prototype as a baseline.
2. Install and test StormPy.
3. Load one small DTMC model from a PRISM model file.
4. Print the number of states, transitions and initial states.
5. identify the error-labelled states.
6. Compute the exact probability of eventually reaching the error state.
7. Generate traces from the same model.
8. Create prefix-based datasets at different learning depths.
9. Train Logistic Regression, Decision Tree and Random Forest models.
10. Extract promising candidate states from the trained models.
11. Verify selected candidates precisely using Storm.
12. Record the effect of trace count, trace length and learning depth on prediction quality.

### Organizational Point

* Confirm registration for the CMS Research Project “Model Checking” with the examination office.
* Clarify the course registration and final-presentation arrangements.
