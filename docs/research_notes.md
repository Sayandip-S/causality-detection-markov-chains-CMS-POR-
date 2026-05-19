@"
# Research Notes: Causality Detection in Markov Chains

## Key Concepts from Meeting (May 16, 2025)

### Definition of Causality (Probability Raising)
- A cause C raises the probability of an effect E
- Formal: P(E | C) > P(E)
- This is the core definition we'll use

### Proposed Approach
1. Simulate runs through the Markov chain
2. Label paths: reach error (positive) or not (negative)
3. Extract prefixes of length k as features
4. Train ML model to predict final outcome
5. Use model as monitor: alert when error probability becomes high

## Questions for Next Meeting
1. What's the baseline performance we're targeting?
2. How do we validate if identified causes are true causes?
3. How to handle probabilistic vs deterministic causality?
4. What prefix length is optimal?
"@ | Out-File -FilePath docs\research_notes.md -Encoding utf8