# Data Preprocessing for Causality Detection

## The Challenge: Variable Length Paths

Markov chain paths have different lengths:
- `[0, 1, 3, 4]` - Reached error state (4 steps)
- `[0, 2, 5]` - Reached safe state (3 steps)  
- `[0, 1, 5]` - Reached safe state (3 steps)

But ML models (Random Forest, Neural Networks) require **fixed-size inputs**.

## Solution: Padding & Truncation

We standardize all prefixes to length `k` (e.g., k=3):

| Original Path | After Padding (k=3) |
|--------------|---------------------|
| `[0, 1, 3, 4]` | `[0, 1, 3]` (truncated) |
| `[0, 2, 5]` | `[0, 2, 5]` (exact) |
| `[0, 1]` | `[0, 1, -1]` (padded) |

**Padding value:** `-1` (since state IDs are 0-5, -1 is unused)

## Why This Works

1. **Early prediction**: We only use first k steps to predict final outcome
2. **Fixed input**: Model always receives same size vector
3. **Padding signals**: Model learns that `-1` means "no state here"

## Impact on Model Performance

With k=3, the model can predict error by seeing:
- State 3 (C) in early steps → high probability of error
- State 5 (SAFE) early → safe outcome
- Missing states (padded) → less information

## Alternative Approaches

1. **Variable-length RNN/LSTM**: Can handle sequences naturally
2. **Different k values**: Trade-off between early detection and accuracy
3. **Different padding values**: Try `0` or `-999` as padding