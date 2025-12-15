## Extension 6.1 – Stage 4: Define a Simple Adaptive Re-ranking Rule

### Goal of Stage 4
- Use the trained logistic regression model to compute, for each query:
  - A **probability that Top‑1 is correct**.
- Define a **simple rule** (with a fixed threshold) to classify queries as:
  - **Easy** → skip re-ranking.
  - **Hard** → re-rank (run image matching).

### Simple decision rule (no tuning)
- We avoid optional complexity and use a **fixed threshold**:
  - If \( p(\text{Top‑1 correct}) \geq 0.5 \) → **easy query** → **skip re-ranking**.
  - If \( p(\text{Top‑1 correct}) < 0.5 \) → **hard query** → **apply re-ranking**.
- This keeps the implementation straightforward while still reflecting the idea of adaptive re-ranking.

### Inputs and outputs of Stage 4 script
- **Inputs**:
  - A trained model file, e.g. `logreg_svox.pkl` (from Stage 3).
  - A feature file for some split, e.g. `features_sf_xs_test.npz`.
- **Outputs**:
  - A new `.npz` file containing:
    - `probs` – predicted \( p(\text{Top‑1 correct}) \) per query.
    - `is_easy` – boolean array: `True` if query is easy (skip re-ranking).
    - `is_hard` – boolean array: `True` if query is hard (apply re-ranking).


