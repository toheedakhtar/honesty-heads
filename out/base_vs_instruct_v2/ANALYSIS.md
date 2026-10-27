# Base vs post-trained honesty-circuit analysis

All figures below are computed from this run. Rates use exact 95% binomial intervals; paired model comparisons use identical stored prompt IDs.

## base

- Model: `Qwen/Qwen3.5-9B-Base`
- Selected circuit: 42 heads using the `min` IG aggregation rule; target met: True.
- Held-out eligible lies: 29.
- Truth preferred at alpha=2.0: 0.483 (exact interval 0.294-0.675).
- Random-head empirical p: 0.05; random-direction empirical p: 0.05.

## instruct

- Model: `Qwen/Qwen3.5-9B`
- Selected circuit: 35 heads using the `mean` IG aggregation rule; target met: True.
- Held-out eligible lies: 34.
- Truth preferred at alpha=2.0: 0.353 (exact interval 0.198-0.535).
- Random-head empirical p: 0.05; random-direction empirical p: 0.05.

## Cross-checkpoint structure

- Mean per-layer direction cosine: 0.814.
- Selected-head overlap: 13; Jaccard: 0.203.
- Fixed-prompt mean improvement difference (base minus post-trained) at max alpha: 0.973 bits; paired sign-flip p=0.1746; McNemar p=0.0391.

## Interpretation guardrails

Cross-transfer is reported under raw, global-norm, per-head-norm, and selection-effect calibration. Treat an apparent asymmetry as robust only when it persists across calibrations. The Yes-push and truth-directed components are reported separately in JSON and the diagnostics figure. Model-specific eligible-item rates and fixed-prompt paired results answer different questions and should both be shown.
