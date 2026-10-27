# Base vs. post-trained analysis: v2 changes

## Status

This document describes the second version of `code/base_vs_instruct.py`. It was implemented in response to the limitations identified after the initial Base-versus-post-trained run documented in `docs/initial_base_v_instruct_results.md`.

The initial results remain useful as exploratory evidence. However, v2 should be run into a new output directory because its JSON schema, statistical analysis, null controls, circuit-selection procedure, and plots are substantially expanded.

The checkpoint pair remains:

| Role | Checkpoint |
|---|---|
| Pretrained Base model | `Qwen/Qwen3.5-9B-Base` |
| Post-trained model | `Qwen/Qwen3.5-9B` |

## Why v2 was needed

The initial analysis suggested three interesting patterns:

1. Both checkpoints contained a causally useful truth-versus-lie steering direction.
2. The learned directions were similar across layers, with mean cosine similarity of approximately 0.814.
3. Exact selected-head overlap was much lower, suggesting that post-training might preserve broad activation geometry while changing its sparse implementation.

Those findings were not yet sufficient for a strong Base-versus-post-training conclusion because:

- the two models contributed different eligible test examples;
- only one random head set and one random direction were evaluated;
- cross-transfer used the same raw steering coefficient despite potentially different direction norms;
- the new selector used only the conservative minimum attribution rule, rather than reproducing the repository's mean-versus-minimum selection procedure;
- the Yes-response bias was not decomposed;
- paired uncertainty estimates and statistical tests were missing; and
- the comparison had not yet been extended to persona, long-form behavior, ordinary-task controls, ablation, or sycophancy.

V2 implements all of these missing analyses.

---

## 1. Repository-parity circuit selection

The original repository forms two integrated-gradient rankings:

- **minimum rule:** favors heads that contribute on both lie and resist examples;
- **mean rule:** favors heads with the largest average contribution across those conditions.

V2 now evaluates prefix curves for both rankings. For each rule, it finds the first prefix whose median effect:

- reaches at least 70% of the every-head effect on both lies and resists; and
- does not overshoot 130% of the every-head effect.

The script then chooses the smallest successful prefix across the two ranking rules. If neither rule passes, it records that the target was missed and chooses the prefix with the best worst-condition effect.

The output records:

- both complete prefix curves;
- the selected aggregation rule;
- whether the target was met;
- the selected head count and identities; and
- the per-head integrated-gradient scores for lie and resist examples.

This change is necessary before comparing the post-trained selection with the repository's original 29-head result.

---

## 2. Shared-prompt, paired Base-versus-post-trained analysis

V1 evaluated each checkpoint on examples whose remeasured margin agreed with that model's stored lie/resist label. This is appropriate for asking whether a model's own naturally occurring lies are steerable, but it conditions the two models on different subsets.

V2 therefore reports two complementary analyses:

### Model-specific eligible analysis

Each checkpoint is evaluated on examples where its float32 remeasurement agrees with the expected lie or resist sign. This answers:

> How effective is steering on the behavior this particular checkpoint naturally exhibits?

The output includes an eligibility funnel for each split and label:

- stored usable candidates;
- candidates whose remeasured margin has the expected sign; and
- final selection/test counts.

### Fixed shared-prompt analysis

Both checkpoints are also evaluated on exactly the same stored test IDs without filtering on either model's baseline margin sign. This answers:

> On the same prompts, how much does each model move toward the intended answer under its native intervention?

The JSON stores per-item:

- item ID;
- unedited margin;
- steered margins at every alpha;
- improvement in bits;
- binary truth/lie flip outcome; and
- answer polarity.

This enables genuinely paired model comparisons rather than comparing unrelated denominators such as 29 and 34.

---

## 3. Calibrated cross-checkpoint transfer

The initial cross-transfer result applied each source direction to the other checkpoint using the same raw coefficient. That can be misleading if direction or activation norms differ after post-training.

V2 evaluates every transfer under four scaling rules:

| Mode | Meaning |
|---|---|
| `raw` | The source direction is transferred unchanged. |
| `global_norm` | The source direction is rescaled to match the target direction's total head-output norm. |
| `per_head_norm` | Every source head slice is rescaled to match the corresponding target head-direction norm. |
| `behavior` | The source direction is rescaled to match the target native direction's median all-head selection effect. |

For each transfer, the output records:

- source and target direction norms;
- native target selection effect;
- raw transferred selection effect;
- behavioral calibration factor; and
- complete dose-response results under all four calibrations.

An apparent transfer asymmetry should be considered robust only if it survives the norm- and behavior-calibrated comparisons.

Cross-transfer still runs only when:

- the stored tokenizer-ID fingerprint matches;
- the number of layers and heads is compatible; and
- every head-output direction has the expected dimensionality.

---

## 4. Repeated matched null interventions

V1 used one matched random head set and one random direction. V2 defaults to 19 of each for every native checkpoint analysis:

- **random head sets:** preserve the selected circuit's layer distribution while replacing its heads;
- **random directions:** preserve the learned direction's per-unit sizes while randomizing direction.

The real intervention and all null interventions are evaluated on the same prompts and steering strengths.

V2 reports empirical rank p-values:

```text
p = (1 + number of null effects at least as large as observed) / (K + 1)
```

With 19 null draws, the smallest attainable p-value is 0.05. The p-values are computed separately for:

- truth-preferred share; and
- median improvement in bits.

Cross-transfer calibration modes use a separate `--cross-random` count, defaulting to eight per mode to control computational cost.

---

## 5. Truth-directed effect versus generic Yes-push

V2 applies the repository's Yes/No/other decomposition to each checkpoint's native intervention and its null controls.

For lie examples:

```text
truth-directed component = (effect on No-lies + effect on Yes-lies) / 2
Yes-push component        = (effect on No-lies - effect on Yes-lies) / 2
```

It separately reports effects on:

- Yes-answer cases;
- No-answer cases;
- answers that are neither Yes nor No;
- the truth-directed component; and
- the generic Yes-push component.

The same decomposition is produced for resist examples with the appropriate reversed intervention sign. Counts for all three polarity groups are retained so small subgroups remain visible.

This analysis is important because the original honesty direction contained a substantial response-polarity component.

---

## 6. Confidence intervals and paired statistical tests

V2 adds uncertainty estimates at the item level.

### Within each model

- Exact Clopper-Pearson 95% intervals for truth/lie flip rates
- Bootstrap intervals for median margin improvement
- Empirical randomization p-values against matched head and direction nulls

### Between models on shared prompts

- Difference in mean improvement in bits
- Difference in median improvement
- Paired bootstrap interval for the median difference
- Paired sign-flip permutation test for mean improvement
- Exact McNemar/binomial test for discordant binary flip outcomes

These analyses are repeated at every requested steering strength rather than only at `alpha = 2`.

The paired analysis is deliberately kept separate from model-specific eligible-item analysis because they answer different scientific questions.

---

## 7. Behavioral and mechanistic extension stages

V2 exposes four expensive optional stages. They reuse the repository's existing implementations instead of duplicating their statistical logic.

### Persona

For each checkpoint:

- run the 135-trait persona evaluation;
- compare the selected direction with 39 same-size random directions by default;
- compute polarity-balanced trait effects;
- run permutation FDR; and
- run Westfall-Young step-down maxT familywise correction.

### Insider-trading audit

For each checkpoint:

- convert the selected head circuit into the audit core format;
- generate 512-token insider-trading replies;
- compare the real intervention with matched random interventions; and
- retain rule-based truth/lie/other grades.

### Ablation and ordinary-task controls

For each checkpoint:

- switch off the selected heads;
- test whether lies are removed and truthful answers are damaged;
- run matched random ablations; and
- measure preservation of ordinary answer and instruction-following tasks.

### Sycophancy-derived direction

For each checkpoint:

- construct a direction from sycophantic lies;
- select a sparse sycophancy head set using integrated gradients;
- test held-out sycophancy;
- test transfer to role-play lies; and
- compare against matched random directions and task controls.

The extension stages write into separate directories under:

```text
OUT/extensions/base/
OUT/extensions/instruct/
```

They can be run after the core experiment without repeating direction construction or the main two-model comparison.

---

## New outputs

The enhanced core run writes:

| Artifact | Contents |
|---|---|
| `base_vs_instruct.json` | Complete native, paired, null, polarity, calibrated-transfer, and extension metadata |
| `base_vs_instruct.png` | Main circuit-selection and behavioral summary |
| `base_vs_instruct_diagnostics.png` | Null distributions, Yes-bias decomposition, paired improvements, and calibrated transfer |
| `ANALYSIS.md` | Automatically generated result summary with statistical guardrails |
| `base.json`, `instruct.json` | Per-checkpoint detailed outputs |
| `base_circuit.pt`, `instruct_circuit.pt` | Directions and selected circuit locations for transfer analysis |
| `base_delta.pt`, `instruct_delta.pt` | Repository-compatible direction tensors |
| `base_heads.json`, `instruct_heads.json` | Repository-compatible selected-head definitions |

Extension outputs are referenced from the combined JSON when the stages complete.

---

## Recommended execution

Preserve the initial exploratory run and write v2 into a new directory:

```bash
uv run code/base_vs_instruct.py \
  --base-model Qwen/Qwen3.5-9B-Base \
  --instruct-model Qwen/Qwen3.5-9B \
  --out out/base_vs_instruct_v2 \
  --random 19 \
  --cross-random 8
```

After the core finishes, run the expensive extensions without repeating it:

```bash
uv run code/base_vs_instruct.py \
  --base-model Qwen/Qwen3.5-9B-Base \
  --instruct-model Qwen/Qwen3.5-9B \
  --out out/base_vs_instruct_v2 \
  --extensions all \
  --extensions-only
```

Individual extensions can also be run separately:

```bash
--extensions persona
--extensions audit
--extensions ablation
--extensions sycophancy
```

This is useful because persona and 512-token generation are substantially more expensive than the token-level circuit comparison.

## Interpretation rule for the eventual v2 results

A strong post-training conclusion should require agreement among several views:

1. native effects exceed both matched null distributions;
2. Base-versus-post-trained differences persist on identical paired prompts;
3. cross-transfer asymmetry survives norm and behavioral calibration;
4. the effect contains a truth-directed component beyond generic Yes-push;
5. selected circuits do not merely destroy ordinary behavior;
6. findings extend beyond answer-token margins to longer responses or persona effects; and
7. conclusions remain consistent when deception type changes from role-play to sycophancy.

Until those results are available, v2 should be described as an implemented confirmatory analysis, not as evidence that the initial interpretation has already been confirmed.
