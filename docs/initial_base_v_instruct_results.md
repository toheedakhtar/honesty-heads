The strongest result is that post-training appears to preserve a transferable honesty-related activation direction, while redistributing which individual heads carry the effect.

| Result at α = 2 | Base | Post-trained |
|---|---:|---:|
| Selected heads | 42 | 60 |
| Held-out lies moved to truth | 14/29 = 48.3% | 12/34 = 35.3% |
| All-head intervention | 17/29 = 58.6% | 20/34 = 58.8% |
| Random matched heads | 1/29 = 3.4% | 2/34 = 5.9% |
| Random direction | 1/29 = 3.4% | 3/34 = 8.8% |

The full outputs are in base_vs_instruct.json (out/base_vs_instruct/base_vs_instruct.json) and base_vs_instruct.png (out/base_vs_instruct/base_vs_instruct.png).

## What the results suggest

### 1. Both models contain a causal, steerable signal

The selected circuits substantially outperform the random-head and random-direction controls in both checkpoints. This suggests the effect is not simply caused by adding arbitrary activation noise or modifying any heads from the same layers.

This is evidence for a causally useful truth-versus-lie direction in both the pretrained and post-trained models.

### 2. Post-training did not create the representation from scratch

The mean per-layer cosine similarity between the two learned directions is 0.814, which is quite high.

That means the broad honest-minus-lie activation geometry is similar before and after post-training. A plausible interpretation is:

pretraining learns much of the representation
             ↓
post-training modifies how it is used and distributed

It does not prove that post-training has no effect; it suggests continuity rather than complete replacement.

### 3. The same broad direction is implemented by different sparse circuits

Only 14 selected heads overlap:

Base set:         42 heads
Post-trained set: 60 heads
Exact overlap:    14 heads
Jaccard overlap:  0.159

High direction similarity combined with low exact head overlap suggests that post-training may preserve the global representation while redistributing causal responsibility across heads.

However, head-set instability can also arise from correlated or interchangeable heads, selection noise, and the ranking procedure. We should not immediately interpret every changed head as mechanistic rewiring.

### 4. Cross-transfer is real and asymmetric

At α = 2:

| Transfer | Selected circuit | All heads |
|---|---:|---:|
| Base direction → post-trained model | 18/34 = 52.9% | 22/34 = 64.7% |
| Post-trained direction → Base model | 11/29 = 37.9% | 15/29 = 51.7% |

The Base direction transfers particularly well into the post-trained model—better than the post-trained model’s native selected direction in raw lie-removal rate.

This could mean the Base direction captures a broadly reusable representation that remains accessible after post-training. But there is an important confound: the two directions may have different activation norms. At the same α, one may simply constitute a larger intervention. Cross-transfer must be repeated after norm or behavioral-effect calibration before concluding that the Base direction is intrinsically better.

### 5. The direction is bidirectional

The experiment reverses the intervention on initially truthful/resistant cases. At α = 2:

- Base native direction induces lie preference on 4/14 cases.
- Post-trained native direction induces it on 8/16.
- Base → post-trained transfer induces it on 9/16.

This strengthens the causal interpretation: the direction does not only suppress a few existing lie tokens; moving the opposite way can push truthful cases toward lies.

It is also a safety warning. This is a behavioral control direction, not an honesty-only feature.

## What we cannot conclude yet

We cannot confidently say that the Base model is “more steerable” than the post-trained model because:

- The eligible test sets differ: 29 Base lies versus 34 post-trained lies.
- These are not necessarily the same prompts.
- Only one random-head set and one random direction were tested.
- The cross-model directions were not norm-calibrated.
- The new script currently ranks using the conservative minimum of lie and resist attribution. The original repository considers both mean and minimum rankings and chooses the smallest successful set. This likely helps explain why the post-trained run selected 60 heads rather than reproducing the repository’s 29.
- The Yes-bias decomposition has not yet been run on these new directions.
- Confidence intervals and paired cross-model tests are not yet reported.
- Samples remain small.

## Highest-value next steps

1. Make selection exactly match the repository. Evaluate both mean and minimum integrated-gradient rankings and select the smallest successful prefix. This is needed before comparing the new 60-head post-trained set to the original 29-head result.
2. Use shared prompts for the model comparison. Report:
   - model-specific eligible lies;
   - prompts where both models lie;
   - all shared prompts, using continuous truth-minus-lie margin changes.

   The shared subset enables paired bootstrap, permutation, and McNemar comparisons.
3. Calibrate cross-transfer strength. Compare directions using:
   - equal per-head norms;
   - equal total intervention norm;
   - equal selection-set behavioral effect;
   - full dose-response curves.

   This will determine whether Base → post-trained transfer is genuinely stronger or merely larger.
4. Run 19–39 null interventions. Use multiple layer-matched head sets and norm-matched directions. This enables empirical randomization p-values instead of comparison against a single random draw.
5. Repeat the Yes-bias analysis. Decompose each native and transferred direction into:
   - truth-directed effect;
   - generic Yes-push;
   - effects on non-Yes/No answers.

   This is essential because the original study found a substantial Yes component.
6. Add exact intervals and paired tests. Report Clopper–Pearson intervals for rates, paired bootstrap intervals for margin differences, and within-prompt permutation tests for Base versus post-trained effects.
7. Extend beyond answer tokens. Run both checkpoints on:
   - insider-trading free generation;
   - persona evaluations;
   - ordinary-task controls;
   - head ablation;
   - sycophancy-derived directions.

My current best interpretation is: pretraining already contains a strong truth-versus-lie activation geometry; post-training preserves much of that geometry but changes its sparse causal implementation and behavioral accessibility. That is an interesting result, but calibration, paired prompts, multiple null directions, and the Yes-bias analysis are needed before treating it as a robust post-training finding.
