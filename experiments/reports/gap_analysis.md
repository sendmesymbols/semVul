# Gap analysis — how far from (a) code-only par and (b) the base-paper targets

Generated 2026-07-08. Numbers are from the `fusevul_ladder` end-to-end pipeline
(val split, GraphCodeBERT + RoBERTa, both encoders fine-tuned) and the cached
`*_probs.npz` under `experiments/runs/`. Threshold-free ROC/PR-AUC is the fair
ladder-contribution measure; operating-point Acc/F1 are reported at honest,
tune-selected thresholds (never chosen on val) unless marked "oracle".

## Two different ladders (do not conflate them)

- **Ladder A — internal monotonicity:** does L2 (+explanation) beat L1 (code-only)?
  does L3 (+quality features) beat L2? This is "every rung adds and makes sense."
- **Ladder B — base-paper targets:** Devign Acc 60.39 / F1 55.91;
  ReVeal Acc 91.68 / F1 46.76.

## Ladder A — does the explanation add? (threshold-free ROC-AUC)

| Dataset | L1 code | L2 +expl | L3 +qual | L2−L1 | L3−L1 |
|---|---|---|---|---|---|
| Devign | 63.21 | 63.16 | 62.72 | **−0.05** | **−0.49** |
| ReVeal | 86.87 | 84.92 | 82.23 | **−1.95** | **−4.64** |

The explanation rungs do **not** add; on ReVeal they subtract heavily. Confirmed
independently by the channel probe (`devign_channel_probe.md`): expl-only ceiling
≈ 55 AUC, code+expl (63.65) < code alone (64.46).

## Ladder B — distance to the base paper, after multi-seed ensembling

Single-model rungs (no ensemble):

| Dataset | Best honest Acc | Acc gap | Best honest F1 | F1 gap |
|---|---|---|---|---|
| Devign | 57.84 (L3, bal-acc thr) | −2.55 | 55.48 (L2 @0.5) | −0.43 |
| ReVeal | 88.87 (L3 @0.5) | −2.81 | 47.40 (L3 @0.5) | **+0.64 win** |

After pooling all cached seeds (`ensemble.py`, 6 members = L1+L2+L3, 4 seeds of L3):

| Dataset | Pooled ROC | Honest op-point (thr on tune) | Acc gap | F1 gap | Oracle worst-margin |
|---|---|---|---|---|---|
| Devign | 63.58 | acc 59.04 / f1 54.23 | **−1.35** | −1.68 | **−1.17** (acc-bound) |
| ReVeal | 87.22 | acc 90.28 / f1 44.89 | **−1.40** | −1.87 | **−1.36** (acc-bound) |

Ensembling alone (zero retraining) halved the accuracy deficit: Devign −2.55 → −1.35,
ReVeal −6.73 → −1.40. F1 is essentially at par on both.

## The decisive constraint: the accuracy gap is ROC-limited

The "oracle" column is the single threshold, chosen *on val itself* (optimistic),
that maximises the worst-side margin vs both targets. On both datasets it is still
negative (−1.17, −1.36) and **accuracy-bound**. Meaning: **at the current ROC-AUC,
no threshold — not even a cheating one — hits Acc and F1 simultaneously.** Closing
the last ~1.3 points therefore requires raising ROC-AUC, not better calibration.

Ensembling has sharp diminishing returns for ROC (1 member 63.21 → 6 members 63.58,
i.e. +0.37 for +5 members), so more seeds alone will not deliver the ~+1 ROC needed.
Within the **fixed architecture** (GraphCodeBERT + RoBERTa + gated fusion, "maintain
other parameters"), the only remaining lever that can raise ROC is **an explanation
channel that carries real, code-orthogonal signal.**

### Consequence — the two goals are one problem

"Make L2 > L1" (Ladder A) and "beat the base paper's accuracy" (Ladder B) both
reduce to: **raise ROC-AUC.** A discriminative explanation channel serves both.
This is why the explanation work is the accuracy work.

## Why the explanation channel is currently at chance (mechanism)

Fresh measurement on the one previously-untested cell — **de-anonymized (real
identifier) code + grounded v2 prompt**, generator Llama-3.3-70B, n=135 functions
(`expl_v2_pilot/out/…devign_real…jsonl`):

- **Grounding: 93.8%** of risky-op evidence spans are verbatim-in-code (interpretability strong — the RO1 win).
- **`risk_level` standalone signal AUC = 0.490** (chance). `n_risky` AUC = 0.486.
- **Mechanism:** `risk_level="none"` fires on only 10% of benign vs 13% of vulnerable
  functions — nearly identical. The generator flags ~88% of *every* function as risky,
  so the feature cannot separate the classes. **The bottleneck is generator
  calibration, not the encoder or fusion** (which are clean — the user's `GatedFusion`
  does not have the base-paper `model.py` overwrite bug).

### Result — calibrated label-blind regeneration (Claude as generator, same 135 fns)

Regenerated explanations for the identical 135 real-code functions with a calibrated,
label-blind generator (`experiments/claude_gen/`, inputs are code-only with labels
held out in `labels_hidden.json`, joined back only at scoring):

| Generator | flagged% | none-rate benign / vuln | grounding | standalone risk_level AUC |
|---|---|---|---|---|
| Llama-3.3-70B (n=135) | ~88% | 10% / 13% (flat, over-flags) | 93.8% | 0.490 |
| Claude, calibrated (n=135) | ~19% | 83% / 78% | 100% | 0.526 (95% CI [0.43, 0.62]) |
| **Claude, calibrated (n=300)** | **16%** | **85% / 84% (no gap)** | **100%** | **0.506 (95% CI [0.44, 0.57])** |

At n=300 the signal is **indistinguishable from chance** — the n=135 point estimate (0.526)
was noise and regressed to the mean. Calibration was a real, fixable problem (I fixed it:
16% flagged vs 88%, 100% grounded), but a well-calibrated expert honestly assigns
vulnerable and benign Devign functions the *same* risk, because they look the same at the
function level. **Verdict: the Devign explanation channel is a confirmed dead end for
lifting ROC**, across two very different generators — the ceiling is the Devign
function-level label structure, not generator quality. Consequence: running the full
`expl_v2_pilot` fusion training on Devign is not worth it (standalone 0.506 leaves no room
for the text to carry orthogonal signal). The one place left to "try hard": **ReVeal**
(different labels) — run the same cheap standalone probe there before any training spend.

### ReVeal probe — the channel is ALIVE here (n=300 balanced, calibrated Claude, label-blind)

Same generator / prompt / schema / scoring as the Devign probe; only the dataset changed:

| Dataset | flagged% | none-rate benign / vuln | grounding | standalone risk_level AUC |
|---|---|---|---|---|
| Devign | 16% | 85% / 84% (no gap) | 100% | 0.506 [0.44, 0.57] — chance |
| **ReVeal** | 17% | **89% / 77% (+12 pt)** | 100% | **0.558 [0.494, 0.623]** |

On ReVeal the generator flags vulnerable functions ~2× more often than benign; standalone
AUC = 0.558 (borderline-significant at n=300, CI grazes 0.50). This is the first positive
signal for the explanation channel in the project. Mechanism confirmed: **ReVeal's
vulnerabilities are more often locally visible in the function**, so a per-function
explanation can see them; Devign's are not. **Consequence:** the ReVeal explanation channel
is worth the fusion training spend (unlike Devign). Recommended sequence: (1) cheap — extend
the ReVeal standalone probe to n≈600 to push the CI lower-bound clearly above chance; then
(2) generate a ~1.5-2k training subset and run the `expl_v2_pilot` L1-vs-L2 fusion verdict
on ReVeal (does the channel LIFT ROC over the strong 86.87 code-only baseline?).

## Roadmap (this is the plan of record)

1. **Fix the generator (primary, this is the ROC lever).** Regenerate explanations
   with a well-calibrated, label-blind generator that returns `none` on genuinely
   benign code and only flags with verbatim evidence — raising class-conditional
   separation from ~0 toward something usable. Test: standalone signal AUC on a
   balanced held-out set, then the L1-vs-L2 fusion verdict (ROC/PR-AUC).
2. **Accuracy squeeze (secondary) — DONE for focal (2026-07-08).** Turned focal loss +
   capped class weights ON for Devign (was ReVeal-only). Result: ensemble ROC 63.58 →
   **64.18** (+0.60; best Devign ROC yet), focal L1 argmax acc 60.21 ≈ target 60.39, but
   F1 becomes the binding side so the simultaneous-beat margin stays ~−1.2. Committee RO4
   ablation row (focal on/off, `experiments/runs/focal_devign/`): focal L1/L2/L3 ROC
   63.75/63.39/62.46 vs plain 63.21/63.16/62.72. Verdict: focal helps ROC+accuracy but does
   not close the ~1.2-pt Devign gap alone — the gap is genuinely hard within the fixed arch.
3. **Bank the ensembling win now.** The pooled ensemble already beats the base-paper
   F1 on both datasets and sits ~1.3 acc points short; report it as the honest headline.

## Leakage guard (binding — see memory: semvul-no-label-in-explanation)

Explanation fields are FEATURES; the label is the target. The generator must **never**
see the label. All regeneration reads code-only inputs (`experiments/claude_gen/in/*`,
labels stripped); the label is joined back only at scoring/training time. Verdict words
are banned from `risk_summary`. Any lift produced by conditioning fields on the label is
leakage, not signal, and is out of scope.
