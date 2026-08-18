# Statistical gap analysis — explanation rungs vs code-only vs FuSEVul targets

Date: 2026-07-08. Sources: saved val/tune probabilities in `experiments/runs/`
(desktop + 3 seed reruns + laptop zip), explanation JSONLs, dataset CSVs.
All CIs are 2000-rep bootstraps; rung comparisons are exact McNemar tests on
identical val samples.

## 1. How much is missing (the headline numbers)

The stated FuSEVul targets are a JOINT operating point: one threshold must
deliver both acc and F1. Best joint shortfall found at ANY threshold
(val-oracle sweep = optimistic upper bound of current models):

| dataset | best single rung | best ensemble (9 members) | joint shortfall |
|---|---|---|---|
| devign (60.39 / 55.91) | L3: min-margin −2.24 | desk_L1+s2025_L3: **−0.95** | ≈ 26–45 flips of 2732 |
| reveal (91.68 / 46.76) | L3: min-margin −1.49 | all-9 @thr 0.60: acc 91.03, F1 47.15 → **−0.65 acc only** | ≈ 15 flips of 2273 |

- Devign: every rung's ROC-AUC sits in 61.5–63.6 (nine runs). The joint target
  needs roughly ROC ≈ 65.5–66. Missing ≈ **+1.5–2 ROC points of genuine
  ranking**, i.e. ~1.0–1.6 joint acc/F1 points.
- Reveal: F1 target is already beaten (48.7 @0.5 by the 5×L3 ensemble;
  47.15 at the acc-max threshold). Missing ≈ **+0.65 accuracy** while holding
  F1 — about 15 net flips.

## 2. What the explanation rungs currently contribute (statistically)

Paired bootstrap ΔROC-AUC on val (positive = rung helps):

| dataset | L2−L1 | L3−L1 | verdict |
|---|---|---|---|
| devign | −0.02 [−1.51, +1.40], p≈0.96 | −0.48 [−2.03, +1.04], p≈0.55 | **exactly zero** |
| reveal | −1.93 [−3.94, −0.08], p≈0.04 | −4.66 [−7.56, −2.01], p≈0.001 | **significantly negative** |

McNemar on reveal confirms the rungs change different samples (L3 fixes 142 of
L1's 342 errors, p<0.001) but the *ranking* gets worse — the accuracy gain
@0.5 is an operating-point shift (recall 69→54), not information gain.

## 3. Why: the explanation channel carries (almost) no label signal

TF-IDF + logistic regression, train→val, on the explanation text itself:

| channel | devign ROC | reveal ROC |
|---|---|---|
| full explanation text | **53.5** | **79.8** |
| best single field | purpose 54.8 | data_flow 78.7 |
| code (same probe, reference) | 57.8 | 85.5 |
| explanation on samples the code model gets WRONG | **38.3** | **25.0** |

The last row is the killer: exactly where help is needed, the explanation
signal is *anti-correlated*. Fusion can therefore only learn to ignore the
channel (devign) or gets dragged (reveal).

Root causes found:
1. **Devign code is 100% VARn/FUNn-anonymized** (reveal 0%). The generator
   hallucinated semantics ("pointer arithmetic on buf" with no `buf` in code).
2. **Verdict fields are inverted**: "no risky operations" appears in
   risk_summary for 18.7% of *vulnerable* vs 22.0% of clean devign train rows
   (reveal: 27.7% vuln vs 22.7% clean — wrong direction).
3. Boilerplate: 506 identical "no risky operations..." summaries in devign
   val; reveal has 70% empty missing_checks, 1.8% placeholder explanations,
   and 409 cross-label duplicate explanation rows in train.
4. expl_v2 pilot (Gemma 4 31B, calibrated prompt) lifted standalone signal
   only to AUC 0.549–0.551 — the ceiling is the anonymized input, not the
   prompt.

## 4. Dataset findings that motivate cleaning/augmentation

| issue | devign | reveal |
|---|---|---|
| train code strings carrying BOTH labels | 185 groups (418 rows) | 213 groups (426 rows) |
| train rows exact-code-equal to a val row (different sample_id → survives the sample_id dedup) | 300 | 69 |
| same-label exact duplicates | 496 | 0 |
| functions longer than the 320-token code window | 33.3% | 29.5% |
| duplicate sample_ids in train JSONL | 845 | — |

Code length alone has AUC 70.8 on reveal (66.6 for the truncated flag) — a
strong feature the 22 v1 quality features did not include.

## 5. What was built in response

- A legacy deterministic post-generation enrichment pipeline produced grounded
  findings, guard indicators, code metrics, calibrated risk levels, and tail
  summaries. That pipeline has been removed; its outputs are incompatible with
  the clean Qwen-only ACTIVE contract and must not be used for current results.
- `experiments/expl_enrich/augment_train.py` — train-only cleaning
  (conflicting labels, exact-code val leaks, same-label dups → devign 21837→20658,
  reveal 18187→17692) + devign VARn/FUNn permutation augmentation
  (`*.clean.aug.jsonl`, 41316 rows; exactly label-preserving).
- `src/quality_features_v2.py` — 44-dim quality block (v1 22 + static 22).
- Env-gated integration (defaults unchanged): `SEMVUL_EXPL_VARIANT=enriched`,
  `SEMVUL_TRAIN_SUFFIX=clean.aug`, `SEMVUL_QUAL_V2=1` in the ladder.
- Gate results: `experiments/reports/enrich_gate_probe.md` (frozen-embedding
  probe; run `experiments/expl_enrich/gate_probe.py`).

Enriched-text standalone signal (TF-IDF): devign 53.5→54.8, reveal 79.8→81.9.

## 6. Honest read + recommended sequence

1. **Reveal is winnable now**: retrain L2/L3 with `SEMVUL_EXPL_VARIANT=enriched
   SEMVUL_QUAL_V2=1 SEMVUL_TRAIN_SUFFIX=clean` and ensemble 3–5 seeds; pick the
   joint threshold on the tune slice. The 0.65-acc gap is ~1 ensemble member of
   variance away; n_words/truncated features alone are worth trying.
2. **Devign needs ranking, not explanations**: raise `--max-code` to 512
   (33% of functions are truncated; single biggest untapped lever), train on
   `clean.aug`, ensemble seeds. Explanations from anonymized code are
   information-theoretically capped (best measured standalone AUC ≈ 0.55–0.57);
   the rung can be made *non-negative* with enrichment, but the acc gap must
   come from the code channel.
3. Re-run `run_ladder.py` per dataset and re-check with
   `experiments/expl_enrich/gate_probe.py` before burning GPU-hours on any new
   explanation generator: if the frozen probe doesn't move, training won't.
