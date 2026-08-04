# Devign deep-dive — where the joint gap actually lives

Date: 2026-07-08. Val n=2732, all 9 saved probability members
(desktop L1–L3, seeds s2024–26 L3, laptop L1–L3); ensemble = probability mean.

## 1. The gap is a truncation problem, not a fusion problem

Stratify val by code length vs the 320-token (~230-word) encoder window:

| stratum | n | ens ROC | ens acc@0.5 | desk_L1 ROC |
|---|---|---|---|---|
| fits the 320-token window | 1833 (67%) | **67.75** | **61.92** | 65.82 |
| truncated at 320, fits 512 | 402 (15%) | **53.03** | 50.50 | 54.18 |
| exceeds even 512 | 497 (18%) | 59.08 | 53.32 | 61.31 |

On the two-thirds of val the encoder can actually read, the current ensemble
**already beats the stated target** (61.92 vs 60.39 acc; ROC 67.75). The
truncated-at-320 stratum is at coin-flip. Truncated samples are 33% of val but
38% of ensemble errors. If the truncated stratum merely matched the short-code
stratum, overall acc ≈ 61.9 → target cleared.

Supporting evidence that window length (not the text channel) is the lever:
frozen GraphCodeBERT-LoRA embeddings built at max_len=512 score ROC ≈ 57 on
the same truncated stratum where the 320-token end-to-end rungs get 53–54.
Conversely, adding the enriched explanation embedding does NOT rescue the
truncated stratum in probe space (code 56.95 → code+enr 55.89): pooled MiniLM
vectors of tail summaries are too coarse; the code encoder needs the tokens.

## 2. Label-noise ceiling (small but real)

- 54 val rows (1.98%) have an identical-code row elsewhere (train or val) with
  the OPPOSITE label; ensemble acc on them is 42.6% vs 59.0% on clean rows.
- 4 identical-code groups WITHIN val carry both labels (cannot get both right).
- Cost ≈ 0.3–0.5 acc points of irreducible ceiling. Val stays untouched
  (benchmark comparability), but this bounds expectations.

## 3. Persistent errors

213 samples (7.8%) are misclassified by ALL 9 members: 146 vulnerable vs 67
clean (a recall floor), slightly longer (275 vs 255 words), 1.7× the
contradicted-label rate, and the ensemble is measurably less confident on them
(|p−0.5| = 0.25 vs 0.33) — consistent with a mix of label noise and genuinely
out-of-window evidence.

## 4. Treatment queued (2026-07-08)

`experiments/expl_enrich/retrain_devign512.py` — waits for the ReVeal enriched
retrain to free the GPU, then trains Devign L1/L2/L3 with:
- `max_code=512` (batch 2 × grad-accum 16 keeps the effective batch of the
  320-token baseline within 8 GB),
- train = `devign_train.enriched.clean.aug.jsonl` (conflict/leak-cleaned +
  VARn/FUNn-permutation augmented, 41,316 rows),
- enriched explanations + 44-dim qual_v2,
- outputs to `experiments/runs/enriched512/` (baselines untouched; resumable).

After it lands: `python experiments/fusevul_ladder/ensemble.py` now auto-scans
`runs/enriched*/` members into the pooled ensemble (val-aligned; their tune
slices differ in size and are auto-skipped for threshold selection).

Follow-up lever if 512 is not enough for the >512-word stratum (18% of val,
ROC 59): head+tail truncation or sliding-window pooling over 512-token chunks.
