# ReVeal treatment: beyond-window tail digest + ReVeal-only focal knobs

**Date:** 2026-07-09
**Status:** design (awaiting user review)
**Scope:** ReVeal arm only. Devign pipeline, parameters, and files are frozen.

## Goal

Push the ReVeal explanation-channel win (+1.39 ROC, CORE fields as a separate
channel — see `memory/devign-real-enrichment-win.md`) further by applying the
mechanism that moved Devign: **inject lexical signal the code channel cannot
see.** On ReVeal the code encoder head-truncates at a 320-subword window, so the
untapped signal is the **function tail** the window drops. Two levers, both
ReVeal-scoped:

- **Lever A (enrichment):** add a label-blind `tail_digest` field built from the
  code *beyond* the encoder window. Additive — the existing whole-function
  `lexical_digest` is retained.
- **Lever B (loss reweight):** expose the already-ReVeal-only focal-loss knobs
  (`alpha_pos`, `gamma`), currently hardcoded/auto, and tune them — configured
  exclusively in `reproduce_reveal.ps1`.

**Configuration mechanism (user directive 2026-07-09):** the treatment introduces
**no new environment variables**. All new knobs (tail offset, focal α/γ, run tag)
are hardcoded literals in `reproduce_reveal.ps1`, passed as **CLI arguments** to
the Python drivers, which forward them as function kwargs. The one existing env
var reused is `SEMVUL_EXPL_FIELDS` (set from `REVEAL_FIELDS`) for text-channel
field selection — routing `tail_digest` any other way requires editing
`data_io`/`data.py`, which Devign also reads, so reuse is the Devign-safe choice.
No existing env var is removed (removal would touch Devign's path).

## Constraints (hard)

1. **Do not disturb Devign.** No change to any file, env, or code path that
   Devign's training reads. Verified touchpoints below.
2. **Respect the 9% skew.** Enrichment is label-blind (class balance unchanged).
   Val stays at natural 9% and row-order-identical (ensemble-alignable). Judge in
   **ROC + tune-selected / calibrated F1 / PR-AUC — never argmax accuracy**
   (all-negative baseline ≈ 90.8%). No resampling → no train/val distribution
   confound.

## Non-goals (explicitly parked)

- **L3 slow warmup is NOT addressed here.** L2 and L3 use identical focal config
  (same `alpha_pos` from the same data, same `gamma`); the L2→L3 gap is the raw
  44-dim quality-feature concat at `model.py:88` / `data.py:141` (no visible
  standardization), not the loss. Reweighting cannot close it. Targeted fix =
  standardize the quality block — separate work.
- **Code-side evidence window** (`SEMVUL_CODE_WINDOW=evidence`, `data.py:71`)
  is left OFF. It attacks the same tail-truncation bottleneck from the code
  channel; enabling it would make the enrichment lift non-isolable. Available to
  toggle later.
- Devign tail digest, de-anon changes, resampling/oversampling.

## Design

### Lever A — additive `tail_digest`

**Where:** `experiments/expl_enrich/apply_real_enrichment.py`, inside the
`deanon=False` (ReVeal) branch of `enrich_row` only.

**What:** a new helper `tail_digest_fields(raw_code, offset)` that reuses the
existing shared read-only utilities (`strip_comments`, `toks`, `C_KEYWORDS`,
`RISKY_APIS`) but extracts callees / risky APIs / string literals from
`toks(...)[offset:]` — the span past the window. It returns a digest string of
the same shape as `digest_fields` but prefixed (`tail_calls …`,
`tail_risky_apis …`, `tail_literals …`). The ReVeal branch sets
`expl["tail_digest"] = <string>`. **`digest_fields` itself is not modified**, so
the Devign branch (`deanon=True`) is byte-identical.

- **Offset:** `--tail-offset` CLI arg on `apply_real_enrichment.py`, default
  module constant `TAIL_OFFSET_TOKENS = 220` regex tokens (≈ 320 GraphCodeBERT
  subwords at ~1.45 subwords/token); the literal `220` is passed from
  `reproduce_reveal.ps1`. This is an approximation by design (the "additive,
  no-tokenizer" option chosen over exact boundary detection). Because the
  whole-function `lexical_digest` is retained, an off-by-some offset only shifts
  how much of the tail is *duplicated* vs *unique* — it cannot drop signal.
- **Short functions** (≤ offset tokens) get an empty `tail_digest`. The lift
  comes exclusively from long/truncated functions — ReVeal's affected set
  (`data.py:20` notes ~58% of vulnerables are truncated).
- **Field selection.** `tail_digest` is added to the JSONL but only enters the
  text channel when listed in `REVEAL_FIELDS` (→ existing `SEMVUL_EXPL_FIELDS`).
  `data_io` renders unknown keys verbatim, so **no change to `data_io`,
  `data.py`, or `train.py`** is needed for the field to flow. No new env var.

**File regeneration:** re-run `apply_real_enrichment.py` to add `tail_digest` to
the ReVeal `.real.jsonl` files. Adding the field is **backward-compatible**:
baseline runs whose `REVEAL_FIELDS` omit `tail_digest` ignore it, so overwriting
does not change baseline behavior, and baseline members already trained overnight
remain valid A/B baselines. To guarantee Devign files are never rewritten, add an
`--only {reveal,devign}` flag to `apply_real_enrichment.py` and use `--only reveal`.

**Toggle for the A/B:** `REVEAL_FIELDS` in `reproduce_real.py` gains `tail_digest`
for the treated arm; the baseline arm omits it. `DEVIGN_FIELDS` untouched.

### Lever B — ReVeal-only focal knobs

**Where:** `experiments/fusevul_ladder/train.py`, inside the `use_focal` branch
only (`use_focal` is `dataset == "reveal"` by default; Devign uses plain
`cross_entropy` and never enters this branch).

**What:** `train_rung` gains kwargs `focal_alpha=None, focal_gamma=2.0`. In the
focal branch:
```python
gamma = focal_gamma
if focal_alpha is not None:
    alpha_pos = float(focal_alpha)
# loss call: focal_ce(logits, yb, alpha_pos, gamma)
```
Defaults reproduce current behavior exactly (`focal_alpha=None` → `alpha_pos`
from the `clip(1-pos_rate,…)` formula = 0.80 for ReVeal; `focal_gamma=2.0`).
Record `focal_gamma` alongside the already-logged `use_focal` / `alpha_pos` in
the payload `config`. **No env reads.**

**Plumbing:** `reproduce_real.py` gains `--focal-alpha` / `--focal-gamma` CLI
args and forwards them **only to the ReVeal job** (via that job's kwargs dict);
the Devign job never receives them. `reproduce_reveal.ps1` passes the hardcoded
literals; `reproduce_devign.ps1` is not touched. Sweep grid: α ∈ {0.80, 0.85,
0.90}, γ ∈ {2, 3}, **selected on tune PR-AUC, never val**, each into a distinct
`--out-tag` dir (below).

### Isolation guarantee (Devign untouched — three independent barriers on Lever B)

| Barrier | Mechanism |
|---|---|
| Focal off for Devign | `use_focal = (dataset=="reveal")`; Devign uses plain CE |
| CLI arg to ReVeal job only | `reproduce_real.py` forwards `focal_alpha/gamma` only into the ReVeal job's kwargs; Devign job dict never gets them |
| Applied only in focal branch | `alpha_pos`/`gamma` override lives inside `if use_focal` |

Lever A: `digest_fields` (the shared function) is unmodified; the new field is
written only in the ReVeal branch; `apply_real_enrichment.py --only reveal`
never rewrites Devign files; `DEVIGN_FIELDS` unchanged.

**Files touched:** `apply_real_enrichment.py` (ReVeal branch + `--only` /
`--tail-offset` flags), `reproduce_real.py` (`tail_digest` in `REVEAL_FIELDS`;
`--focal-alpha` / `--focal-gamma` / `--out-tag` args forwarded to the ReVeal job),
`train.py` (`focal_alpha` / `focal_gamma` kwargs, focal branch),
`reproduce_reveal.ps1` (hardcoded literals). Nothing Devign reads is modified.

## Measurement plan

**Clean A/B, not pooled.** `dual_eval` averages all `runs/enriched*` members into
one ensemble, which would blend arms — do **not** read the ENSEMBLE row as the
verdict. Use:

1. **Per-member rows** from `dual_eval` (each config listed individually) for the
   headline ROC / PR / F1 per arm.
2. **Paired bootstrap ROC-delta** over the two matched saved `val_prob` vectors
   (same rung, same seed, baseline vs treated). Val rows are identical and
   order-aligned (`*_val.enriched.real.jsonl` unchanged), so the pairing is valid.
   Report the delta with a 95% CI (mirrors the +1.39 CI [+0.31,+2.63] framing).
   This is a small new script over the `_probs.npz` files.

**Metric discipline:** headline = val ROC + tune-selected F1 / PR-AUC. Argmax
accuracy is reported only as the base-paper-comparability column, never as the
verdict.

## Experiment staging (bounded by GPU cost)

ReVeal is ~3.6 h / rung / seed, so run a staged, one-variable-at-a-time matrix
and **log what is deferred** (no silent caps):

1. **Lever A, L2, seed 1337:** treated (baseline focal + `tail_digest`) vs the
   overnight baseline L2 member. One variable = the tail field. Paired-bootstrap
   ROC delta. This is the fastest real signal and where the +1.39 lives.
2. If A shows lift → **Lever B sweep, L2, seed 1337** (focal grid, no tail),
   tune-PR-AUC select, then optionally combine winner-A + winner-B.
3. Expand to **seed 2024** and **L3** only for configs that showed lift at L2;
   L3 is expected to warm up slowly regardless (see Non-goals).

Distinct focal/tail configs write to distinct `runs/enriched_real*/` subdirs via
`--out-tag` (hardcoded per experiment in the ps1) so runs don't clobber and
members stay individually scannable by `dual_eval`'s `runs/enriched*` glob.

## Risks & honesty caveats

- **Offset is approximate** (regex tokens ≠ GraphCodeBERT subwords). Mitigated by
  the additive design; documented, and `--tail-offset`-tunable from the ps1.
- **Config-on-val risk:** as with the original +1.39, if the focal grid is
  selected on anything val-derived it is optimistic. Selection is pinned to tune
  PR-AUC; the paired bootstrap is the honest readout.
- **Coverage:** ReVeal `.real.jsonl` is full-coverage (deanon=False path treats
  100%), so unlike Devign there is no ~67% coverage caveat for this arm.
- **tail_digest may be inert** if ReVeal's truncated tails carry little unique
  lexical signal — the cheap per-member A/B settles this before scaling to all
  rungs/seeds.

## Verification

- Regeneration is idempotent; assert ReVeal `.real.jsonl` row counts and order
  are unchanged vs the pre-treatment files (only the added key differs), and that
  Devign `.real.jsonl` files are byte-identical (unchanged mtime / hash) after
  `--only reveal`.
- Assert baseline-arm training with `REVEAL_FIELDS` lacking `tail_digest`
  produces the same text channel as before (field is inert unless selected).
- Confirm Devign payload `config.use_focal == False` unchanged; ReVeal payload
  records the swept `alpha_pos` / `gamma`.
