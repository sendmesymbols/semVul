# Ablation Study Protocol — SemVul Component Ladder

**Date:** 2026-07-05
**Author:** abdul
**Status:** Protocol for supervisor sign-off (forward-looking — result tables are placeholders)
**Scope:** Core component ladder + within-rung leave-one-out, on Devign and Reveal.

---

## 1. Objective & scope

The SemVul detector combines three components on top of a fine-tuned code encoder:

- **C — Code channel:** GraphCodeBERT, fine-tuned; token-level representation of the (normalized) function.
- **E — Explanation channel + fusion:** a fine-tuned RoBERTa encoder over the LLM-generated local
  explanation, fused with the code tokens by multi-head **self-attention**.
- **Q — Quality features:** 22 structured features distilled from the explanation, concatenated at the head.

This study **isolates the marginal contribution of each component** (C, E, Q) and of the **fusion
mechanism** (self-attention vs. naive concatenation), on **both** benchmark datasets, and benchmarks
the full model against FuSEVul's stated results.

**In scope:** the component ladder and its within-rung leave-one-out. **Out of scope** (noted only as
contingency pointers): explanation v1-vs-v2 content, evidence-token variants, ensemble/L4, and the
raw channel probe — these are separate studies, not part of this protocol.

**Reference targets (FuSEVul, stated):**

| Dataset | Acc | F1 | Precision | Recall |
|---|---|---|---|---|
| Devign | 60.39 | 55.91 | — | — |
| Reveal | 91.68 | 46.76 | 57.24 | 39.52 |

---

## 2. Research questions & pre-registered hypotheses

The hypotheses are fixed **before** the runs. Each maps to specific configuration comparisons
(defined in §3). All configurations are reported regardless of outcome (§8).

| RQ | Question | Comparison | Hypothesis (H) |
|---|---|---|---|
| **RQ1** | Does adding each component improve detection? (ladder) | L1 → L2 → L3 | monotone **L1 < L2 < L3** on Acc & F1, both datasets |
| **RQ2** | Does the LLM-explanation channel contribute? | full vs. **−expl** | **full > −expl** |
| **RQ3** | Do the 22 quality features add signal beyond the explanation? | full vs. **−QF** | **full > −QF** |
| **RQ4** | Does self-attention fusion beat naive concatenation? | full vs. **concat** | **full > concat** |

RQ1 is the **additive (bottom-up)** view — the ladder narrative. RQ2–RQ4 are the **leave-one-out
(top-down)** view — each component's marginal value *in the presence of the others*. Reporting both
is deliberate: the ladder shows components *help when stacked*; leave-one-out shows each is *still
necessary in the full model*.

---

## 3. Configurations under test

Components present in each configuration (C = code, E = explanation+fusion, Q = quality features):

| Config | C | E (fusion) | Q | Role |
|---|:---:|:---:|:---:|---|
| **L1** | ✓ | — | — | ladder base (code-only) |
| **L2** | ✓ | ✓ (self-attn) | — | ladder + explanation |
| **L3 = full** | ✓ | ✓ (self-attn) | ✓ | ladder + QF / **full model** |
| **−expl** | ✓ | — | ✓ | leave-one-out: remove explanation |
| **−QF** | ✓ | ✓ (self-attn) | — | leave-one-out: remove QF *(≡ L2)* |
| **concat** | ✓ | ✓ (**concat**) | ✓ | fusion variant: self-attn → concatenation |

**Distinct trainings:** `{L1, L2 (≡ −QF), L3=full, −expl, concat}` = **5 configurations per dataset.**
`−QF` is intentionally identical to `L2`, so it is trained once and read into both the ladder table
and the leave-one-out table.

---

## 4. Datasets & experimental protocol

- **Datasets:** Devign, Reveal.
- **Split & comparability:** evaluate on the **same benchmark validation split** FuSEVul used, so
  "beats stated results" is a direct same-benchmark comparison.
- **Leakage control:** drop within-train duplicate functions and any **train** function that also
  appears in **val** — applied to **TRAIN only**; the val set is left byte-identical to the benchmark.
  Alignment and dedup key: `sample_id = sha1(normalized_code)[:16]`.
- **Imbalance (Reveal only):** focal loss + capped class weight; Devign uses plain cross-entropy.
- **Invariance:** across configurations, the datasets, splits, tokenization, optimizer, schedule, and
  epoch budget are held **identical** — only the ablated component changes. This is what makes any
  measured delta attributable to the component.

---

## 5. Metrics & decision thresholds

**Threshold-free (primary ablation headline).** ROC-AUC and PR-AUC on val. These are the fair
measure of a component's contribution because they are immune to threshold gaming — RQ1–RQ4 verdicts
are decided here first.

**Comparability metrics (vs. FuSEVul).** Accuracy, F1, Precision, Recall on val, reported under a
transparent set of decision-threshold policies:

| Policy | Purpose |
|---|---|
| fixed 0.5 | direct comparability to FuSEVul's reported operating point |
| max balanced-accuracy | balanced honest headline |
| max F1 | best-F1 operating point |
| both-tuned (on tune slice) | joint Acc+F1 sweet spot |

**Non-circular selection.** The decision threshold *and* the early-stopping epoch are chosen on a
**stratified tune slice carved from TRAIN** (`split_seed = 1337`, fixed across seeds), never on val.
Val is used only for the final reported numbers.

**Reveal caveat (stated up front).** Reveal is ~90.8% negative, so accuracy ≈ the majority-class base
rate and is near-uninformative; **F1 is the meaningful metric on Reveal** and is reported as such.

---

## 6. Statistical rigor

- **5 seeds per configuration.** Report **mean ± std** for every metric.
- **Noise floor.** Single-seed rung deltas of ≈ ±2 F1 are within seed noise. A component is judged to
  **contribute** only if its delta **exceeds the seed std** and survives a **paired comparison across
  the shared seeds** (paired because configs share seeds/splits).
- Optional: bootstrap confidence interval on the val metric for the full-vs-leave-one-out deltas.

---

## 7. Controls & validity checks

- **Gradient check:** in `L1` and `−expl`, the explanation encoder must receive **zero gradient** —
  asserts the channel is genuinely off, not merely down-weighted.
- **Flow check:** `L2` output must differ from `L1` output — confirms the explanation actually reaches
  the head (guards against the FuSEVul-style bug where the fused vector is discarded).
- **Shared-input check:** where a toggle only changes the head input, the upstream cached
  representations are reused so the comparison is apples-to-apples.
- **Sanity:** `L1` lands near a plausible code-only baseline before any ablation deltas are trusted.

---

## 8. Pre-registration & honest reporting

Hypotheses (§2) are fixed before running. **Every configuration is reported regardless of whether it
supports its hypothesis**, including null and negative results — e.g. if `−expl ≥ full` on a dataset,
that is reported as *the explanation channel not contributing on that dataset*, with the
threshold-free AUC as the arbiter. This pre-registration is a deliberate strength: the study commits
to its verdict before seeing the numbers, and the contribution of each component is reported as
measured rather than as hoped. (This concerns **our own components**; it is not a comparison against
or critique of FuSEVul's protocol.)

---

## 9. Run matrix & budget

- **Runs:** 5 configurations × 2 datasets × 5 seeds = **50 training runs.**
- **Status:** the L3/full and single-seed ladder rows already exist; the outstanding work is the
  **5-seed L1/L2 rows** and the leave-one-out (`−expl`, `concat`) rows at 5 seeds.
- **Hardware:** 8 GB VRAM; bf16 + gradient checkpointing + small batch + grad-accum; resumable
  (runs skip if their JSON/NPZ already exist).
- **Estimated cost:** _(fill after a single-run timing: per-run wall-clock × 50)_.

---

## 10. Deliverables

For **each dataset**, two tables benchmarked against FuSEVul stated numbers:

**(a) Ladder table (RQ1)** — one row per rung.

| Rung | ROC-AUC | PR-AUC | Acc@0.5 | F1@0.5 | P | R | vs. FuSEVul |
|---|---|---|---|---|---|---|---|
| L1 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | — |
| L2 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | — |
| L3 (full) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | Acc Δ / F1 Δ |

**(b) Leave-one-out table (RQ2–RQ4)** — deltas vs. full (negative Δ = the component helps).

| Config | ROC-AUC (Δ) | PR-AUC (Δ) | F1@0.5 (Δ) | Contributes? |
|---|---|---|---|---|
| full (L3) | _pending_ | _pending_ | _pending_ | — |
| −expl (RQ2) | _pending_ | _pending_ | _pending_ | _pending_ |
| −QF (RQ3) | _pending_ | _pending_ | _pending_ | _pending_ |
| concat (RQ4) | _pending_ | _pending_ | _pending_ | _pending_ |

Plus: a one-line **verdict per RQ** (supported / not supported, with the deciding AUC delta and
whether it clears the seed-noise floor), and go/no-go notes on any contingency (e.g. explanation
regeneration) that a null RQ2 would trigger.

---

## 11. Mapping to existing artifacts

The protocol reuses machinery already in the repo — it does not require new infrastructure:

| Protocol element | Produced by |
|---|---|
| Ladder table (RQ1) | `experiments/reports/{devign,reveal}_ladder_progression.md` |
| Leave-one-out table (RQ2–RQ4) | `experiments/reports/{devign,reveal}_L*_component_ablation.md` |
| Training / seeds / tune-slice | `experiments/fusevul_ladder/{train.py, run_ladder.py}` |
| Threshold-free AUC + threshold policies | run JSON + saved val/tune probs |

Outstanding: extend the ladder runs to **5 seeds for L1/L2** and add the leave-one-out configs at 5
seeds, then regenerate the two tables per dataset.
