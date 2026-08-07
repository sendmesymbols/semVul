# Devign real-code explanation enrichment (label-blind) — gate results

**Date:** 2026-07-08
**Goal:** make the explanation channel carry the real-code lexical signal that Devign's
identifier normalization strips, WITHOUT ever using a sample's label to author its text.

## Why the previous gate failed

Real-code-derived explanations (static, label-blind) added nothing over anon code
(58.46 vs 58.71 CV ROC) because the explanation *prose* described patterns but the
*tokens* were still `VARn`/`FUNn` — the +ROC signal from real code is lexical
(real function names, called APIs, literals), and the text never contained it.

## The two enhancements (both deterministic, label-blind)

1. **De-anonymization** — align `anon_code` ↔ `raw_code` token streams per function
   (lockstep for 99.3% of functions; difflib opcodes fallback) to recover the
   `VARn/FUNn → real identifier` map, then substitute real names into every
   explanation field (`purpose`, `data_flow`, `risky_operations`, `missing_checks`,
   `evidence_tokens`, `risk_summary`).
2. **Lexical digest** — extract from `raw_code`: function name, top-30 callees,
   risky-API hits, up to 10 string literals. Stored as structured fields
   (`function_name`, `called_functions`, `risky_apis`, `string_literals`,
   `lexical_digest`).

## Gate (TF-IDF 1-2gram LR; labels used only for scoring)

### 5-fold CV on devign_real val (n=1829)

| features | ROC |
|---|---|
| anon code only (benchmark input) | 55.81 |
| anon + deanon explanation | 58.40 |
| anon + digest | 57.10 |
| **anon + deanon explanation + digest** | **59.31** |
| anon + real code (upper bound) | 59.51 |
| deanon explanation + digest alone (no code) | 59.32 |

### train (n=14,752) → val (n=1829) — the decisive setting

| features | ROC |
|---|---|
| anon code only | 59.20 |
| anon + deanon explanation | 64.59 |
| anon + digest | 64.82 |
| **anon + deanon explanation + digest** | **66.65 (+7.45)** |
| anon + real code (upper bound) | 69.77 |
| real code only (reference) | 71.91 |

**Conclusion:** the enriched explanation channel recovers ~70% of the real-code gap
(+7.45 of +10.6 ROC) while the code input stays the normalized benchmark input.
On CV it recovers essentially all of it. This is the first setting in the project
where the explanation channel adds material signal over the code channel — because
it injects information the code input physically lacks, rather than paraphrasing it.

## Integrity notes (for the thesis)

- Text generation never sees the label: de-anonymization and digest are deterministic
  functions of the code only. Labels are copied through for training/scoring.
- Covers the devign_real subset only (~67% of the benchmark val set, 1829/16,581
  functions aligned by `sample_id`). Any benchmark-level claim must report this as a
  representation change on that subset, not apples-to-apples vs FuSEVul's stated input.
- TF-IDF LR is a floor; the fused encoder on the desktop run may extract more or less.

## Benchmark-facing deployment (added later same day)

All six benchmark Devign files were treated in place-preserving fashion
(`devign_{train,val}.{enriched.clean,clean}[.aug].real.jsonl`): samples with real
code (65.6% of val, 66.6% of train rows incl. aug) get de-anonymized explanation
text + digest keys and a `real_enrich: "deanon+digest-v1"` tag; uncovered samples
pass through byte-identical; the code input stays the anonymized benchmark input.

## ReVeal control (same treatment, gated)

ReVeal was never anonymized — its code channel already carries real identifiers —
so only the digest lever applies. Gate (TF-IDF LR, train 18,187 → val 2,273, ~10% pos):

| features | ROC |
|---|---|
| real code only (benchmark input) | 85.42 |
| code + enriched explanation | 85.05 |
| code + digest | 85.89 |
| code + explanation + digest | 85.24 |
| explanation + digest alone | 83.93 |

Neutral, as the mechanism predicts: the digest tokens are already in the code input.
This is the clean contrast for the thesis — the explanation channel adds signal
exactly when (and because) the code representation is information-poor (Devign
normalization), and is redundant when the code is already real (ReVeal).
ReVeal files were still treated (`reveal_{train,val}.{enriched.clean,clean}.real.jsonl`,
`real_enrich: "digest-v1"`, 100% coverage) so both datasets share one schema.

## ReVeal lift (follow-up to the neutral control)

The -0.4 from "code + explanation" was fixable. Field ablation + channel-separation
(TF-IDF LR, train 17,692 → val 2,196, clean fold):

| config | ROC |
|---|---|
| code only | 84.69 |
| code + ALL fields (concatenated) | 85.07 |
| code + CORE fields (concatenated) | 85.37 |
| **code (+) CORE, separate vectorizers** | **86.09 (+1.39, CI [+0.31, +2.63], P>0 = .997)** |
| window-proxy (code[:512]) only | 84.44 |
| window-proxy + CORE | 85.89 (+1.45) |

CORE = risky_operations, missing_checks, evidence_tokens, safety_indicators,
tail_facts, function_name, called_functions, risky_apis, string_literals —
i.e. drop llm_v1 (hallucination-prone) and generic prose (purpose/data_flow/
risk_summary). Two mechanisms: (1) noise removal, (2) feeding the explanation as
a SEPARATE channel instead of concatenating — which is exactly the desktop
gated-fusion architecture. The window-proxy row shows the explanation also
recovers beyond-window (truncation) information (246/2196 val functions exceed
512 tokens).

Caveats: config was selected on this val fold (multiple comparisons), so treat
+1.39 as an upper-ish estimate pending the desktop encoder run; TF-IDF proxy only.

Deployment note: CORE selection is applied at TRAIN time via
`SEMVUL_EXPL_FIELDS` (see `experiments/expl_enrich/reproduce_real.py`), NOT via
trimmed files — quality_features_v2 needs code_metrics/risk_level/confidence,
so the `.real.jsonl` files keep every field. The earlier
`*.core.real.jsonl` files were removed for this reason.

## Artifacts

- `devign_real_train.enriched.jsonl` (14,752) / `devign_real_val.enriched.jsonl` (1829)
- Same schema as source, `explanation` object de-anonymized + 5 new digest keys.
- Generator: session scratchpad `gen_enriched.py`; gate: `enrich_gate.py`.
