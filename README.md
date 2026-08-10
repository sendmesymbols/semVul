# SemanticVul

**Evidence-Grounded Structured LLM Explanations with Adaptive Gated Fusion for Vulnerability Detection**

A function-level C/C++ vulnerability detector built as a controlled **L1→L3
fusion ladder**. Local explanations are generated *offline* by Qwen2.5-Coder
(served through Ollama), scrubbed of verdict words and grounded in quoted code
substrings, then fused with code semantics under a per-sample adaptive gate.
Trained and evaluated on the audited **Devign** and **Reveal** benchmarks.

This README is the single starting point for taking the project over. It covers
what the project is (aim, objectives, questions), the folder/file layout, the
environment setup, and step-by-step reproduction of explanation generation, the
L1–L3 training ladder, and every RQ/RO analysis. Every command and flag below was
confirmed against the code in this repository — see
[§8 Verification notes](#8-verification-notes).

---

## Table of contents

1. [Aim, objectives, and questions](#1-aim-objectives-and-questions)
2. [Repository layout](#2-repository-layout)
3. [Environment setup](#3-environment-setup)
4. [Reproduce — explanation generation (RO1 / RQ1)](#4-reproduce--explanation-generation-ro1--rq1)
5. [Reproduce — the L1–L3 training ladder](#5-reproduce--the-l1l3-training-ladder)
6. [Reproduce — the RQ / RO analyses and figures](#6-reproduce--the-rq--ro-analyses-and-figures)
7. [Suggested order for a fresh takeover](#7-suggested-order-for-a-fresh-takeover)
8. [Verification notes](#8-verification-notes)

---

## 1. Aim, objectives, and questions

**Aim.** Design and evaluate SemanticVul, a vulnerability detector that combines
locally generated, evidence-grounded explanations with code semantics through a
controlled L1–L3 fusion ladder, while separately auditing explanation quality and
testing whether quality-aware fusion mechanisms add predictive value. The thesis
assesses explanation faithfulness, fusion effectiveness, predictive performance,
and imbalance-handling strategy on the audited Devign and Reveal benchmarks under
explicitly declared resource and evaluation constraints.

The objectives (RO) and questions (RQ) are paired one-to-one:

| # | Research Objective | Research Question | Reproduced by |
|---|--------------------|-------------------|---------------|
| **1** | Design & evaluate a local explanation pipeline using structured JSON output, evidence-token grounding, and explicit leakage controls. | To what extent do locally generated, verdict-scrubbed, evidence-grounded explanations improve explanation faithfulness and downstream detection utility vs FuSEVul-style free-form LLM explanations? | `generate_explanations.*` + `src/rqs/rq1.py` |
| **2** | Develop & evaluate a lightweight gated fusion module over cached code embeddings, explanation embeddings, and label-free quality features. | How does quality-aware adaptive gated fusion compare with static fusion, single-modality models, and classical cached-feature baselines in performance and training efficiency? | `src/rqs/rq2.py` (+ `rq2_oracle_gate.py`) |
| **3** | Evaluate SemanticVul against FuSEVul and baselines under audited, explicitly declared protocols. | Under the audited Devign/Reveal splits, how does SemanticVul compare with FuSEVul and baselines in predictive performance, threshold robustness, and low-resource feasibility? | `src/rqs/rq3.py` + `aggregate_seeds.py` |
| **4** | Investigate focal loss, capped class weighting, validation-based threshold tuning, and multi-seed ensembling via controlled ablations. | What are the individual and combined effects of imbalance-aware loss, threshold tuning, and multi-seed ensembling on minority-class detection and the P/R trade-off? | `src/rqs/rq4.py` |

### The L1–L3 ladder (spine of the whole project)

| Rung | What it is | Isolates |
|------|------------|----------|
| **L1** | Code-only baseline (CodeT5+ code channel; text channel present but unused). | The code-only floor. |
| **L2** | Code + explanation (static concatenation of the two channels). | **L2 − L1 = the explanation contribution.** |
| **L3** | L2 + a soft routing gate. | **L3 − L2 = the gate mechanism's value.** |

The L3 gate:

```
w      = sigmoid(MLP([expl_pooled ; confidence]))
pooled = w * code_pooled + (1 - w) * expl_pooled
```

Each rung is trained for **5 seeds (1–5)** on both datasets, producing the six
final cache folders analysed by the RQ scripts.

---

## 2. Repository layout

```
semVul/
├── data/                         # Raw inputs (CSV): the audited Devign & Reveal splits
│   ├── devign/{devign_train.csv, devign_val.csv}
│   └── reveal/{reveal_train.csv, reveal_val.csv}
│
├── explanations/SemanticVul/     # Generated explanation datasets (JSONL)
│   ├── ACTIVE/{devign,reveal}/{train.jsonl,val.jsonl}   # <- THE files training reads
│   ├── devign/, reveal/          # long-named canonical build outputs + full_code/
│   └── devign_real/              # read-only de-anonymisation inputs (stage 5 needs these)
│
├── src/                          # Core library (imported as the `src` package)
│   ├── config.py                 # paths + env-var switches (SEMVUL_*)
│   ├── data_io.py                # dataset + ACTIVE/ JSONL loading, field rendering
│   ├── encode_code.py            # code channel (CodeT5+ / GraphCodeBERT)
│   ├── encode_text.py, encode_text_lora.py   # explanation (text) channel (RoBERTa)
│   ├── model.py                  # fusion model + the L3 routing gate
│   ├── train.py                  # training loop / per-rung trainer
│   ├── eval.py, reports.py       # metrics + reporting
│   ├── quality_features*.py      # label-free quality vector (legacy; proven not useful)
│   └── rqs/                      # ALL analysis entry points (read-only over caches)
│       ├── aggregate_seeds.py    # headline L1–L3 five-seed aggregation
│       ├── rq1.py rq2.py rq3.py rq4.py rq2_oracle_gate.py
│       └── plots.py              # regenerates every RQ figure
│
├── experiments/
│   ├── explanation/              # THE explanation generator
│   │   ├── pipeline.py           #   6-stage orchestrator (generate → … → prefix)
│   │   ├── generate.py           #   stage 1: structured LLM generation (Ollama)
│   │   ├── prompt.py             #   the structured JSON prompt (no label, no verdict words)
│   │   └── work/                 #   default (safe) build output tree
│   ├── expl_enrich/              # explanation post-processing + the trainer wrapper
│   │   ├── reproduce_real.py     #   the trainer the final_*.ps1 launchers call
│   │   ├── apply_real_enrichment.py  # builds ACTIVE/ (stage 5); reveal prereq
│   │   ├── run_enrich.py static_enrich.py correct_val.py augment_train.py build_prefix.py
│   │   └── make_ladder.py        #   gathers per-rung caches
│   ├── cache/                    # frozen encoder embeddings (.npz/.npy) for RQ2/RQ4
│   │   └── lora_ckpt/            #   *.pt LoRA checkpoints (git-ignored, ~520MB each)
│   └── runs/                     # training outputs, one folder per run
│       ├── final_devign_l{1,2,3}_cache/   # <- the six FINAL ladder caches (5 seeds each)
│       ├── final_reveal_l{1,2,3}_cache/   #    per seed: semanticvul_<ds>_<rung>.json + _probs.npz
│       └── rq2_cache/            #    RQ2 frozen-fusion caches (large *nfull*.npz are git-ignored)
│
├── reports/plots/                # generated RQ figures (*.png)
│
├── final_devign_l{1,2,3}.ps1     # Windows final-ladder launchers  (+ matching .sh)
├── final_reveal_l{1,2,3}.ps1     # Windows final-ladder launchers  (+ matching .sh)
├── generate_explanations.ps1/.sh # THE explanation entry point (thin shim over pipeline.py)
├── requirements.txt              # Python deps (UTF-16; torch installed separately, see §3)
└── README.md                     # this guide
```

**The one input contract that matters:** training reads exactly two files per
dataset — `explanations/SemanticVul/ACTIVE/<dataset>/{train,val}.jsonl`. Everything
in `experiments/explanation/` exists to *produce* those two files; everything in
`src/rqs/` exists to *analyse* the caches produced by training on them.

> **Cache filenames.** The six final caches store per-seed results as
> `semanticvul_<ds>_<rung>.json` (+ `_probs.npz`). Older runs used the
> `fusevul_ladder_*` prefix. Every analysis script accepts **both** prefixes
> (`aggregate_seeds.py:157`, `rq3/rq4/plots.py` `_CACHE_TAGS`), so mixed trees
> still aggregate correctly.

---

## 3. Environment setup

**Prerequisites**

- **Python 3.12+** with a virtual environment at `.venv/` in the repo root. The
  launchers call `.venv\Scripts\python.exe` (Windows) directly.
- An **NVIDIA GPU**. Everything is tuned to fit **8 GB** at `-Batch512 2`; pass
  `-Batch512 4` on ≥16 GB. Effective batch stays 32 via gradient accumulation.
- **Ollama** (for explanation generation *only*, not for training) serving
  `qwen2.5-coder:14b`, reachable at `http://localhost:9999`.

```bash
# 1. create + activate the venv (repo root)
python -m venv .venv
#   Windows PowerShell:
.\.venv\Scripts\Activate.ps1
#   Unix:
source .venv/bin/activate

# 2. install PyTorch FIRST, matched to your CUDA (torch is commented out in
#    requirements.txt precisely so you pick the right wheel). The runs used:
#      torch==2.12.1+cu132  torchvision==0.27.1+cu132
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132

# 3. install the rest
pip install -r requirements.txt
```

Key packages: `transformers`, `peft`, `sentence-transformers`, `scikit-learn`,
`numpy`, `pandas`, `scipy`. (`requirements.txt` is UTF-16 encoded — `pip` reads it
fine; a plain text editor may show a BOM.)

> **GPU-free path.** The RQ/RO analyses in `src/rqs/` are *read-only over the
> committed caches* and need **no GPU and no training** — you can reproduce every
> headline number and figure ([§6](#6-reproduce--the-rq--ro-analyses-and-figures))
> on CPU as long as `experiments/runs/final_*` and `experiments/cache/` are present
> (they ship in the repo).

---

## 4. Reproduce — explanation generation (RO1 / RQ1)

> **Slow, optional.** The shipped `ACTIVE/` files already contain the final
> generated set. You do **not** need to regenerate to reproduce training or the
> RQs — use `-Smoke` only to prove the pipeline works end-to-end.

Explanation generation is **fully separate** from detector training. One entry
point, `generate_explanations.ps1` (Windows) / `.sh` (Unix), a thin shim over
`experiments/explanation/pipeline.py`. It runs six stages and ends with a complete
`ACTIVE/{devign,reveal}/{train,val}.jsonl` pair:

| Stage | Script | Produces |
|-------|--------|----------|
| 1 generate | `explanation/generate.py` | `purpose, data_flow, risky_operations, missing_checks, evidence_tokens, safety_indicators, risk_summary, risk_level, confidence` (**confidence measured from decode-time logprobs**, not self-reported) |
| 2 install | (pipeline) | stage-1 output → the filename later stages read |
| 3 enrich | `expl_enrich/run_enrich.py` | `llm_v1, code_metrics, tail_facts, enrich` |
| 4 clean/aug | `correct_val.py`, `augment_train.py` | `.clean` / `.clean.aug` variants |
| 5 real | `apply_real_enrichment.py` | `function_name, called_functions, risky_apis, string_literals, lexical_digest, real_enrich, tail_digest` (+ refreshes `ACTIVE/`) |
| 6 prefix | `build_prefix.py` | `prefix, prefix_recipe` (+ `ACTIVE/README.md`) |

> The older `organize_explanations.ps1` is **obsolete** — its work is now stages 5
> and 6. Do not call it.

```powershell
# Windows — smoke test: 6 rows per split, end-to-end proof of the whole pipeline
.\generate_explanations.ps1 -Smoke

# Full default build into experiments\explanation\work\ (does NOT touch shipped data)
.\generate_explanations.ps1

# Stratified sample with parallel workers
.\generate_explanations.ps1 -Stratified 300 -Workers 4

# Promote: rebuild directly in the SHIPPED explanations\SemanticVul\ tree (opt-in, overwrites)
.\generate_explanations.ps1 -Promote
```

```bash
# Unix equivalents
./generate_explanations.sh --smoke
./generate_explanations.sh
./generate_explanations.sh --promote
```

**Defaults** (all overridable as flags): model `qwen2.5-coder:14b`, host
`http://localhost:9999`, `mode=auto`, `num-ctx=8192`, `timeout=600`,
`tail-offset=220`. The launcher preflights Ollama and fails early if the model is
missing.

**Leakage control.** The ground-truth label is **never** in the prompt; labels are
copied onto output rows *only after* generation, purely so the rows can be used for
supervised training. This is the property RQ1 audits.

> **Runtime.** Full generation is ~70 s/sample on `qwen2.5-coder:14b`. All four
> splits ≈ 70,802 rows ≈ **57 days sequential** (~7 days at `-Workers 8`).
> Generation is resumable (finished `sample_id`s are skipped).

---

## 5. Reproduce — the L1–L3 training ladder

The six per-dataset, per-rung launchers train 5 seeds each and write independent
cache folders under `experiments/runs/final_*`. They call
`experiments/expl_enrich/reproduce_real.py` with a shared config; only the
per-dataset treatment knobs differ.

```powershell
# Activate the venv
.\.venv\Scripts\Activate.ps1

# Devign final ladder  (balanced -> plain cross-entropy, no focal)
.\final_devign_l1.ps1        # code-only baseline
.\final_devign_l2.ps1        # code + explanation
.\final_devign_l3.ps1        # + soft routing gate

# Reveal final ladder  (imbalanced -> focal loss)
.\final_reveal_l1.ps1
.\final_reveal_l2.ps1
.\final_reveal_l3.ps1

# ...matching Unix launchers: ./final_devign_l1.sh, ./final_reveal_l1.sh, etc.
# Pass -Batch512 4 on a >=16GB GPU (default 2 fits 8GB). Runs are RESUMABLE:
# a finished rung JSON is skipped, so re-running continues an interrupted ladder.
```

**Shared config** (all rungs, both datasets), passed to `reproduce_real.py`:
`--code-enc codet5p` (CodeT5+, FuSEVul's encoder), `--max-text 512`, `--epochs 12`,
`--seeds 1 2 3 4 5`, and the 8-column text channel:

```
confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose
```

**Per-dataset specifics** (confirmed in the launchers):

- **Devign** — code window **512** (the reproducer's built-in Devign default,
  `reproduce_real.py:103` `devign_kw = dict(max_code=512, ...)`; the launchers do
  not pass `--max-code`). Balanced classes → plain cross-entropy (focal auto-off).
  Requires `ACTIVE/devign/{train,val}.jsonl` to already exist (the Devign launchers
  do **not** rebuild it and will error if missing).
- **Reveal** — explicitly `--max-code 512`, plus `--focal-alpha 0.85`,
  `--focal-gamma 2.0`, tail-offset 220. **Prerequisite:** the Reveal launchers
  first run `apply_real_enrichment.py --check --only reveal`; if `ACTIVE/reveal` is
  absent they auto-build it with `--tail-offset 220`. (This step is implicit — it
  runs for you — but is why a fresh checkout can train Reveal without a manual
  enrichment step.)
- **L3 (both)** — enables the gate: env `SEMVUL_QUAL_V2=0` (old quality-vector gate
  off), `SEMVUL_QUAL_GATE=1`, `SEMVUL_GATE_LR_MULT=100` (gate LR ×100 so it moves),
  plus the `--qual-gate` flag. Encoders **fine-tune** here (no `SEMVUL_FROZEN`), so
  L3 sits in the same regime as L1/L2 and aggregates on one scale. An alternative
  hard if/else arm is available via `-HardConfSwitch [-HardConfThreshold 85]`
  (writes to a separate `*_hardswitch_cache`).

**Outputs.** Six folders, 5 seed subfolders each, each seed holding
`semanticvul_<ds>_<rung>.json`, `..._partial.json`, and `..._probs.npz`:

```
experiments/runs/final_devign_l1_cache, final_devign_l2_cache, final_devign_l3_cache,
                 final_reveal_l1_cache, final_reveal_l2_cache, final_reveal_l3_cache
```

**Aggregate the ladder** (headline L1–L3 numbers, both scoring protocols side by
side — the FuSEVul-comparable circular protocol and the non-circular
validation-split diagnostic):

```powershell
python src\rqs\aggregate_seeds.py both
python src\rqs\aggregate_seeds.py both --json aggregate_live.json   # also dump JSON
```

---

## 6. Reproduce — the RQ / RO analyses and figures

These are **read-only** over the caches from [§5](#5-reproduce--the-l1l3-training-ladder)
(and `experiments/cache/` for RQ2) — no training, no GPU required.

```powershell
# RQ1 / RO1 — explanation quality, leakage, and L1->L2 utility
python src\rqs\rq1.py both

# RQ2 / RO2 — frozen-cache fusion + quality-aware gate mechanism study
python -m src.rqs.rq2 --dataset devign --seeds 1,2,3,4,5
python -m src.rqs.rq2 --dataset reveal --seeds 1,2,3,4,5
#   Diagnostic only (NOT a headline RQ2 result): oracle confidence gate
python -m src.rqs.rq2_oracle_gate --dataset reveal --seeds 1,3,7

# RQ3 / RO3 — FuSEVul-comparable benchmark, stability, cost, audit
python src\rqs\rq3.py both

# RQ4 / RO4 — imbalance-aware loss, threshold tuning, ensembling
python src\rqs\rq4.py both

# Regenerate every RQ figure into reports/plots/
python src\rqs\plots.py both --cache-prefix final
```

`rq1/rq3/rq4/plots/aggregate_seeds` take a positional dataset (`reveal | devign |
both`); with no argument they show an interactive menu. `rq2` and `rq2_oracle_gate`
are **modules** (`python -m src.rqs.rq2 …`) and require `--dataset`.

**Data-quality audit** (Section 3.2.1, Table 3.2): reproduced with the same hashing
workflow as Appendix B applied to the distributed Devign files — 124 within-train
label-conflicting rows (62 pairs), 12 within-val rows (6 pairs), 27 cross-split
rows with reversed labels (~1.0% of the 2,705 unique validation functions),
bounding un-deduplicated validation accuracy near 2678/2705 ≈ 99.0%.

---

## 7. Suggested order for a fresh takeover

1. **Set up the env** ([§3](#3-environment-setup)) and confirm
   `experiments/runs/final_*` and `experiments/cache/` are present (they ship in
   the repo).
2. **Reproduce the analyses first** ([§6](#6-reproduce--the-rq--ro-analyses-and-figures))
   — no GPU, fastest way to confirm the caches are intact and match the paper's
   tables/figures.
3. `aggregate_seeds.py both` for the headline ladder numbers
   ([§5](#5-reproduce--the-l1l3-training-ladder)).
4. Only if you need to *rebuild* results: run the six ladder launchers
   ([§5](#5-reproduce--the-l1l3-training-ladder)). This needs a GPU but no LLM.
5. Only if you need to *regenerate explanations*: stand up Ollama and run
   `generate_explanations.ps1 -Smoke` first, then scale
   ([§4](#4-reproduce--explanation-generation-ro1--rq1)). This is the slow,
   optional path.

---

## 8. Verification notes

All commands/flags below were checked against the code on this machine (scripts,
argparse definitions, and the on-disk caches). Confirmed:

- All six `final_*.ps1` launchers exist (with matching `.sh`); flags `--code-enc
  codet5p`, `--max-text 512`, `--epochs 12`, seeds 1–5, and the exact 8-field text
  channel all match the code.
- Devign's 512-token code window is real (reproducer default, not a launcher flag).
- Reveal's `--max-code 512`, `--focal-alpha 0.85`, `--focal-gamma 2.0` match.
- L3's `SEMVUL_QUAL_V2=0`, `SEMVUL_QUAL_GATE=1`, `SEMVUL_GATE_LR_MULT=100`,
  `--qual-gate` all match.
- All RQ entry points, argument styles (positional vs `-m … --dataset`), and
  `aggregate_seeds.py --json` match.
- `generate_explanations.*` defaults (`qwen2.5-coder:14b`, `localhost:9999`,
  `mode=auto`, `num-ctx=8192`, `timeout=600`) and the six pipeline stages match.
