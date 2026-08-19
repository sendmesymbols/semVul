# SemanticVul

**Evidence-Grounded Structured LLM Explanations with Adaptive Gated Fusion for Vulnerability Detection**

A function-level C/C++ vulnerability detector built as a controlled **L1→L3
fusion ladder**. Local explanations are generated *offline* by Qwen2.5-Coder
(served through Ollama), scrubbed of verdict words and grounded in quoted code
substrings, then fused with code semantics under a quality-aware routing gate.
Trained and evaluated on the audited **Devign** and **Reveal** benchmarks.

> **Result-provenance hold.** The checked-in `ACTIVE/` rows and the existing
> `experiments/runs/final_*_cache` results predate removal of deterministic
> post-generation enrichment. They reproduce the historical results only; they
> are not evidence for the clean Qwen-only pipeline described below. Regenerate
> `ACTIVE/` and rerun all six canonical caches before using the numbers in a new
> paper submission.

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
| **4** | Evaluate operating-point choices and multi-seed ensembling under class imbalance. | How do the fixed focal-loss recipe, validation-based threshold tuning, and ensembling change minority-class performance and the precision/recall trade-off? | `src/rqs/rq4.py` |

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
│   └── devign/, reveal/          # canonical clean-Qwen build outputs
│
├── src/                          # Shared data/feature code + RQ analyses
│   ├── config.py                 # paths + env-var switches (SEMVUL_*)
│   ├── data_io.py                # dataset + ACTIVE/ JSONL loading, field rendering
│   ├── quality_features*.py      # label-free quality vectors used by L3/RQ2
│   └── rqs/                      # ALL analysis entry points (read-only over caches)
│       ├── aggregate_seeds.py    # headline L1–L3 five-seed aggregation
│       ├── rq1.py rq2.py rq3.py rq4.py rq2_oracle_gate.py
│       └── plots.py              # regenerates every RQ figure
│
├── experiments/
│   ├── fusevul_ladder/           # final detector implementation
│   │   ├── data.py               #   L1–L3 inputs and quality features
│   │   ├── model.py              #   encoders, fusion, and L3 routing gate
│   │   └── train.py              #   final per-rung training loop
│   ├── explanation/              # THE explanation generator
│   │   ├── pipeline.py           #   3-stage orchestrator (generate/install/validate)
│   │   ├── generate.py           #   stage 1: structured LLM generation (Ollama)
│   │   ├── prompt.py             #   the structured JSON prompt (no label, no verdict words)
│   │   ├── validate_clean.py      #   rejects missing/legacy ACTIVE fields
│   │   └── work/                 #   default (safe) build output tree
│   ├── expl_enrich/              # final-ladder trainer wrapper (historical directory name)
│   │   ├── reproduce_real.py     #   the trainer the final_*.ps1 launchers call
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

> **GPU-free historical path.** The RQ/RO analyses in `src/rqs/` are read-only
> over the committed caches and need no GPU. They reproduce the historical
> numbers, not yet the clean-Qwen experiment. Clean headline numbers require the
> regeneration and six training runs described below.

---

## 4. Reproduce — explanation generation (RO1 / RQ1)

> **Required for clean results.** The shipped `ACTIVE/` files contain legacy
> fields and are deliberately rejected by `validate_clean.py`. A smoke run proves
> mechanics only; regenerate and promote the complete dataset before training.

Explanation generation is **fully separate** from detector training. One entry
point, `generate_explanations.ps1` (Windows) / `.sh` (Unix), a thin shim over
`experiments/explanation/pipeline.py`. It runs three stages and ends with a complete
`ACTIVE/{devign,reveal}/{train,val}.jsonl` pair:

| Stage | Script | Produces |
|-------|--------|----------|
| 1 generate | `explanation/generate.py` | `purpose, data_flow, risky_operations, missing_checks, evidence_tokens, safety_indicators, risk_summary, risk_level, confidence` (**confidence measured from decode-time logprobs**, not self-reported) |
| 2 install | (pipeline) | stage-1 output → canonical dataset files |
| 3 validate/promote | `explanation/validate_clean.py` | rejects legacy fields, validates types, and activates only clean Qwen rows |

```powershell
# Windows — smoke test: 6 rows per split, end-to-end mechanics only
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
`http://localhost:9999`, `mode=auto`, `num-ctx=8192`, and `timeout=600`. The
launcher preflights Ollama and fails early if the model is missing.

**Leakage control.** The ground-truth label is **never** in the prompt; labels are
copied onto output rows *only after* generation, purely so the rows can be used for
supervised training. This is the property RQ1 audits.

> **Runtime.** Full generation is ~70 s/sample on `qwen2.5-coder:14b`. All four
> splits ≈ 70,802 rows ≈ **57 days sequential** (~7 days at `-Workers 8`).
> Generation is resumable (finished `sample_id`s are skipped).

---

## 5. Reproduce — the L1–L3 training ladder

The six per-dataset, per-rung launchers train 5 seeds each and write the six
canonical `experiments/runs/final_*_cache` folders. They call
`experiments/expl_enrich/reproduce_real.py` with a shared config; only the
per-dataset treatment knobs differ.

For the shipped/original results, use the single entry point below. It maps each
dataset/rung to its matching `experiments/runs/final_<dataset>_<rung>_cache`
folder; completed seed JSONs are skipped and only missing seeds are trained:

```powershell
.\run_final_sequence.ps1              # reuse/generate missing original caches
.\run_final_sequence.ps1 -Dataset reveal -Rungs L1,L2,L3
```

The original per-rung commands below remain the canonical commands and are the
scripts invoked by this wrapper. Clean-Qwen generation is separate and opt-in:
`.\run_final_sequence.ps1 -GenerateCleanQwen`.

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
`--seeds 1 2 3 4 5`, and the eight-field clean Qwen text channel:

```
confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary
```

**Per-dataset specifics** (confirmed in the launchers):

- **Devign** — code window **512** (the reproducer's built-in Devign default,
  `reproduce_real.py:103` `devign_kw = dict(max_code=512, ...)`; the launchers do
  not pass `--max-code`). Balanced classes → plain cross-entropy (focal auto-off).
  Reuses `final_devign_l{1,2,3}_cache`; completed seed JSONs are skipped and
  missing seeds are trained into the same cache folders.
- **Reveal** — explicitly `--max-code 512`, plus `--focal-alpha 0.85` and
  `--focal-gamma 2.0`. Reuses `final_reveal_l{1,2,3}_cache` in the same way.
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

Each newly initialized folder receives `.clean_qwen_contract.json`, binding it
to hashes of the ACTIVE train/validation files and the training configuration.
The currently checked-in folders contain unmarked historical results. The
trainer therefore refuses to skip or overwrite them: archive their contents
outside these six canonical folders after preserving any historical analysis,
then start the clean rerun. This prevents old metrics from being silently
presented as results of the new input contract.

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
giving a conditional train-label-memorization consistency bound of
2678/2705 ≈ 99.0%, not a universal model-accuracy ceiling.

---

## 7. Suggested order for a fresh takeover

1. **Set up the environment** ([§3](#3-environment-setup)).
2. Optionally run the analyses in [§6](#6-reproduce--the-rq--ro-analyses-and-figures)
   to reproduce the explicitly historical checked-in results.
3. Start Ollama and run `generate_explanations.ps1 -Smoke` to verify the clean
   three-stage pipeline.
4. Run the full generator with `-Promote`; confirm `validate_clean.py` accepts
   both datasets.
5. Preserve the historical cache contents outside the six canonical
   `final_*_cache` folders, then run the six final launchers. The trainer creates
   and verifies a clean-input contract in each folder.
6. Rerun aggregation and all RQ analyses; only these regenerated results support
   claims about the clean Qwen-only pipeline.

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
  `mode=auto`, `num-ctx=8192`, `timeout=600`) and the three pipeline stages match.

---

## Acknowledgments

This work was completed under the supervision and guidance of:

- **Assoc Prof Dr. Ihtesham Ul Islam** (Supervisor) — Department of Electrical Engineering, Military College of Signals, NUST
- **Asst Prof Dr. Rabia Khan** (Co-Supervisor) — Department of Computer Software Engineering, Military College of Signals, NUST

Committee Members:
- **Asst Prof Dr. Muhammad Sohail** — Department of Computer Software Engineering, Military College of Signals, NUST
- **Asst Prof Dr. Nazia Bibi** — Department of Computer Software Engineering, Military College of Signals, NUST

I extend our gratitude to the supervisor, co-supervisor, and committee members for their invaluable guidance, insightful feedback, and unwavering support throughout this research journey.
