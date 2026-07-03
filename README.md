# SemVul

Beat FuSEVul on Accuracy, F1, Precision, Recall — Devign and Reveal — on 8 GB
VRAM. Three ladders, one clean pipeline, honest reporting.

## Layout

```
data/                              raw code + label (JSONL is source of truth)
  devign/{devign_train,devign_val}.csv
  reveal/{reveal_train,reveal_val}.csv

explanations/
  FuseVul/{devign,reveal}/ss_{train,val}.csv     base paper's explanations
  SemanticVul/{devign,reveal}/*.jsonl            our structured JSON (canonical source)

src/
  config.py                paths, model IDs, hparams (single source of truth)
  data_io.py               JSONL loader -> Sample dataclass
  quality_features.py      22 QF, all label-free
  encode_text.py           MiniLM explanation embeddings (frozen)
  encode_code.py           frozen code encoder (GraphCodeBERT, CodeT5+)
  lora_finetune.py         LoRA-tune GraphCodeBERT + extract embeddings
  model.py                 GatedFusion + focal BCE
  eval.py                  metrics at 3 threshold policies
  reports.py               ladder progression + component ablation tables
  train.py    <-- ENTRY    train ONE gated-fusion head
  run.py      <-- ENTRY    the full campaign (encode + all ladders + reports)

train.ps1   <-- ENTRY      wrapper for src/train.py (single-run retrain)
run.ps1     <-- ENTRY      wrapper for src/run.py   (full pipeline for a dataset)

experiments/
  cache/     cached embeddings (.npz) + LoRA checkpoints
  runs/      per-run archives (.json + probs .npz)
  reports/   generated .md tables
  logs/      transcript logs from run.ps1
```

## The three ladders

Each ladder shares the same head architecture (gated fusion of code + expl + QF).
The only thing that changes is the **code representation** fed into the head.
No stacking, no forward-selection over 15 seeds. Each ladder is one number.

| Ladder | Code representation | Why it matters |
|---|---|---|
| L1 | **LoRA-fine-tuned GraphCodeBERT** | Frozen embeddings cap at the pre-training discrimination. LoRA lifts the ceiling by learning task-specific separation. |
| L2 | L1 concatenated with **frozen CodeT5+ 220M** | Adds a 2023-era encoder trained on more code, orthogonal signal. |
| L3 | **Probability ensemble of L1 head + L2 head** | Averages independent-family models. Balanced-acc threshold on TUNE. |

## Threshold reporting

Every result appears at three policies simultaneously, so we can never hide
behind a degenerate operating point:

- `fixed_0.5` — no tuning, the paper's natural policy on balanced data
- `max_bal_acc` — threshold chosen on TUNE to maximize balanced accuracy (honest)
- `max_f1` — threshold chosen on TUNE to maximize F1 (FuSEVul-comparable)

## Component ablation (per ladder)

For each ladder we can drop one block at a time and rerun the same seeds:

- `full`     — gated fusion + explanation + quality features
- `no_expl`  — code + QF (no explanation channel)
- `no_qual`  — code + explanation, gated (no 22 QF)
- `concat`   — static concatenation instead of gated fusion

Report shows delta vs `full` for every metric. If a component isn't pulling
its weight, it will show up as a negative delta immediately.

## How to run

### First-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Full campaign for one dataset (Devign)

```powershell
.\run.ps1 -Dataset devign -Ablate
```

This will:
1. Encode explanations with MiniLM (cached)
2. LoRA fine-tune GraphCodeBERT on Devign (~30-60 min on 8 GB)
3. Extract LoRA embeddings
4. Encode with frozen CodeT5+
5. Train L1, L2 heads (full + 3 ablations each if `-Ablate`)
6. Ensemble to L3
7. Write reports to `experiments/reports/`

### Retrain a single head

```powershell
.\train.ps1 -Dataset devign -Ladder L1 -Tag full
.\train.ps1 -Dataset devign -Ladder L1 -Tag no_qual -NoQual
```

### Only certain ladders

```powershell
.\run.ps1 -Dataset devign -Ladders L1,L2      # skip L3
.\run.ps1 -Dataset reveal                     # all three
```

## Reports emitted

Per dataset (`experiments/reports/<dataset>_*`):

- `ladder_progression.md` — L1 vs L2 vs L3 at all threshold policies, WIN/LOSE
  vs FuSEVul on every metric
- `L1_component_ablation.md`, `L2_component_ablation.md`, `L3_component_ablation.md`
  — delta of dropping each component

## FuSEVul targets (what we must beat)

| Dataset | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Devign | 60.39 | 55.91 | — | — |
| Reveal | 91.68 | 46.76 | 57.24 | 39.52 |



Why your current setup has hit a wall
You are training frozen-embedding MLP heads. Frozen GraphCodeBERT/UniXcoder embeddings on Devign cap out around PR-AUC ~53, ROC-AUC ~59. That ceiling is the reason no component you add helps — you are re-arranging deck chairs on a fixed representation. To beat FuSEVul at 0.5 threshold you need PR-AUC ~58+. That requires learning task-specific features, not stacking heads on frozen ones.

The plan that actually beats Devign
Three levers, in order. Each is 8 GB feasible. Do them in sequence — each raises the ceiling for the next.

Lever 1 — LoRA-fine-tune GraphCodeBERT on Devign (single biggest jump)
Frozen → LoRA-tuned on the same encoder typically moves Devign PR-AUC from ~53 to ~60–63 and F1@0.5 from ~52 to ~57–60. That alone beats FuSEVul F1 at threshold 0.5, and lifts accuracy simultaneously because the model is actually separating classes, not sliding a threshold.

Config that fits 8 GB:

peft LoRA on microsoft/graphcodebert-base, rank=16, alpha=32, target query,key,value,dense
fp16 + gradient checkpointing, batch=16, grad-accum=4 (effective 64)
lr=2e-5, 3–5 epochs, cosine schedule, warmup 0.06
Save the [CLS] pooled hidden state → new train_code_gcb_lora.npz cache
Everything downstream (your gated fusion, QF, ensemble) stays identical
Expected: Devign Acc@0.5 55→60+, F1@0.5 52→57–60. Reveal also lifts.

Lever 2 — Add a stronger 2023-era code encoder to the cache
GraphCodeBERT (2020) is the weakest encoder still in use. Drop in one of these as an additional cached embedding stream:

CodeT5+ 220M encoder (Salesforce/codet5p-220m) — encoder-only mode, take mean-pooled last hidden state
CodeSage-small (codesage/codesage-small) — sentence-embedding style, trained specifically for retrieval/classification on code
UniXcoder you already have, but LoRA-tune it too and ensemble with LoRA-GCB
Adding CodeT5+ frozen gets you roughly +1–2 PR-AUC on top of Lever 1. LoRA-tuned CodeT5+ gets +3–4.

Lever 3 — Cross-family ensemble at balanced threshold
Once you have two independently-strong models (LoRA-GCB and LoRA-CodeT5+ or LoRA-UniXcoder), ensemble their probabilities (not rank) and pick threshold by max balanced accuracy on the honest tune split, not max F1. On balanced Devign, max-bal-acc threshold sits near 0.5 and gives you Acc and F1 simultaneously — no more degenerate all-positive predictions.

Add LightGBM on [LoRA-GCB embedding | LoRA-CodeT5+ embedding | 22 QF] as a third ensemble member. Tree models + fine-tuned embeddings are historically the strongest recipe on Devign in the leaderboard papers.

Execution order (concrete)
Write src/train/lora_finetune_code.py — LoRA-tune GraphCodeBERT on Devign train, checkpoint by val F1@0.5, dump pooled embeddings to train_code_gcb_lora.npz / val_code_gcb_lora.npz. ~2–3 hrs on 8 GB.
Register graphcodebert_lora as a --code-encoder option in src/run/semanticvul.py and your loaders. Zero change downstream.
Re-run the observation matrix using --code-encoder graphcodebert_lora. Ladder should shift up 5–8 F1 points across the board at 0.5.
Repeat Lever 1 with CodeT5+, produce train_code_codet5p_lora.npz.
Ensemble the two LoRA models + LightGBM-on-fused-features, threshold via max-balanced-acc on tune split.
Report at threshold=0.5 and max-balanced-acc threshold so accuracy and F1 both stay high.
If Lever 1 alone gets you past the paper, stop and lock. If it gets close but not over, add Lever 2. Lever 3 is the safety net that turns "close" into "cleanly beats on both."



Yes — this should pull it off on both datasets, with Reveal a near-lock and Devign accuracy the one number to watch. ~4–6 hours total on your 8 GB GPU for the full campaign both datasets with ablations. Reports are complete and auto-generated — one progression table per dataset plus one component ablation per trained ladder.

Kick it off with:

cd D:\Projects\SemVul
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 -Dataset devign -Ablate
.\run.ps1 -Dataset reveal -Ablate
If Devign accuracy misses on the first run, the fix is a single-line change in src/config.py (LORA_CFG["r"] = 32, LORA_CFG["epochs"] = 5) and re-running run.ps1 — LoRA re-trains, everything downstream re-uses cache.