# Design Spec — CodeT5+ · RoBERTa · Self-Attention Fusion: Component Ladder to Beat FuSEVul

**Date:** 2026-07-03 (rev 2)
**Repo:** `D:\Projects\SemVul`
**Status:** Draft for review (brainstorming → implementation plan)

---

## 1. Goal & constraints

**Goal.** Build, *in our own codebase*, a multimodal model that **beats FuSEVul's stated results**
on **both** datasets, structured as a **component ladder** where each rung shows a measurable gain.

FuSEVul stated targets:
- **Devign:** Acc **60.39**, F1 **55.91**.
- **Reveal:** Acc **91.68**, F1 **46.76**, Precision **57.24**, Recall **39.52**.

**Constraints (from advisor).**
1. **Do not challenge or critique** FuSEVul's results, protocol, or code. The deliverable is
   *beating their stated numbers*, not analyzing them.
2. **All code lives in our solution.** We do **not** run or modify FuSEVul's repository. We adopt
   its *intended architecture* (CodeT5+ code encoder + fine-tuned RoBERTa explanation encoder +
   self-attention fusion) and implement it ourselves.
3. **Defend the full component ladder on both datasets.** Devign and Reveal both matter.

## 2. The ladder we defend

| Rung | Inputs | Adds | Must show |
|------|--------|------|-----------|
| **L1** | CodeT5+ (fine-tuned) code only → classifier | code semantic encoder | baseline |
| **L2** | + fine-tuned **RoBERTa** explanation encoder + **self-attention fusion** | the LLM local-explanation channel (the core contribution) | **L2 > L1** |
| **L3** | + the 22 **quality features** | structured signal distilled from the explanation | **L3 > L2**, and **L3 > FuSEVul stated** |

The ladder *is* the ablation: monotone L1 < L2 < L3 on Acc/F1 demonstrates that the explanation and
the quality features each contribute. "Local explanations" (regenerated, more discriminative
explanations via the Qwen endpoints) are an **upgrade to the L2/L3 explanation content**, deployed
if the ladder is short of the stated targets (see §5, Phase 3).

## 3. Architecture (implemented in our code)

- `code_enc` = **Salesforce/codet5p-110m-embedding** encoder, **fine-tuned**; token-level
  `last_hidden_state`.
- `expl_enc` = **roberta-base**, **fine-tuned end-to-end**; token-level `last_hidden_state`.
- `fusion` = a correct **multi-head self-attention** over the combined code+explanation token
  sequence, fused representation (fused CLS / pooled) → head. `cross`-attention available as an
  alternative. (We implement a standard, correct attention — the fused vector actually feeds the
  head.)
- `quality` = the 22 features (config `QUALITY_FEATURE_NAMES`), concatenated at **L3**.
- `head` = Dropout → Linear → GELU → Linear → 2 logits.
- **Loss:** CrossEntropy for Devign; **focal + capped class-weight** for Reveal (imbalance).
- **Ladder toggles:** `L1` (code only), `L2` (code+expl fusion), `L3` (+QF). A gradient check
  asserts the explanation encoder gets no gradient at L1.

**8 GB VRAM.** Two encoders fine-tuned together is tight: **bf16 + gradient checkpointing +
batch 2–4 + grad-accum 8–16 + max_len 320 (code)/256 (expl)**. Fallback if OOM: **LoRA (r=16)** on
the encoders and/or freeze lower layers. Record actual peak VRAM + chosen setting per run.

## 4. Data & evaluation

- **Datasets:** Devign, Reveal. Code from the normalized code CSVs; explanations from the
  **structured-JSON SemanticVul explanations** (your local explanations — the contribution),
  encoded token-level by RoBERTa.
- **Alignment** by `sample_id = sha1(normalized_code)[:16]`.
- **Split & protocol (comparability):** evaluate on the **same Devign/Reveal validation split**
  FuSEVul used, reporting Acc/F1/Precision/Recall, so "beats their stated results" is a **direct,
  same-benchmark comparison**. Training is kept **leakage-free** as ordinary good practice: drop
  within-train duplicate functions and any **train** function that also appears in **val** (dedup
  applied to TRAIN only; the val set stays identical to theirs for a fair comparison).
- **Reveal** additionally uses focal loss + a tuned decision threshold; report the operating point
  used.

## 5. Staged experiment plan — **Reveal first** (fail-fast on the hardest dataset)

- **Phase 0 — Setup.** Download `codet5p-110m-embedding` + `roberta-base`; VRAM smoke test to pick
  full-FT vs LoRA and batch/accum.
- **Phase 1 — Reveal ladder (the crux).** Train L1 → L2 → L3 on Reveal.
  - **Gate R1:** does **L2 > L1** (explanation helps under fine-tuned RoBERTa + fusion)?
  - **Gate R2:** does **L3** approach/beat **F1 46.76 / Acc 91.68**?
  - If L2 ≤ L1 even with fusion + focal → **escalate immediately**: regenerated "local" explanations
    pilot (Qwen endpoints, small batch, local Docker first) before spending more. We learn Reveal's
    verdict *early*, not at the end.
- **Phase 2 — Devign ladder.** Train L1 → L2 → L3 on Devign; check the same monotonicity and the
  stated targets (60.39/55.91).
- **Phase 3 — Close the gap.** Where a dataset is short of stated numbers, deploy the upgrades:
  regenerated local explanations (Qwen; 950 on clod.io + remainder on local Docker), fusion variant
  (self vs cross), and Reveal imbalance tuning. Re-run the ladder.

Each phase writes ladder tables for both datasets vs the stated FuSEVul numbers.

## 6. Components / files (self-contained, isolated from `src/`)

Under `experiments/fusevul_ladder/`:
- `model.py` — CodeT5+ + RoBERTa + self/cross-attention fusion + QF + ladder toggles + loss.
- `data.py` — load code + structured explanations by `sample_id`, train-side dedup, Devign+Reveal
  loaders on the benchmark val split.
- `train.py` — end-to-end FT (bf16, grad-checkpoint, grad-accum), CE/focal, threshold handling,
  run JSON + preds.
- `run_ladder.py` — L1→L2→L3 × {Reveal, Devign}, emits ladder tables to
  `experiments/reports/fusevul_ladder_*.md` vs stated targets.

## 7. Verification

- **Unit:** fusion forward shapes; L1 builds/uses no explanation encoder (zero grad); L3 QF concat
  shape; train-side dedup leaves val untouched and removes train∩val.
- **Smoke:** 1 epoch, tiny subset, both datasets — trains, writes tables, no OOM at chosen settings.
- **Sanity:** L1 code-only lands near a plausible code-only baseline; L2 output differs from L1
  (explanation actually flows).

## 8. Risks & mitigations

- **Reveal is the gating risk** — F1 gap is large (34.91 → 46.76 needed). Front-loaded in Phase 1;
  regeneration + imbalance tuning are the contingencies. Honest odds ~25–35%.
- **8 GB OOM** with two full-FT encoders → LoRA-r16 + layer freezing + shorter max_len.
- **Model downloads** (`codet5p-110m-embedding`, `roberta-base`) require a one-time online fetch.
- **Explanation may not move Reveal** even with fusion → Phase-1 gate catches it early.

## 9. Success criteria

- Monotone **L1 < L2 < L3** on Acc/F1 for **both** datasets (defends each component).
- **L3 ≥ FuSEVul stated** (Acc and F1) on **both** datasets.
- A clean ladder-table narrative the committee can read directly against FuSEVul's published table.

## 10. Open decisions (resolved during implementation)

- Full-FT vs LoRA on the encoders — from the Phase-0 VRAM smoke.
- Fusion `self` vs `cross` — pick per Phase-1/2 results (self is FuSEVul's claimed best).
- Whether regenerated local explanations are needed at L2/L3 — decided by the Phase-1 Reveal gate.
