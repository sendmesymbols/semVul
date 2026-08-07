===============================================================================
 SemanticVul
 Quality-Aware Fusion of Code Semantics and Local Explanations
 for Vulnerability Detection
===============================================================================

A function-level C/C++ vulnerability detector built as a controlled L1->L3
fusion ladder. Local explanations are generated OFFLINE by Qwen2.5-Coder (served
through Ollama), scrubbed of verdict words and grounded in quoted code
substrings, then fused with code semantics under a quality-aware routing gate.
Trained and evaluated on the audited Devign and Reveal benchmarks.

This README is the quick-start. For the full handover -- folder map, environment
setup, and step-by-step reproduction of every result -- see HANDOVER.md.

-------------------------------------------------------------------------------
 1. AIM, OBJECTIVES, QUESTIONS
-------------------------------------------------------------------------------

AIM
  Design and evaluate SemanticVul, a vulnerability detector that combines
  locally generated, evidence-grounded explanations with code semantics through
  a controlled L1-L3 fusion ladder, while separately auditing explanation
  quality and testing whether quality-aware fusion mechanisms add predictive
  value. Assessed on the audited Devign and Reveal benchmarks under explicitly
  declared resource and evaluation constraints.

Objectives and questions are paired one-to-one:

  RO1 / RQ1  INPUT.  A local explanation pipeline (structured JSON, evidence-
             token grounding, leakage controls). Do verdict-scrubbed, grounded
             explanations improve faithfulness and downstream utility vs
             FuSEVul-style free-form explanations?

  RO2 / RQ2  ARCHITECTURE.  A lightweight gated fusion module over cached code
             embeddings, explanation embeddings, and label-free quality
             features. Does quality-aware adaptive gating beat static fusion,
             single-modality models, and classical cached-feature baselines in
             performance and training efficiency?

  RO3 / RQ3  COMPARISON.  Evaluate against FuSEVul and baselines under audited,
             explicitly declared protocols. How does SemanticVul compare on
             predictive performance, threshold robustness, and low-resource
             feasibility?

  RO4 / RQ4  OPTIMIZATION.  Investigate focal loss, capped class weighting,
             validation-based threshold tuning, and multi-seed ensembling. What
             are their individual and combined effects on minority-class
             detection and the precision-recall trade-off?

THE LADDER (spine of the whole project)
  L1  code-only baseline (CodeT5+ code channel).
  L2  code + explanation (static concatenation).   L2 - L1 = explanation value.
  L3  L2 + soft routing gate:                       L3 - L2 = the gate's value.
        w = sigmoid(MLP([expl_pooled; confidence]))
        pooled = w * code_pooled + (1 - w) * expl_pooled
  Each rung is trained for 5 seeds (1-5) on both datasets.

-------------------------------------------------------------------------------
 2. WHAT IS WHERE  (top-level map; full detail in HANDOVER.md)
-------------------------------------------------------------------------------

  data/                          Raw CSV inputs: audited Devign & Reveal splits.
  explanations/SemanticVul/      Generated explanation datasets (JSONL).
      ACTIVE/<ds>/{train,val}.jsonl   <- THE two files training actually reads.
  src/                           Core library (the `src` package).
      config.py data_io.py encode_* model.py train.py eval.py reports.py
      rqs/                       ALL analysis entry points (read-only over caches).
  experiments/
      explanation/               THE explanation generator (pipeline.py + generate.py
                                 + prompt.py); builds ACTIVE/ in 6 stages.
      expl_enrich/               Post-processing + reproduce_real.py (the trainer
                                 the final_*.ps1 launchers call).
      cache/                     Frozen encoder embeddings for RQ2/RQ4 (.npz/.npy);
                                 lora_ckpt/*.pt are git-ignored (~520MB each).
      runs/final_*_cache/        The SIX final ladder caches (5 seeds each).
  reports/plots/                 Generated RQ figures (*.png).
  final_devign_l{1,2,3}.ps1/.sh  Devign ladder launchers.
  final_reveal_l{1,2,3}.ps1/.sh  Reveal ladder launchers.
  generate_explanations.ps1/.sh  Explanation entry point (shim over pipeline.py).
  requirements.txt               Python deps (UTF-16; install torch separately).

  INPUT CONTRACT: training reads exactly
     explanations/SemanticVul/ACTIVE/<dataset>/{train,val}.jsonl
  Everything in experiments/explanation/ produces those files; everything in
  src/rqs/ analyses the caches produced by training on them.

  CACHE FILENAMES: per-seed results are semanticvul_<ds>_<rung>.json (+ _probs.npz).
  Older runs used fusevul_ladder_*; every analysis script accepts BOTH prefixes.

-------------------------------------------------------------------------------
 3. ENVIRONMENT SETUP
-------------------------------------------------------------------------------

Prerequisites:
  * Python 3.12+ with a virtualenv at .venv/ in the repo root. Launchers call
    .venv\Scripts\python.exe (Windows) directly.
  * NVIDIA GPU. Tuned for 8GB at -Batch512 2; use -Batch512 4 on >=16GB.
    Effective batch stays 32 via gradient accumulation.
  * Ollama serving qwen2.5-coder:14b at http://localhost:9999
    (ONLY for explanation generation, not for training or the RQ analyses).

Steps:
  1) Create + activate the venv (repo root):
        python -m venv .venv
        Windows:  .\.venv\Scripts\Activate.ps1
        Unix:     source .venv/bin/activate

  2) Install PyTorch FIRST, matched to your CUDA (torch is commented out in
     requirements.txt so you pick the right wheel). The runs used:
        torch==2.12.1+cu132   torchvision==0.27.1+cu132
        pip install torch==2.12.1 torchvision==0.27.1 \
            --index-url https://download.pytorch.org/whl/cu132

  3) Install the rest:
        pip install -r requirements.txt

  GPU-FREE PATH: the src/rqs/ analyses are read-only over the committed caches
  and need no GPU and no training. You can reproduce every headline number and
  figure (section 5) on CPU as long as experiments/runs/final_* and
  experiments/cache/ are present (they ship in the repo).

-------------------------------------------------------------------------------
 4. REPRODUCE -- EXPLANATION GENERATION  (RO1 / RQ1)   [SLOW, OPTIONAL]
-------------------------------------------------------------------------------

One entry point over a 6-stage pipeline (generate -> install -> enrich ->
clean/aug -> real -> prefix), ending with a complete ACTIVE/ pair per dataset.
confidence is MEASURED from decode-time token logprobs, not self-reported. The
ground-truth label is NEVER in the prompt (copied to rows only after generation).
The old organize_explanations.ps1 is OBSOLETE (now stages 5-6); do not call it.

  Windows:
     .\generate_explanations.ps1 -Smoke                 # 6 rows/split, end-to-end proof
     .\generate_explanations.ps1                        # full build into experiments\explanation\work\
     .\generate_explanations.ps1 -Stratified 300 -Workers 4
     .\generate_explanations.ps1 -Promote               # overwrite shipped explanations\SemanticVul\

  Unix:
     ./generate_explanations.sh --smoke
     ./generate_explanations.sh
     ./generate_explanations.sh --promote

  Defaults: model qwen2.5-coder:14b, host http://localhost:9999, mode auto,
  num-ctx 8192, timeout 600, tail-offset 220.

  WARNING: full generation is ~70 s/sample; all four splits (~70,802 rows) is
  ~57 days sequential (~7 days at -Workers 8). Resumable (finished rows skipped).
  The shipped ACTIVE/ files ALREADY contain the final set -- you do NOT need to
  regenerate to reproduce training or the RQs. Use -Smoke only to prove the
  pipeline works.

-------------------------------------------------------------------------------
 5. REPRODUCE -- TRAINING LADDER + ANALYSES
-------------------------------------------------------------------------------

TRAIN THE SIX FINAL CACHES  (GPU; no LLM needed):
  # activate venv, then:
  .\final_devign_l1.ps1   .\final_devign_l2.ps1   .\final_devign_l3.ps1
  .\final_reveal_l1.ps1   .\final_reveal_l2.ps1   .\final_reveal_l3.ps1
  # (Unix: the matching .sh scripts.  Add -Batch512 4 on >=16GB. RESUMABLE:
  #  a finished rung JSON is skipped.)

  Shared config (all rungs, both datasets): --code-enc codet5p, --max-text 512,
  --epochs 12, seeds 1 2 3 4 5, and the 8-column text channel:
     confidence,risky_operations,missing_checks,function_name,
     called_functions,risky_apis,risk_summary,purpose

  Devign : code window 512 (reproducer default); balanced -> plain cross-entropy.
           Requires ACTIVE/devign to already exist (does NOT rebuild it).
  Reveal : --max-code 512, --focal-alpha 0.85, --focal-gamma 2.0, tail-offset 220.
           Auto-builds ACTIVE/reveal via apply_real_enrichment.py if absent.
  L3     : SEMVUL_QUAL_V2=0, SEMVUL_QUAL_GATE=1, SEMVUL_GATE_LR_MULT=100,
           --qual-gate. Encoders fine-tune (same regime as L1/L2).

  Outputs: experiments/runs/final_{devign,reveal}_l{1,2,3}_cache/, 5 seeds each,
  per seed: semanticvul_<ds>_<rung>.json, _partial.json, _probs.npz.

AGGREGATE THE LADDER (headline L1-L3 numbers, both scoring protocols):
  python src\rqs\aggregate_seeds.py both
  python src\rqs\aggregate_seeds.py both --json aggregate_live.json

REPRODUCE THE RQ ANALYSES  (read-only over caches; no GPU):
  python src\rqs\rq1.py both                                     # RQ1/RO1
  python -m src.rqs.rq2 --dataset devign --seeds 1,2,3,4,5       # RQ2/RO2
  python -m src.rqs.rq2 --dataset reveal --seeds 1,2,3,4,5
  python -m src.rqs.rq2_oracle_gate --dataset reveal --seeds 1,3,7   # diagnostic only
  python src\rqs\rq3.py both                                     # RQ3/RO3
  python src\rqs\rq4.py both                                     # RQ4/RO4
  python src\rqs\plots.py both --cache-prefix final              # all figures -> reports/plots/

  Note: rq1/rq3/rq4/plots/aggregate_seeds take a positional dataset
  (reveal|devign|both); no arg -> interactive menu. rq2 and rq2_oracle_gate are
  modules (python -m ...) and require --dataset.

DATA-QUALITY AUDIT (Devign; Section 3.2.1 / Table 3.2): same hashing workflow as
  Appendix B -- 124 within-train label-conflicting rows (62 pairs), 12 within-val
  (6 pairs), 27 cross-split reversed-label rows (~1.0% of 2,705 unique val
  functions), bounding un-deduplicated val accuracy near 2678/2705 ~= 99.0%.

-------------------------------------------------------------------------------
 6. SUGGESTED ORDER FOR A FRESH TAKEOVER
-------------------------------------------------------------------------------

  1) Set up the env (section 3); confirm experiments/runs/final_* and
     experiments/cache/ are present.
  2) Run the RQ analyses first (section 5) -- no GPU, fastest integrity check.
  3) aggregate_seeds.py both for the headline ladder numbers.
  4) Only to REBUILD results: run the six ladder launchers (GPU, no LLM).
  5) Only to REGENERATE explanations: stand up Ollama, run -Smoke, then scale.

-------------------------------------------------------------------------------
 7. NOTES / GOTCHAS
-------------------------------------------------------------------------------

  * requirements.txt is UTF-16 (pip reads it fine; editors may show a BOM).
  * torch/torchvision are intentionally NOT pinned in requirements.txt -- install
    the CUDA-matched wheel yourself first (section 3).
  * Large caches (experiments/runs/rq2_cache/*nfull*.npz, ~112-329MB) and LoRA
    checkpoints (experiments/cache/lora_ckpt/*.pt, ~520MB) are git-ignored: they
    exceed GitHub's file limit and are regenerable. Keep local copies.
  * The final caches use the semanticvul_* filename prefix; analysis scripts also
    accept the legacy fusevul_ladder_* prefix.

  Full details, folder-by-folder, in HANDOVER.md.
===============================================================================
