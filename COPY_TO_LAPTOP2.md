# Copy to laptop 2 (Devign overnight, 16 GB GPU) — no git available

You can't pull, so copy files directly. Two options.

## Option A (simplest): copy the whole project

Copy all of `D:\Projects\SemVul` **except**:
- `.venv\`  (recreate on laptop 2 — see below)
- `experiments\runs\`  (old caches; not needed, and large)

Put it at the same path if you can (`D:\Projects\SemVul`). Done.

## Option B (minimal, ~130 MB): copy only what a Devign run needs

The run reads exactly **two** files — the ACTIVE pair — plus the code:

1. **Code**
   - `src\`  (whole folder)
   - `experiments\`  (whole folder; or at least `experiments\expl_enrich\` + `experiments\fusevul_ladder\`)
2. **The ACTIVE Devign data** (this is the only data needed now)
   - `explanations\SemanticVul\ACTIVE\devign\train.jsonl`
   - `explanations\SemanticVul\ACTIVE\devign\val.jsonl`
   - (keep the `explanations\SemanticVul\ACTIVE\devign\` path). The loaders read
     ACTIVE directly — you do **not** need the long-named `.real` files,
     `devign_real\`, or `full_code\`.
3. **The PowerShell launchers** (repo root)
   - `reproduce_devign.ps1`, `reproduce_devign_l1.ps1`, `reproduce_devign_l2.ps1`,
     `reproduce_devign_l3.ps1`, `produce_devign.ps1`, `make_ladder_devign.ps1`
4. **`src\config.py` paths** — already relative to the repo root; nothing to edit.

## Environment on laptop 2 (needed either way)

- **Python packages**: recreate the venv, or copy `.venv\` only if laptop 2 has
  no internet AND the project sits at the exact same path `D:\Projects\SemVul`.
  To recreate: `py -m venv .venv; .\.venv\Scripts\pip install torch transformers scikit-learn numpy`
  (match the torch CUDA build to the GPU).
- **Model weights** (downloaded on first run): `microsoft/graphcodebert-base` and
  `roberta-base`. If laptop 2 has internet, they download automatically (~600 MB).
  If it's **offline**, also copy the HuggingFace cache from this machine:
  `C:\Users\<you>\.cache\huggingface\`  ->  same path on laptop 2.

## Run it (laptop 2, 16 GB -> batch 4)

```powershell
cd D:\Projects\SemVul
.\reproduce_devign.ps1                 # L1 -> L2 -> L3 (batch 4), then the ladder report
# or one rung at a time:
.\reproduce_devign_l1.ps1
.\reproduce_devign_l2.ps1
.\reproduce_devign_l3.ps1
.\make_ladder_devign.ps1               # gather + report (no training)
```

Per-rung ~2.5 h at batch 4; 3 rungs, 1 seed ≈ one night. Caches land in
`experiments\runs\l1_devign_cache\`, `l2_devign_cache\`, `l3_devign_cache\`.

## Bringing results back

Copy `experiments\runs\l1_devign_cache\`, `l2_devign_cache\`, `l3_devign_cache\`
off laptop 2. On any machine with the repo, drop them into `experiments\runs\`
and run `.\produce_devign.ps1` for the merged ladder + ensemble verdict.

## Note on the L1 you already copied from the first desktop

That earlier cache is `experiments\runs\enriched512_real\...`, a different folder
name than the new `l1_devign_cache`. It won't be auto-picked by the new ladder
scripts. Simplest: just let laptop 2 retrain L1 fresh (it's one rung). If you want
to reuse it, rename that folder to `l1_devign_cache` and move its `s1337` subdir in.
