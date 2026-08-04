# ACTIVE explanations — the ONLY files a run consumes

Non-destructive snapshot (originals untouched). Each dataset uses exactly **two**
files. Verified against `src/data_io.py`, `experiments/fusevul_ladder/data.py`,
and `experiments/expl_enrich/reproduce_real.py`.

This folder IS the run input. The loaders read it directly (reproduce_real.py
sets `SEMVUL_ACTIVE_DIR=1`; when ACTIVE is absent they fall back to the long-named
`.real` files). `apply_real_enrichment.py` refreshes these four files on every run,
so ACTIVE always reflects the current data — including ReVeal's per-run
`tail_digest`. Copying just this folder onto another machine is enough to run.

| dataset | role  | ACTIVE copy        | real source the run reads                    | rows  |
|---------|-------|--------------------|----------------------------------------------|-------|
| devign  | train | devign/train.jsonl | devign_train.enriched.clean.aug.real.jsonl   | 41316 |
| devign  | val   | devign/val.jsonl   | devign_val.enriched.real.jsonl               | 2732  |
| reveal  | train | reveal/train.jsonl | reveal_train.enriched.clean.real.jsonl       | 17692 |
| reveal  | val   | reveal/val.jsonl   | reveal_val.enriched.real.jsonl               | 2273  |

**Devign** — explanation text = de-anonymized fields + `evidence_tokens` +
`lexical_digest`; `raw_code` stays the anon benchmark code (comparable to
FuSEVul). val = full benchmark val (2732; ~67% carry de-anon/digest, rest pass
through unchanged).

**ReVeal** — real code already; explanation = CORE fields + `lexical_digest` +
`tail_digest` (beyond-window callees/literals). The ReVeal `.real` files are
**regenerated with `tail_digest` every run** by `apply_real_enrichment.py --only
reveal`; re-run `organize_explanations.ps1` after a ReVeal run to refresh this
snapshot.

Everything else under `devign/` and `reveal/` (`.clean`, `.enriched.clean`,
plain `.real`, `devign_real/`, `full_code/`) is intermediate/source and is **not
read at train time**.
