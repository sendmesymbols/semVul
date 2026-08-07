# ACTIVE explanations -- the ONLY files a run consumes

Written by experiments/explanation/pipeline.py (stage 6). Each dataset uses
exactly two files; the training wrappers read these via SEMVUL_ACTIVE_DIR=1.

| dataset | role  | ACTIVE copy        | built from                                 |
|---------|-------|--------------------|--------------------------------------------|
| devign  | train | devign/train.jsonl | devign_train.enriched.clean.aug.real.jsonl |
| devign  | val   | devign/val.jsonl   | devign_val.enriched.real.jsonl             |
| reveal  | train | reveal/train.jsonl | reveal_train.enriched.clean.real.jsonl     |
| reveal  | val   | reveal/val.jsonl   | reveal_val.enriched.real.jsonl             |

explanation.confidence is MEASURED from the generator's decode-time token
logprobs over the risk_level verdict span (experiments/explanation/generate.py),
not self-reported and not derived from the label.

explanation.prefix is rebuilt by experiments/expl_enrich/build_prefix.py. Check
its byte-fidelity against a reference tree at any time with `--verify`.
