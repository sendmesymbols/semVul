# Annotator-ceiling pilot — can the strongest LLM annotator beat the code channel?

Date: 2026-07-08. Direct test of Part C's premise, with Claude itself as the
annotator (8 parallel subagents), judging **blind** (code only, labels withheld
by the harness) under a patch-oriented rubric: "would a security patch need to
change this function?"

Sample: 60 val functions per dataset, stratified 30 vulnerable / 30 benign,
seed 20260708. Annotator emits a 0-1 risk per function. Scored against the
withheld labels and against the code channel (GraphCodeBERT-LoRA L1 val prob on
the same functions). Bootstrap 5000.

## Result

| | annotator (Claude, blind, code-only) | code channel (L1) | delta annotator-code (95% CI) |
|---|---|---|---|
| Devign | AUC 0.569 [0.42, 0.72] | 0.642 | -0.072 [-0.25, +0.12] |
| ReVeal | AUC 0.585 [0.43, 0.73] | 0.901 | **-0.317 [-0.48, -0.16]** |

- Both annotator CIs include 0.5 -> not reliably above chance on the raw task.
- Annotator never beats code; on ReVeal it is significantly worse.
- Complementarity (annotator AUC on samples the code model gets wrong): 0.44
  (Devign) / 0.58 (ReVeal) -> no orthogonal signal to rescue code's errors.
- 50/50 blend of code+annotator: Devign 0.646 (~=code 0.642), ReVeal 0.858
  (< code 0.901) -> adding the annotator does not help and can hurt.

## Interpretation

This is the annotator ceiling measured at the source, not inferred from a weaker
generator. A frontier model reading the code directly, with the best rubric,
cannot classify these functions better than chance and cannot beat the code
encoder. The subagents flagged plausible, verbatim-cited bugs — but "looks like
a bug" does not equal "is a function a commit labeled vulnerable." Devign benign
samples are frequently the patched twin of a vulnerable function and still
contain scary-looking patterns, so pattern-matching does not track the label.

## Decision

Part C (re-annotation for PREDICTION) is closed on both datasets: no annotator
adds signal over the code channel. This confirms the incremental-over-code gate
(experiments/expl_enrich/probe_incremental.py: devign -1.36, reveal +0.26 neutral).
The explanation channel's defensible contribution is grounding/faithfulness
(RQ1), already delivered by the static enrichment. Accuracy levers remain the
code window (512 / evidence-centered) and the ensemble.

Artifacts: scratchpad/pilot/{batch,ann,key}_*.json; score_pilot.py.
Caveat: n=60/dataset, wide CIs; the finding is "annotator << code," which is
robust at this n, not a precise annotator-AUC estimate.
