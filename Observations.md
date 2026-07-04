Here are few observations noted by my professor and committee members, I want to make sure I have these covered in my reports



Member Name: Ihtesham Ul Islam __________________________________ Member Role: Advisor ________________________________

Observation:

+ plan of ablation study is not there.

Action To be taken:

+ The student will perform experiments to report the significance in terms of accuracy/performance measures of individual blocks in the ML Pipeline designed for the problem.



Member Name: Nazia Bibi __________________________________ Member Role: Committee Member ________________________________

Observation:

The proposed methodology requires further justification regarding selected techniques and

the evaluation strategy.

Action To be taken:

Provide detailed methodological justification and define evaluation metrics with expected

outcomes.

These need to be done for both, paper ready, furnished



Here are my ROs and RQs

Canonical 4-RQ / 4-RO defence scheme (locked 2026-06-03 in decisions.md D10.3–D10.4, one-to-one mapping):

#	Research Objective (RO)	Research Question (RQ)	Evidence (where answered)	Output requirement (deliverable)
1	RO1. Design and evaluate a local explanation pipeline using structured JSON output, evidence-token grounding, and explicit leakage controls.	RQ1. To what extent do locally generated, verdict-scrubbed, evidence-grounded explanations improve explanation faithfulness and downstream detection utility vs FuSEVul-style free-form LLM explanations?	Methodology §3.4–§3.6 (JSON schema, evidence tokens, leakage_controls.md); Results §4.8 (faithfulness metrics). Generator = Qwen2.5-Coder via Ollama, replacing FuSEVul's ChatGPT.	Structured JSON explanations generated for Devign + Reveal + faithfulness/leakage table (evidence-token overlap, verdict-token scrub audit, faithfulness vs FuSEVul deltas).
2	RO2. Develop and evaluate a lightweight gated fusion module combining cached code embeddings, explanation embeddings, and label-free quality features.	RQ2. How does quality-aware adaptive gated fusion compare with static fusion, single-modality models, and classical cached-feature baselines in performance and training efficiency?	Methodology §3.8 (locked verbatim in D10.6: per-dim sigmoid gate, fused = [code × (1−gate), explanation × gate, code × explanation, quality_features]); Results §4.3 ablation ladder.	Ablation table: code-only vs exp-only vs static self-attention (FuSEVul) vs adaptive gated fusion, on both datasets, ≥5 seeds, with training-efficiency (wall-clock, GPU-hours).
3	RO3. Evaluate SemanticVul against FuSEVul + representative baselines under audited, explicitly declared experimental protocols.	RQ3. Under audited Devign + Reveal splits, how does SemanticVul compare with FuSEVul and representative baselines in Accuracy, F1, Recall, threshold robustness, and low-resource training feasibility?	Methodology §3.9 (protocol, experimental_setup.md); Results §4.5 headline op-point + §4.6 encoder×classifier benchmark; results_record.md.	Headline table beating FuSEVul on Acc and F1 on both datasets (one op-point per dataset per D1); 5-seed reporting; data-quality audit; 8 GB-feasibility statement.
4	RO4. Investigate focal loss, capped class weighting, validation-threshold tuning, multi-seed ensembling through controlled ablation.	RQ4. What are the individual and combined effects of imbalance-aware loss, threshold tuning, and multi-seed ensembling on minority-class detection and the precision–recall trade-off?	Methodology §3.10; Results §4.7 (imbalance ablation, threshold sweep, ensemble gain — Reveal is the harder case).	Per-lever ablation table (focal on/off, capped CW on/off, val-threshold on/off, ensemble on/off) with minority-class Precision/Recall/F1 + PR curves.


Co-supervisor observation: JSON-schema + Quality-Features effectiveness study
This is a committed experiment, recorded in two independent places in the grounding pack — treat both together as binding:

1. Proposal-presentation examiner directive (2026-06-24) — decisions.md D11.3, risk K21:

Run an ablation with and without the structured JSON explanation schema, and with and without the 22 quality features, on BOTH Devign and Reveal, to quantify how much each contributes. The examiner stressed the study must be carried out on both datasets, not one.

Motivation (from author's own framing): this is the empirical answer to the "INPUT-only / just prompt engineering" critique on contribution C2 (risk K3) and the label-only-input gap G2 (risk K14). Analogous to how the base paper (FuSEVul) isolates the explanation channel's F1 contribution, SemanticVul must isolate the schema and quality-feature contributions to F1.
Where numbers go: scaffold already in results_record.md — cells TBD until run. Do not fabricate; the whole point is to measure the effect (D8/D11.3).
Where the write-up goes: Results §4.3 propagation is pending the actual numbers.
2. Advisor Ihtesham Ul Islam — written observation sheet — proposal-defence-examiner-observations:

"Plan of ablation study is not there." Action: perform experiments reporting the significance (accuracy/performance measures) of individual blocks in the ML pipeline.

Broader scope than D11.3 — covers per-block isolation of: explanation channel, JSON schema, 22 quality features, the gate, focal loss, threshold tuning, ensemble.
Ties into: Methodology needs an explicit ablation-study plan; Results must report per-block deltas (a significance table).
Required output (both observations combined):

Condition	Devign — Acc / F1 / Recall (±σ)	Reveal — Acc / F1 / Recall (±σ)
Full SemanticVul (JSON schema + 22 QF + gated fusion)	TBD	TBD
Full − JSON schema (free-form explanations)	TBD	TBD
Full − 22 quality features (gate on code+exp only)	TBD	TBD
Full − JSON schema − 22 quality features	TBD	TBD
Also owed by Nazia Bibi's parallel observation: a per-technique justification ("why this, not that") + a metrics-definition table with expected outcomes.



why do we need more JSON schema (free form)?

Let me explain again

My base paper reported that when he ran his experiment  and said that when I fed my algorithm raw code, it drops F1 to 8.1%. He argued that when I created explnations and fed it to the algo i saw 8 % increase in F1.

This inspired my reviewer and asked me to create ablation study for comparison 