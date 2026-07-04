Where each field goes
JSON field	Where used	Channel
sample_id	dedup + alignment	plumbing (not a feature)
label	training target	supervision
raw_code	tokenized by GraphCodeBERT	code encoder input
explanation.purpose	joined into explanation_text → RoBERTa	explanation text (L2, L3)
explanation.data_flow	same	explanation text (L2, L3)
explanation.risky_operations[]	joined into explanation_text (as space-separated strings)	explanation text (L2, L3)
explanation.missing_checks[]	same	explanation text (L2, L3)
explanation.risk_summary	same	explanation text (L2, L3)
explanation.evidence_tokens[]	❌ NOT in explanation_text — but IS in quality features (count + code overlap)	quality features only (L3)
What "explanation_text" concretely is
Per src/data_io.py:48, the string handed to RoBERTa is:

purpose + " " + data_flow + " " + " ".join(risky_operations)
       + " " + " ".join(missing_checks) + " " + risk_summary
No JSON braces, no field labels — a flat paragraph. evidence_tokens is deliberately omitted from the text.

What the 22 quality features (L3) extract from the JSON
Per src/config.py:58-66 and src/quality_features.py:

Length signals: len_purpose, len_dataflow, len_risk_summary, n_sentences_dataflow
Cardinality: n_risky_ops, n_missing_checks, n_evidence_tokens
Evidence↔code overlap: evidence_char_total, evidence_overlap_code (uses evidence_tokens)
Keyword counters over the explanation text: kw_memory, kw_pointer, kw_bounds, kw_validation, kw_integer, kw_input, kw_null, kw_concurrency
API counters over the code: api_mem_ops, api_str_ops, api_alloc_ops, api_io_ops
Structural: has_missing_check_language
Per-rung recap
L1 — only raw_code + label used. Explanation JSON ignored.
L2 — raw_code, label, and the joined explanation_text (5 fields, no evidence_tokens).
L3 — everything L2 uses, plus the 22 quality features (which is where evidence_tokens finally enters, via count + overlap with the code).