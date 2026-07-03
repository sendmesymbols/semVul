import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
EXPL_DIR = ROOT / "explanations" / "SemanticVul"

# All HuggingFace models cache under the project (moved out of ~/.cache/huggingface).
# Layout: models/hub/models--<owner>--<name>/...  (this is HF's own convention).
# Must be set BEFORE transformers/sentence-transformers/huggingface_hub import.
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR / "hub"))

EXP_DIR    = ROOT / "experiments"
CACHE_DIR  = EXP_DIR / "cache"
RUNS_DIR   = EXP_DIR / "runs"
REPORTS_DIR = EXP_DIR / "reports"

for d in (CACHE_DIR, RUNS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = ("devign", "reveal")
SPLITS   = ("train", "val")

FUSEVUL_TARGETS = {
    "devign": {"accuracy": 60.39, "f1": 55.91, "precision": None, "recall": None},
    "reveal": {"accuracy": 91.68, "f1": 46.76, "precision": 57.24, "recall": 39.52},
}

CODE_ENCODERS = {
    "graphcodebert": "microsoft/graphcodebert-base",
    # unixcoder replaces codet5p as the L2 partner: RoBERTa-based (drop-in for
    # AutoModel + mean-pool), but pretrained with a different objective
    # (cross-modal understanding+generation) so signal is orthogonal.
    "unixcoder":     "microsoft/unixcoder-base",
}
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

LORA_CFG = dict(
    r=32, alpha=32, dropout=0.1,
    target_modules=["query", "key", "value", "dense"],
    max_len=512, batch_size=8, grad_accum=8,
    lr=1e-4, epochs=5, weight_decay=0.01, warmup_ratio=0.06,
)

HEAD_CFG = dict(
    proj_dim=256, hidden=256, dropout=0.3,
    batch_size=256, lr=2e-4, weight_decay=0.01, epochs=40,
    early_stop_patience=6, focal_gamma=1.0, alpha_pos_cap=0.75,
    seeds=(1337, 2024, 42),
    tune_frac=0.15, tune_seed=1337,
)

QUALITY_FEATURE_NAMES = [
    "len_purpose", "len_dataflow", "len_risk_summary",
    "n_risky_ops", "n_missing_checks", "n_evidence_tokens",
    "evidence_char_total", "evidence_overlap_code",
    "kw_memory", "kw_pointer", "kw_bounds", "kw_validation",
    "kw_integer", "kw_input", "kw_null", "kw_concurrency",
    "api_mem_ops", "api_str_ops", "api_alloc_ops", "api_io_ops",
    "n_sentences_dataflow", "has_missing_check_language",
]
