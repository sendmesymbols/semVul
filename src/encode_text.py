"""MiniLM sentence embeddings for the flat explanation text."""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import CACHE_DIR, TEXT_ENCODER
from src.data_io import load_split


def cache_path(dataset: str, split: str) -> Path:
    return CACHE_DIR / f"{dataset}_{split}_text_minilm.npz"


def encode(dataset: str, split: str, batch_size: int = 64, force: bool = False) -> Path:
    out = cache_path(dataset, split)
    if out.exists() and not force:
        return out
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(TEXT_ENCODER, device=device)

    samples = load_split(dataset, split)
    texts = [s.explanation_text for s in samples]
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    ids = np.asarray([s.sample_id for s in samples], dtype=object)

    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)

    np.savez_compressed(out, embeddings=embs, labels=labels, sample_ids=ids)
    print(f"[text-enc] {out.name} shape={embs.shape}")
    return out
