"""Frozen code encoders: GraphCodeBERT / CodeT5+. Mean-pooled [CLS]/last hidden state."""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import CACHE_DIR, CODE_ENCODERS
from src.data_io import load_split


def cache_path(dataset: str, split: str, encoder: str, tag: str = "frozen") -> Path:
    return CACHE_DIR / f"{dataset}_{split}_code_{encoder}_{tag}.npz"


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode(dataset: str, split: str, encoder: str = "graphcodebert",
           batch_size: int = 16, max_len: int = 512, force: bool = False) -> Path:
    out = cache_path(dataset, split, encoder, "frozen")
    if out.exists() and not force:
        return out
    if encoder not in CODE_ENCODERS:
        raise ValueError(f"Unknown code encoder: {encoder}")

    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = CODE_ENCODERS[encoder]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    if device == "cuda":
        model = model.half()

    samples = load_split(dataset, split)
    codes = [s.code for s in samples]
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    ids = np.asarray([s.sample_id for s in samples], dtype=object)

    all_embs = []
    for i in tqdm(range(0, len(codes), batch_size), desc=f"code:{encoder}:{split}"):
        batch = codes[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        out_h = model(**enc)
        if hasattr(out_h, "last_hidden_state"):
            h = out_h.last_hidden_state
        else:
            h = out_h[0]
        pooled = _mean_pool(h, enc["attention_mask"]).float().cpu().numpy()
        all_embs.append(pooled)

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.savez_compressed(out, embeddings=embs, labels=labels, sample_ids=ids)
    print(f"[code-enc] {out.name} shape={embs.shape}")

    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return out
