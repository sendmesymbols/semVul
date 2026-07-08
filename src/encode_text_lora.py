"""Fair-fight explanation encoder: LoRA fine-tune roberta-base on the flat
explanation text (binary vuln label), then extract mean-pooled embeddings.

Mirror of src/lora_finetune.py (the code side) so the explanation channel gets
the SAME training budget as the code channel — the frozen-MiniLM baseline in
encode_text.py was the "rigged fight". roberta-base is FuSEVul's stated
explanation encoder (docs/superpowers/specs/2026-07-03-fusevul-multimodal-design.md).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import CACHE_DIR, LORA_CFG
from src.data_io import load_split


TEXT_ENCODER_LORA = "roberta-base"   # HF id; resolved from models/hub via config env
TEXT_TAG = "roberta_lora"
TEXT_MAX_LEN = 256                   # explanations are short (spec: 256 for expl)

CKPT_DIR = CACHE_DIR / "lora_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(dataset: str, split: str) -> Path:
    return CACHE_DIR / f"{dataset}_{split}_text_{TEXT_TAG}.npz"


class _TextDS(Dataset):
    def __init__(self, samples, tok, max_len):
        self.samples = samples
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        enc = self.tok(
            s.explanation_text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(s.label, dtype=torch.long),
        }


class _LoRAClassifier(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.cls = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled, self.cls(pooled)


def _make():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER_LORA)
    base = AutoModel.from_pretrained(TEXT_ENCODER_LORA)
    lora = LoraConfig(
        r=LORA_CFG["r"], lora_alpha=LORA_CFG["alpha"],
        lora_dropout=LORA_CFG["dropout"],
        target_modules=LORA_CFG["target_modules"],
        bias="none", task_type="FEATURE_EXTRACTION",
    )
    base = get_peft_model(base, lora)
    return tok, _LoRAClassifier(base)


def train(dataset: str, force: bool = False) -> Path:
    """LoRA fine-tune the explanation encoder. Returns saved state_dict path."""
    ckpt = CKPT_DIR / f"{dataset}_text_{TEXT_TAG}.pt"
    if ckpt.exists() and not force:
        return ckpt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = _make()
    model.to(device)
    model.base.gradient_checkpointing_enable()

    train_s = load_split(dataset, "train")
    ds = _TextDS(train_s, tok, TEXT_MAX_LEN)
    dl = DataLoader(ds, batch_size=LORA_CFG["batch_size"], shuffle=True,
                    num_workers=0, pin_memory=(device == "cuda"))

    pos = sum(1 for s in train_s if s.label == 1)
    neg = len(train_s) - pos
    pos_w = torch.tensor([1.0, neg / max(1, pos)], device=device, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=pos_w)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LORA_CFG["lr"],
                            weight_decay=LORA_CFG["weight_decay"])
    steps_total = (len(dl) // LORA_CFG["grad_accum"]) * LORA_CFG["epochs"]
    warmup = max(1, int(steps_total * LORA_CFG["warmup_ratio"]))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    step = 0
    model.train()
    for ep in range(LORA_CFG["epochs"]):
        opt.zero_grad()
        pbar = tqdm(dl, desc=f"lora roberta/{dataset} ep{ep+1}", mininterval=30.0)
        running = 0.0
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                _, logits = model(input_ids, attn)
                loss = loss_fn(logits, y) / LORA_CFG["grad_accum"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * LORA_CFG["grad_accum"]
            if (i + 1) % LORA_CFG["grad_accum"] == 0:
                for pg in opt.param_groups:
                    pg["lr"] = LORA_CFG["lr"] * lr_at(step)
                if scaler is not None:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                step += 1
                pbar.set_postfix(loss=f"{running/(i+1):.4f}",
                                 lr=f"{opt.param_groups[0]['lr']:.2e}")

    torch.save({"state_dict": model.state_dict()}, ckpt)
    print(f"[text-lora] saved -> {ckpt.name}")
    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return ckpt


@torch.no_grad()
def encode(dataset: str, split: str, batch_size: int = 64, force: bool = False) -> Path:
    """Extract mean-pooled embeddings from the LoRA-tuned explanation encoder."""
    out = cache_path(dataset, split)
    if out.exists() and not force:
        return out

    ckpt = CKPT_DIR / f"{dataset}_text_{TEXT_TAG}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing text-LoRA checkpoint: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = _make()
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.to(device).eval()

    samples = load_split(dataset, split)
    texts = [s.explanation_text for s in samples]
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    ids = np.asarray([s.sample_id for s in samples], dtype=object)

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"text-lora-enc {split}"):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=TEXT_MAX_LEN, return_tensors="pt").to(device)
        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            pooled, _ = model(enc["input_ids"], enc["attention_mask"])
        all_embs.append(pooled.float().cpu().numpy())

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.savez_compressed(out, embeddings=embs, labels=labels, sample_ids=ids)
    print(f"[text-lora-enc] {out.name} shape={embs.shape}")
    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "devign"
    train(ds)
    encode(ds, "train")
    encode(ds, "val")
