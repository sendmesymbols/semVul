"""FuSEVul-style component-ladder model, implemented correctly in our code.

L1: CodeT5+ (fine-tuned) code tokens -> mean-pool -> head.
L2: + RoBERTa (fine-tuned) explanation tokens, fused with code via multi-head
    self/cross attention (code queries attend over [code; explanation] tokens),
    residual+LayerNorm, pooled -> head.   <-- the explanation actually flows.
L3: L2 + adaptive gate fed by pooled explanation representation (768-dim) +
    confidence (1-dim). The gate learns per-dimension scaling of the fusion
    output: g = sigmoid(MLP([expl_pooled; conf])). Quality features (22-dim)
    removed — proven useless (near-zero per-sample variance).

Both encoders are 768-dim (CodeT5+-110m-embedding encoder and roberta-base), so
their token states share a space for the attention fusion.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn


class LadderModel(nn.Module):
    def __init__(self, code_enc, text_enc, hidden=768, qual_dim=22,
                 rung="L3", fusion="self", n_heads=8, dropout=0.3, code_kind=""):
        super().__init__()
        assert rung in ("L1", "L2", "L3")
        assert fusion in ("self", "cross")
        self.rung = rung
        self.fusion = fusion
        # code_kind="t5" -> read token states from .encoder (CodeT5+). Default
        # "" -> call the encoder directly (GraphCodeBERT / RoBERTa-style).
        self.code_kind = code_kind
        self.use_expl = rung in ("L2", "L3")
        self.text_only = os.environ.get("SEMVUL_TEXT_ONLY") == "1" and self.use_expl
        # Hard confidence switch (SEMVUL_HARD_CONF_SWITCH=1): per-sample if/else
        # on the raw explanation.confidence field. Confidence >= threshold ->
        # code ALONE; below -> explanation ALONE. Validated direction: HIGH
        # confidence -> code (oracle-bound AUC 0.72-0.85 across 5 seeds).
        self.hard_conf_switch = (os.environ.get("SEMVUL_HARD_CONF_SWITCH") == "1"
                                 and rung == "L3" and self.use_expl)
        self.hard_conf_thresh = float(os.environ.get("SEMVUL_HARD_CONF_THRESH", "50"))
        self.hard_conf_flip = os.environ.get("SEMVUL_HARD_CONF_LEGACY_DIR") != "1"
        self.hard_conf_legacy_fusion = os.environ.get("SEMVUL_HARD_CONF_LEGACY_FUSION") == "1"
        # Adaptive gate (L3): g = sigmoid(MLP([expl_pooled; confidence])).
        # Input = pooled explanation (768) + confidence scalar (1) = 769.
        # Unlike the old quality-feature gate (inert because qual had no
        # per-sample variance), this gate sees the actual explanation content
        # plus confidence — both vary meaningfully per sample.
        self.use_gate = (os.environ.get("SEMVUL_QUAL_GATE") == "1"
                         and rung == "L3" and self.use_expl)

        self.code_enc = code_enc
        self.text_enc = text_enc if self.use_expl else None
        if self.use_expl:
            self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                              batch_first=True)
            self.ln = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )
        if self.use_gate:
            gate_in = hidden + 1  # pooled explanation (768) + confidence (1)
            self.gate = nn.Sequential(
                nn.Linear(gate_in, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, 0.0)

    def _code_tokens(self, ids, mask):
        # GraphCodeBERT / RoBERTa-style encoder: returns token states directly.
        # CodeT5+ (codet5p-*-embedding): its top-level forward returns a 256-dim
        # normalized embedding, so reach INTO .encoder for 768-dim token states
        # (the FuSEVul-comparable path). Selected via code_kind="t5".
        if getattr(self, "code_kind", "") == "t5":
            return self.code_enc.encoder(input_ids=ids,
                                         attention_mask=mask).last_hidden_state
        return self.code_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    @staticmethod
    def _pool(h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, code_ids, code_mask, text_ids, text_mask, qual, conf=None):
        if self.text_only:
            th = self.text_enc(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state
            return self.head(self._pool(th, text_mask))
        if self.hard_conf_switch and conf is not None:
            ch = self._code_tokens(code_ids, code_mask)
            code_pooled = self._pool(ch, code_mask)
            th = self.text_enc(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state
            if self.hard_conf_legacy_fusion:
                if self.fusion == "self":
                    kv = torch.cat([ch, th], dim=1)
                    kvm = torch.cat([code_mask, text_mask], dim=1)
                else:
                    kv = th
                    kvm = text_mask
                attn_out, _ = self.attn(ch, kv, kv, key_padding_mask=(kvm == 0))
                fused = self.ln(ch + attn_out)
                expl_pooled = self._pool(fused, code_mask)
            else:
                expl_pooled = self._pool(th, text_mask)
            cf = conf.to(expl_pooled.dtype)
            code_side = (cf >= self.hard_conf_thresh) if self.hard_conf_flip else (cf < self.hard_conf_thresh)
            pooled = torch.where(code_side.view(-1, 1), code_pooled, expl_pooled)
            return self.head(pooled)
        ch = self._code_tokens(code_ids, code_mask)
        if not self.use_expl:
            pooled = self._pool(ch, code_mask)
        elif self.use_gate and conf is not None:
            th = self.text_enc(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state
            code_pooled = self._pool(ch, code_mask)               # [B, H]
            expl_pooled = self._pool(th, text_mask)               # [B, H]
            cf = conf.to(expl_pooled.dtype).view(-1, 1)           # [B, 1]
            gate_in = torch.cat([expl_pooled, cf], dim=-1)        # [B, H+1]
            w = torch.sigmoid(self.gate(gate_in))                 # [B, 1]
            pooled = w * code_pooled + (1 - w) * expl_pooled      # soft routing
        else:
            th = self.text_enc(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state
            if self.fusion == "self":
                kv = torch.cat([ch, th], dim=1)
                kvm = torch.cat([code_mask, text_mask], dim=1)
            else:
                kv = th
                kvm = text_mask
            attn_out, _ = self.attn(ch, kv, kv, key_padding_mask=(kvm == 0))
            fused = self.ln(ch + attn_out)
            pooled = self._pool(fused, code_mask)
        return self.head(pooled)

    def enable_grad_checkpointing(self):
        for m in (self.code_enc, getattr(self.code_enc, "encoder", None), self.text_enc):
            if m is None:
                continue
            try:
                m.gradient_checkpointing_enable()
                if hasattr(m, "config"):
                    m.config.use_cache = False
            except Exception:
                pass


def focal_ce(logits, targets, alpha_pos=0.75, gamma=2.0):
    """Focal cross-entropy for imbalanced (Reveal). alpha_pos weights positives."""
    logp = torch.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    alpha = torch.where(targets == 1, torch.as_tensor(alpha_pos, device=logits.device),
                        torch.as_tensor(1.0 - alpha_pos, device=logits.device))
    return -(alpha * (1.0 - pt).pow(gamma) * logpt).mean()
