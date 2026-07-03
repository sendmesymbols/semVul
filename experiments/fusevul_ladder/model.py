"""FuSEVul-style component-ladder model, implemented correctly in our code.

L1: CodeT5+ (fine-tuned) code tokens -> mean-pool -> head.
L2: + RoBERTa (fine-tuned) explanation tokens, fused with code via multi-head
    self/cross attention (code queries attend over [code; explanation] tokens),
    residual+LayerNorm, pooled -> head.   <-- the explanation actually flows.
L3: + 22 quality features concatenated before the head.

Both encoders are 768-dim (CodeT5+-110m-embedding encoder and roberta-base), so
their token states share a space for the attention fusion.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LadderModel(nn.Module):
    def __init__(self, code_enc, text_enc, hidden=768, qual_dim=22,
                 rung="L3", fusion="self", n_heads=8, dropout=0.3):
        super().__init__()
        assert rung in ("L1", "L2", "L3")
        assert fusion in ("self", "cross")
        self.rung = rung
        self.fusion = fusion
        self.use_expl = rung in ("L2", "L3")
        self.use_qual = rung == "L3"

        self.code_enc = code_enc
        self.text_enc = text_enc if self.use_expl else None
        if self.use_expl:
            self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                              batch_first=True)
            self.ln = nn.LayerNorm(hidden)

        feat = hidden + (qual_dim if self.use_qual else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def _code_tokens(self, ids, mask):
        # GraphCodeBERT / RoBERTa-style encoder: returns token states directly.
        return self.code_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    @staticmethod
    def _pool(h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, code_ids, code_mask, text_ids, text_mask, qual):
        ch = self._code_tokens(code_ids, code_mask)              # [B, Tc, H]
        if not self.use_expl:
            pooled = self._pool(ch, code_mask)
        else:
            th = self.text_enc(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state  # [B,Tt,H]
            if self.fusion == "self":
                kv = torch.cat([ch, th], dim=1)
                kvm = torch.cat([code_mask, text_mask], dim=1)
            else:  # cross: code queries attend over explanation only
                kv = th
                kvm = text_mask
            attn_out, _ = self.attn(ch, kv, kv, key_padding_mask=(kvm == 0))
            fused = self.ln(ch + attn_out)                       # residual
            pooled = self._pool(fused, code_mask)
        if self.use_qual:
            pooled = torch.cat([pooled, qual], dim=-1)
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
