"""Gated fusion + ablation-friendly heads. Kept small and readable."""
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GatedFusion(nn.Module):
    """Ladder-agnostic head.

    Inputs:
      code (B, Dc)       -- concatenation of any number of code embeddings (L1/L2)
      expl (B, De)  or 0 -- explanation embedding (or None to ablate)
      qual (B, Dq)  or 0 -- quality features (or None to ablate)

    Config knobs:
      fusion: "gated" (default) or "concat"
      use_expl / use_qual: switch off channels for ablation
    """
    def __init__(self, code_dim: int, expl_dim: int, qual_dim: int,
                 proj_dim: int = 256, hidden: int = 256, dropout: float = 0.3,
                 fusion: str = "gated",
                 use_expl: bool = True, use_qual: bool = True):
        super().__init__()
        self.use_expl = use_expl
        self.use_qual = use_qual and qual_dim > 0
        self.fusion = fusion

        self.code_proj = nn.Linear(code_dim, proj_dim)
        self.expl_proj = nn.Linear(expl_dim, proj_dim) if self.use_expl else None

        gate_in = proj_dim * (2 if self.use_expl else 1) + (qual_dim if self.use_qual else 0)
        self.gate = None
        if fusion == "gated" and self.use_expl:
            self.gate = nn.Sequential(
                nn.Linear(gate_in, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Sigmoid(),
            )

        fused_dim = proj_dim
        if self.use_expl:
            fused_dim += proj_dim  # code and expl always both flow to head
        if self.use_qual:
            fused_dim += qual_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, code, expl=None, qual=None):
        c = self.code_proj(code)
        parts = [c]
        if self.use_expl:
            e = self.expl_proj(expl)
            if self.gate is not None:
                cat = torch.cat([c, e] + ([qual] if self.use_qual else []), dim=-1)
                g = self.gate(cat)
                e = g * e + (1 - g) * c
            parts.append(e)
        if self.use_qual:
            parts.append(qual)
        fused = torch.cat(parts, dim=-1)
        return self.head(fused).squeeze(-1)


def focal_bce(logits, targets, gamma: float = 1.0, alpha_pos: float = 0.5):
    """Focal BCE with a class-balance weight alpha_pos in (0,1)."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p).clamp(1e-6, 1 - 1e-6)
    w  = torch.where(targets == 1, alpha_pos, 1 - alpha_pos)
    loss = -w * (1 - pt).pow(gamma) * torch.log(pt)
    return loss.mean()
