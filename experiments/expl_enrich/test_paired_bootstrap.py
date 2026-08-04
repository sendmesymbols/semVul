"""Self-running tests for paired_roc_delta (no pytest).
Run: .venv\\Scripts\\python.exe experiments\\expl_enrich\\test_paired_bootstrap.py
"""
import numpy as np
from paired_bootstrap import paired_roc_delta


def test_treated_strictly_better_has_positive_ci():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.1).astype(int)          # ~9% positives (ReVeal-like)
    p_base = rng.random(n)                          # uninformative
    p_treat = np.clip(0.4 * p_base + 0.6 * y + 0.05 * rng.random(n), 0, 1)  # tracks y
    r = paired_roc_delta(y, p_base, p_treat, n_boot=500, seed=1)
    assert r["delta"] > 0
    assert r["ci"][0] > 0          # lower CI bound above zero -> real lift


def test_identical_probs_delta_zero():
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.random(n) < 0.1).astype(int)
    p = rng.random(n)
    r = paired_roc_delta(y, p, p.copy(), n_boot=300, seed=3)
    assert abs(r["delta"]) < 1e-9
    assert r["ci"][0] <= 0 <= r["ci"][1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print("ALL PASS")
