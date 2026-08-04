"""Self-running tests for reproduce_real._build_jobs (no pytest).
Run: .venv\\Scripts\\python.exe experiments\\expl_enrich\\test_build_jobs.py
"""
from reproduce_real import _build_jobs


class _Args:
    only = None; seeds = [1337]; batch512 = 2; rungs = None
    tail_digest = False; focal_alpha = None; focal_gamma = 2.0; out_tag = ""


def _args(**kw):
    a = _Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _job(jobs, ds):
    return next(j for j in jobs if j[0] == ds)


def test_devign_never_gets_focal_or_tail():
    jobs = _build_jobs(_args(focal_alpha=0.9, focal_gamma=3.0, tail_digest=True))
    ds, rungs, sub, suffix, fields, kw = _job(jobs, "devign")
    assert "focal_alpha" not in kw and "focal_gamma" not in kw
    assert "tail_digest" not in fields


def test_reveal_gets_focal_and_tail():
    jobs = _build_jobs(_args(focal_alpha=0.9, focal_gamma=3.0,
                             tail_digest=True, out_tag="_tail_a90"))
    ds, rungs, sub, suffix, fields, kw = _job(jobs, "reveal")
    assert kw.get("focal_alpha") == 0.9 and kw.get("focal_gamma") == 3.0
    assert "tail_digest" in fields
    assert sub == "enriched_real_tail_a90"


def test_reveal_baseline_has_no_tail_or_focal():
    jobs = _build_jobs(_args())
    ds, rungs, sub, suffix, fields, kw = _job(jobs, "reveal")
    assert "tail_digest" not in fields
    assert "focal_alpha" not in kw and "focal_gamma" not in kw
    assert sub == "enriched_real"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print("ALL PASS")
