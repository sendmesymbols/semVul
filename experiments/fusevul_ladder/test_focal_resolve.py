"""Self-running tests for _resolve_focal (no pytest).
Run: .venv\\Scripts\\python.exe experiments\\fusevul_ladder\\test_focal_resolve.py
"""
from train import _resolve_focal


def test_devign_plain_ce_unaffected():
    # use_focal False (Devign): overrides must not change the computed alpha
    alpha, gamma = _resolve_focal(False, 0.5, 0.99, 3.0)
    assert alpha == 0.5


def test_reveal_default_alpha_kept():
    alpha, gamma = _resolve_focal(True, 0.80, None, 2.0)
    assert alpha == 0.80 and gamma == 2.0


def test_reveal_override_applied():
    alpha, gamma = _resolve_focal(True, 0.80, 0.90, 3.0)
    assert alpha == 0.90 and gamma == 3.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print("ALL PASS")
