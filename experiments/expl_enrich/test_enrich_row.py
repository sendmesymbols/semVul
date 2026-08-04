"""Self-running tests for enrich_row's ReVeal/Devign branch behavior (no pytest).
Run: .venv\\Scripts\\python.exe experiments\\expl_enrich\\test_enrich_row.py
"""
from apply_real_enrichment import enrich_row

_LONG = ("int f() {\n" + ("  y = y + 1;\n" * 40) + "  strcpy(d, e);\n}\n")


def test_reveal_branch_sets_tail_digest():
    r = {"sample_id": "s1", "raw_code": _LONG, "explanation": {}, "label": 1}
    treated = enrich_row(r, real_idx=None, deanon=False, tail_offset=30)
    assert treated is True
    assert "strcpy" in r["explanation"]["tail_digest"]
    assert r["explanation"]["real_enrich"] == "digest-v1"


def test_devign_branch_has_no_tail_digest():
    # deanon path: provide a matching real index so the row is treated
    r = {"sample_id": "s1", "raw_code": _LONG, "explanation": {}, "label": 0}
    treated = enrich_row(r, real_idx={"s1": _LONG}, deanon=True, tail_offset=30)
    assert treated is True
    assert "tail_digest" not in r["explanation"]        # Devign never gets it
    assert r["explanation"]["real_enrich"] == "deanon+digest-v1"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print("ALL PASS")
