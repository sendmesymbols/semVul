"""Self-running unit tests for tail_digest_fields (no pytest).
Run: .venv\\Scripts\\python.exe experiments\\expl_enrich\\test_tail_digest.py
"""
from apply_real_enrichment import tail_digest_fields


def test_short_code_is_empty():
    assert tail_digest_fields("int f() { return 0; }", offset=220) == ""


def test_only_tail_content_included():
    head = "int f() { memcpy(a, b, c);\n"     # memcpy is in the HEAD
    filler = "  y = y + 1;\n" * 40             # pushes the rest past offset=30
    tail = "  strcpy(d, e);\n  log(\"boom\");\n}\n"
    d = tail_digest_fields(head + filler + tail, offset=30)
    assert "strcpy" in d          # tail callee present
    assert "memcpy" not in d      # head callee excluded
    assert "tail_risky_apis strcpy" in d
    assert 'tail_literals "boom"' in d


def test_no_tail_callees_gives_empty():
    # long function whose tail has only arithmetic (no calls/literals)
    code = "int f() {\n" + ("  z = z + 1;\n" * 80) + "}\n"
    assert tail_digest_fields(code, offset=20) == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"ok {fn.__name__}")
    print("ALL PASS")
