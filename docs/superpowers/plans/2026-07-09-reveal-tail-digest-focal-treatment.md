# ReVeal Tail-Digest + Focal-Knobs Treatment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a label-blind beyond-window `tail_digest` explanation field and ReVeal-only focal-loss knobs (α/γ), both driven by hardcoded literals in `reproduce_reveal.ps1`, without touching any Devign code path.

**Architecture:** Two levers, ReVeal-scoped. (A) A new pure helper `tail_digest_fields` extracts lexical facts from the code *past* the encoder window and is written into the ReVeal branch of `enrich_row`; it flows to the text channel via the existing `SEMVUL_EXPL_FIELDS` mechanism, toggled by a `--tail-digest` flag. (B) `train_rung` gains `focal_alpha`/`focal_gamma` kwargs applied only inside the existing ReVeal-only focal branch; `reproduce_real.py` forwards them to the ReVeal job only. A paired-bootstrap script gives the honest ROC-delta readout.

**Tech Stack:** Python 3, NumPy, scikit-learn (for ROC), PyTorch/transformers (training, untouched by tests), PowerShell driver. Tests are self-running assert scripts (`python file.py`) — no pytest.

## Global Constraints

- **No new environment variables.** New knobs are hardcoded literals in `reproduce_reveal.ps1`, passed as CLI args → function kwargs. The only env var reused is the existing `SEMVUL_EXPL_FIELDS` (for field selection).
- **Devign is frozen.** Do not modify `digest_fields`, `DEVIGN_FIELDS`, `reproduce_devign.ps1`, or any code path Devign training reads. Devign uses plain `cross_entropy` (`use_focal == False`) and must stay byte-identical.
- **Label-blind.** No generated text may depend on labels. Enrichment cannot change class balance.
- **Skew discipline (ReVeal ≈ 9% positives).** Val stays at natural 9% and row-order-identical (ensemble-alignable). Headline metrics are **ROC + tune-selected F1 / PR-AUC**; argmax accuracy is never the verdict (all-negative baseline ≈ 90.8%).
- **Selection on tune, never val.** Focal grid picked on tune PR-AUC; ROC-delta reported via paired bootstrap.
- **Venv Python:** `D:\Projects\SemVul\.venv\Scripts\python.exe`. Run test scripts from repo root.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Task 0: Branch for isolation

Repo is on `master`. Create a working branch before any edit.

- [ ] **Step 1: Create and switch to the branch**

```bash
git checkout -b reveal-tail-digest-focal
```

- [ ] **Step 2: Confirm clean tree on the new branch**

Run: `git status`
Expected: `On branch reveal-tail-digest-focal` / `nothing to commit, working tree clean` (the spec/plan docs may show as untracked — that is fine).

---

## Task 1: `tail_digest_fields` pure helper

**Files:**
- Modify: `experiments/expl_enrich/apply_real_enrichment.py` (add constant + helper; no other behavior change)
- Test: `experiments/expl_enrich/test_tail_digest.py` (create)

**Interfaces:**
- Produces: `TAIL_OFFSET_TOKENS = 220` (module constant); `tail_digest_fields(raw_code: str, offset: int = TAIL_OFFSET_TOKENS) -> str` — a digest string of `tail_calls …`, `tail_risky_apis …`, `tail_literals …` built from tokens past `offset`; `""` when the function is ≤ `offset` tokens. Reuses existing module utilities `strip_comments`, `toks`, `C_KEYWORDS`, `RISKY_APIS`, `Counter`.

- [ ] **Step 1: Write the failing test**

Create `experiments/expl_enrich/test_tail_digest.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_tail_digest.py`
Expected: FAIL — `ImportError: cannot import name 'tail_digest_fields'`.

- [ ] **Step 3: Add the constant and helper**

In `experiments/expl_enrich/apply_real_enrichment.py`, add the constant next to the existing `RISKY_APIS` block (after line 51):

```python
TAIL_OFFSET_TOKENS = 220  # code-token offset ~= 320 GraphCodeBERT subwords (ReVeal window)
```

Then add the helper immediately after `digest_fields` (after line 129), leaving `digest_fields` unchanged:

```python
def tail_digest_fields(raw_code, offset=TAIL_OFFSET_TOKENS):
    """Digest of the code BEYOND the encoder window (ReVeal-only, label-blind).
    Same extraction as digest_fields but over tokens past `offset`, so it carries
    lexical signal the head-truncated code channel cannot see. Empty for functions
    at or below the window length."""
    ts = toks(strip_comments(raw_code), norm_strings=False)
    if len(ts) <= offset:
        return ""
    tail = ts[offset:]
    callees, lits = [], []
    for i, t in enumerate(tail[:-1]):
        if re.match(r'^[A-Za-z_]\w*$', t) and t not in C_KEYWORDS \
                and tail[i + 1] == '(':
            callees.append(t)
        if t.startswith('"') and len(t) > 2:
            lits.append(t)
    callee_list = [c for c, _ in Counter(callees).most_common(30)]
    risky = [c for c in callee_list if c in RISKY_APIS]
    lit_list = list(dict.fromkeys(lits))[:10]
    parts = []
    if callee_list:
        parts.append("tail_calls " + " ".join(callee_list))
    if risky:
        parts.append("tail_risky_apis " + " ".join(risky))
    if lit_list:
        parts.append("tail_literals " + " ".join(lit_list))
    return ". ".join(parts)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_tail_digest.py`
Expected: `ok test_no_tail_callees_gives_empty` / `ok test_only_tail_content_included` / `ok test_short_code_is_empty` / `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add experiments/expl_enrich/apply_real_enrichment.py experiments/expl_enrich/test_tail_digest.py
git commit -m "feat(reveal-enrich): add beyond-window tail_digest_fields helper + tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Write `tail_digest` in the ReVeal branch + `--only` / `--tail-offset` flags

**Files:**
- Modify: `experiments/expl_enrich/apply_real_enrichment.py` (`enrich_row` signature + ReVeal branch; `main` argparse + TARGETS filter)
- Test: `experiments/expl_enrich/test_enrich_row.py` (create)

**Interfaces:**
- Consumes: `tail_digest_fields` (Task 1).
- Produces: `enrich_row(r, real_idx, deanon, tail_offset=TAIL_OFFSET_TOKENS)` — the ReVeal branch (`deanon=False`) additionally sets `r["explanation"]["tail_digest"]`; the Devign branch (`deanon=True`) does **not** set that key. `main` accepts `--only {reveal,devign}` (filters `TARGETS`) and `--tail-offset N`.

- [ ] **Step 1: Write the failing test**

Create `experiments/expl_enrich/test_enrich_row.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_enrich_row.py`
Expected: FAIL — `AssertionError` on `test_reveal_branch_sets_tail_digest` (key `tail_digest` absent) or a `TypeError` for the unexpected `tail_offset` kwarg.

- [ ] **Step 3: Thread the offset through `enrich_row` and set the field in the ReVeal branch only**

In `experiments/expl_enrich/apply_real_enrichment.py`, change the signature (line 147) and the ReVeal branch. Replace:

```python
def enrich_row(r, real_idx, deanon):
    """Returns True if the row was treated (mutates r)."""
    if deanon:
```

with:

```python
def enrich_row(r, real_idx, deanon, tail_offset=TAIL_OFFSET_TOKENS):
    """Returns True if the row was treated (mutates r)."""
    if deanon:
```

Then, in the `else` (ReVeal) branch, after `tag = "digest-v1"` (line 164), and after the shared `digest_fields` block sets the standard fields (after line 172, i.e. after `expl["lexical_digest"] = dig`), add the ReVeal-only field. The cleanest placement is inside the `else` branch itself; set a flag and apply it after the shared block:

```python
    else:
        raw = r.get("raw_code") or ""   # reveal code is already real
        expl = r.get("explanation") or {}
        tag = "digest-v1"
        _add_tail = True
```

and in the `if deanon:` branch add `_add_tail = False` after `tag = "deanon+digest-v1"`. Then, after the shared digest assignment block (after `expl["real_enrich"] = tag`, line 172), before `r["explanation"] = expl` (line 173):

```python
    if _add_tail:
        expl["tail_digest"] = tail_digest_fields(raw, tail_offset)
```

- [ ] **Step 4: Add `--only` and `--tail-offset` to `main` and pass the offset through**

In `main` (argparse block around line 178) add:

```python
    ap.add_argument("--only", choices=["reveal", "devign"], default=None,
                    help="restrict to one dataset's TARGETS (Devign-safe)")
    ap.add_argument("--tail-offset", type=int, default=TAIL_OFFSET_TOKENS)
```

Filter the targets right after args are parsed (before the `--check` block uses them is fine; place after `args = ap.parse_args()`):

```python
    targets = [t for t in TARGETS if args.only is None or t[0] == args.only]
```

Replace the two `for ds, base, deanon in TARGETS:` loops (in the `--check` block, line 186, and the main loop, line 196) with `... in targets:`. In the main enrichment loop, pass the offset:

```python
                hit += enrich_row(r, real_idx, deanon, args.tail_offset)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_enrich_row.py`
Expected: `ok test_devign_branch_has_no_tail_digest` / `ok test_reveal_branch_sets_tail_digest` / `ALL PASS`.

- [ ] **Step 6: Verify Devign files are byte-identical under `--only reveal` (isolation gate)**

This regenerates the ReVeal `.real.jsonl` (adds `tail_digest`) and proves Devign files are untouched. Run:

```bash
.venv/Scripts/python.exe - <<'PY'
import hashlib, glob, subprocess, os
root = "explanations/SemanticVul/devign"
before = {p: hashlib.md5(open(p,'rb').read()).hexdigest()
          for p in glob.glob(os.path.join(root, "*.real.jsonl"))}
subprocess.run(["./.venv/Scripts/python.exe",
                "experiments/expl_enrich/apply_real_enrichment.py",
                "--only", "reveal", "--tail-offset", "220"], check=True)
after = {p: hashlib.md5(open(p,'rb').read()).hexdigest()
         for p in glob.glob(os.path.join(root, "*.real.jsonl"))}
assert before and before == after, f"Devign files changed! {before} vs {after}"
print(f"OK: {len(before)} Devign .real.jsonl files byte-identical")
PY
```

Expected: `OK: N Devign .real.jsonl files byte-identical` (N ≈ 7). Then confirm ReVeal gained the field **and** row count/order is preserved (one output row per source row):

```bash
.venv/Scripts/python.exe - <<'PY'
import json
base = "explanations/SemanticVul/reveal"
def count(p): return sum(1 for ln in open(p, encoding="utf-8") if ln.strip())
for stem in ("reveal_val.enriched", "reveal_train.enriched.clean"):
    src, dst = f"{base}/{stem}.jsonl", f"{base}/{stem}.real.jsonl"
    assert count(src) == count(dst), f"{stem}: row count changed"
r = json.loads(open(f"{base}/reveal_val.enriched.real.jsonl", encoding="utf-8").readline())
assert "tail_digest" in r["explanation"], "tail_digest missing"
print("OK: ReVeal row counts preserved; tail_digest present")
PY
```

Expected: `OK: ReVeal row counts preserved; tail_digest present`.

- [ ] **Step 7: Commit**

```bash
git add experiments/expl_enrich/apply_real_enrichment.py experiments/expl_enrich/test_enrich_row.py
git commit -m "feat(reveal-enrich): write tail_digest in ReVeal branch; add --only/--tail-offset (Devign untouched)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: ReVeal-only focal knobs in `train.py`

**Files:**
- Modify: `experiments/fusevul_ladder/train.py` (add `_resolve_focal`, `train_rung` kwargs, apply in focal branch, record γ)
- Test: `experiments/fusevul_ladder/test_focal_resolve.py` (create)

**Interfaces:**
- Produces: `_resolve_focal(use_focal: bool, computed_alpha: float, focal_alpha, focal_gamma) -> (alpha_pos, gamma)` — override applies only when `use_focal`; `focal_alpha=None` keeps the computed alpha. `train_rung(..., focal_alpha=None, focal_gamma=2.0)`.
- Consumes (downstream, Task 4): `reproduce_real.py` calls `train_rung(..., focal_alpha=?, focal_gamma=?)` for the ReVeal job only.

- [ ] **Step 1: Write the failing test**

Create `experiments/fusevul_ladder/test_focal_resolve.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe experiments\fusevul_ladder\test_focal_resolve.py`
Expected: FAIL — `ImportError: cannot import name '_resolve_focal'`.

- [ ] **Step 3: Add the helper and wire it into `train_rung`**

In `experiments/fusevul_ladder/train.py`, add the pure helper above `train_rung` (after `_metrics_at`, line 85):

```python
def _resolve_focal(use_focal, computed_alpha, focal_alpha, focal_gamma):
    """Actual (alpha_pos, gamma) used. Overrides apply ONLY when focal is on
    (ReVeal); Devign (use_focal=False) uses plain CE and is unaffected."""
    if not use_focal:
        return computed_alpha, float(focal_gamma)
    alpha = computed_alpha if focal_alpha is None else float(focal_alpha)
    return alpha, float(focal_gamma)
```

Add the kwargs to the `train_rung` signature (line 88-90) — append to the existing keyword list:

```python
def train_rung(dataset, rung, *, epochs=12, patience=3, batch=4, grad_accum=8,
               max_code=320, max_text=256, lr=2e-5, fusion="self", tune_frac=0.12,
               subset=None, seed=1337, split_seed=None, out_dir=RUNS, focal="auto",
               focal_alpha=None, focal_gamma=2.0):
```

Replace the alpha/use_focal block (lines 132-136) — keep the computation, then resolve:

```python
    pos_rate = float(ytr.mean())
    computed_alpha = float(np.clip(1.0 - pos_rate, 0.5, 0.80))
    # focal="auto" preserves prior behavior (focal on ReVeal only); "on"/"off" override
    # so the RO4 focal-loss ablation can be run on Devign (currently plain CE).
    use_focal = {"on": True, "off": False}.get(focal, dataset == "reveal")
    alpha_pos, gamma = _resolve_focal(use_focal, computed_alpha, focal_alpha, focal_gamma)
```

Update the loss call (line 179-180) to pass gamma:

```python
                loss = (focal_ce(logits, yb, alpha_pos, gamma) if use_focal
                        else nn.functional.cross_entropy(logits, yb))
```

Record γ in the payload `config` (line 273-276) — add `focal_gamma=gamma` to the `dict(...)`:

```python
        "config": dict(epochs=epochs, patience=patience, batch=batch,
                       grad_accum=grad_accum, max_code=max_code, max_text=max_text,
                       lr=lr, tune_frac=tune_frac, seed=seed, split_seed=split_seed,
                       subset=subset, use_focal=use_focal, alpha_pos=alpha_pos,
                       focal_gamma=gamma),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe experiments\fusevul_ladder\test_focal_resolve.py`
Expected: `ok test_devign_plain_ce_unaffected` / `ok test_reveal_default_alpha_kept` / `ok test_reveal_override_applied` / `ALL PASS`.

- [ ] **Step 5: Sanity-check that nothing else broke on import**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'experiments/fusevul_ladder'); import train; print('train import OK')"`
Expected: `train import OK` (no exceptions).

- [ ] **Step 6: Commit**

```bash
git add experiments/fusevul_ladder/train.py experiments/fusevul_ladder/test_focal_resolve.py
git commit -m "feat(ladder): ReVeal-only focal alpha/gamma kwargs via _resolve_focal (Devign CE unaffected)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `reproduce_real.py` plumbing — `_build_jobs`, `--tail-digest`, focal + out-tag

**Files:**
- Modify: `experiments/expl_enrich/reproduce_real.py` (extract `_build_jobs`, add args, forward focal to ReVeal job only, out-tag subdir)
- Test: `experiments/expl_enrich/test_build_jobs.py` (create)

**Interfaces:**
- Consumes: `train_rung(..., focal_alpha, focal_gamma)` (Task 3).
- Produces: `_build_jobs(args) -> list[tuple]` where each tuple is `(ds, rungs, sub, suffix, fields, kw)`. For the ReVeal job: `fields` includes `,tail_digest` iff `args.tail_digest`; `kw` includes `focal_alpha`/`focal_gamma` iff provided. The Devign job never receives focal keys and its fields never include `tail_digest`. `sub` = base dir + `args.out_tag`.

- [ ] **Step 1: Write the failing test**

Create `experiments/expl_enrich/test_build_jobs.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_build_jobs.py`
Expected: FAIL — `ImportError: cannot import name '_build_jobs'`.

- [ ] **Step 3: Extract `_build_jobs` and add args**

In `experiments/expl_enrich/reproduce_real.py`, add the argparse options in `main` (after line 51, before `args = ap.parse_args()`):

```python
    ap.add_argument("--tail-digest", action="store_true",
                    help="ReVeal only: append tail_digest to the text channel")
    ap.add_argument("--focal-alpha", type=float, default=None)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix on the run subdir, e.g. _tail_a85 (keeps A/B arms separate)")
```

Add the `_build_jobs` function above `main` (after line 43):

```python
def _build_jobs(args):
    """Construct the (ds, rungs, sub, suffix, fields, kw) job tuples. ReVeal-only
    knobs (tail_digest, focal alpha/gamma) are attached to the ReVeal job ONLY;
    the Devign job is never given them."""
    ga = max(1, 32 // args.batch512)
    reveal_fields = REVEAL_FIELDS + (",tail_digest" if args.tail_digest else "")
    reveal_kw = {}
    if args.focal_alpha is not None:
        reveal_kw["focal_alpha"] = args.focal_alpha
        reveal_kw["focal_gamma"] = args.focal_gamma
    jobs = [
        ("reveal", args.rungs or ["L2", "L3", "L1"],
         "enriched_real" + args.out_tag, "clean.real", reveal_fields, reveal_kw),
        ("devign", args.rungs or ["L1", "L2", "L3"],
         "enriched512_real" + args.out_tag, "clean.aug.real", DEVIGN_FIELDS,
         dict(max_code=512, batch=args.batch512, grad_accum=ga)),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    return jobs
```

Replace the inline job list + filter in `main` (lines 54-66) with:

```python
    from train import train_rung  # after sys.path setup
    jobs = _build_jobs(args)
```

The existing training loop (lines 68-80) already reads `sub`, `suffix`, `fields`, `kw` from each job tuple and calls `train_rung(ds, rung, ..., **kw)` — it needs no change, since `focal_alpha`/`focal_gamma` now ride inside `kw` for the ReVeal job.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_build_jobs.py`
Expected: `ok test_devign_never_gets_focal_or_tail` / `ok test_reveal_baseline_has_no_tail_or_focal` / `ok test_reveal_gets_focal_and_tail` / `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add experiments/expl_enrich/reproduce_real.py experiments/expl_enrich/test_build_jobs.py
git commit -m "feat(reproduce): _build_jobs with ReVeal-only --tail-digest/--focal-*/--out-tag

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Paired-bootstrap ROC-delta measurement script

**Files:**
- Create: `experiments/expl_enrich/paired_bootstrap.py`
- Test: `experiments/expl_enrich/test_paired_bootstrap.py` (create)

**Interfaces:**
- Produces: `paired_roc_delta(y, p_base, p_treat, n_boot=2000, seed=1337) -> dict` with keys `base`, `treat`, `delta` (all ×100 ROC points), `ci` (`[lo, hi]` 95% percentile of the paired delta), `n_boot` (usable resamples). `main` resolves two `_probs.npz` by `--base-dir`/`--treat-dir`/`--ds`/`--rung`/`--seed` and prints the dict.

- [ ] **Step 1: Write the failing test**

Create `experiments/expl_enrich/test_paired_bootstrap.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_paired_bootstrap.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'paired_bootstrap'`.

- [ ] **Step 3: Implement the script**

Create `experiments/expl_enrich/paired_bootstrap.py`:

```python
"""Paired bootstrap ROC-delta between two ladder members (baseline vs treated).

The honest A/B readout for the ReVeal treatment: dual_eval POOLS all
runs/enriched* members into one ensemble, which blends arms. This instead
compares two matched members (same rung + seed) on the SAME row-aligned val set
and bootstraps the paired ROC difference with a 95% CI.

  .venv/Scripts/python.exe experiments/expl_enrich/paired_bootstrap.py \\
      --base-sub enriched_real --treat-sub enriched_real_tail_a85 \\
      --ds reveal --rung L2 --seed 1337
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "experiments", "runs")


def paired_roc_delta(y, p_base, p_treat, n_boot=2000, seed=1337):
    y = np.asarray(y); p_base = np.asarray(p_base); p_treat = np.asarray(p_treat)
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, p_base)
    treat = roc_auc_score(y, p_treat)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = y[idx].sum()
        if s == 0 or s == n:            # need both classes for ROC
            continue
        deltas.append(roc_auc_score(y[idx], p_treat[idx])
                      - roc_auc_score(y[idx], p_base[idx]))
    lo, hi = (np.percentile(deltas, [2.5, 97.5]) if deltas else (0.0, 0.0))
    return dict(base=100 * base, treat=100 * treat,
                delta=100 * (treat - base),
                ci=[100 * float(lo), 100 * float(hi)], n_boot=len(deltas))


def _load(sub, ds, rung, seed):
    """sub = run subdir name directly under experiments/runs (e.g. enriched_real)."""
    p = os.path.join(RUNS, sub, f"s{seed}", f"fusevul_ladder_{ds}_{rung}_probs.npz")
    d = np.load(p)
    return d["val_prob"], d["val_y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sub", default="enriched_real",
                    help="baseline run subdir under experiments/runs")
    ap.add_argument("--treat-sub", required=True,
                    help="treated run subdir, e.g. enriched_real_tail_a85")
    ap.add_argument("--ds", default="reveal")
    ap.add_argument("--rung", default="L2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    pb, yb = _load(args.base_sub, args.ds, args.rung, args.seed)
    pt, yt = _load(args.treat_sub, args.ds, args.rung, args.seed)
    assert np.array_equal(yb, yt), "val labels/order differ -> not paired"
    r = paired_roc_delta(yb, pb, pt, n_boot=args.n_boot, seed=args.seed)
    print(f"{args.ds} {args.rung} s{args.seed}: base ROC={r['base']:.2f}  "
          f"treat ROC={r['treat']:.2f}  delta={r['delta']:+.2f}  "
          f"95% CI=[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  (n_boot={r['n_boot']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe experiments\expl_enrich\test_paired_bootstrap.py`
Expected: `ok test_identical_probs_delta_zero` / `ok test_treated_strictly_better_has_positive_ci` / `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add experiments/expl_enrich/paired_bootstrap.py experiments/expl_enrich/test_paired_bootstrap.py
git commit -m "feat(eval): paired-bootstrap ROC-delta for clean baseline-vs-treated A/B

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Wire the hardcoded treatment into `reproduce_reveal.ps1`

**Files:**
- Modify: `reproduce_reveal.ps1`

**Interfaces:**
- Consumes: `apply_real_enrichment.py --only reveal --tail-offset` (Task 2), `reproduce_real.py --tail-digest --focal-alpha/--focal-gamma --out-tag` (Task 4), `paired_bootstrap.py` (Task 5).

- [ ] **Step 1: Replace the body of `reproduce_reveal.ps1`**

Keep the header comments and the `param(...)` / `$ErrorActionPreference` / `Set-Location` / `$py = ...` setup (through line 15). Replace everything from the `# 1.` comment to the end of the file (the three numbered step blocks) with:

```powershell
# ---- ReVeal treatment knobs (HARDCODED here; no env vars) ----
$TailOffset = 220          # code-token offset ~= 320 GraphCodeBERT subwords (window)
$FocalAlpha = 0.85         # focal positive weight (baseline auto = 0.80)
$FocalGamma = 2.0          # focal gamma (baseline = 2.0)
$OutTag     = "_tail_a85"  # -> runs/enriched_real_tail_a85/  (A/B vs runs/enriched_real)
# --------------------------------------------------------------

# 1. (re)generate ReVeal *.real.jsonl WITH tail_digest. --only reveal never
#    touches Devign files and needs no devign_real index.
& $py experiments\expl_enrich\apply_real_enrichment.py --only reveal --tail-offset $TailOffset
if ($LASTEXITCODE -ne 0) { throw "apply_real_enrichment (reveal) failed" }

# 2. train the ReVeal TREATED arm (tail_digest channel + focal knobs), reveal-only.
$seedArgs = $Seeds | ForEach-Object { "$_" }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --seeds @seedArgs `
      --tail-digest --focal-alpha $FocalAlpha --focal-gamma $FocalGamma --out-tag $OutTag
if ($LASTEXITCODE -ne 0) { Write-Warning "reproduce_real exited $LASTEXITCODE (partial results kept)" }

# 3. pooled ensemble view (all arms) + the honest paired A/B for L2 seed 1337.
& $py experiments\expl_enrich\dual_eval.py
& $py experiments\expl_enrich\paired_bootstrap.py `
      --base-sub enriched_real --treat-sub "enriched_real$OutTag" `
      --ds reveal --rung L2 --seed 1337
```

- [ ] **Step 2: Static-parse the PowerShell to confirm no syntax error**

Run: `powershell -NoProfile -Command "[void][System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw reproduce_reveal.ps1),[ref]$null); 'ps1 parse OK'"`
Expected: `ps1 parse OK`.

- [ ] **Step 3: Confirm the treated-arm text channel actually includes tail_digest**

This runs no GPU — it just checks that with `--tail-digest` the ReVeal field list resolves to include the key (mirrors what `data_io._expl_field_set` will receive):

```bash
.venv/Scripts/python.exe - <<'PY'
import sys; sys.path.insert(0, "experiments/expl_enrich")
from reproduce_real import _build_jobs
class A:
    only="reveal"; seeds=[1337]; batch512=2; rungs=["L2"]
    tail_digest=True; focal_alpha=0.85; focal_gamma=2.0; out_tag="_tail_a85"
fields = next(j for j in _build_jobs(A()) if j[0]=="reveal")[4]
assert "tail_digest" in fields.split(","), fields
print("OK treated fields:", fields)
PY
```

Expected: `OK treated fields: ...,tail_digest`.

- [ ] **Step 4: Commit**

```bash
git add reproduce_reveal.ps1
git commit -m "feat(reveal-run): hardcode tail_digest + focal knobs in reproduce_reveal.ps1; add paired A/B

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Full regression gate + spec/plan commit

**Files:** none (verification only) + commit the design docs.

- [ ] **Step 1: Run every unit test**

```bash
.venv/Scripts/python.exe experiments/expl_enrich/test_tail_digest.py
.venv/Scripts/python.exe experiments/expl_enrich/test_enrich_row.py
.venv/Scripts/python.exe experiments/fusevul_ladder/test_focal_resolve.py
.venv/Scripts/python.exe experiments/expl_enrich/test_build_jobs.py
.venv/Scripts/python.exe experiments/expl_enrich/test_paired_bootstrap.py
```

Expected: each prints `ALL PASS`.

- [ ] **Step 2: Re-confirm Devign isolation end-to-end**

Re-run the hash gate from Task 2 Step 6. Expected: `OK: N Devign .real.jsonl files byte-identical`. Also confirm `reproduce_devign.ps1` is unchanged:

Run: `git status --short reproduce_devign.ps1 experiments/fusevul_ladder/model.py src/data_io.py src/data.py`
Expected: no output (none of these files modified).

- [ ] **Step 3: Commit the spec + plan**

```bash
git add docs/superpowers/specs/2026-07-09-reveal-tail-digest-focal-treatment-design.md docs/superpowers/plans/2026-07-09-reveal-tail-digest-focal-treatment.md
git commit -m "docs: ReVeal tail-digest + focal-knobs treatment spec and plan

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes on execution (GPU stage — not part of the code tasks)

The code tasks above are CPU-only and fast. The actual treatment run (`.\reproduce_reveal.ps1`) is the expensive GPU stage (~3.6 h / rung / seed). Per the spec's staging, run **L2 / seed 1337 first**, compare against the overnight baseline member via `paired_bootstrap.py`, and only expand to the focal grid / seed 2024 / L3 on a positive, CI-clear lift. L3's slow warmup is expected and out of scope (quality-feature normalization is separate work).
