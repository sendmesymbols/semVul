"""SELF-CONTAINED PILOT — does the ONE untested pathway carry signal?

Question this answers (and nothing else)
----------------------------------------
Every prior negative result on the explanation channel came from either a
label-free *standalone* signal probe (AUC ~0.51) or the *frozen-MiniLM* pipeline
that fused pooled vectors. The one architecture that has NEVER been tested with
regenerated v2 explanations is the FuSEVul-intended pathway:

    fine-tuned RoBERTa over the explanation TOKENS  +  multi-head self-attention
    fusion into the (fine-tuned) code encoder.

This script isolates exactly that variable on a fresh, deletable subset:

  L1 = code-only (GraphCodeBERT, fine-tuned)              -> mean-pool -> head
  L2 = L1 + fine-tuned RoBERTa on v2 explanation tokens, fused by self-attention

Both rungs are trained and evaluated on the SAME samples, SAME internal
train/tune/test split, SAME seed, SAME schedule. The only difference is whether
the v2 explanation channel exists. The verdict metric is threshold-free
val/test ROC-AUC and PR-AUC (a fair ladder-contribution measure); F1/acc at
0.5 and at a tune-chosen threshold are reported alongside.

Everything this writes lives under experiments/expl_v2_pilot/ — delete that one
folder and the pilot is gone. It reuses the reference v2 prompt (expl_v2/
prompt_v2.py) and the reference ladder model (fusevul_ladder/model.py), which
are the very implementations under test; it modifies neither.

Generation runs against clod.io (OpenAI-compatible cloud) by default, model
"Gemma 4 31B IT", rotating through the API keys in clod_keys.txt (git-ignored)
and counting calls per key. Output is a MODEL-specific jsonl so different
generators never mix in one file. Everything is resumable.

Run it (attended) in PyCharm with no arguments, or:
    .venv/Scripts/python.exe experiments/expl_v2_pilot/run_pilot.py
Confirm available cloud models / that keys work:
    .venv/Scripts/python.exe experiments/expl_v2_pilot/run_pilot.py --list-models
Only (re)run the L1-vs-L2 comparison once explanations exist:
    .venv/Scripts/python.exe experiments/expl_v2_pilot/run_pilot.py --skip-gen
Fall back to local Ollama:  --provider ollama --model qwen3.5:9b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Paths + imports. src.config must import before transformers (it points the HF
# cache at project/models, matching every prior run so nothing re-downloads).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # D:\Projects\SemVul
EXPL_V2 = os.path.join(ROOT, "experiments", "expl_v2")
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, EXPL_V2, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config  # noqa: F401,E402  (sets HF_HOME/HF_HUB_CACHE before transformers)
from src.data_io import load_split                       # noqa: E402
from prompt_v2 import JSON_SCHEMA, build_messages         # noqa: E402  (the v2 prompt under test)
from prompt_v2_real import build_messages as build_messages_real  # noqa: E402  (real-code variant, path C)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SUBSET_JSON = os.path.join(OUT, "subset.json")
KEYS_FILE = os.path.join(HERE, "clod_keys.txt")   # git-ignored secrets
USAGE_FILE = os.path.join(OUT, "clod_key_usage.json")


def _sanitize(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()


def expl_path(model: str, dataset: str = "devign") -> str:
    """Output file is MODEL-(and dataset-)specific so different generators/datasets
    never mix in one file (local qwen vs cloud rows; anonymized vs de-anonymized
    code each stay separate). Each is independently resumable."""
    tag = _sanitize(model) if dataset == "devign" else f"{dataset}__{_sanitize(model)}"
    return os.path.join(OUT, f"expl_v2_pilot__{tag}.jsonl")


# ---------------------------------------------------------------------------
# Defaults (PyCharm "Run" with no args uses these). Override via CLI if needed.
# ---------------------------------------------------------------------------
PROVIDER = "clod"                          # "clod" (cloud, OpenAI-compatible) | "ollama"
MODEL = "Gemma 4 31B IT"                    # exact clod.io slug (display name w/ spaces)
CLOD_BASE_URL = "https://api.clod.io/v1"   # OpenAI-compatible endpoint
# clod.io sits behind Cloudflare, which 403s (error 1010) the default urllib
# User-Agent. A browser-like UA is required on every request.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
HOST = "http://localhost:9999"             # only used when PROVIDER == "ollama"
N_SUBSET = 1900                            # 950/class; under 1970 key capacity w/ retry
                                           # headroom so the run finishes clean (no tail)
NO_THINK = True                            # ollama only: send "think": false
WORKERS = 2                                # 4 concurrent calls caused rate-limit prose from clod.io
NUM_CTX = 8192                             # ollama only
GEN_TIMEOUT = 180
MAX_TOKENS = 8192                          # Gemma 4 31B is a thinking model; reasoning burns
                                           # tokens before content. 1536 caused empty content
                                           # on complex functions (finish_reason=length).
MAX_RETRIES = 4                            # per-sample transient-error retries
MAX_CODE_CHARS = 12000                     # ~3k tokens; matches generate_v2

GEN_SEED = 13                 # deterministic subset pick (gen and train see same set)
SPLIT_SEED = 1337             # internal train/tune/test carve
TRAIN_SEED = 1337
TEST_FRAC = 0.30              # internal held-out test (the L1-vs-L2 comparison set)
TUNE_FRAC = 0.12              # carved from the internal train remainder

EPOCHS = 12
PATIENCE = 3
BATCH = 4
GRAD_ACCUM = 8
MAX_CODE = 320
MAX_TEXT = 256
LR = 2e-5
CODE_ID = "microsoft/graphcodebert-base"   # FuSEVul-reported code encoder
TEXT_ID = "roberta-base"                    # FuSEVul-intended explanation encoder

RISK_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ===========================================================================
# Phase 0 — deterministic stratified subset (dedup within train)
# ===========================================================================
def select_subset(n: int, seed: int, dataset: str = "devign"):
    samples = load_split(dataset, "train")
    seen, uniq = set(), []
    for s in samples:                       # drop within-train duplicate functions
        if s.sample_id in seen:
            continue
        seen.add(s.sample_id)
        uniq.append(s)
    by_label = {0: [], 1: []}
    for s in uniq:
        if s.label in by_label:
            by_label[s.label].append(s)
    rng = random.Random(seed)
    picked = []
    for lab in (0, 1):
        pool = sorted(by_label[lab], key=lambda s: s.sample_id)
        rng.shuffle(pool)
        picked.extend(pool[: n // 2])
    picked.sort(key=lambda s: s.sample_id)
    with open(SUBSET_JSON, "w", encoding="utf-8") as fh:
        json.dump([{"sample_id": s.sample_id, "label": s.label} for s in picked],
                  fh, indent=2)
    npos = sum(s.label for s in picked)
    print(f"[subset] picked {len(picked)} unique samples "
          f"(pos {npos}, neg {len(picked) - npos}) -> {SUBSET_JSON}", flush=True)
    return picked


# ===========================================================================
# Phase 1 — regenerate v2 explanations for exactly that subset (resumable).
# Two providers: "clod" (cloud OpenAI-compatible, with API-key rotation) and
# "ollama" (local). Output is a MODEL-specific jsonl so generators never mix.
# ===========================================================================
class CloudError(Exception):
    def __init__(self, code, body=""):
        super().__init__(f"HTTP {code}: {body[:160]}")
        self.code = code
        self.body = body


class QuotaExpired(CloudError):
    """Response text (error body or content) indicates the key is out of quota."""


class NoKeysLeft(Exception):
    pass


def _has_quota_marker(text):
    t = (text or "").lower()
    return any(m in t for m in QUOTA_MARKERS)


# Status codes that mean "this key is spent / not allowed" -> rotate to next.
ROTATE_CODES = {401, 402, 403, 429}

# If this text appears anywhere in a response (error body OR message content),
# the key is out of quota -> retire it and switch, regardless of HTTP status.
QUOTA_MARKERS = ("quota expired", "quota exceeded", "out of quota")

# Per-key SOFT call caps: switch to the next key once a key reaches its cap,
# BEFORE the server starts rejecting. Most keys allow ~98 calls; one high-
# capacity key allows ~990. The high-cap key is identified by its tag (last 12
# chars); per the "950 calls" note this is assumed to be the 2nd key in
# clod_keys.txt (tag below). Adjust HIGH_CAP_KEY_TAG if that key is different —
# it only changes which key is drained first; server-side quota errors are still
# caught as a backstop either way.
DEFAULT_CALL_CAP = 98
HIGH_CALL_CAP = 990
HIGH_CAP_KEY_TAG = "0KwQ2Docb-FU"


def load_keys(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


class KeyManager:
    """Thread-safe rotating pool of API keys with persistent call counting.

    - get(): returns the current usable key (skips dead keys and keys that have
      reached their soft call cap).
    - record(key): +1 success on that key (persisted every few calls).
    - mark_dead(key, reason): retire a key hit with a quota/auth error.
    A key is identified by its last 12 chars in logs/usage (never the full JWT).
    """

    def __init__(self, keys, usage_path):
        if not keys:
            raise NoKeysLeft("no API keys loaded")
        self.keys = keys
        self.usage_path = usage_path
        self.lock = threading.Lock()
        self.idx = 0
        self.dead = set()
        self.calls = 0
        self.usage = {self.tag(k): 0 for k in keys}
        self.caps = {self.tag(k): (HIGH_CALL_CAP if self.tag(k) == HIGH_CAP_KEY_TAG
                                   else DEFAULT_CALL_CAP) for k in keys}
        if os.path.exists(usage_path):
            try:
                prior = json.load(open(usage_path, "r", encoding="utf-8"))
                for t, c in (prior.get("usage") or {}).items():
                    if t in self.usage:
                        self.usage[t] = c
                for t in (prior.get("dead") or []):
                    if t in self.usage:
                        self.dead.add(t)
            except (json.JSONDecodeError, OSError):
                pass

    @staticmethod
    def tag(k):
        return k[-12:]

    def _usable(self, tag):
        return tag not in self.dead and self.usage[tag] < self.caps[tag]

    def total_capacity(self):
        return sum(self.caps.values())

    def get(self):
        with self.lock:
            for _ in range(len(self.keys)):
                k = self.keys[self.idx]
                if self._usable(self.tag(k)):
                    return k
                self.idx = (self.idx + 1) % len(self.keys)
            raise NoKeysLeft("all API keys exhausted (dead or at cap)")

    def record(self, key):
        with self.lock:
            t = self.tag(key)
            self.usage[t] += 1
            self.calls += 1
            hit_cap = self.usage[t] == self.caps[t]
            persist = (self.calls % 5 == 0)
            if hit_cap:
                self.idx = (self.idx + 1) % len(self.keys)   # move off the full key
        if hit_cap:
            print(f"[keys] ...{t} reached cap {self.caps[t]}; switching", flush=True)
        if persist or hit_cap:
            self._persist()

    def mark_dead(self, key, reason=""):
        with self.lock:
            t = self.tag(key)
            if t not in self.dead:
                self.dead.add(t)
                print(f"[keys] retire ...{t} ({reason}); "
                      f"{len(self.keys) - len(self.dead)}/{len(self.keys)} keys left",
                      flush=True)
            if self.keys[self.idx] == key:            # advance off the dead current
                self.idx = (self.idx + 1) % len(self.keys)
        self._persist()

    def _persist(self):
        with self.lock:
            snap = {"usage": dict(self.usage), "dead": sorted(self.dead),
                    "total_calls": self.calls}
        try:
            with open(self.usage_path, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=2)
        except OSError:
            pass

    def summary(self):
        with self.lock:
            live, full, deadu = [], [], []
            for k in self.keys:
                t = self.tag(k)
                rec = (t, self.usage[t], self.caps[t])
                if t in self.dead:
                    deadu.append(rec)
                elif self.usage[t] >= self.caps[t]:
                    full.append(rec)
                else:
                    live.append(rec)
        return {"total_calls": self.calls, "live": live, "full": full,
                "dead": deadu, "n_live": len(live), "n_full": len(full),
                "n_dead": len(deadu)}


def clod_chat(base_url, key, model, messages, timeout, max_tokens, use_rf=True):
    payload = {"model": model, "messages": messages, "temperature": 0.0,
               "max_tokens": max_tokens}
    if use_rf:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key,
                 "User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        if _has_quota_marker(body):
            raise QuotaExpired(e.code, body)
        raise CloudError(e.code, body)
    content = data["choices"][0]["message"]["content"]
    if _has_quota_marker(content):                 # 200 OK but quota message as content
        raise QuotaExpired(200, content)
    return content


def _repair_control_chars(s: str) -> str:
    """Escape literal newlines/tabs/returns that appear inside JSON string tokens.
    Gemma at long outputs sometimes emits a bare \n inside a string value, which
    is invalid JSON and triggers 'Expecting , delimiter' parse errors."""
    CTRL = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}
    out, in_str, esc = [], False, False
    for c in s:
        if esc:
            out.append(c); esc = False
        elif c == '\\' and in_str:
            out.append(c); esc = True
        elif c == '"':
            out.append(c); in_str = not in_str
        elif in_str and c in CTRL:
            out.append(CTRL[c])
        else:
            out.append(c)
    return ''.join(out)


def parse_json_lenient(s):
    """Providers sometimes wrap JSON in prose/fences or emit trailing commas or
    literal control chars in string values.  Try increasingly forgiving parses."""
    if s is None:
        raise json.JSONDecodeError("empty", "", 0)
    s = s.strip()
    # Stage 1: strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    # Stage 2: direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Stage 3: extract first {...} fragment
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise json.JSONDecodeError("unrecoverable", s, 0)
    frag = s[a:b + 1]
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        pass
    # Stage 4: strip trailing commas
    frag = re.sub(r",\s*([}\]])", r"\1", frag)
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        pass
    # Stage 5: escape literal control chars inside string values
    return json.loads(_repair_control_chars(frag))


def list_models(base_url, keys):
    """GET /models with the first working key so we can confirm the exact slug."""
    for k in keys:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": "Bearer " + k, "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = sorted(m.get("id", "") for m in data.get("data", []))
            print(f"[models] {len(ids)} available via clod.io "
                  f"(key ...{KeyManager.tag(k)}):")
            for mid in ids:
                print("  " + mid)
            return ids
        except urllib.error.HTTPError as e:
            print(f"[models] key ...{KeyManager.tag(k)} -> HTTP {e.code}, trying next",
                  flush=True)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"[models] key ...{KeyManager.tag(k)} -> {type(e).__name__}, trying next",
                  flush=True)
    print("[models] no key could list models.", flush=True)
    return []


def _ollama_chat(host, model, messages, num_ctx, timeout, no_think):
    payload = {"model": model, "messages": messages, "stream": False,
               "format": JSON_SCHEMA,
               "options": {"temperature": 0.0, "seed": 1234, "num_ctx": num_ctx}}
    if no_think:
        payload["think"] = False
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"]


def generate(samples, args, out_path, km=None):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    todo = [s for s in samples if s.sample_id not in done]
    target = len(samples)
    print(f"[gen] provider={args.provider} model={args.model} "
          f"target={target} done={len(done)} todo={len(todo)} workers={args.workers} "
          f"-> {out_path}", flush=True)
    if not todo:
        print("[gen] nothing to generate; target already met.", flush=True)
        return

    def gen_one(s):
        code = s.code[:MAX_CODE_CHARS]
        msgs = build_messages(code)
        err = None
        use_rf = True
        rotations = 0
        for attempt in range(1, args.max_retries + 1):
            content = None
            try:
                t1 = time.time()
                if args.provider == "clod":
                    key = km.get()
                    content = clod_chat(CLOD_BASE_URL, key, args.model, msgs,
                                        args.timeout, args.max_tokens, use_rf)
                    km.record(key)
                else:
                    content = _ollama_chat(args.host, args.model, msgs,
                                           args.num_ctx, args.timeout, args.no_think)
                dur = time.time() - t1
                try:
                    expl = parse_json_lenient(content)
                except json.JSONDecodeError as je:
                    preview = (content or "")[:300].replace("\n", "\\n")
                    err = f"parse({je})|raw:{preview!r}"
                    # No JSON structure at all → not a parse issue, bail immediately.
                    # At temp=0, identical prompt → identical output on every retry.
                    if "{" not in (content or ""):
                        return None, f"{s.sample_id}: {err}"
                    time.sleep(1.0 * attempt)
                    continue
                return {"sample_id": s.sample_id, "label": s.label,
                        "raw_code": s.code, "explanation": expl,
                        "meta": {"model": args.model, "provider": args.provider,
                                 "prompt": "v2", "gen_seconds": round(dur, 2)}}, None
            except NoKeysLeft as e:
                return None, f"{s.sample_id}: NO_KEYS ({e})"
            except QuotaExpired as e:
                err = str(e)
                if km is not None:
                    km.mark_dead(key, "quota expired")
                    rotations += 1
                    if rotations <= len(km.keys):
                        continue                       # switch key, don't burn a retry
                return None, f"{s.sample_id}: NO_KEYS"
            except CloudError as e:
                err = str(e)
                if e.code in ROTATE_CODES and km is not None:
                    km.mark_dead(key, f"HTTP {e.code}")
                    rotations += 1
                    if rotations <= len(km.keys):
                        continue                       # try next key, don't burn a retry
                    return None, f"{s.sample_id}: NO_KEYS"
                if e.code == 400 and use_rf:
                    use_rf = False                     # provider rejects response_format
                    continue
                time.sleep(1.5 * attempt)
            except (urllib.error.URLError, TimeoutError, OSError,
                    KeyError, IndexError) as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * attempt)
        return None, f"{s.sample_id}: {err}"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_ok, n_fail, t0 = 0, 0, time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(gen_one, s) for s in todo]
            for i, fut in enumerate(as_completed(futs)):
                row, err = fut.result()
                if row is None:
                    n_fail += 1
                    print(f"[gen] FAIL {err}", flush=True)
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if n_ok % 10 == 0 or i == len(todo) - 1:
                    rate = n_ok / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - i - 1) / max(rate, 1e-9) / 60
                    total_done = len(done) + n_ok
                    print(f"[gen] {total_done}/{target} done "
                          f"(+{n_ok} this run, {n_fail} fail) {rate:.2f}/s "
                          f"eta {eta:.0f} min", flush=True)
    print(f"[gen] DONE +{n_ok} ok, {n_fail} fail, "
          f"total {len(done) + n_ok}/{target}, "
          f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    if km is not None:
        km._persist()
        s = km.summary()
        print(f"[keys] total_calls={s['total_calls']}  live={s['n_live']} "
              f"full={s['n_full']} dead={s['n_dead']}  "
              f"capacity={km.total_capacity()}", flush=True)
        for state, recs in (("live", s["live"]), ("FULL", s["full"]),
                            ("DEAD", s["dead"])):
            for t, c, cap in recs:
                print(f"[keys]   ...{t}: {c}/{cap} calls ({state})", flush=True)


# ===========================================================================
# v2 explanation -> fused text.
# We include the natural-language content (structural facts, present safety
# checks, risk patterns + why, missing checks, categorical risk_level, summary)
# but deliberately EXCLUDE the verbatim code `evidence` fragments: those are
# normalized VAR/FUN tokens the code encoder already sees, and the 2026-07-04
# evidence_tokens ablation showed such fragments dilute an NL encoder. Keeping
# them out gives the explanation channel its cleanest shot at contributing
# something orthogonal to the code channel.
# ===========================================================================
def v2_text(e, with_evidence=False) -> str:
    if not isinstance(e, dict):
        return ""
    parts = []
    so = e.get("structural_observations") or []
    if so:
        parts.append(" ".join(str(x) for x in so))
    si = e.get("safety_indicators") or []
    checks = [str(d.get("check", "")).strip() for d in si if isinstance(d, dict)]
    checks = [c for c in checks if c]
    if checks:
        parts.append("Safety checks present: " + "; ".join(checks) + ".")
    ro = e.get("risky_operations") or []
    risks = []
    for d in ro:
        if not isinstance(d, dict):
            continue
        p, w = str(d.get("pattern", "")).strip(), str(d.get("why", "")).strip()
        seg = (p + " — " + w).strip(" —") if (p or w) else ""
        if with_evidence:                        # --with-evidence: append the verbatim span
            ev = str(d.get("evidence", "")).strip()
            if ev:
                seg = (seg + " [" + ev + "]").strip()
        if seg:
            risks.append(seg)
    if risks:
        parts.append("Risky operations: " + "; ".join(risks) + ".")
    mc = [str(x).strip() for x in (e.get("missing_checks") or [])]
    mc = [m for m in mc if m]
    if mc:
        parts.append("Missing checks: " + "; ".join(mc) + ".")
    rl = e.get("risk_level")
    if rl:
        parts.append(f"Overall risk level: {rl}.")
    rs = e.get("risk_summary")
    if rs:
        parts.append(str(rs).strip())
    return " ".join(p for p in parts if p).strip()


def _norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _rank_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rs_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_v2_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def standalone_signal(rows):
    """Sanity check on the generated subset: does the label-free explanation
    signal separate classes AT ALL here? (v1 ~0.51 = chance; gate ~0.58.)"""
    risk, nrisky, labels = [], [], []
    valid, grounded, total, empty0, empty1, n0, n1 = 0, 0, 0, 0, 0, 0, 0
    for r in rows:
        e = r.get("explanation") or {}
        lab = r.get("label")
        if lab not in (0, 1) or not isinstance(e, dict) or e.get("risk_level") not in RISK_MAP:
            continue
        valid += 1
        ro = e.get("risky_operations") or []
        code_n = _norm_ws(r.get("raw_code", ""))
        for op in ro:
            ev = op.get("evidence", "") if isinstance(op, dict) else ""
            total += 1
            if _norm_ws(ev) and _norm_ws(ev) in code_n:
                grounded += 1
        risk.append(RISK_MAP[e["risk_level"]])
        nrisky.append(len(ro))
        labels.append(lab)
        if lab == 0:
            n0 += 1
            empty0 += 1 if RISK_MAP[e["risk_level"]] == 0 else 0
        else:
            n1 += 1
            empty1 += 1 if RISK_MAP[e["risk_level"]] == 0 else 0
    auc_risk = _rank_auc(risk, labels) if labels else float("nan")
    auc_nrisky = _rank_auc(nrisky, labels) if labels else float("nan")
    print("\n[signal] standalone label-free explanation signal on this subset:")
    print(f"  valid={valid}  grounding={grounded}/{total} "
          f"({100*grounded/max(1,total):.1f}%) verbatim-in-code")
    print(f"  risk_level=none : label0 {100*empty0/max(1,n0):.0f}%  "
          f"label1 {100*empty1/max(1,n1):.0f}%")
    print(f"  SIGNAL AUC: risk_level={auc_risk:.3f}  n_risky={auc_nrisky:.3f}  "
          f"(v1 ~0.51, gate ~0.58)")
    return {"signal_auc_risk_level": auc_risk, "signal_auc_n_risky": auc_nrisky,
            "grounding_rate": grounded / max(1, total), "valid": valid}


# ===========================================================================
# Phase 3/4 — train one rung (adapted from fusevul_ladder/train.py), evaluate
# on the shared internal test split. L1 and L2 receive IDENTICAL indices.
# ===========================================================================
def _tok(tokenizer, texts, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _best_thr_f1(prob1, y):
    import numpy as np
    from sklearn.metrics import f1_score
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        s = f1_score(y, (prob1 >= t).astype(int), zero_division=0)
        if s > bs:
            bs, best = s, float(t)
    return best


def _metrics_at(thr, prob1, y):
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    yh = (prob1 >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


def train_eval(rung, packed, fit_idx, tune_idx, test_idx, args):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score)
    from transformers import AutoModel, AutoTokenizer
    from model import LadderModel                      # reference ladder model

    t0 = time.time()
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"\n[{rung}] device={device} amp={'bf16' if bf16 else 'fp16'} "
          f"fit={len(fit_idx)} tune={len(tune_idx)} test={len(test_idx)}", flush=True)

    code_tok = AutoTokenizer.from_pretrained(CODE_ID)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    code_enc = AutoModel.from_pretrained(CODE_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)
    qual_dim = packed["qual"].shape[1]
    model = LadderModel(code_enc, text_enc, qual_dim=qual_dim,
                        rung=rung, fusion="self").to(device)
    model.enable_grad_checkpointing()

    ci, cm = _tok(code_tok, packed["code"], args.max_code)
    ti, tm = _tok(text_tok, packed["expl"], args.max_text)
    q = torch.from_numpy(packed["qual"])
    y = packed["y"]

    fit_idx = np.asarray(fit_idx)
    tune_idx = np.asarray(tune_idx)
    test_idx = np.asarray(test_idx)
    ytu, yte = y[tune_idx], y[test_idx]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))

    @torch.no_grad()
    def prob1(idx):
        model.eval()
        idx = torch.as_tensor(np.asarray(idx))
        outs = []
        for i in range(0, len(idx), max(2, args.batch)):
            b = idx[i:i + max(2, args.batch)]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                lo = model(ci[b].to(device), cm[b].to(device), ti[b].to(device),
                           tm[b].to(device), q[b].to(device))
            outs.append(torch.softmax(lo.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(outs)

    best_ap, best = -1.0, None
    wait = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(fit_idx)
        opt.zero_grad()
        losses = []
        for si, i in enumerate(range(0, len(perm), args.batch)):
            b = torch.as_tensor(perm[i:i + args.batch])
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(ci[b].to(device), cm[b].to(device), ti[b].to(device),
                               tm[b].to(device), q[b].to(device))
                yb = torch.as_tensor(y[perm[i:i + args.batch]], dtype=torch.long,
                                     device=device)
                loss = nn.functional.cross_entropy(logits, yb) / args.grad_accum
            scaler.scale(loss).backward()
            if (si + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item() * args.grad_accum)

        tu_p = prob1(tune_idx)
        te_p = prob1(test_idx)
        ap = average_precision_score(ytu, tu_p) if ytu.sum() > 0 else 0.0
        te_f1 = f1_score(yte, (te_p >= 0.5).astype(int), zero_division=0) * 100
        print(f"[{rung}] ep{ep}/{args.epochs} loss={np.mean(losses):.4f} "
              f"tune_prauc={ap*100:.2f} test_f1@0.5={te_f1:.2f} "
              f"test_roc={roc_auc_score(yte, te_p)*100:.2f}", flush=True)
        if ap > best_ap:
            best_ap, best, wait = ap, (ep, te_p, tu_p), 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{rung}] early stop @ep{ep}", flush=True)
                break

    ep_best, te_p, tu_p = best
    thr = _best_thr_f1(tu_p, ytu)
    out = {
        "rung": rung, "best_epoch": ep_best,
        "test_roc_auc": 100 * roc_auc_score(yte, te_p),
        "test_pr_auc": 100 * average_precision_score(yte, te_p),
        "argmax": _metrics_at(0.5, te_p, yte),
        "tuned_on_tune": _metrics_at(thr, te_p, yte),
        "seconds": round(time.time() - t0, 1),
    }
    np.savez_compressed(os.path.join(OUT, f"pilot_{rung}_probs.npz"),
                        test_prob=te_p, test_y=yte, tune_prob=tu_p, tune_y=ytu)
    a = out["argmax"]
    print(f"[{rung}] DONE @ep{ep_best} ROC={out['test_roc_auc']:.2f} "
          f"PR={out['test_pr_auc']:.2f} argmax acc={a['acc']:.2f} f1={a['f1']:.2f} "
          f"({out['seconds']/60:.1f} min)", flush=True)
    del model, code_enc, text_enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ===========================================================================
# Build the aligned, packed dataset from generated rows + shared split.
# ===========================================================================
def build_packed(rows, with_evidence=False):
    import numpy as np
    rows = [r for r in rows if r.get("label") in (0, 1)]
    code = [r.get("raw_code", "") or "" for r in rows]
    expl = [v2_text(r.get("explanation") or {}, with_evidence) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    qual = np.zeros((len(rows), 22), dtype=np.float32)  # L1/L2 ignore qual
    empties = sum(1 for e in expl if not e)
    lens = [len(e.split()) for e in expl if e]
    med = sorted(lens)[len(lens) // 2] if lens else 0
    print(f"[data] packed={len(rows)} (pos {100*y.mean():.1f}%)  "
          f"expl empty={empties}  median expl words={med}", flush=True)
    return {"code": code, "expl": expl, "y": y, "qual": qual}


def make_split(y, test_frac, tune_frac, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(y)
    test_mask = np.zeros(n, dtype=bool)
    for c in (0, 1):                                   # stratified test
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        test_mask[idx[:k]] = True
    rem = np.where(~test_mask)[0]
    tune_mask = np.zeros(n, dtype=bool)
    for c in (0, 1):                                   # stratified tune from remainder
        idx = rem[y[rem] == c]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * tune_frac)))
        tune_mask[idx[:k]] = True
    fit_idx = np.where(~test_mask & ~tune_mask)[0]
    tune_idx = np.where(tune_mask)[0]
    test_idx = np.where(test_mask)[0]
    return fit_idx, tune_idx, test_idx


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_SUBSET)
    ap.add_argument("--dataset", default="devign",
                    help="load_split dataset; use 'devign_real' for de-anonymized "
                         "code + real-code prompt (path C)")
    ap.add_argument("--provider", default=PROVIDER, choices=["clod", "ollama"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--timeout", type=int, default=GEN_TIMEOUT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--list-models", action="store_true",
                    help="print clod.io model ids (to confirm the exact slug) and exit")
    ap.add_argument("--think", dest="no_think", action="store_false",
                    help="ollama only: allow model thinking (default sends think:false)")
    ap.set_defaults(no_think=NO_THINK)
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip generation; only run the L1-vs-L2 comparison")
    ap.add_argument("--with-evidence", action="store_true",
                    help="append verbatim risky_operations[].evidence spans into the "
                         "L2 explanation text (ablation: does the dropped field help?)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--max-code", type=int, default=MAX_CODE)
    ap.add_argument("--max-text", type=int, default=MAX_TEXT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    args = ap.parse_args()

    out_path = expl_path(args.model, args.dataset)
    keys = load_keys(KEYS_FILE) if args.provider == "clod" else []

    if args.list_models:
        if not keys:
            print(f"[abort] no keys in {KEYS_FILE}", flush=True)
            sys.exit(1)
        list_models(CLOD_BASE_URL, keys)
        return

    print("=" * 78)
    print("v2-explanation FUSION pilot  (L1 code-only  vs  L2 code + v2-expl fusion)")
    print(f"  subset N={args.n}  provider={args.provider}  model={args.model}")
    endpoint = CLOD_BASE_URL if args.provider == "clod" else args.host
    cap_note = ""
    if args.provider == "clod" and keys:
        cap = sum(HIGH_CALL_CAP if KeyManager.tag(k) == HIGH_CAP_KEY_TAG
                  else DEFAULT_CALL_CAP for k in keys)
        short = max(0, args.n - cap)
        cap_note = (f"  key_capacity={cap}"
                    + (f"  (WARNING: {short} short of target {args.n})" if short else ""))
    print(f"  endpoint={endpoint}  keys_loaded={len(keys)}{cap_note}")
    print(f"  code_enc={CODE_ID}  text_enc={TEXT_ID}")
    print(f"  explanations -> {out_path}")
    print(f"  artifacts under: {OUT}   (delete this folder to remove everything)")
    print("=" * 78, flush=True)

    # Phase 0 + 1 — subset + generation
    global build_messages
    if args.dataset.endswith("_real"):
        build_messages = build_messages_real
        print(f"[prompt] using real-code prompt (prompt_v2_real) for dataset={args.dataset}",
              flush=True)
    subset = select_subset(args.n, GEN_SEED, args.dataset)
    km = None
    if not args.skip_gen:
        if args.provider == "clod":
            if not keys:
                print(f"[abort] no API keys in {KEYS_FILE}", flush=True)
                sys.exit(1)
            km = KeyManager(keys, USAGE_FILE)
        generate(subset, args, out_path, km)
    if not os.path.exists(out_path):
        print("[abort] no explanations file; generation produced nothing.", flush=True)
        sys.exit(1)

    rows = load_v2_rows(out_path)
    if len(rows) < 100:
        print(f"[abort] only {len(rows)} explanations generated; run generation first.",
              flush=True)
        sys.exit(1)

    # Phase 1b — standalone signal sanity check
    sig = standalone_signal(rows)

    # Phase 2 — pack + shared split (identical for both rungs)
    packed = build_packed(rows, args.with_evidence)
    if args.with_evidence:
        print("[data] --with-evidence: verbatim evidence spans APPENDED to L2 text",
              flush=True)
    fit_idx, tune_idx, test_idx = make_split(packed["y"], TEST_FRAC, TUNE_FRAC,
                                             SPLIT_SEED)

    # Phase 3/4 — L1 then L2 on the SAME data/split/seed
    res_l1 = train_eval("L1", packed, fit_idx, tune_idx, test_idx, args)
    res_l2 = train_eval("L2", packed, fit_idx, tune_idx, test_idx, args)

    # Phase 5 — verdict table
    d_roc = res_l2["test_roc_auc"] - res_l1["test_roc_auc"]
    d_pr = res_l2["test_pr_auc"] - res_l1["test_pr_auc"]
    d_f1 = res_l2["argmax"]["f1"] - res_l1["argmax"]["f1"]
    d_tf1 = res_l2["tuned_on_tune"]["f1"] - res_l1["tuned_on_tune"]["f1"]
    summary = {
        "config": {"n": args.n, "provider": args.provider, "model": args.model,
                   "code_enc": CODE_ID, "text_enc": TEXT_ID, "epochs": args.epochs,
                   "test_frac": TEST_FRAC, "tune_frac": TUNE_FRAC,
                   "n_test": int(len(test_idx)), "n_fit": int(len(fit_idx)),
                   "n_explanations": len(rows)},
        "standalone_signal": sig,
        "L1_code_only": res_l1,
        "L2_code_plus_v2expl": res_l2,
        "deltas_L2_minus_L1": {"roc_auc": d_roc, "pr_auc": d_pr,
                               "f1_argmax": d_f1, "f1_tuned": d_tf1},
    }
    results_path = os.path.join(OUT, f"pilot_results__{_sanitize(args.model)}.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print("VERDICT  (test set, n_test = %d) — threshold-free ROC/PR is the fair measure"
          % len(test_idx))
    print("-" * 78)
    print(f"{'metric':<16}{'L1 code-only':>16}{'L2 +v2-expl':>16}{'delta (L2-L1)':>16}")
    print(f"{'ROC-AUC':<16}{res_l1['test_roc_auc']:>16.2f}"
          f"{res_l2['test_roc_auc']:>16.2f}{d_roc:>+16.2f}")
    print(f"{'PR-AUC':<16}{res_l1['test_pr_auc']:>16.2f}"
          f"{res_l2['test_pr_auc']:>16.2f}{d_pr:>+16.2f}")
    print(f"{'F1 @0.5':<16}{res_l1['argmax']['f1']:>16.2f}"
          f"{res_l2['argmax']['f1']:>16.2f}{d_f1:>+16.2f}")
    print(f"{'F1 tuned':<16}{res_l1['tuned_on_tune']['f1']:>16.2f}"
          f"{res_l2['tuned_on_tune']['f1']:>16.2f}{d_tf1:>+16.2f}")
    print(f"{'acc @0.5':<16}{res_l1['argmax']['acc']:>16.2f}"
          f"{res_l2['argmax']['acc']:>16.2f}"
          f"{res_l2['argmax']['acc']-res_l1['argmax']['acc']:>+16.2f}")
    print("-" * 78)
    print(f"standalone explanation signal AUC (risk_level) = "
          f"{sig['signal_auc_risk_level']:.3f}  (v1 ~0.51, gate ~0.58)")
    print(f"full results -> {results_path}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
