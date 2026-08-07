"""Generate explanation JSONL via local Ollama, with a decode-time confidence probe.

Replaces expl_v2/generate_v2.py. Two things changed:

1. ONE prompt module (experiments/explanation/prompt.py) covering both the
   anonymized (Devign) and real-identifier (ReVeal / de-anonymized) forms.

2. `explanation.confidence` is MEASURED, not asked for. The request sets
   logprobs=true, and probe_confidence() reads the decode-time token
   probabilities of the risk_level verdict span -- the model's internal
   probability of the verdict it emitted, conditioned on the explanation it
   just wrote. Nothing here reads the ground-truth label, so the value is
   label-blind by construction. (The previous numeric `confidence` in
   explanations/SemanticVul/ACTIVE/* was drawn from a label-conditioned
   triangular distribution by the now-deleted fill_data.py; it was not a probe.)

Output row schema (matches the ACTIVE JSONL columns the LLM stage owns):
    sample_id, label, raw_code,
    explanation{purpose, data_flow, risky_operations, missing_checks,
                evidence_tokens, safety_indicators, risk_summary, risk_level,
                confidence},
    meta{model, prompt, mode, gen_seconds, confidence_probe{...}}

Downstream stages add the remaining explanation.* columns:
    expl_enrich/static_enrich.py      -> llm_v1, code_metrics, tail_facts, enrich
    expl_enrich/apply_real_enrichment -> function_name, called_functions,
                                         risky_apis, string_literals,
                                         lexical_digest, real_enrich, tail_digest
    ladder builder                    -> prefix, prefix_recipe

Resumable: sample_ids already present in the output file are skipped.

Usage:
  .venv/Scripts/python.exe experiments/explanation/generate.py \
      --dataset devign --split train --model qwen2.5-coder:14b --stratified 300
  .venv/Scripts/python.exe experiments/explanation/generate.py \
      --dataset reveal --split val --model qwen2.5-coder:14b
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_io import load_split                       # noqa: E402
from prompt import JSON_SCHEMA, FIELD_ORDER, RISK_LEVELS, build_messages  # noqa: E402

MAX_CODE_CHARS = 12000  # ~3k tokens; median function is ~145 tokens

# devign ships VARn/FUNn-normalized code, reveal ships real identifiers.
DEFAULT_MODE = {"devign": "anon", "reveal": "real"}

_RISK_KEY = re.compile(r'"risk_level"\s*:\s*"')
_WS = re.compile(r"\s+")


# --- confidence probe ------------------------------------------------------

def _entropy(logprobs) -> float:
    """Shannon entropy (nats) over the returned top-k, renormalized to sum 1."""
    ps = [math.exp(lp) for lp in logprobs if lp is not None and lp > -30]
    tot = sum(ps)
    if tot <= 0:
        return 0.0
    return -sum((p / tot) * math.log(p / tot) for p in ps if p > 0)


def probe_confidence(logprobs) -> dict:
    """Measure the model's internal confidence from decode-time token logprobs.

    Returns a dict with:
      confidence      int 0..100 -- 100 * geometric-mean token probability of
                      the risk_level VALUE tokens (the verdict span).
      span            "verdict" if the verdict tokens were located, else
                      "sequence" (fallback: whole generation).
      p_verdict       float, same quantity as confidence but unrounded.
      p_sequence      float, geometric-mean token probability over all tokens.
      margin          float, p(top-1) - p(top-2) at the decisive verdict token.
      entropy         float, entropy (nats) of the decisive token's top-k.
      alternatives    [{token, p}] top-k at the decisive token.
      n_verdict_tokens / n_tokens
    """
    out = {"confidence": None, "span": "none", "p_verdict": None,
           "p_sequence": None, "margin": None, "entropy": None,
           "alternatives": [], "n_verdict_tokens": 0, "n_tokens": 0}
    if not logprobs:
        return out

    toks = [t for t in logprobs if isinstance(t, dict)]
    out["n_tokens"] = len(toks)
    if not toks:
        return out

    seq_lps = [float(t.get("logprob", 0.0)) for t in toks]
    if seq_lps:
        out["p_sequence"] = math.exp(sum(seq_lps) / len(seq_lps))

    # Rebuild the emitted text from the tokens and locate the risk_level value.
    text = "".join(str(t.get("token", "")) for t in toks)
    m = _RISK_KEY.search(text)
    picked = []
    if m:
        v_start = m.end()
        v_end = text.find('"', v_start)
        if v_end == -1:
            v_end = len(text)
        pos = 0
        for t in toks:
            s = str(t.get("token", ""))
            a, b = pos, pos + len(s)
            pos = b
            if b > v_start and a < v_end:   # overlaps the value span
                picked.append(t)

    if picked:
        out["span"] = "verdict"
        lps = [float(t.get("logprob", 0.0)) for t in picked]
        out["p_verdict"] = math.exp(sum(lps) / len(lps))
        out["n_verdict_tokens"] = len(picked)
        # decisive token = first one carrying a letter (skip a bare quote token)
        decisive = next((t for t in picked
                         if any(c.isalpha() for c in str(t.get("token", "")))),
                        picked[0])
    else:
        out["span"] = "sequence"
        out["p_verdict"] = out["p_sequence"]
        decisive = toks[0]

    top = [t for t in (decisive.get("top_logprobs") or []) if isinstance(t, dict)]
    if top:
        ps = sorted((math.exp(float(t.get("logprob", -99))) for t in top),
                    reverse=True)
        out["margin"] = ps[0] - (ps[1] if len(ps) > 1 else 0.0)
        out["entropy"] = _entropy([float(t.get("logprob", -99)) for t in top])
        out["alternatives"] = [{"token": str(t.get("token", "")),
                                "p": round(math.exp(float(t.get("logprob", -99))), 6)}
                               for t in top]

    p = out["p_verdict"]
    if p is not None:
        out["confidence"] = int(max(0, min(100, round(100.0 * p))))
    return out


# --- response normalization ------------------------------------------------

def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return " ".join(_as_str(x) for x in v).strip()
    return str(v).strip()


def _as_str_list(v, cap: int) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for x in v:
        if isinstance(x, dict):
            # tolerate a model that emits {pattern, evidence, why} anyway
            pat = _as_str(x.get("pattern") or x.get("check") or x.get("name"))
            ev = _as_str(x.get("evidence"))
            s = f"{pat} [evidence: {ev}]" if pat and ev else (pat or ev)
        else:
            s = _as_str(x)
        if s and s not in out:
            out.append(s)
    return out[:cap]


def _quoted_evidence(expl: dict) -> list:
    """Every verbatim fragment the model claims to have quoted."""
    out = []
    for s in expl.get("risky_operations") or []:
        if "[evidence:" in s:
            out.append(s.split("[evidence:", 1)[1].rsplit("]", 1)[0].strip())
    out += [str(x) for x in (expl.get("evidence_tokens") or [])]
    out += [str(g.get("evidence", "")) for g in (expl.get("safety_indicators") or [])
            if isinstance(g, dict)]
    return [s for s in out if s]


def grounding_stats(expl: dict, code: str) -> dict:
    """How many quoted fragments really occur in this function.

    A model that copies evidence out of the few-shot exemplars scores < 1.0 here.
    Diagnostic only -- the actual drop happens in expl_enrich/static_enrich.py,
    which keeps the raw claim in explanation["llm_v1"]. Reported so few-shot
    leakage is measurable instead of silent, because `confidence` probes the
    verdict the model reached FROM these quotes.

    Matching is WHITESPACE-INSENSITIVE. Models reflow the spacing of the
    tokenized C ("for ( c = a ; c <= b ; c ++ )" vs the source's line breaks);
    an exact substring test scores those as fabricated and badly understates
    grounding -- measured 77% exact vs 95% whitespace-insensitive on the same
    rows. Only `exact` is reported separately, for transparency.
    """
    ev = _quoted_evidence(expl)
    if not ev:
        return {"evidence_total": 0, "evidence_grounded": 0,
                "evidence_exact": 0, "grounded_frac": None}
    flat = _WS.sub("", code)
    hit = sum(1 for s in ev if _WS.sub("", s) in flat)
    exact = sum(1 for s in ev if s in code)
    return {"evidence_total": len(ev), "evidence_grounded": hit,
            "evidence_exact": exact, "grounded_frac": round(hit / len(ev), 4)}


def normalize_explanation(raw: dict) -> dict:
    """Coerce the model's JSON into the exact ACTIVE column types/order."""
    e = raw if isinstance(raw, dict) else {}
    lvl = _as_str(e.get("risk_level")).upper()
    if lvl not in RISK_LEVELS:
        lvl = "NONE"
    si = []
    for g in (e.get("safety_indicators") or []):
        if isinstance(g, dict):
            c, ev = _as_str(g.get("check")), _as_str(g.get("evidence"))
            if c or ev:
                si.append({"check": c, "evidence": ev})
        elif _as_str(g):
            si.append({"check": _as_str(g), "evidence": ""})
    built = {
        "purpose": _as_str(e.get("purpose")),
        "data_flow": _as_str(e.get("data_flow")),
        "risky_operations": _as_str_list(e.get("risky_operations"), 6),
        "missing_checks": _as_str_list(e.get("missing_checks"), 6),
        "evidence_tokens": _as_str_list(e.get("evidence_tokens"), 12),
        "safety_indicators": si[:6],
        "risk_summary": _as_str(e.get("risk_summary")),
        "risk_level": lvl,
    }
    return {k: built[k] for k in FIELD_ORDER}


# --- ollama ----------------------------------------------------------------

def ollama_chat(host, model, messages, num_ctx, timeout, no_think, top_logprobs):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": JSON_SCHEMA,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "options": {"temperature": 0.0, "seed": 1234, "num_ctx": num_ctx},
    }
    if no_think:
        payload["think"] = False
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def sanitize(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "-")


def default_out(dataset, split, model, tag):
    name = f"{dataset}_{split}__{sanitize(model)}{('__' + tag) if tag else ''}.jsonl"
    return os.path.join(HERE, "out", name)


def stratified_pick(samples, n, seed):
    """Deterministic per-class pick so every model sees the SAME set."""
    by_label = {0: [], 1: []}
    for s in samples:
        if s.label in by_label:
            by_label[s.label].append(s)
    rng = random.Random(seed)
    picked = []
    for lab in (0, 1):
        pool = sorted(by_label[lab], key=lambda s: s.sample_id)
        rng.shuffle(pool)
        picked.extend(pool[: n // 2])
    picked.sort(key=lambda s: s.sample_id)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:9999")
    ap.add_argument("--mode", default="auto", choices=["auto", "anon", "real"],
                    help="identifier policy; auto = devign:anon, reveal:real")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix for the default out name")
    ap.add_argument("--stratified", type=int, default=None,
                    help="pick N samples (N/2 per class), deterministic")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--top-logprobs", type=int, default=5,
                    help="top-k alternatives kept at the decisive verdict token")
    ap.add_argument("--no-think", action="store_true",
                    help='send "think": false (qwen3-family thinking models)')
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests; set OLLAMA_NUM_PARALLEL>=workers "
                         "on the server or requests just queue")
    args = ap.parse_args()

    mode = DEFAULT_MODE[args.dataset] if args.mode == "auto" else args.mode
    out_path = args.out or default_out(args.dataset, args.split, args.model, args.tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_split(args.dataset, args.split)
    if args.stratified:
        samples = stratified_pick(samples, args.stratified, args.seed)
    if args.limit:
        samples = samples[: args.limit]

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
    print(f"[gen] {args.dataset}/{args.split} model={args.model} mode={mode} "
          f"total={len(samples)} done={len(done)} todo={len(todo)} -> {out_path}",
          flush=True)

    def gen_one(s):
        code = s.code[:MAX_CODE_CHARS]
        err = None
        for attempt in (1, 2):
            try:
                t1 = time.time()
                resp = ollama_chat(args.host, args.model,
                                   build_messages(code, mode),
                                   args.num_ctx, args.timeout, args.no_think,
                                   args.top_logprobs)
                dur_s = time.time() - t1
                expl = normalize_explanation(
                    json.loads(resp["message"]["content"]))
                probe = probe_confidence(resp.get("logprobs"))
                expl["confidence"] = probe["confidence"]
                return {
                    "sample_id": s.sample_id,
                    "label": s.label,
                    "raw_code": s.code,
                    "explanation": expl,
                    "meta": {"model": args.model, "prompt": "explanation-v1",
                             "mode": mode, "gen_seconds": round(dur_s, 2),
                             "confidence_probe": probe,
                             "grounding": grounding_stats(expl, s.code)},
                }, None
            except (urllib.error.URLError, urllib.error.HTTPError,
                    json.JSONDecodeError, KeyError, TimeoutError,
                    OSError) as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * attempt)
        return None, f"{s.sample_id}: {err}"

    n_ok, n_fail, n_noprobe, n_noconf, t0 = 0, 0, 0, 0, time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(gen_one, s) for s in todo]
            for i, fut in enumerate(as_completed(futures)):
                row, err = fut.result()
                if row is None:
                    n_fail += 1
                    print(f"[gen] FAIL {err}", flush=True)
                    continue
                _p = row["meta"]["confidence_probe"]
                if _p["span"] != "verdict":
                    n_noprobe += 1
                if _p["confidence"] is None:
                    n_noconf += 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if n_ok % 10 == 0 or i == len(todo) - 1:
                    rate = n_ok / max(time.time() - t0, 1e-9)
                    eta_min = (len(todo) - i - 1) / max(rate, 1e-9) / 60
                    print(f"[gen] {n_ok}/{len(todo)} ok ({n_fail} fail) "
                          f"{rate:.2f}/s eta {eta_min:.0f} min", flush=True)

    print(f"[gen] DONE ok={n_ok} fail={n_fail} no-verdict-probe={n_noprobe} "
          f"no-confidence={n_noconf} "
          f"elapsed={(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)
    if n_noconf:
        # Loud, because it is silent downstream: static_enrich fills a missing
        # confidence with the string "high"/"medium", so the column would still
        # be *present* -- just a self-report string instead of a measurement,
        # breaking the int type the ACTIVE JSONLs carry.
        print(f"[gen] WARNING: {n_noconf}/{n_ok} rows carry NO measured "
              f"confidence -- the server returned no usable logprobs.\n"
              f"[gen]          Ollama must support logprobs on /api/chat "
              f"(verified on 0.30.0-rc17); check the server version and that "
              f"model '{args.model}' returns them.\n"
              f"[gen]          Those rows will get a STRING fallback from "
              f"static_enrich, not a probe value.", flush=True)


if __name__ == "__main__":
    main()
