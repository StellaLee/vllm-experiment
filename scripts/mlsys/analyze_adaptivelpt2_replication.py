#!/usr/bin/env python3
"""Per-trial replication check for §5.7's adaptivelpt2 result. Reuses the same phase-
reconstruction logic as scripts/analyze_lengthgate.py (bucket_lengthgate), applied
separately to each of the 3 adaptivelpt2 trials, plus the (single-instance) static
baselines for comparison context."""
import glob, json, os, sys, statistics as st
sys.path.insert(0, "/root/pli/vllm-experiment/scripts")
from replay_timing import parse_phase_schedule, phase_type_at, token_times

SCHEDULE = "6:0.0@60,1.0:0.2@60"
SLO = 500.0
sched = parse_phase_schedule(SCHEDULE)


def _ptype(t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"


def load(pattern):
    R = []
    for f in glob.glob(pattern):
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def bucket(recs):
    starts = [r["ts"] - r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
    out = {"S": {"ttft": [], "tbt": [], "n": 0}, "W": {"ttft": [], "tbt": [], "n": 0}}
    if not starts:
        return out
    t0 = min(starts)
    for r in recs:
        ts, lat, ttft = r.get("ts"), r.get("latency"), r.get("ttft")
        if ts is None or lat is None:
            continue
        start = ts - lat
        pt = _ptype(start - t0)
        out[pt]["n"] += 1
        if ttft is not None:
            out[pt]["ttft"].append(ttft)
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            out[_ptype(emit - t0)]["tbt"].append(ms)
    return out


def w_goodput(recs):
    starts = [r["ts"] - r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
    if not starts:
        return float("nan"), 0
    t0 = min(starts)
    w_recs = [r for r in recs if r.get("tbt_ms") and _ptype((r["ts"] - r["latency"]) - t0) == "W"]
    ok = sum(1 for r in w_recs if max(r["tbt_ms"]) <= SLO)
    return (100.0 * ok / len(w_recs) if w_recs else float("nan")), len(w_recs)


def report(label, recs):
    b = bucket(recs)
    S, W = b["S"], b["W"]
    s_ttm = 1000.0 * st.mean(S["ttft"]) if S["ttft"] else float("nan")
    w_p99 = pctl(W["tbt"], 99)
    w_max = max(W["tbt"]) if W["tbt"] else float("nan")
    gp, wn = w_goodput(recs)
    print(f"{label:>28} | S:TTFTmean={s_ttm:7.0f}ms S:n={S['n']:4d} | "
          f"W:TBTp99={w_p99:7.1f}ms W:TBTmax={w_max:7.1f}ms W:goodput={gp:5.1f}% W:n={wn:4d}")
    return s_ttm, w_p99, w_max, gp


print(f"adaptivelpt2 replication check -- schedule={SCHEDULE} SLO={SLO}ms\n")
print("--- static baselines (single instance, 2026-07-22) ---")
report("mono (16384)", load("logs/2026-07-22-lgate-b16384-t1.jsonl"))
report("static-512", load("logs/2026-07-22-lgate-b512-t1.jsonl"))
report("static-2048", load("logs/2026-07-22-lgate-b2048-t1.jsonl"))
report("static-16384lpt512", load("logs/2026-07-22-lgate-b16384lpt512-t1.jsonl"))

print("\n--- adaptivelpt2, per trial ---")
report("trial 1 (2026-07-23)", load("logs/2026-07-23-lgate-badaptivelpt2-t1.jsonl"))
report("trial 2 (2026-07-24)", load("logs/2026-07-24-lgate-badaptivelpt2-rep2-t1.jsonl"))
report("trial 3 (2026-07-24)", load("logs/2026-07-24-lgate-badaptivelpt2-rep3-t1.jsonl"))
