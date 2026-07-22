#!/usr/bin/env python3
"""Latency DECOMPOSITION for the chunk-size sweep (reads existing logs only, no run).
Question: is chunk's TTFT win economically meaningful, or does decode dominate end-to-end latency
at fixed concurrency? Reports per-arm mean output_tokens, mean TTFT, mean decode time, mean E2E
latency, TTFT-share of latency, and implied throughput (concurrency / mean_latency, Little's law)."""
import json, glob, statistics as st
CONC = 40
def rows(b):
    R = []
    for f in glob.glob(f"logs/2026-07-21-sweep-b{b}-t1.jsonl"):
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R
budgets = ["16384", "512", "1024", "2048"]
print("budget |  n  out_tok  TTFTs  decode_s   E2E_s  TTFT%%  thrpt(req/s)  [arm]")
base_lat = None
for b in budgets:
    R = rows(b)
    out = st.mean([r["output_tokens"] for r in R if r.get("output_tokens")])
    ttft = st.mean([r["ttft"] for r in R if r.get("ttft") is not None])
    lat = st.mean([r["latency"] for r in R if r.get("latency") is not None])
    dec = lat - ttft
    thr = CONC / lat
    tag = "mono" if b == "16384" else "chunk"
    print("%6s | %3d %7.0f %6.2f %8.2f %7.2f %5.1f%% %11.2f  [%s]" % (
        b, len(R), out, ttft, dec, lat, 100 * ttft / lat, thr, tag))
    if b == "16384":
        base_lat = lat
print()
mono = rows("16384"); ml = st.mean([r["latency"] for r in mono if r.get("latency")])
print("end-to-end latency delta vs mono (drives throughput at fixed concurrency):")
for b in budgets[1:]:
    R = rows(b); l = st.mean([r["latency"] for r in R if r.get("latency")])
    print("  chunk-%-5s dE2E=%+6.1f%%  (=> throughput %+.1f%%)" % (
        b, (l - ml) / ml * 100, (ml / l - 1) * 100))
