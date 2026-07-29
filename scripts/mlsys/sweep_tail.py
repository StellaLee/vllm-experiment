#!/usr/bin/env python3
"""Extreme-tail check for the chunk-size sweep: per-request TPOT p50/p90/p95/p99/max and TTFT tail,
plus p99/max TPOT delta vs mono. NOTE: per-request-mean TPOT dilutes transient whale stalls; true
per-token ITL (vllm bench serve) would be sharper. This is a first proxy."""
import json, glob
def rows(b):
    R = []
    for f in glob.glob(f"logs/2026-07-21-sweep-b{b}-t1.jsonl"):
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R
def pctl(x, p):
    y = sorted(x); return y[min(len(y) - 1, int(p / 100 * len(y)))]
budgets = ["16384", "512", "1024", "2048"]
hdr = ("budget | TPOTp50  p90  p95  p99   max  | TTFTp50  p90  p99    max")
print(hdr)
for b in budgets:
    R = rows(b)
    tp = [r["tpot"] * 1000 for r in R if r.get("tpot")]
    tt = [r["ttft"] * 1000 for r in R if r.get("ttft")]
    tag = "mono" if b == "16384" else "chunk"
    print("%6s | %7.1f %5.1f %5.1f %5.1f %6.1f | %7.0f %5.0f %5.0f %6.0f  [%s]" % (
        b, pctl(tp, 50), pctl(tp, 90), pctl(tp, 95), pctl(tp, 99), max(tp),
        pctl(tt, 50), pctl(tt, 90), pctl(tt, 99), max(tt), tag))
mono = rows("16384"); mtp = [r["tpot"] * 1000 for r in mono if r.get("tpot")]
m99 = pctl(mtp, 99); mmax = max(mtp)
print("\nTPOT extreme-tail delta vs mono (does chunking protect the worst decodes?):")
for b in budgets[1:]:
    tp = [r["tpot"] * 1000 for r in rows(b) if r.get("tpot")]
    print("  chunk-%-5s dP99=%+6.1f%%  dMax=%+6.1f%%" % (
        b, (pctl(tp, 99) - m99) / m99 * 100, (max(tp) - mmax) / mmax * 100))
