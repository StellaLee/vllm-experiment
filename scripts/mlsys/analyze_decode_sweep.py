#!/usr/bin/env python3
"""Decode-length sweep analysis (14B TP=2, Cs2=0). For each max-tokens, compare
mono vs chunk on TTFT (prefill wait) and TPOT (decode step latency = the decode-stall
signal). Hypothesis: chunk's TPOT advantage should GROW with decode length (more decode
to protect from prefill stalls). Pooled over trials; NEG chunk Δ% = chunk wins."""
import json, glob, statistics as st

MTS = ["128", "512", "1024"]
TRIALS = ["1", "2", "3"]

def rows(p):
    return [r for r in (json.loads(l) for l in open(p) if l.strip())]
def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")
def vals(mt, arm, field):
    xs = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*cs2repl_mt{mt}-{arm}-cv0-t{tr}.jsonl"):
            xs += [r[field] * 1000 for r in rows(f) if r.get(field) is not None]
    return sorted(xs)
def summ(xs):
    return (st.median(xs), pct(xs, 95)) if xs else (float("nan"), float("nan"))
def d(c, m):
    return (c - m) / m * 100 if (m == m and m) else float("nan")

print("DECODE-LENGTH SWEEP — 14B TP=2, Cs2=0, mono vs chunk (pooled trials).")
print("NEG chunk Δ% = chunk wins. KEY: does chunk's TPOT advantage grow with max-tokens?\n")
for field, lab in (("ttft", "TTFT (prefill wait)"), ("tpot", "TPOT (decode step latency)")):
    print(f"=== {lab} ===")
    print(f"{'max_tok':>8} | {'mono p50/p95 ms':>18} | {'chunk p50/p95 ms':>18} | {'d_p50%':>8} | {'d_p95%':>8}")
    for mt in MTS:
        m50, m95 = summ(vals(mt, "mono", field))
        c50, c95 = summ(vals(mt, "chunk", field))
        print(f"{mt:>8} | {m50:8.0f}/{m95:8.0f} | {c50:8.0f}/{c95:8.0f} | {d(c50,m50):+7.1f} | {d(c95,m95):+7.1f}")
    print()

print("=== saturation sanity (mono TTFT first50->last50 per max-tokens; flat = sub-sat) ===")
for mt in MTS:
    tt = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*cs2repl_mt{mt}-mono-cv0-t{tr}.jsonl"):
            tt += [(r["ts"], r["ttft"] * 1000) for r in rows(f)
                   if r.get("ttft") is not None and r.get("ts") is not None]
    tt.sort()
    v = [x for _, x in tt]
    if len(v) >= 100:
        f50, l50 = st.median(v[:50]), st.median(v[-50:])
        flag = "SATURATED!" if l50 > 2 * f50 else "sub-sat OK"
        print(f"  mt={mt:>4}: first50={f50:.0f}ms  last50={l50:.0f}ms  ratio={l50/f50:.1f}x  [{flag}]")
