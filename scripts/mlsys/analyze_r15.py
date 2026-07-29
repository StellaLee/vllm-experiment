#!/usr/bin/env python3
"""Paired mono-vs-chunk crossover analysis across trials t1,t2,t3 for the RATE=1.5 run.
Per cv2 point: per-trial chunk Delta% = (chunk_mean - mono_mean)/mono_mean*100 (paired,
same pad seed within a trial), then mean +/- std across trials and wins/N (chunk<mono).
Also prints a drift sanity (first50 vs last50 median) to confirm sub-saturation.
"""
import json, glob, statistics as st

GRID = [("0", "~0.01"), ("1", "~1.06"), ("2", "~2.0"), ("4", "~4.0")]
TRIALS = ["1", "2", "3"]

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r["ttft"] * 1000 for r in rows if r.get("ttft") is not None]

def find(arm, cv, tr):
    g = glob.glob(f"logs/*cs2repl-{arm}-cv{cv}-t{tr}.jsonl")
    return g[0] if g else None

print("=" * 78)
print("CROSSOVER RESULT  (14B, TP=2, RATE=1.0, NUM=150)  -- paired chunk Delta% vs mono")
print("=" * 78)
print(f"{'Cs2':>6} | {'mono/trial (ms)':>22} | {'chunk/trial (ms)':>22} | {'chunkD%':>16} | wins")
print("-" * 78)
for cv, cs in GRID:
    monos, chunks, deltas = [], [], []
    for tr in TRIALS:
        mp, cp = find("mono", cv, tr), find("chunk", cv, tr)
        if not mp or not cp:
            continue
        m, c = st.mean(load(mp)), st.mean(load(cp))
        monos.append(m); chunks.append(c); deltas.append((c - m) / m * 100)
    if not deltas:
        print(f"{cs:>6} | (no data)"); continue
    wins = sum(1 for d in deltas if d < 0)
    dmean = st.mean(deltas)
    dstd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
    ms = "/".join(f"{x:.0f}" for x in monos)
    csr = "/".join(f"{x:.0f}" for x in chunks)
    star = "  <-- chunk WINS" if dmean < 0 else ""
    print(f"{cs:>6} | {ms:>22} | {csr:>22} | {dmean:+6.1f}+-{dstd:>4.1f} | {wins}/{len(deltas)}{star}")
print("-" * 78)
print("Delta% = (chunk-mono)/mono. NEGATIVE = chunk wins. Prediction: lose at Cs2<1, win at Cs2>1.")

print("\n=== drift sanity (median TTFT first50 -> last50; GROWING would mean saturated) ===")
for cv, cs in GRID:
    line = f"  Cs2 {cs:>6}: "
    for tr in TRIALS:
        for arm in ("mono", "chunk"):
            p = find(arm, cv, tr)
            if not p:
                continue
            v = load(p)
            f50, l50 = st.median(v[:50]), st.median(v[-50:])
            tag = "!" if l50 > 2 * f50 else ""
            line += f"{arm[0]}{tr}:{f50:.0f}->{l50:.0f}{tag}  "
    print(line)
