#!/usr/bin/env python3
import json, glob, statistics as st

THR = ["0","512"]
WF = ["05","15"]
CONC = ["20","40"]
NBINS = 4

def load(t, wf, conc):
    f = f"logs/2026-07-23-cs2wt-t{t}-w{wf}-c{conc}-t1.jsonl"
    try:
        return [json.loads(l) for l in open(f) if l.strip()]
    except FileNotFoundError:
        return []

def pctl(x, p):
    if not x: return float("nan")
    y = sorted(x)
    return y[min(len(y)-1, int(p/100*len(y)))]

def bin_stats(rows, key):
    vals = [(r["ts"], r[key]*1000) for r in rows if r.get(key) is not None and r.get("ts") is not None]
    if not vals:
        return []
    vals.sort()
    t0, t1 = vals[0][0], vals[-1][0]
    span = max(t1 - t0, 1e-9)
    bins = [[] for _ in range(NBINS)]
    for ts, v in vals:
        idx = min(NBINS-1, int((ts - t0) / span * NBINS))
        bins[idx].append(v)
    out = []
    for b in bins:
        if b:
            out.append((st.mean(b), pctl(b,95), len(b)))
        else:
            out.append((float("nan"), float("nan"), 0))
    return out

for wf in WF:
    for conc in CONC:
        print(f"\n{'='*90}\nwf={wf} conc={conc}\n{'='*90}")
        for t in THR:
            rows = load(t, wf, conc)
            if not rows:
                print(f"  [thr={t}] NO DATA"); continue
            whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
            short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
            print(f"  [thr={t}] n={len(rows)} (whale={len(whale)}, short={len(short)})")
            for label, pop in [("W", whale), ("S", short)]:
                for metric in ["ttft", "tpot"]:
                    bins = bin_stats(pop, metric)
                    cells = " | ".join(f"q{i+1}:mean={m:.0f},p95={p:.0f},n={n}" for i,(m,p,n) in enumerate(bins))
                    print(f"    {label} {metric:4s}: {cells}")
