#!/usr/bin/env python3
import json, statistics as st

THR = ["0","512"]
WF = ["05","15"]
CONC = ["20","40"]

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

def stats(rows, key):
    v = [r[key]*1000 for r in rows if r.get(key) is not None]
    if not v: return (float("nan"),)*3 + (0,)
    return (st.mean(v), pctl(v,95), pctl(v,99), len(v))

for wf in WF:
    for conc in CONC:
        print(f"\n{'='*100}\nwf={wf} conc={conc}\n{'='*100}")
        print(f"{'arm':8s} {'pop':3s} {'n':4s} | {'TTFTmean':>9s} {'TTFTp95':>9s} {'TTFTp99':>9s} | {'TPOTmean':>9s} {'TPOTp95':>9s} {'TPOTp99':>9s}")
        for t in THR:
            rows = load(t, wf, conc)
            if not rows:
                print(f"[thr={t}] NO DATA"); continue
            whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
            short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
            for label, pop in [("W", whale), ("S", short)]:
                tm, t95, t99, n = stats(pop, "ttft")
                pm, p95, p99, _ = stats(pop, "tpot")
                print(f"t={t:4s} {label:3s} {n:4d} | {tm:9.0f} {t95:9.0f} {t99:9.0f} | {pm:9.1f} {p95:9.1f} {p99:9.1f}")
