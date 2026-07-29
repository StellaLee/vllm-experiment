#!/usr/bin/env python3
"""Utilization sweep: wf=05 fixed, conc in {20 (reuse existing), 26, 32, 38}, threshold {0,512}.
Tests whether the admission-side S:TTFT benefit grows with concurrency (utilization proxy) as
Eq. genuine's rho/(1-rho) term predicts."""
import json, glob, statistics as st

CONCS = ["20", "26", "32", "38"]

SOURCES = {
    (0, "20"): "logs/2026-07-23-cs2wt-t0-w05-c20-t1.jsonl",
    (512, "20"): "logs/2026-07-23-cs2wt-t512-w05-c20-t1.jsonl",
}
for c in ("26", "32", "38"):
    SOURCES[(0, c)] = f"logs/2026-07-24-cs2wt-t0-w05-c{c}-t1.jsonl"
    SOURCES[(512, c)] = f"logs/2026-07-24-cs2wt-t512-w05-c{c}-t1.jsonl"


def load(f):
    try:
        return [json.loads(l) for l in open(f) if l.strip()]
    except FileNotFoundError:
        return []


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def stats(rows, key):
    v = [r[key] * 1000 for r in rows if r.get(key) is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    return (st.mean(v), pctl(v, 95), pctl(v, 99), len(v))


print("Utilization sweep -- wf=05 fixed, threshold={0,512}, conc swept 20->38\n")
print(f"{'conc':>5} {'thr':>4} {'pop':>3} {'n':>4} | {'TTFTmean':>9} {'TTFTp95':>9} {'TTFTp99':>9}")
results = {}
for c in CONCS:
    for thr in (0, 512):
        rows = load(SOURCES[(thr, c)])
        if not rows:
            print(f"{c:>5} {thr:>4} NO DATA")
            continue
        whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
        short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
        for label, pop in [("W", whale), ("S", short)]:
            tm, t95, t99, n = stats(pop, "ttft")
            results[(c, thr, label)] = (tm, t95, t99, n)
            print(f"{c:>5} {thr:>4} {label:>3} {n:>4} | {tm:9.0f} {t95:9.0f} {t99:9.0f}")
    print()

print("=== S:TTFT delta (threshold=512 vs mono) as concurrency rises ===\n")
print(f"{'conc':>5} | {'S:TTFTmean d%':>14} {'S:TTFTp95 d%':>14} {'S:TTFTp99 d%':>14}")
for c in CONCS:
    mono = results.get((c, 0, "S"))
    chunk = results.get((c, 512, "S"))
    if not mono or not chunk:
        continue
    def d(a, b):
        return (b - a) / a * 100 if a else float("nan")
    print(f"{c:>5} | {d(mono[0], chunk[0]):+14.1f} {d(mono[1], chunk[1]):+14.1f} {d(mono[2], chunk[2]):+14.1f}")
