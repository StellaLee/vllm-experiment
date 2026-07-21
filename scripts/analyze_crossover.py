#!/usr/bin/env python3
"""Cs^2 crossover sweep analysis (3-arm mono/chunk/ours-slocvar). Per Cs^2 point: TTFT (mean +
p50/p95) and TPOT (p50/p95) per arm, the crossover signal (chunk TTFT-MEAN vs mono; theory: >0
below Cs^2=1, ~0 at 1, <0 above), ours-vs-mono deltas, a saturation flag (late/early TTFT
ratio), and the slocvar budget trajectory (start/last/min/max, transitions, dwell) from the
chunktrace csv. Ends with a one-line-per-point crossover summary to see the sign flip."""
import json, glob, statistics as st, csv
from collections import Counter

POINTS = [("cross05", 0.5), ("cross10", 1.0), ("cross15", 1.5)]
ARMS = ["mono", "chunk", "ours"]
TRIALS = ["1", "2", "3"]


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def vals(prefix, arm, field):
    xs = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-{prefix}-{arm}-cv0-t{tr}.jsonl"):
            xs += [r[field] * 1000 for r in rows(f) if r.get(field) is not None]
    return xs


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(p / 100 * len(ys)))] if ys else float("nan")


def sat(prefix, arm):
    """Per-trial late/early TTFT ratio, averaged. >~1.3 hints saturation/drift."""
    ratios = []
    for tr in TRIALS:
        xs = []
        for f in sorted(glob.glob(f"logs/*-{prefix}-{arm}-cv0-t{tr}.jsonl")):
            xs += [r["ttft"] for r in rows(f) if r.get("ttft") is not None]
        if len(xs) >= 60 and st.mean(xs[:30]):
            ratios.append(st.mean(xs[-30:]) / st.mean(xs[:30]))
    return st.mean(ratios) if ratios else float("nan")


def traj(prefix):
    tf = sorted(glob.glob(f"logs/*-{prefix}-ours-chunktrace.csv"))
    if not tf:
        return None
    ch = []
    for row in csv.DictReader(open(tf[0])):
        try:
            ch.append(int(row["chunk"]))
        except (KeyError, ValueError):
            pass
    return ch or None


print("Cs^2 CROSSOVER SWEEP — 14B/TP=2, mt=1024, rate=0.64.  mono(16384) vs chunk(512) vs ours(slocvar start=512)")
print("Crossover signal = chunk TTFT-MEAN vs mono: theory >0 (Cs^2<1) -> ~0 (=1) -> <0 (Cs^2>1).\n")

summary = []
for prefix, cv in POINTS:
    print(f"================ Cs^2 = {cv}  ({prefix}) ================")
    mono_ttft = vals(prefix, "mono", "ttft")
    if not mono_ttft:
        print("  (no data)\n")
        continue
    m_mean = st.mean(mono_ttft)
    print(f"{'arm':>6} | {'TTFTmean':>9} {'TTFTp50':>8} {'TTFTp95':>8} | {'TPOTp50':>8} {'TPOTp95':>8} | {'ttft d_mean%':>12} {'sat':>5}")
    for a in ARMS:
        vt = vals(prefix, a, "ttft")
        vp = vals(prefix, a, "tpot")
        if not vt:
            print(f"{a:>6} | (no data)")
            continue
        tmean = st.mean(vt)
        dmean = (tmean - m_mean) / m_mean * 100
        print(f"{a:>6} | {tmean:9.0f} {st.median(vt):8.0f} {pct(vt,95):8.0f} | "
              f"{st.median(vp):8.0f} {pct(vp,95):8.0f} | {dmean:+11.1f} {sat(prefix,a):5.2f}")
        if a == "chunk":
            summary.append((cv, dmean))
    ch = traj(prefix)
    if ch:
        d = Counter(ch); n = len(ch)
        dwell = " ".join(f"{b}:{100*d[b]/n:.0f}%" for b in sorted(d))
        trans = sum(1 for x, y in zip(ch, ch[1:]) if x != y)
        print(f"  ours budget: start={ch[0]} last={ch[-1]} min={min(ch)} max={max(ch)} "
              f"transitions={trans} dwell[{dwell}]")
    print()

print("=== CROSSOVER SUMMARY: chunk TTFT-mean vs mono (sign flip across Cs^2=1 = crossover) ===")
for cv, d in summary:
    print(f"  Cs^2={cv}:  chunk d_mean = {d:+6.1f}%   ({'chunk WORSE' if d > 0 else 'chunk BETTER'})")
