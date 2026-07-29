#!/usr/bin/env python3
"""Cs^2 crossover sweep analysis (3-arm mono/chunk/ours-slocvar). Per Cs^2 point it reports the
REALIZED service-size Cs^2 (from prompt_tokens_approx -- the nominal pad-cv2 knob is compressed
by lognormal clipping, so we never trust it), TTFT (mean + p50/p95), TPOT (p50/p95) with deltas
vs mono for BOTH TTFT-mean and TPOT-p50/p95, the crossover signal (chunk TTFT-mean vs mono;
theory: >0 below Cs^2=1, ~0 at 1, <0 above), a saturation flag, and the slocvar budget
trajectory. Ends with a one-line-per-point summary (ordered by realized Cs^2) to see the flip."""
import json, glob, statistics as st, csv
from collections import Counter

# (prefix, nominal pad-cv2 knob). Realized Cs^2 is measured, not assumed.
POINTS = [("cross05", 0.5), ("cross10", 1.0), ("cross15", 1.5),
          ("cross175", 1.75), ("cross225", 2.25)]
ARMS = ["mono", "chunk", "ours"]
TRIALS = [str(i) for i in range(1, 9)]  # up to 8 trials; missing files simply absent


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def vals(prefix, arm, field):
    xs = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-{prefix}-{arm}-cv0-t{tr}.jsonl"):
            xs += [r[field] * 1000 for r in rows(f) if r.get(field) is not None]
    return xs


def realized_cs2(prefix):
    """Service-size Cs^2 from prompt word counts (mono arm; sizes paired across arms)."""
    xs = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-{prefix}-mono-cv0-t{tr}.jsonl"):
            xs += [r["prompt_tokens_approx"] for r in rows(f)
                   if r.get("prompt_tokens_approx") is not None]
    if len(xs) < 3:
        return None
    m = st.mean(xs)
    return st.pvariance(xs) / (m * m)


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
print("Crossover signal = chunk TTFT-MEAN vs mono: theory >0 (Cs^2<1) -> ~0 (=1) -> <0 (Cs^2>1).")
print("Deltas (d..%) are vs the mono arm at the same point.\n")

summary = []
for prefix, nom in POINTS:
    mono_ttft = vals(prefix, "mono", "ttft")
    if not mono_ttft:
        continue
    rcs2 = realized_cs2(prefix)
    rcs2_s = f"{rcs2:.3f}" if rcs2 is not None else "  n/a"
    print(f"================ realized Cs^2 = {rcs2_s}  (nominal {nom}, {prefix}) ================")
    m_mean = st.mean(mono_ttft)
    mono_tp50 = st.median(vals(prefix, "mono", "tpot"))
    mono_tp95 = pct(vals(prefix, "mono", "tpot"), 95)
    print(f"{'arm':>6} | {'TTFTmean':>8} {'p50':>6} {'p95':>6} | {'dTTFTm%':>8} | "
          f"{'TPOTp50':>7} {'TPOTp95':>7} | {'dTP50%':>7} {'dTP95%':>7} | {'sat':>4}")
    for a in ARMS:
        vt = vals(prefix, a, "ttft")
        vp = vals(prefix, a, "tpot")
        if not vt:
            print(f"{a:>6} | (no data)")
            continue
        tmean = st.mean(vt)
        dmean = (tmean - m_mean) / m_mean * 100
        tp50, tp95 = st.median(vp), pct(vp, 95)
        dtp50 = (tp50 - mono_tp50) / mono_tp50 * 100 if mono_tp50 else float("nan")
        dtp95 = (tp95 - mono_tp95) / mono_tp95 * 100 if mono_tp95 else float("nan")
        print(f"{a:>6} | {tmean:8.0f} {st.median(vt):6.0f} {pct(vt,95):6.0f} | {dmean:+8.1f} | "
              f"{tp50:7.0f} {tp95:7.0f} | {dtp50:+7.1f} {dtp95:+7.1f} | {sat(prefix,a):4.2f}")
        if a == "chunk":
            summary.append((rcs2 if rcs2 is not None else nom, dmean))
    ch = traj(prefix)
    if ch:
        d = Counter(ch); n = len(ch)
        trans = sum(1 for x, y in zip(ch, ch[1:]) if x != y)
        print(f"  ours budget: start={ch[0]} last={ch[-1]} min={min(ch)} max={max(ch)} "
              f"transitions={trans} dwell[512:{100*d[512]/n:.0f}% 16384:{100*d[16384]/n:.0f}%]")
    print()

print("=== CROSSOVER SUMMARY: chunk TTFT-mean vs mono, ordered by REALIZED Cs^2 (sign flip = crossover) ===")
for cs2, d in sorted(summary):
    print(f"  Cs^2={cs2:5.3f}:  chunk d_mean = {d:+6.1f}%   ({'chunk WORSE' if d > 0 else 'chunk BETTER'})")
