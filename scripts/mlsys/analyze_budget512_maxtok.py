#!/usr/bin/env python3
"""Compare step-wide budget mono(16384) vs chunk(512), threshold OFF, at wf=05/conc=20,
crossed with MAXTOK {256,1024}. Mono baselines come from the two prior sweeps (threshold
sweep's thr=0 arm, and its maxtok=1024 variant's thr=0 arm); chunk=512 is the new run from
orchestrate_budget512_maxtok.sh. Disaggregates whale (W) vs short (S) as usual."""
import json, os, glob, statistics as st

DATE = os.environ.get("DATE", "")

SOURCES = {
    (16384, 256): "logs/2026-07-23-cs2wt-t0-w05-c20-t1.jsonl",
    (16384, 1024): "logs/2026-07-24-cs2wtmt1024-t0-w05-c20-t1.jsonl",
    (512, 256): f"logs/{DATE}-budget512mt-mt256-w05-c20-t1.jsonl",
    (512, 1024): f"logs/{DATE}-budget512mt-mt1024-w05-c20-t1.jsonl",
}


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


print("Step-wide budget mono(16384) vs chunk(512), threshold OFF, wf=05/conc=20, "
      "crossed with MAXTOK {256,1024}\n")
print(f"{'budget':7s} {'maxtok':6s} {'pop':3s} {'n':4s} | "
      f"{'TTFTmean':>9s} {'TTFTp95':>9s} {'TTFTp99':>9s} | "
      f"{'TPOTmean':>9s} {'TPOTp95':>9s} {'TPOTp99':>9s}")

results = {}
for (budget, mt), f in SOURCES.items():
    rows = load(f)
    if not rows:
        print(f"{budget:<7d} {mt:<6d} NO DATA ({f})")
        continue
    whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
    short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
    for label, pop in [("W", whale), ("S", short)]:
        tm, t95, t99, n = stats(pop, "ttft")
        pm, p95, p99, _ = stats(pop, "tpot")
        results[(budget, mt, label)] = (tm, t95, t99, pm, p95, p99, n)
        print(f"{budget:<7d} {mt:<6d} {label:3s} {n:4d} | "
              f"{tm:9.0f} {t95:9.0f} {t99:9.0f} | {pm:9.1f} {p95:9.1f} {p99:9.1f}")
    print()

print("\n=== Delta table: chunk=512 vs mono=16384, same MAXTOK, threshold OFF ===\n")
for mt in (256, 1024):
    for label in ("W", "S"):
        mono = results.get((16384, mt, label))
        chunk = results.get((512, mt, label))
        if not mono or not chunk:
            continue
        mtm, mt95, mt99, mpm, mp95, mp99, mn = mono
        ctm, ct95, ct99, cpm, cp95, cp99, cn = chunk
        def d(a, b):
            return (b - a) / a * 100 if a else float("nan")
        print(f"maxtok={mt:<5d} {label} | TTFT mean {mtm:.0f}->{ctm:.0f} ({d(mtm,ctm):+.1f}%) "
              f"p95 {mt95:.0f}->{ct95:.0f} ({d(mt95,ct95):+.1f}%) "
              f"p99 {mt99:.0f}->{ct99:.0f} ({d(mt99,ct99):+.1f}%)")
        print(f"{'':13s} | TPOT mean {mpm:.1f}->{cpm:.1f} ({d(mpm,cpm):+.1f}%) "
              f"p95 {mp95:.1f}->{cp95:.1f} ({d(mp95,cp95):+.1f}%) "
              f"p99 {mp99:.1f}->{cp99:.1f} ({d(mp99,cp99):+.1f}%)")
