#!/usr/bin/env python3
"""Compare trial 1 (original whale/threshold/concurrency sweep) vs trials 2/3
(orchestrate_cs2_whale_threshold_replicate.sh) at wf in {05,15}, conc=20, threshold {0,512}."""
import json, glob, statistics as st

def load(pattern):
    files = glob.glob(pattern)
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows

def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")

def stats(rows, key):
    v = [r[key] * 1000 for r in rows if r.get(key) is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    return (st.mean(v), pctl(v, 95), pctl(v, 99), len(v))

TRIAL1_PATTERNS = {
    (0, "05"): "logs/2026-07-23-cs2wt-t0-w05-c20-t1.jsonl",
    (0, "15"): "logs/2026-07-23-cs2wt-t0-w15-c20-t1.jsonl",
    (512, "05"): "logs/2026-07-23-cs2wt-t512-w05-c20-t1.jsonl",
    (512, "15"): "logs/2026-07-23-cs2wt-t512-w15-c20-t1.jsonl",
}

for wf in ("05", "15"):
    print(f"\n{'='*100}\nwf={wf} conc=20\n{'='*100}")
    for trial in (1, 2, 3):
        for thr in (0, 512):
            if trial == 1:
                rows = load(TRIAL1_PATTERNS[(thr, wf)])
            else:
                rows = load(f"logs/*-cs2wtrep{trial}-t{thr}-w{wf}-c20-t{trial}.jsonl")
            if not rows:
                print(f"trial={trial} thr={thr}: NO DATA")
                continue
            whale = [r for r in rows if (r.get("pad_chars") or 0) >= 40000]
            short = [r for r in rows if (r.get("pad_chars") or 0) < 40000]
            for label, pop in [("W", whale), ("S", short)]:
                tm, t95, t99, n = stats(pop, "ttft")
                pm, p95, p99, _ = stats(pop, "tpot")
                print(f"trial={trial} thr={thr:<4d} {label} n={n:<4d} | "
                      f"TTFT mean={tm:7.0f} p95={t95:7.0f} p99={t99:7.0f} | "
                      f"TPOT mean={pm:6.1f} p95={p95:6.1f} p99={p99:6.1f}")
        print()

print("\n=== Delta summary across trials: threshold=512 vs mono, S population ===\n")
for wf in ("05", "15"):
    print(f"-- wf={wf} --")
    for trial in (1, 2, 3):
        if trial == 1:
            mono = load(TRIAL1_PATTERNS[(0, wf)])
            chunk = load(TRIAL1_PATTERNS[(512, wf)])
        else:
            mono = load(f"logs/*-cs2wtrep{trial}-t0-w{wf}-c20-t{trial}.jsonl")
            chunk = load(f"logs/*-cs2wtrep{trial}-t512-w{wf}-c20-t{trial}.jsonl")
        mono_s = [r for r in mono if (r.get("pad_chars") or 0) < 40000]
        chunk_s = [r for r in chunk if (r.get("pad_chars") or 0) < 40000]
        if not mono_s or not chunk_s:
            continue
        mtm, mt95, mt99, _ = stats(mono_s, "ttft")
        ctm, ct95, ct99, _ = stats(chunk_s, "ttft")
        mpm, mp95, mp99, _ = stats(mono_s, "tpot")
        cpm, cp95, cp99, _ = stats(chunk_s, "tpot")
        d = lambda a, b: (b - a) / a * 100 if a else float("nan")
        print(f"  trial={trial}: dTTFT mean={d(mtm,ctm):+6.1f}% p95={d(mt95,ct95):+6.1f}% p99={d(mt99,ct99):+6.1f}% | "
              f"dTPOT mean={d(mpm,cpm):+6.1f}% p95={d(mp95,cp95):+6.1f}% p99={d(mp99,cp99):+6.1f}%")
