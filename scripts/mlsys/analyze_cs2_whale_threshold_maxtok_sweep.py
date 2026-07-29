#!/usr/bin/env python3
"""Analyze orchestrate_cs2_whale_threshold_sweep_maxtok.sh. Same whale/short
disaggregation as analyze_cs2_whale_threshold_sweep.py, but files are tagged with
MAXTOK and DATE to distinguish from the original MAXTOK=256 run."""
import json, os, statistics as st

THRESHOLDS = os.environ.get("THRESHOLDS", "0 512").split()
MONO = THRESHOLDS[0]
WHALE_FRACS = os.environ.get("WHALE_FRACS", "0.05 0.15").split()
CONCS = os.environ.get("CONCS", "20 40").split()
MAXTOK = os.environ.get("MAXTOK", "1024")
DATE = os.environ.get("DATE", "")


def wtag(wf):
    return f"{round(float(wf) * 100):02d}"


def rows(t, wf, conc):
    f = f"logs/{DATE}-cs2wtmt{MAXTOK}-t{t}-w{wtag(wf)}-c{conc}-t1.jsonl"
    try:
        return [json.loads(l) for l in open(f) if l.strip()]
    except FileNotFoundError:
        return []


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def cs2(vals):
    if len(vals) < 2:
        return float("nan")
    mean = st.mean(vals)
    return st.variance(vals) / (mean ** 2) if mean else float("nan")


def stats(rows_, key):
    v = [r[key] * 1000 for r in rows_ if r.get(key) is not None]
    if not v:
        return (float("nan"),) * 3 + (0,)
    return (st.mean(v), pctl(v, 95), pctl(v, 99), len(v))


print(f"Whale/threshold/concurrency/maxtok sweep -- 14B/TP2, single-turn, budget FIXED 16384, "
      f"maxtok={MAXTOK} (compare vs the MAXTOK=256 run in "
      f"findings/2026-07-23-whale-threshold-concurrency-tradeoff.md)")
print(f"thresholds={THRESHOLDS} (mono={MONO})  whale_fracs={WHALE_FRACS}  concs={CONCS}\n")

print(f"{'wf':4s} {'conc':4s} {'thr':4s} {'pop':3s} {'n':4s} | "
      f"{'TTFTmean':>9s} {'TTFTp95':>9s} {'TTFTp99':>9s} | "
      f"{'TPOTmean':>9s} {'TPOTp95':>9s} {'TPOTp99':>9s}")

for wf in WHALE_FRACS:
    for conc in CONCS:
        for t in THRESHOLDS:
            R = rows(t, wf, conc)
            if not R:
                print(f"{wf:4s} {conc:4s} {t:4s} NO DATA")
                continue
            whale = [r for r in R if (r.get("pad_chars") or 0) >= 40000]
            short = [r for r in R if (r.get("pad_chars") or 0) < 40000]
            for label, pop in [("W", whale), ("S", short)]:
                tm, t95, t99, n = stats(pop, "ttft")
                pm, p95, p99, _ = stats(pop, "tpot")
                print(f"{wf:4s} {conc:4s} {t:4s} {label:3s} {n:4d} | "
                      f"{tm:9.0f} {t95:9.0f} {t99:9.0f} | {pm:9.1f} {p95:9.1f} {p99:9.1f}")
        pt = [r["prompt_tokens_approx"] for r in rows(MONO, wf, conc) if r.get("prompt_tokens_approx")]
        print(f"  (realized Cs^2 mono: {cs2(pt):.3f}, n={len(pt)})\n")
