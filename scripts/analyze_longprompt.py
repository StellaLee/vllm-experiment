#!/usr/bin/env python3
"""Analyze the long-prompt (whale) test. First budget in $BUDGETS = mono baseline. Reports, per arm:
  - TTFT mean/p50/p95, and delta vs mono
  - TPOT (per-request mean) p50/p95  [the coarse metric]
  - P99 / max TBT pooled across ALL output tokens  [the SHARP metric; Sarathi's P99 TBT]
  - whale count & prompt-size stats, completion count
The hypothesis: mono's giant whale-prefill iterations spike POOLED P99 TBT even though per-request
mean TPOT looks fine; chunk (2048/512) bounds the iteration => lower P99 TBT (at some TTFT cost)."""
import json, glob, os, statistics as st

BUDGETS = os.environ.get("BUDGETS", "16384 2048 512").split()
MONO = BUDGETS[0]
TRIALS = ["1", "2", "3"]


def rows(b):
    R = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-longp-b{b}-t{tr}.jsonl"):
            R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def pooled_tbt(R):
    t = []
    for r in R:
        t.extend(r.get("tbt_ms") or [])
    return t


data = {b: rows(b) for b in BUDGETS}
mono = data[MONO]
if not mono:
    print("no mono data"); raise SystemExit

pt = [r["prompt_tokens_approx"] for r in mono if r.get("prompt_tokens_approx")]
whales = sum(1 for r in mono if (r.get("pad_chars") or 0) >= 40000)
print("Long-prompt whale test — 14B/TP=2, single-turn, ISOLATED sequential arms")
print(f"prompts (mono, words): n={len(pt)} p50={st.median(pt):.0f} p95={pctl(pt,95):.0f} "
      f"max={max(pt)} | whales(>=40k pad chars)={whales} ({100*whales/len(pt):.0f}%)\n")

m_tt = [r["ttft"] * 1000 for r in mono if r.get("ttft") is not None]
m_ttm = st.mean(m_tt)
m_tbt = pooled_tbt(mono); m_p99 = pctl(m_tbt, 99); m_max = max(m_tbt) if m_tbt else float("nan")

print(f"{'budget':>7} | {'n':>4} {'TTFTmean':>8} {'p95':>6} {'dTTFT%':>7} | "
      f"{'TPOTp50':>7} {'TPOTp95':>7} | {'TBTp50':>6} {'TBTp99':>7} {'TBTmax':>7} | {'dP99':>7} {'dMax':>7}  [arm]")
for b in BUDGETS:
    R = data[b]
    tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
    tp = [r["tpot"] * 1000 for r in R if r.get("tpot") is not None]
    tbt = pooled_tbt(R)
    tag = "mono" if b == MONO else "chunk"
    if not tt:
        print(f"{b:>7} | (no data)  [{tag}]"); continue
    ttm = st.mean(tt); dtt = (ttm - m_ttm) / m_ttm * 100
    p99 = pctl(tbt, 99); mx = max(tbt) if tbt else float("nan")
    dp99 = (p99 - m_p99) / m_p99 * 100; dmx = (mx - m_max) / m_max * 100
    print(f"{b:>7} | {len(tt):>4} {ttm:8.0f} {pctl(tt,95):6.0f} {dtt:+7.1f} | "
          f"{pctl(tp,50):7.1f} {pctl(tp,95):7.1f} | {pctl(tbt,50):6.1f} {p99:7.1f} {mx:7.1f} | "
          f"{dp99:+7.1f} {dmx:+7.1f}  [{tag}]")

# Controller engagement for any dynamic arm (non-numeric budget name).
import csv
for b in BUDGETS:
    if b.isdigit():
        continue
    tf = glob.glob(f"logs/*-longp-b{b}-chunktrace.csv")
    if not (tf and os.path.getsize(tf[0]) > 0):
        continue
    ch = [int(r["chunk"]) for r in csv.DictReader(open(tf[0])) if r.get("chunk", "").strip().isdigit()]
    if not ch:
        continue
    from collections import Counter
    c = Counter(ch); n = len(ch)
    interior = sum(1 for x in ch if min(ch) < x < max(ch))
    trans = sum(1 for x, y in zip(ch, ch[1:]) if x != y)
    print(f"\n[{b}] controller: n_steps={n} start={ch[0]} last={ch[-1]} min={min(ch)} max={max(ch)} "
          f"transitions={trans} dwell[floor:{100*c[min(ch)]/n:.0f}% ceil:{100*c[max(ch)]/n:.0f}% "
          f"interior:{100*interior/n:.0f}%]")

print("\nRead: chunk beats mono on TBTp99/max (Sarathi decode-protection). The DYNAMIC arm wins only")
print("if it matches chunk's low TBT WHILE keeping mono-like TTFT -- i.e. drops budget in time for")
print("whales. If its TBTp99 stays near mono, reactive control was too slow (whale = 1 iteration).")
