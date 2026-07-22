#!/usr/bin/env python3
"""Analyze the chunk-size (prefill token-budget) sweep. First budget in $BUDGETS is the mono baseline
(no chunking); the rest are chunk sizes. Reports TTFT (mean/p50/p95) and TPOT (p50/p95) with deltas
vs mono, the realized service Cs^2, and a saturation hint (completion count + late/early TTFT ratio).
Then picks the 'best' chunk budget: lowest TTFT-mean among budgets whose TPOT-p95 regression vs mono
is <= TPOT_TOL% (default 5). This encodes the goal: win TTFT without paying TPOT."""
import json, glob, os, statistics as st

BUDGETS = os.environ.get("BUDGETS", "16384 512 1024 2048").split()
MONO = BUDGETS[0]
TPOT_TOL = float(os.environ.get("TPOT_TOL", "5"))
TRIALS = ["1", "2", "3"]


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def recs(budget):
    R = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-sweep-b{budget}-t{tr}.jsonl"):
            R += rows(f)
    return R


def pct(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def field(R, k, scale=1000.0):
    return [r[k] * scale for r in R if r.get(k) is not None]


def cs2(x):
    m = st.mean(x); return st.pvariance(x) / (m * m) if m else float("nan")


data = {b: recs(b) for b in BUDGETS}
mono = data[MONO]
if not mono:
    print("no mono data"); raise SystemExit

il = [r["prompt_tokens_approx"] for r in mono if r.get("prompt_tokens_approx")]
print("Chunk-size sweep — 14B/TP=2, single-turn, heavy-tail prompts, near-saturation")
print(f"realized service Cs^2 (prompt words, mono) = {cs2(il):.3f}  n={len(il)} "
      f"mean={st.mean(il):.0f} p50={st.median(il):.0f} p95={pct(il,95):.0f} max={max(il)}\n")

m_tt = field(mono, "ttft"); m_tp = field(mono, "tpot")
m_ttm = st.mean(m_tt); m_tp50 = st.median(m_tp); m_tp95 = pct(m_tp, 95)

print(f"{'budget':>7} | {'n':>4} {'TTFTmean':>8} {'p50':>6} {'p95':>6} | {'dTTFTm%':>8} | "
      f"{'TPOTp50':>7} {'TPOTp95':>7} | {'dTP50%':>7} {'dTP95%':>7} | sat(late/early)")
best = None
for b in BUDGETS:
    R = data[b]
    tt = field(R, "ttft"); tp = field(R, "tpot")
    tag = "mono" if b == MONO else "chunk"
    if not tt:
        print(f"{b:>7} | (no data)  [{tag}]"); continue
    ttm = st.mean(tt); dm = (ttm - m_ttm) / m_ttm * 100
    tp50 = st.median(tp); tp95 = pct(tp, 95)
    d50 = (tp50 - m_tp50) / m_tp50 * 100; d95 = (tp95 - m_tp95) / m_tp95 * 100
    # saturation hint: TTFT of last-third vs first-third arrivals (needs ordered records)
    order = [r for r in R if r.get("ttft") is not None]
    n = len(order); k = max(1, n // 3)
    early = st.mean([r["ttft"] for r in order[:k]]) or 1e-9
    late = st.mean([r["ttft"] for r in order[-k:]])
    sat = late / early if early else float("nan")
    print(f"{b:>7} | {len(tt):>4} {ttm:8.0f} {pct(tt,50):6.0f} {pct(tt,95):6.0f} | {dm:+8.1f} | "
          f"{tp50:7.1f} {tp95:7.1f} | {d50:+7.1f} {d95:+7.1f} | {sat:5.2f}  [{tag}]")
    if b != MONO and d95 <= TPOT_TOL:
        if best is None or ttm < best[1]:
            best = (b, ttm, dm, d95)

print()
if best:
    print(f">> best chunk budget = {best[0]}  (TTFT {best[2]:+.1f}% vs mono, TPOT-p95 {best[3]:+.1f}%, "
          f"within {TPOT_TOL:.0f}% TPOT tolerance)")
else:
    print(f">> no chunk budget beat mono on TTFT within {TPOT_TOL:.0f}% TPOT tolerance "
          f"(every chunk size regressed TPOT-p95 beyond tolerance)")
