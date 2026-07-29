#!/usr/bin/env python3
"""Analyze the closed-loop multi-turn ShareGPT 3-arm run. Overall + PER-TURN TTFT/TPOT (mean +
p50/p95) with deltas vs mono. Per-turn matters here: history accumulates each turn, so later-turn
prompts are longer -> that is where chunking's TPOT-tail protection (if any) should show. Also
reports the realized service Cs^2 (prompt_tokens_approx) and the slocvar budget engagement, plus a
saturation hint (late/early TTFT ratio). TTFT/TPOT in ms."""
import json, glob, statistics as st

ARMS = ["mono", "chunk", "ours", "feed"]
TRIALS = ["1", "2", "3"]


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def recs(arm):
    R = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*-sharecl-{arm}-t{tr}.jsonl"):
            R += rows(f)
    return R


def pct(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def field(R, k, scale=1000.0):
    return [r[k] * scale for r in R if r.get(k) is not None]


def cs2(x):
    m = st.mean(x); return st.pvariance(x) / (m * m) if m else float("nan")


data = {a: recs(a) for a in ARMS}
mono = data["mono"]
if not mono:
    print("no mono data"); raise SystemExit

il = [r["prompt_tokens_approx"] for r in mono if r.get("prompt_tokens_approx")]
print("Closed-loop multi-turn ShareGPT 3-arm — 14B/TP=2, mt=1024, prefix-caching ON, real prompts")
print(f"realized service Cs^2 (prompt words, mono) = {cs2(il):.3f}  n={len(il)} "
      f"mean={st.mean(il):.0f} p50={st.median(il):.0f} max={max(il)}\n")


def block(title, subset):
    mr = subset("mono")
    if not mr:
        return
    m_tt = field(mr, "ttft"); m_tp = field(mr, "tpot")
    if not m_tt:
        return
    m_ttm = st.mean(m_tt); m_tp50 = st.median(m_tp) if m_tp else float("nan"); m_tp95 = pct(m_tp, 95)
    print(f"--- {title} ---")
    print(f"{'arm':>6} | {'n':>4} {'TTFTmean':>8} {'p50':>6} {'p95':>6} | {'dTTFTm%':>8} | "
          f"{'TPOTp50':>7} {'TPOTp95':>7} | {'dTP50%':>7} {'dTP95%':>7}")
    for a in ARMS:
        R = subset(a)
        tt = field(R, "ttft"); tp = field(R, "tpot")
        if not tt:
            print(f"{a:>6} | (no data)"); continue
        ttm = st.mean(tt)
        dm = (ttm - m_ttm) / m_ttm * 100 if m_ttm else float("nan")
        tp50 = st.median(tp) if tp else float("nan"); tp95 = pct(tp, 95)
        d50 = (tp50 - m_tp50) / m_tp50 * 100 if m_tp50 else float("nan")
        d95 = (tp95 - m_tp95) / m_tp95 * 100 if m_tp95 else float("nan")
        print(f"{a:>6} | {len(tt):>4} {ttm:8.0f} {pct(tt,50):6.0f} {pct(tt,95):6.0f} | {dm:+8.1f} | "
              f"{tp50:7.1f} {tp95:7.1f} | {d50:+7.1f} {d95:+7.1f}")
    print()


block("OVERALL (all turns)", lambda a: data[a])
turns = sorted({r.get("turn") for r in mono if r.get("turn")})
for t in turns:
    block(f"turn {t}", lambda a, t=t: [r for r in data[a] if r.get("turn") == t])

# controller engagement
import os, csv
for a in ["ours", "feed"]:
    tf = glob.glob(f"logs/*-sharecl-{a}-chunktrace.csv")
    if tf and os.path.getsize(tf[0]) > 0:
        ch = [int(r["chunk"]) for r in csv.DictReader(open(tf[0])) if r.get("chunk", "").isdigit()]
        if ch:
            from collections import Counter
            c = Counter(ch); n = len(ch)
            print(f"ours budget: start={ch[0]} last={ch[-1]} min={min(ch)} max={max(ch)} "
                  f"transitions={sum(1 for x,y in zip(ch,ch[1:]) if x!=y)} "
                  f"dwell[512:{100*c[512]/n:.0f}% 16384:{100*c[16384]/n:.0f}%] n_steps={n}")
