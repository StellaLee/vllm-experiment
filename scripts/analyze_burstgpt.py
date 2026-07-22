#!/usr/bin/env python3
"""Analyze a BurstGPT one-trial 3-arm run (burstgpt-bench detail logs). Per arm: TTFT
(first_chunk_time) mean/p50/p95, TPOT ((total-first)/(tokens-1)) p50/p95, deltas vs mono, and
the realized service-size Cs^2 from in_len. TTFT/TPOT in ms. Uses out_len_expected as the token
count (exact because we run with --ignore_eos, so the model emits exactly that many tokens)."""
import json, glob, statistics as st

ARMS = ["mono", "chunk", "ours"]


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip() and l.lstrip().startswith("{")]


def load(arm):
    fs = glob.glob(f"logs/burstgpt_one-{arm}-detail.jsonl")
    if not fs:
        return None
    R = [r for r in rows(fs[0]) if "first_chunk_time" in r and "total_chunk_time" in r]
    ttft, tpot, inlen = [], [], []
    for r in R:
        ft = r.get("first_chunk_time"); tot = r.get("total_chunk_time")
        ol = r.get("out_len_expected", 0); il = r.get("in_len")
        if ft:
            ttft.append(ft * 1000.0)
        if ft is not None and tot is not None and ol and ol > 1:
            tpot.append((tot - ft) / (ol - 1) * 1000.0)
        if il:
            inlen.append(il)
    return ttft, tpot, inlen


def pct(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def cs2(x):
    m = st.mean(x); return st.pvariance(x) / (m * m) if m else float("nan")


data = {a: load(a) for a in ARMS}
mono = data.get("mono")
if not mono or not mono[0]:
    print("no mono data"); raise SystemExit

il = mono[2]
print("BurstGPT one-trial 3-arm (real trace window) — 14B/TP=2, native api_server /generate, ignore_eos")
print(f"realized service Cs^2 (in_len, mono arm) = {cs2(il):.3f}   n={len(il)} "
      f"mean={st.mean(il):.0f} p50={st.median(il):.0f} max={max(il)}\n")
m_ttft = st.mean(mono[0])
m_tp50 = st.median(mono[1]) if mono[1] else float("nan")
m_tp95 = pct(mono[1], 95)
print(f"{'arm':>6} | {'n':>4} {'TTFTmean':>8} {'p50':>6} {'p95':>6} | {'dTTFTm%':>8} | "
      f"{'TPOTp50':>7} {'TPOTp95':>7} | {'dTP50%':>7} {'dTP95%':>7}")
for a in ARMS:
    d = data.get(a)
    if not d or not d[0]:
        print(f"{a:>6} | (no data)"); continue
    tt, tp, _ = d
    tmean = st.mean(tt)
    dm = (tmean - m_ttft) / m_ttft * 100 if m_ttft else float("nan")
    tp50 = st.median(tp) if tp else float("nan")
    tp95 = pct(tp, 95)
    dtp50 = (tp50 - m_tp50) / m_tp50 * 100 if m_tp50 else float("nan")
    dtp95 = (tp95 - m_tp95) / m_tp95 * 100 if m_tp95 else float("nan")
    print(f"{a:>6} | {len(tt):>4} {tmean:8.0f} {pct(tt,50):6.0f} {pct(tt,95):6.0f} | {dm:+8.1f} | "
          f"{tp50:7.1f} {tp95:7.1f} | {dtp50:+7.1f} {dtp95:+7.1f}")

# controller budget trajectory if present
import os
tf = glob.glob("logs/burstgpt_one-ours-chunktrace.csv")
if tf and os.path.getsize(tf[0]) > 0:
    import csv
    ch = [int(r["chunk"]) for r in csv.DictReader(open(tf[0])) if r.get("chunk", "").isdigit()]
    if ch:
        from collections import Counter
        c = Counter(ch); n = len(ch)
        print(f"\nours budget: start={ch[0]} last={ch[-1]} min={min(ch)} max={max(ch)} "
              f"transitions={sum(1 for x,y in zip(ch,ch[1:]) if x!=y)} "
              f"dwell[512:{100*c[512]/n:.0f}% 16384:{100*c[16384]/n:.0f}%]")
