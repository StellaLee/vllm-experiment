#!/usr/bin/env python3
"""Analyze the Cs^2 threshold sweep (orchestrate_cs2_threshold_sweep.sh). Budget is fixed large
(16384); long_prefill_token_threshold is the swept server-side axis instead -- a genuine
per-request cap independent of the aggregate budget, testing whether it shows a cleaner
Cs^2-correlated TTFT signal than swapping the aggregate budget did (analyze_cs2_openloop_sweep.py
found no clean trend). Metric: TTFT (admission-queue wait), matching Eq. genuine."""
import json, glob, os, statistics as st

THRESHOLDS = os.environ.get("THRESHOLDS", "0 512").split()
MONO = THRESHOLDS[0]  # threshold=0 (off) -- FCFS-equivalent baseline
CV2_LIST = os.environ.get("CV2_LIST", "0.1 1 4 8").split()
RATE = os.environ.get("RATE", "0.05")
NCONV = os.environ.get("NCONV", "30")
WORDS_TO_TOKENS = 1.3  # prompt_tokens_approx is a word count; Cs^2 itself is scale-invariant


def ctag(cv2):
    return f"{round(float(cv2) * 100):04d}"


def rows(t, cv2):
    R = []
    for f in glob.glob(f"logs/*-cs2th-t{t}-c{ctag(cv2)}-r{RATE}-n{NCONV}-t1.jsonl"):
        R += [json.loads(l) for l in open(f) if l.strip()]
    return R


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def cs2(vals):
    if len(vals) < 2:
        return float("nan")
    mean = st.mean(vals)
    return st.variance(vals) / (mean ** 2) if mean else float("nan")


print("Cs^2 threshold sweep -- 14B/TP=2, single-turn, continuous lognormal prompt length "
      "(no whale mode), budget FIXED large, long_prefill_token_threshold swept")
print(f"thresholds={THRESHOLDS} (mono={MONO})  cv2={CV2_LIST}  rate={RATE} conv/s  nconv={NCONV}\n")

header = (f"{'cv2':>6} | {'realizedCs2':>11} {'E[S]tok':>8} {'n':>4} | "
          + " | ".join(f"{'dTTFTmean%':>10} {'dTTFTp95%':>9} {'dTTFTp99%':>9} [thr={t}]"
                       for t in THRESHOLDS if t != MONO))
print(header)

for cv2 in CV2_LIST:
    mono_R = rows(MONO, cv2)
    if not mono_R:
        print(f"{cv2:>6} | (no mono data)")
        continue
    pt = [r["prompt_tokens_approx"] for r in mono_R if r.get("prompt_tokens_approx")]
    realized_cs2 = cs2(pt)
    e_s = st.mean(pt) * WORDS_TO_TOKENS if pt else float("nan")

    m_tt = [r["ttft"] * 1000 for r in mono_R if r.get("ttft") is not None]
    m_ttm = st.mean(m_tt) if m_tt else float("nan")
    m_p95 = pctl(m_tt, 95); m_p99 = pctl(m_tt, 99)

    cells = []
    for t in THRESHOLDS:
        if t == MONO:
            continue
        R = rows(t, cv2)
        if not R:
            cells.append(f"{'(no data)':>34}")
            continue
        tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
        ttm = st.mean(tt) if tt else float("nan")
        p95 = pctl(tt, 95); p99 = pctl(tt, 99)
        dtm = (ttm - m_ttm) / m_ttm * 100 if m_ttm else float("nan")
        d95 = (p95 - m_p95) / m_p95 * 100 if m_p95 else float("nan")
        d99 = (p99 - m_p99) / m_p99 * 100 if m_p99 else float("nan")
        cells.append(f"{dtm:10.1f} {d95:9.1f} {d99:9.1f} ")
    print(f"{cv2:>6} | {realized_cs2:11.3f} {e_s:8.0f} {len(pt):4d} | " + " | ".join(cells))

print("\nRead: if the per-request threshold is the genuine-PS mechanism (vs. budget-swapping,")
print("which still lets one request monopolize a step), dTTFT should show a cleaner move toward")
print("negative (threshold better) as realized Cs^2 rises, compared to the noisy no-trend result")
print("from the budget sweep (cs2ol_r0.05_n30_ANALYSIS.txt).")
