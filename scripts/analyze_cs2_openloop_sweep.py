#!/usr/bin/env python3
"""Analyze the open-loop Cs^2 sweep (orchestrate_cs2_openloop_sweep.sh). Single continuous
lognormal prompt-length distribution (no whale mode), open-loop Poisson arrivals, large fixed
mean so E[S_prefill] stays roughly constant while pad-cv2 sweeps realized Cs^2. Metric: TTFT
(admission-queue wait) -- this is what Eq. genuine actually predicts, unlike TBT."""
import json, glob, os, statistics as st

BUDGETS = os.environ.get("BUDGETS", "16384 2048").split()
MONO = BUDGETS[0]
CV2_LIST = os.environ.get("CV2_LIST", "0.1 0.5 1 2 4 8").split()
RATE = os.environ.get("RATE", "0.15")
NCONV = os.environ.get("NCONV", "90")


def ctag(cv2):
    return f"{round(float(cv2) * 100):04d}"


def rows(b, cv2):
    R = []
    for f in glob.glob(f"logs/*-cs2ol-b{b}-c{ctag(cv2)}-r{RATE}-n{NCONV}-t1.jsonl"):
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


# prompt_tokens_approx is len(prompt.split()) -- a WORD count, not a real tokenizer token
# count. Project convention (see 2026-07-21-longprompt-tbt-win.md) is ~1.3 tok/word; Cs^2
# itself is scale-invariant (ratio of variance to mean-squared) so the word/token distinction
# doesn't affect realized Cs^2, only the E[S] column's absolute units.
WORDS_TO_TOKENS = 1.3


print("Open-loop Cs^2 sweep -- 14B/TP=2, single-turn, continuous lognormal prompt length "
      "(no whale mode)")
print(f"budgets={BUDGETS} (mono={MONO})  cv2={CV2_LIST}  rate={RATE} conv/s  nconv={NCONV}\n")

header = (f"{'cv2':>6} | {'realizedCs2':>11} {'E[S]tok':>8} {'n':>4} | "
          + " | ".join(f"{'dTTFTmean%':>10} {'dTTFTp95%':>9} {'dTTFTp99%':>9} [{b}]"
                       for b in BUDGETS if b != MONO))
print(header)

for cv2 in CV2_LIST:
    mono_R = rows(MONO, cv2)
    if not mono_R:
        print(f"{cv2:>6} | (no mono data)")
        continue
    pt = [r["prompt_tokens_approx"] for r in mono_R if r.get("prompt_tokens_approx")]
    realized_cs2 = cs2(pt)  # scale-invariant -- words vs tokens doesn't matter here
    e_s = st.mean(pt) * WORDS_TO_TOKENS if pt else float("nan")

    m_tt = [r["ttft"] * 1000 for r in mono_R if r.get("ttft") is not None]
    m_ttm = st.mean(m_tt) if m_tt else float("nan")
    m_p95 = pctl(m_tt, 95); m_p99 = pctl(m_tt, 99)

    cells = []
    for b in BUDGETS:
        if b == MONO:
            continue
        R = rows(b, cv2)
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

print("\nRead: Eq. genuine predicts dTTFT (chunk vs mono) should move from ~0/positive (chunk")
print("worse, pure overhead) toward increasingly negative (chunk better) as realized Cs^2 crosses")
print("above 1, AT ROUGHLY CONSTANT E[S] (check the E[S]tok column stays flat across rows -- if")
print("it drifts a lot, cv2 and E[S_prefill] aren't cleanly decoupled and the read is confounded).")
