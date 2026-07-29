#!/usr/bin/env python3
"""Per-trial breakdown of the §5.6 (longprompt-tbt-win) replication, matching how §5.8's
replication was reported. Trial 1 = original 2026-07-21 run; trials 2/3 = 2026-07-24 reruns
(orchestrate_longprompt.sh TRIALS="2 3", seeds 1002/1003)."""
import json, glob, statistics as st

BUDGETS = ["16384", "2048", "512"]
MONO = "16384"

DATE_BY_TRIAL = {1: "2026-07-24", 2: "2026-07-24", 3: "2026-07-24"}


def rows(budget, trial):
    date = DATE_BY_TRIAL[trial]
    pattern = f"logs/{date}-longp-b{budget}-t{trial}.jsonl"
    files = glob.glob(pattern)
    R = []
    for f in files:
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


print("§5.6 replication: pooled P99/max TBT + TTFT, per trial, mono(16384) vs chunk(2048/512)\n")

for trial in (1, 2, 3):
    mono = rows(MONO, trial)
    if not mono:
        print(f"trial {trial}: NO MONO DATA")
        continue
    m_tt = [r["ttft"] * 1000 for r in mono if r.get("ttft") is not None]
    m_ttm = st.mean(m_tt)
    m_tbt = pooled_tbt(mono)
    m_p99 = pctl(m_tbt, 99)
    m_max = max(m_tbt) if m_tbt else float("nan")
    print(f"--- trial {trial} ---")
    print(f"  {'budget':>7} | {'n':>4} {'TTFTmean':>8} {'dTTFT%':>7} | {'TBTp99':>7} {'TBTmax':>7} | {'dP99%':>7} {'dMax%':>7}")
    for b in BUDGETS:
        R = rows(b, trial)
        if not R:
            print(f"  {b:>7} | NO DATA")
            continue
        tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
        ttm = st.mean(tt)
        dtt = (ttm - m_ttm) / m_ttm * 100
        tbt = pooled_tbt(R)
        p99 = pctl(tbt, 99)
        mx = max(tbt) if tbt else float("nan")
        dp99 = (p99 - m_p99) / m_p99 * 100
        dmx = (mx - m_max) / m_max * 100
        tag = "mono" if b == MONO else "chunk"
        print(f"  {b:>7} | {len(tt):>4} {ttm:8.0f} {dtt:+7.1f} | {p99:7.1f} {mx:7.1f} | {dp99:+7.1f} {dmx:+7.1f}  [{tag}]")
    print()

print("=== Summary: dTBT-p99% / dTBT-max% / dTTFT% vs mono, per trial ===\n")
for b in ("2048", "512"):
    print(f"budget={b}:")
    for trial in (1, 2, 3):
        mono = rows(MONO, trial)
        R = rows(b, trial)
        if not mono or not R:
            continue
        m_tt = [r["ttft"] * 1000 for r in mono if r.get("ttft") is not None]
        m_ttm = st.mean(m_tt)
        m_tbt = pooled_tbt(mono)
        m_p99 = pctl(m_tbt, 99)
        m_max = max(m_tbt) if m_tbt else float("nan")
        tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
        ttm = st.mean(tt)
        tbt = pooled_tbt(R)
        p99 = pctl(tbt, 99)
        mx = max(tbt) if tbt else float("nan")
        dtt = (ttm - m_ttm) / m_ttm * 100
        dp99 = (p99 - m_p99) / m_p99 * 100
        dmx = (mx - m_max) / m_max * 100
        print(f"  trial={trial}: dTTFT={dtt:+.1f}%  dTBTp99={dp99:+.1f}%  dTBTmax={dmx:+.1f}%")
