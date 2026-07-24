#!/usr/bin/env python3
"""Analyze the Cs^2 fraction-sweep (orchestrate_cs2_frac_sweep.sh). For each whale fraction,
reports the MONO arm's realized Cs^2 of prompt-length (prefill-service-time proxy), then each
budget's TTFT/TBT-p99/TBT-max delta vs mono AT THAT SAME FRACTION. The crossover question:
does chunk's TBT-p99 advantage over mono grow monotonically with realized Cs^2, starting near
0 at frac=0% (control) and reaching the already-measured ~-82%/-94% by frac=15%?"""
import json, glob, os, statistics as st

BUDGETS = os.environ.get("BUDGETS", "16384 2048 512").split()
MONO = BUDGETS[0]
FRAC_PCTS = os.environ.get("FRAC_PCTS", "0 5 10 15").split()
NCONV = os.environ.get("NCONV", "200")


def rows(b, pct):
    ftag = f"{int(pct):02d}"
    R = []
    for f in glob.glob(f"logs/*-cs2f-b{b}-f{ftag}-n{NCONV}-t1.jsonl"):
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


def cs2(vals):
    if len(vals) < 2:
        return float("nan")
    mean = st.mean(vals)
    if mean == 0:
        return float("nan")
    return st.variance(vals) / (mean ** 2)


print("Cs^2 fraction sweep -- 14B/TP=2, single-turn, whale size fixed 44-50k chars (~12k tok)")
print(f"budgets={BUDGETS} (mono={MONO})  fracs%={FRAC_PCTS}\n")

header = (f"{'frac%':>6} | {'realizedCs2':>11} {'nwhale':>6} | "
          + " | ".join(f"{'dTTFT%':>8} {'dTBTp99%':>9} {'dTBTmax%':>9} [{b}]" for b in BUDGETS if b != MONO))
print(header)

for pct in FRAC_PCTS:
    mono_R = rows(MONO, pct)
    if not mono_R:
        print(f"{pct:>6} | (no mono data)")
        continue
    pt = [r["prompt_tokens_approx"] for r in mono_R if r.get("prompt_tokens_approx")]
    nwhale = sum(1 for r in mono_R if (r.get("pad_chars") or 0) >= 40000)
    realized_cs2 = cs2(pt)

    m_tt = [r["ttft"] * 1000 for r in mono_R if r.get("ttft") is not None]
    m_ttm = st.mean(m_tt) if m_tt else float("nan")
    m_tbt = pooled_tbt(mono_R)
    m_p99 = pctl(m_tbt, 99); m_max = max(m_tbt) if m_tbt else float("nan")

    cells = []
    for b in BUDGETS:
        if b == MONO:
            continue
        R = rows(b, pct)
        if not R:
            cells.append(f"{'(no data)':>30}")
            continue
        tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
        tbt = pooled_tbt(R)
        ttm = st.mean(tt) if tt else float("nan")
        p99 = pctl(tbt, 99); mx = max(tbt) if tbt else float("nan")
        dtt = (ttm - m_ttm) / m_ttm * 100 if m_ttm else float("nan")
        dp99 = (p99 - m_p99) / m_p99 * 100 if m_p99 else float("nan")
        dmx = (mx - m_max) / m_max * 100 if m_max else float("nan")
        cells.append(f"{dtt:8.1f} {dp99:9.1f} {dmx:9.1f} ")
    print(f"{pct:>6} | {realized_cs2:11.3f} {nwhale:6d} | " + " | ".join(cells))

print("\nRead: chunk (2048/512) columns show delta vs mono AT THE SAME FRACTION. If chunk's TBT-p99")
print("delta stays ~0 at frac=0% (control, no whales) and grows toward the already-measured -82%/-94%")
print("by frac=15%, that traces the crossover -- the effect turning on as realized Cs^2 rises, gated")
print("by whale presence (Eq. genuine's Cs^2 term), NOT a step function or noise.")
