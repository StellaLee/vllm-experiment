#!/usr/bin/env python3
"""Analyze the whale/threshold/concurrency sweep (orchestrate_cs2_whale_threshold_sweep.sh).
Budget fixed large (16384); long_prefill_token_threshold is the swept server-side mechanism;
whale fraction and concurrency are the client-side axes controlling Cs^2-exposure and
utilization respectively. Reports BOTH TTFT (admission-queue wait -- the genuine-PS mechanism)
and pooled TBT (decode-protection -- the mechanism that's already won cleanly elsewhere), since
these are different things and a threshold win on one need not show on the other."""
import json, glob, os, statistics as st

THRESHOLDS = os.environ.get("THRESHOLDS", "0 512").split()
MONO = THRESHOLDS[0]
WHALE_FRACS = os.environ.get("WHALE_FRACS", "0.05 0.15").split()
CONCS = os.environ.get("CONCS", "20 40").split()


def wtag(wf):
    return f"{round(float(wf) * 100):02d}"


def rows(t, wf, conc):
    R = []
    for f in glob.glob(f"logs/*-cs2wt-t{t}-w{wtag(wf)}-c{conc}-t1.jsonl"):
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
    return st.variance(vals) / (mean ** 2) if mean else float("nan")


print("Whale/threshold/concurrency sweep -- 14B/TP=2, single-turn, budget FIXED 16384, "
      "long_prefill_token_threshold swept")
print(f"thresholds={THRESHOLDS} (mono={MONO})  whale_fracs={WHALE_FRACS}  concs={CONCS}\n")

for wf in WHALE_FRACS:
    for conc in CONCS:
        mono_R = rows(MONO, wf, conc)
        if not mono_R:
            print(f"wf={wf} conc={conc}: (no mono data)")
            continue
        pt = [r["prompt_tokens_approx"] for r in mono_R if r.get("prompt_tokens_approx")]
        nwhale = sum(1 for r in mono_R if (r.get("pad_chars") or 0) >= 40000)
        realized_cs2 = cs2(pt)

        m_tt = [r["ttft"] * 1000 for r in mono_R if r.get("ttft") is not None]
        m_ttm = st.mean(m_tt) if m_tt else float("nan")
        m_tbt = pooled_tbt(mono_R)
        m_p99 = pctl(m_tbt, 99); m_max = max(m_tbt) if m_tbt else float("nan")

        print(f"--- wf={wf} conc={conc}: n={len(pt)} nwhale={nwhale} realizedCs2={realized_cs2:.3f} "
              f"| mono[thr={MONO}] TTFTmean={m_ttm:.0f}ms TBTp99={m_p99:.1f}ms TBTmax={m_max:.1f}ms")
        for t in THRESHOLDS:
            if t == MONO:
                continue
            R = rows(t, wf, conc)
            if not R:
                print(f"    [thr={t}] (no data)")
                continue
            tt = [r["ttft"] * 1000 for r in R if r.get("ttft") is not None]
            tbt = pooled_tbt(R)
            ttm = st.mean(tt) if tt else float("nan")
            p99 = pctl(tbt, 99); mx = max(tbt) if tbt else float("nan")
            dtt = (ttm - m_ttm) / m_ttm * 100 if m_ttm else float("nan")
            dp99 = (p99 - m_p99) / m_p99 * 100 if m_p99 else float("nan")
            dmx = (mx - m_max) / m_max * 100 if m_max else float("nan")
            print(f"    [thr={t}] n={len(tt)} TTFTmean={ttm:.0f}ms(dTTFT{dtt:+.1f}%) "
                  f"TBTp99={p99:.1f}ms(d{dp99:+.1f}%) TBTmax={mx:.1f}ms(d{dmx:+.1f}%)")
        print()

print("Read: threshold=512 should show TBT protection (matches the already-established win) at")
print("every (wf,conc). The NEW question is TTFT: does raising concurrency (more simultaneous")
print("admission competition) make dTTFT move negative (threshold helps admission-side too),")
print("unlike the low-utilization open-loop sweep where it was uniformly worse?")
