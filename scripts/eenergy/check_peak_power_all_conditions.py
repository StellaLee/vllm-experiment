"""Real per-trial peak-power comparison, coincidence_ceiling vs lmetric (power-blind) and vs
round_robin, across all 7 conditions -- with actual t-tests, not just point-estimate
comparison, per the standard this project adopted after the Table-1 significance catch
(findings.md Part 17)."""
import sys
sys.path.insert(0, "scripts/eenergy")
import compare_closedloopheavy_duration as c
from scipy import stats
import statistics as st

CONDITIONS = {
    "Heavy/CL short": ("closedloopheavypergpu", range(1, 7)),
    "Heavy/CL long": ("closedloopheavylongpergpu", range(1, 7)),
    "Heavy/Matched": ("openloopwhalelongoutmatchedpergpu", range(1, 4)),
    "Light/Cachehit": ("cachehitpergpu", range(1, 4)),
    "BurstGPT": ("burstgptpergpu", range(1, 4)),
    "Ramp & Route": ("rampandroutepergpu", range(1, 4)),
    "WildChat": ("wildchatnaturalpergpu", range(1, 4)),
}


def peaks_only(prefix, arm, trials):
    out = []
    for t in trials:
        pow_path = f"{c.LOGDIR}/{prefix}_power_trace_{arm}_t{t}.csv"
        power_rows = c.load_power(pow_path)
        peak, _, _, _ = c.ramp_stats(power_rows)
        out.append(peak)
    return out


print(f"{'Condition':<18}{'n':>3}{'cc peak':>14}{'lmetric peak':>16}{'diff%':>9}{'p (cc vs lm)':>14}{'rr peak':>12}{'p (cc vs rr)':>14}")
for label, (prefix, trials) in CONDITIONS.items():
    trials = list(trials)
    cc = peaks_only(prefix, "coincidence_ceiling", trials)
    lm = peaks_only(prefix, "lmetric", trials)
    rr = peaks_only(prefix, "round_robin", trials)
    m_cc, m_lm, m_rr = st.mean(cc), st.mean(lm), st.mean(rr)
    _, p_lm = stats.ttest_ind(cc, lm)
    _, p_rr = stats.ttest_ind(cc, rr)
    diff_pct = (m_cc - m_lm) / m_lm * 100
    sig_lm = "*" if p_lm < 0.05 else " "
    sig_rr = "*" if p_rr < 0.05 else " "
    print(f"{label:<18}{len(trials):>3}{m_cc:>14.1f}{m_lm:>16.1f}{diff_pct:>8.1f}%{p_lm:>10.4f}{sig_lm}{m_rr:>12.1f}{p_rr:>10.4f}{sig_rr}")
print("\n* = p<0.05 (unpaired t-test). Negative diff% = coincidence_ceiling has lower peak power.")
