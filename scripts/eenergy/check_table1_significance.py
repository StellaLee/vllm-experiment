"""Proper significance testing for the paper's headline Table 1 (Heavy/Closed-Loop short,
n=6): coincidence_ceiling vs lmetric (power-blind) and vs lmetric_power_coincidence_ceiling
(strongest unsafe alternative). Pulls real per-trial peak/TTFT/TBT (not just the reported
mean+-std) and runs paired+unpaired t-tests -- written after a simulated peer-review pass
caught that the paper's "dominates on every metric" claim was never actually significance-
tested, findings.md Part 17."""
import sys
sys.path.insert(0, "scripts/eenergy")
import compare_closedloopheavy_duration as c
from scipy import stats
import statistics as st

PREFIX = "closedloopheavypergpu"


def per_trial(arm, trials):
    peaks, ttfts, tbts = [], [], []
    for t in trials:
        rec_path = f"{c.LOGDIR}/{PREFIX}_records_{arm}_t{t}.jsonl"
        pow_path = f"{c.LOGDIR}/{PREFIX}_power_trace_{arm}_t{t}.csv"
        ttft_list, tbt_list = c.load_records(rec_path)
        power_rows = c.load_power(pow_path)
        peak, mean_ramp, p99_ramp, max_ramp = c.ramp_stats(power_rows)
        peaks.append(peak)
        ttfts.append(sum(ttft_list) / len(ttft_list))
        tbts.append(sum(tbt_list) / len(tbt_list))
    return peaks, ttfts, tbts


def report(label, a, b):
    print(f"\n{label}")
    for name, idx in [("peak", 0), ("ttft", 1), ("tbt", 2)]:
        x, y = a[idx], b[idx]
        t_ind, p_ind = stats.ttest_ind(x, y)
        t_pair, p_pair = stats.ttest_rel(x, y) if len(x) == len(y) else (None, None)
        mx, sx = st.mean(x), st.stdev(x)
        my, sy = st.mean(y), st.stdev(y)
        pair_str = f"  paired t={t_pair:.3f} p={p_pair:.4f}" if t_pair is not None else ""
        print(f"  {name}: a={mx:.4f}+-{sx:.4f}  b={my:.4f}+-{sy:.4f}  diff={mx-my:.4f}  "
              f"unpaired t={t_ind:.3f} p={p_ind:.4f}{pair_str}")


def main():
    cc = per_trial("coincidence_ceiling", range(1, 7))
    lm = per_trial("lmetric", range(1, 7))
    lp = per_trial("lmetric_power_coincidence_ceiling", range(1, 7))
    report("coincidence_ceiling vs lmetric (power-blind), n=6 vs n=6", cc, lm)
    report("coincidence_ceiling vs lmetric_power_coincidence_ceiling (unsafe), n=6 vs n=6", cc, lp)


if __name__ == "__main__":
    main()
