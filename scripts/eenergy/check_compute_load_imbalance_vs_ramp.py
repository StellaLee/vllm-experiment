"""Does compute/load-BALANCED routing structurally constrain fleet-aggregate ramp?

Follow-up to check_compute_leads_ramp_lag_correlation.py, which showed incoming compute
predicts a single replica's OWN ramp a few seconds later (a signal-level, single-policy
result). This checks the structural/policy-level claim directly: across the whole set of
already-collected Heavy/Closed-Loop-long policies (12 policies, 72 trials total -- everything
from round_robin, which is totally compute/load-blind, through compute_only/load_only/
drf_no_power, which balance compute+load but never read power, to the power-aware family),
does how EVENLY a policy spreads incoming compute/load across replicas IN TIME predict how
severe the resulting fleet-aggregate ramp is?

Method, per (policy, trial):
  1. Bin the assignment log's new_tokens (compute) and dispatch counts (load proxy) into 5s
     windows across the trial's wall-clock span, per GPU.
  2. For each window with nonzero total dispatch, compute the cross-GPU coefficient of
     variation (std/mean) of new_tokens-per-window and of dispatch-count-per-window --
     "how unevenly was this window's incoming pressure spread across the 6 replicas."
  3. Average across windows -> one compute_imbalance and one load_imbalance scalar per trial.
  4. Compute fleet-aggregate ramp stats (mean_ramp, p99_ramp, max_ramp) from the power trace
     using the EXACT same method as compare_closedloopheavy_duration.ramp_stats (reused, not
     reimplemented) -- so these numbers are directly comparable to every existing table in the
     findings doc.
  5. Pool all (policy, trial) pairs and report the Pearson correlation between each imbalance
     metric and each ramp metric. A positive, significant correlation (more imbalance -> more
     ramp) is direct empirical support for "balancing compute/load structurally constrains
     ramp" as a general property of the policy family, not just a same-signal lag artifact.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import compare_closedloopheavy_duration as c  # noqa: E402

LOG_DIR = "/root/pli/vllm-experiment/logs"
PREFIX = "closedloopheavylongpergpu"
WINDOW_S = 5.0

POLICIES = {
    "round_robin": range(1, 10),
    "drf_fixed": range(1, 7),
    "drf_power_tiebreak_full": range(1, 7),
    "lmetric": range(1, 10),
    "lmetric_power": range(1, 7),
    "weighted_sum": range(1, 7),
    "weighted_sum_no_power": range(1, 4),
    "drf_no_power": range(1, 4),
    "compute_only": range(1, 7),
    "load_only": range(1, 7),
    "coincidence_ceiling": range(1, 10),
    "drf_peak_power_tiebreak_full": range(1, 4),
}


def imbalance_metrics(assign_path):
    rows = []
    with open(assign_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["wall_time"]), row["gpu_index"], int(row["new_tokens"])))
    if not rows:
        return None, None
    t0 = min(t for t, _, _ in rows)
    t1 = max(t for t, _, _ in rows)
    n_windows = max(1, int((t1 - t0) / WINDOW_S) + 1)
    gpus = sorted(set(g for _, g, _ in rows))

    tok_bins = defaultdict(lambda: defaultdict(int))   # window -> gpu -> tokens
    count_bins = defaultdict(lambda: defaultdict(int))  # window -> gpu -> dispatch count
    for t, g, tok in rows:
        w = int((t - t0) / WINDOW_S)
        tok_bins[w][g] += tok
        count_bins[w][g] += 1

    def cv_series(bins):
        cvs = []
        for w in range(n_windows):
            vals = np.array([bins[w].get(g, 0) for g in gpus], dtype=float)
            total = vals.sum()
            if total <= 0:
                continue
            mean = vals.mean()
            std = vals.std()
            if mean > 0:
                cvs.append(std / mean)
        return float(np.mean(cvs)) if cvs else None

    return cv_series(tok_bins), cv_series(count_bins)


def main():
    rows = []
    for policy, trials in POLICIES.items():
        for trial in trials:
            assign_path = f"{LOG_DIR}/{PREFIX}_assignment_{policy}_t{trial}.csv"
            pow_path = f"{LOG_DIR}/{PREFIX}_power_trace_{policy}_t{trial}.csv"
            try:
                compute_imb, load_imb = imbalance_metrics(assign_path)
                power_rows = c.load_power(pow_path)
            except FileNotFoundError:
                continue
            if compute_imb is None:
                continue
            peak, mean_ramp, p99_ramp, max_ramp = c.ramp_stats(power_rows)
            rows.append(dict(policy=policy, trial=trial, compute_imb=compute_imb,
                              load_imb=load_imb, mean_ramp=mean_ramp, p99_ramp=p99_ramp,
                              max_ramp=max_ramp))

    print(f"pooled (policy, trial) rows: {len(rows)}\n")

    print(f"{'policy':<32}{'n':>3}{'compute_imb':>13}{'load_imb':>10}"
          f"{'mean_ramp':>11}{'p99_ramp':>10}{'max_ramp':>10}")
    for policy in POLICIES:
        prows = [r for r in rows if r["policy"] == policy]
        if not prows:
            print(f"{policy:<32}  0  MISSING")
            continue
        n = len(prows)
        ci = np.mean([r["compute_imb"] for r in prows])
        li = np.mean([r["load_imb"] for r in prows])
        mr = np.mean([r["mean_ramp"] for r in prows])
        p99 = np.mean([r["p99_ramp"] for r in prows])
        mx = np.mean([r["max_ramp"] for r in prows])
        print(f"{policy:<32}{n:>3}{ci:>13.4f}{li:>10.4f}{mr:>11.1f}{p99:>10.1f}{mx:>10.1f}")

    print("\nPooled Pearson correlation across ALL (policy, trial) rows:")
    compute_imb = np.array([r["compute_imb"] for r in rows])
    load_imb = np.array([r["load_imb"] for r in rows])
    for metric in ("mean_ramp", "p99_ramp", "max_ramp"):
        y = np.array([r[metric] for r in rows])
        r_c, p_c = stats.pearsonr(compute_imb, y)
        r_l, p_l = stats.pearsonr(load_imb, y)
        print(f"  compute_imb vs {metric:<10}: r={r_c:+.3f} (p={p_c:.4f})   "
              f"load_imb vs {metric:<10}: r={r_l:+.3f} (p={p_l:.4f})")


if __name__ == "__main__":
    sys.exit(main())
