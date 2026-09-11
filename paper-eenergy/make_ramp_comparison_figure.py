"""Generates figs/ramp_comparison.pdf (paper.tex/paper.md's trace-comparison figure in SS6):
fleet-aggregate power and ramp rate for the proposed rule (drf_power_tiebreak_full_
coincidence_ceiling, logged under the short outname "coincidence_ceiling"), the unsafe
lmetric_power_coincidence_ceiling, and weighted_sum_coincidence_ceiling (external validity
check, Pareto-safe but not threshold-safe), 3 trials each, Heavy/Closed-Loop (short)
condition (the condition Table 1 reports) -- under the corrected per-GPU-calibrated ramp
ceiling, not the old uniform 450 W/s constant.

drf_fixed is omitted from the trace plot (already fully reported in Table 1, and not part of
the coincidence-ceiling family) to keep the figure at 3 arms x 3 trials = 9 lines per panel --
already denser than the previous 2-arm version, so a 4th arm was judged to cost more
readability than it adds.

Source data: raw per-GPU power_trace CSVs from the closedloopheavypergpu_ batch
(outnames "coincidence_ceiling", "lmetric_power_coincidence_ceiling",
"weighted_sum_coincidence_ceiling", trials 1-3), pulled from the 8x4090 server
(/root/pli/vllm-experiment/logs/ on 183.147.142.123) into DATA_DIR below. Not committed to
this repo (raw trace CSVs, not source) -- re-pull that batch's power_trace files to rerun.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = os.environ.get(
    "EENERGY_TRACE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
)
GPU_INDICES = [2, 3, 4, 5, 6, 7]
# Per-GPU calibrated ramp ceilings (W/s), from the 2026-09-03 per-GPU calibration check
# (scripts/eenergy/calibrate_ramp_ceiling.py, logs/ramp_ceiling_per_gpu.json) -- replaces
# the old uniform 450 W/s constant. Shown as a band, not a single line, since the ceiling
# now varies per replica.
PER_GPU_CEILING_W_PER_S = {2: 450.2, 3: 509.8, 4: 449.6, 5: 409.2, 6: 512.9, 7: 359.5}
CEILING_LO = min(PER_GPU_CEILING_W_PER_S.values())
CEILING_HI = max(PER_GPU_CEILING_W_PER_S.values())
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "ramp_comparison.pdf")
WINDOW_S = 100.0
TRIALS = [1, 2, 3]

C_PROPOSED = "#4C6E9E"
C_UNSAFE = "#C0562B"
C_EXTCHECK = "#5B8C5A"
C_CEIL = "#8A8478"


def load_power(path):
    per_gpu = {i: [] for i in GPU_INDICES}
    with open(path) as f:
        for row in csv.DictReader(f):
            gi = int(row["gpu_index"])
            if gi in per_gpu:
                per_gpu[gi].append((float(row["wall_time"]), float(row["power_w"])))
    for i in per_gpu:
        per_gpu[i].sort()
    return per_gpu


def aggregate_series(per_gpu):
    """Sums power across all GPUs per 50ms tick, then zeroes the time axis to the trial's
    own start -- matches the methodology used throughout findings.md and the artifact."""
    buckets = {}
    for i, series in per_gpu.items():
        for t, p in series:
            key = round(t / 0.05)
            buckets.setdefault(key, {})[i] = p
    ticks = sorted(buckets)
    t0 = ticks[0] * 0.05
    return [(k * 0.05 - t0, sum(buckets[k].values())) for k in ticks]


def ramp_series(agg):
    out = []
    for (t0, p0), (t1, p1) in zip(agg, agg[1:]):
        dt = t1 - t0
        if dt > 0:
            out.append((t1, (p1 - p0) / dt))
    return out


def windowed(series, w=WINDOW_S):
    return [(t, v) for t, v in series if t <= w]


def trial_path(arm, trial):
    return os.path.join(DATA_DIR, f"closedloopheavypergpu_power_trace_{arm}_t{trial}.csv")


def plot_arm(ax1, ax2, arm, color, alpha, label):
    max_ramps = []
    for i, t in enumerate(TRIALS):
        agg = windowed(aggregate_series(load_power(trial_path(arm, t))))
        ramp = windowed(ramp_series(agg))
        max_ramps.append(max(abs(r) for _, r in ramp))
        ax1.plot([x for x, _ in agg], [y for _, y in agg],
                  color=color, lw=0.7, alpha=alpha, label=label if i == 0 else None)
        ax2.plot([x for x, _ in ramp], [y for _, y in ramp], color=color, lw=0.6, alpha=alpha)
    return max_ramps


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#3a362c",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.3), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.12})

    proposed_max = plot_arm(ax1, ax2, "coincidence_ceiling", C_PROPOSED, 0.6,
                             "Proposed rule (coincidence_ceiling, Pareto- and threshold-safe)")
    extcheck_max = plot_arm(ax1, ax2, "weighted_sum_coincidence_ceiling", C_EXTCHECK, 0.6,
                             "weighted_sum_coincidence_ceiling (external check, Pareto-safe)")
    unsafe_max = plot_arm(ax1, ax2, "lmetric_power_coincidence_ceiling", C_UNSAFE, 0.75,
                           "lmetric_power_coincidence_ceiling (unsafe)")

    ax1.set_ylabel("Fleet power (W)")
    ax1.legend(loc="lower right", handlelength=1.6, borderaxespad=0.3)
    ax1.margins(x=0.01)

    ax2.axhspan(CEILING_LO, CEILING_HI, color=C_CEIL, alpha=0.18, lw=0,
                label=f"Per-GPU ramp ceiling ({CEILING_LO:.0f}–{CEILING_HI:.0f} W/s)")
    ax2.axhspan(-CEILING_HI, -CEILING_LO, color=C_CEIL, alpha=0.18, lw=0)
    ax2.set_ylabel("Fleet power ramp (W/s)")
    ax2.set_xlabel("Time (s)")
    ax2.margins(x=0.01)
    ax2.legend(loc="upper right", handlelength=1.6)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

    fig.align_ylabels([ax1, ax2])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    print("saved:", OUT)
    print("coincidence_ceiling per-trial max |ramp|:", [round(x, 1) for x in proposed_max])
    print("weighted_sum_coincidence_ceiling per-trial max |ramp|:", [round(x, 1) for x in extcheck_max])
    print("lmetric_power_coincidence_ceiling per-trial max |ramp|:", [round(x, 1) for x in unsafe_max])


if __name__ == "__main__":
    main()
