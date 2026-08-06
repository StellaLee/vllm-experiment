#!/usr/bin/env python3
"""W8 follow-up figure (draft, not yet wired into the paper): combine the reserve-capacity
saving (Section VI's model) with the whale-TTFT cost (Fig 6's other side) into one decision
curve across all three budgets, so a reader can see the tradeoff directly instead of having
to mentally combine Table IV and Fig 6.

Reserve reduction, mono baseline:
  - chunk=512:  Table IV (headline, published) -- 100-trial library, tight bootstrap CI.
  - chunk=2048: isolated 2-arm nested bootstrap, SAME methodology/seed-order as Table IV,
    computed 2026-07-31 (cf_reserve_2048_isolated.out) -- only a 20-trial library so far
    (still growing), hence the much wider error bars. NOT padded to look matched with 512.

Whale TTFT (mean, ms), same convdiverse_conc10_gpu0 library used for the reserve numbers
above (self-consistent data source, NOT the older 3-trial "wide" dataset Fig 6 used) --
from cf_reserve_2048_and_tradeoff.out, Part B.

Panel B adds short-request TBT p99 and TPOT p99 (ms), same library, computed per-trial and
checked for outliers (all 20 b2048 trials individually inspected: p99 range 246-264ms, tight
and reproducible, NOT a single bad-trial artifact). NOTE the genuinely surprising, reproduced
result here: chunk=2048's short-request tail (TBT p99=258.7, TPOT p99=168.2) is WORSE than
BOTH mono (95.1, 150.3) and chunk=512 (81.3, 71.1) on this traffic condition -- non-monotonic
in budget, echoing the wf=5% p99 sign-flip already documented in
findings/2026-07-24-whale-fraction-size-grid.md (chunking's tail-protection claim is
frequency/condition-dependent, not unconditional). Flagged here rather than smoothed over.
"""
import matplotlib.pyplot as plt
import numpy as np

BUDGETS = ["mono\n(16384)", "chunk\n(2048)", "chunk\n(512)"]

# Reserve reduction vs. mono at 99% reliability (mean, std). Mono is the baseline (0 by definition).
RESERVE_99_MEAN = [0.0, 15.3, 16.9]
RESERVE_99_STD = [0.0, 14.3, 6.8]
N_TRIALS_LABEL = ["100 trials", "20 trials", "100 trials"]

# Whale TTFT mean (ms), same library for all three budgets.
WHALE_TTFT_MEAN = [1893, 2053, 2595]

# Short-request tail metrics (ms), same library, all non-whale requests pooled.
SHORT_TBT_P99 = [95.1, 258.7, 81.3]
SHORT_TPOT_P99 = [150.3, 168.2, 71.1]

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9.6, 3.4))

x = np.arange(len(BUDGETS))
color_reserve = "#4c72b0"
color_ttft = "#c44e52"
color_tbt = "#55a868"
color_tpot = "#8172b2"

ax1.errorbar(x, RESERVE_99_MEAN, yerr=RESERVE_99_STD, fmt="o-", color=color_reserve,
             capsize=4, label="reserve-capacity reduction\nvs. mono, 99% reliability")
ax1.set_ylabel("reserve-capacity reduction (%)", color=color_reserve)
ax1.tick_params(axis="y", labelcolor=color_reserve)
ax1.axhline(0, color="gray", linewidth=0.6, linestyle=":")
ax1.set_xticks(x)
ax1.set_xticklabels(BUDGETS)

for xi, lbl in zip(x, N_TRIALS_LABEL):
    ax1.annotate(lbl, (xi, RESERVE_99_MEAN[xi] + RESERVE_99_STD[xi] + 1.5),
                 ha="center", fontsize=6.5, color="gray")

ax2 = ax1.twinx()
ax2.plot(x, WHALE_TTFT_MEAN, "s--", color=color_ttft, label="whale TTFT mean (ms)")
ax2.set_ylabel("whale TTFT, mean (ms)", color=color_ttft)
ax2.tick_params(axis="y", labelcolor=color_ttft)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper left")
ax1.set_title("(a) Reserve saving vs. whale-TTFT cost\n"
               "(2048's reserve estimate is noisier: 20-trial library)", fontsize=8.5)

ax3.plot(x, SHORT_TBT_P99, "o-", color=color_tbt, label="short-req TBT p99 (ms)")
ax3.set_ylabel("short-req TBT p99 (ms)", color=color_tbt)
ax3.tick_params(axis="y", labelcolor=color_tbt)
ax3.set_xticks(x)
ax3.set_xticklabels(BUDGETS)

ax4 = ax3.twinx()
ax4.plot(x, SHORT_TPOT_P99, "s--", color=color_tpot, label="short-req TPOT p99 (ms)")
ax4.set_ylabel("short-req TPOT p99 (ms)", color=color_tpot)
ax4.tick_params(axis="y", labelcolor=color_tpot)

lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, fontsize=6.5, loc="upper left")
ax3.set_title("(b) Short-request tail: NON-monotonic --\n2048 is worse than BOTH mono and 512 here",
              fontsize=8.5, color="#b03030")

fig.suptitle("Reserve saving, whale-TTFT cost, and short-request tail, by budget", y=1.04)
fig.tight_layout()
out = "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_reserve_latency_decision_DRAFT.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
