"""Generates figs/architecture.pdf: the disaggregated gate's request flow (Section 5.1) --
client -> gate shim (dual per-pool budget check, budgets held in-process) -> Nixl proxy
(round-robin routing, unmodified) -> 4 prefill instances --NIXL KV transfer--> 4 decode
instances -> response back through the shim. A visual complement to the prose description,
not a replacement for it."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "architecture.pdf")

PREFILL_COLOR = "#c0392b"
DECODE_COLOR = "#2874a6"
GATE_COLOR = "#6c3483"
PROXY_COLOR = "#555555"
NEUTRAL = "#f4f4f4"


def box(ax, x, y, w, h, text, edgecolor, facecolor=NEUTRAL, fontsize=9, fontweight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                 linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=edgecolor)


def arrow(ax, xy1, xy2, color="black", style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=13, linewidth=lw,
                         color=color, connectionstyle=connectionstyle)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Row 1: client, gate shim, nixl proxy
    box(ax, 0.2, 5.0, 1.5, 0.7, "Client", "black", fontweight="bold")
    box(ax, 2.5, 4.5, 3.3, 1.7,
        "Gate shim\n\nbudget_prefill, budget_decode\n(held in-process)\n\n"
        "admit only if BOTH clear\nvia dual_budget_would_exceed", GATE_COLOR, fontsize=8.2)
    box(ax, 6.6, 4.75, 3.1, 1.2, "Nixl proxy (unmodified)\nround-robins within each pool",
        PROXY_COLOR, fontsize=8.5)

    arrow(ax, (1.7, 5.4), (2.5, 5.4))
    ax.text(2.1, 5.55, "POST\n/v1/completions", ha="center", va="bottom", fontsize=6.8)
    arrow(ax, (5.8, 5.5), (6.6, 5.5))
    ax.text(6.2, 5.65, "forward\nunchanged", ha="center", va="bottom", fontsize=6.8)
    arrow(ax, (6.6, 5.0), (5.8, 5.0), color="#888888")
    arrow(ax, (2.5, 5.0), (1.7, 5.1), color="#888888")
    ax.text(4.15, 4.4, "streamed response", ha="center", va="top", fontsize=6.8,
            color="#888888")

    # Row 2: prefill / decode pools, with power poll arrows going up to the gate shim
    box(ax, 0.6, 1.6, 3.6, 1.7, "", PREFILL_COLOR, facecolor="#fbeae8")
    ax.text(2.4, 3.05, "Prefill pool (GPU 0-3)", ha="center", fontsize=9.5, fontweight="bold",
            color=PREFILL_COLOR)
    for i, gx in enumerate([0.85, 1.65, 2.45, 3.25]):
        box(ax, gx, 1.75, 0.65, 1.0, f"P{i}\nGPU{i}", PREFILL_COLOR, fontsize=7)

    box(ax, 5.6, 1.6, 3.6, 1.7, "", DECODE_COLOR, facecolor="#e9f1f8")
    ax.text(7.4, 3.05, "Decode pool (GPU 4-7)", ha="center", fontsize=9.5, fontweight="bold",
            color=DECODE_COLOR)
    for i, gx in enumerate([5.85, 6.65, 7.45, 8.25]):
        box(ax, gx, 1.75, 0.65, 1.0, f"D{i}\nGPU{i + 4}", DECODE_COLOR, fontsize=7)

    # Routing legs from proxy down to each pool
    arrow(ax, (7.3, 4.75), (3.2, 3.3), color=PROXY_COLOR, connectionstyle="arc3,rad=-0.2")
    ax.text(5.2, 4.05, "prefill leg", ha="center", fontsize=7.2, color=PROXY_COLOR,
            rotation=-18)
    arrow(ax, (8.4, 4.75), (7.9, 3.3), color=PROXY_COLOR, connectionstyle="arc3,rad=0.15")
    ax.text(8.55, 4.05, "decode leg", ha="center", fontsize=7.2, color=PROXY_COLOR)

    arrow(ax, (4.2, 2.25), (5.6, 2.25), color="black", lw=1.8)
    ax.text(4.9, 2.4, "NIXL KV-cache transfer", ha="center", fontsize=7.5)

    # Power-poll arrows straight up to the gate shim (budgets live inside it -- no separate
    # boxes/back-arrows, which would misrepresent an in-process object as a network hop)
    arrow(ax, (2.4, 3.3), (3.3, 4.5), color=PREFILL_COLOR, lw=1.2,
          connectionstyle="arc3,rad=0.15")
    arrow(ax, (7.4, 3.3), (5.7, 4.5), color=DECODE_COLOR, lw=1.2,
          connectionstyle="arc3,rad=-0.15")
    ax.text(2.2, 3.9, "NVML\npoll", ha="center", fontsize=6.8, color=PREFILL_COLOR)
    ax.text(7.6, 3.9, "NVML\npoll", ha="center", fontsize=6.8, color=DECODE_COLOR)

    ax.set_title("Disaggregated peak-shaving gate: request flow and per-pool budgets "
                  "(Section 5.1)", fontsize=11)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
