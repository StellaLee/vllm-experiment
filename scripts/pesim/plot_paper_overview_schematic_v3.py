#!/usr/bin/env python3
"""Three-panel roadmap schematic, v3: single-GPU measured power shape -> data-center-scale
aggregate power profile (model-free bootstrap resampling of real traces, N=6 shown) -> model-
free bootstrap reserve procurement (N=10,000). Panel (a) unchanged from v1/v2. Panel (b)
restored to a time-series aggregate view (like v1) but built from the same model-free bootstrap
resampling used for the paper's real reserve number, not v1's OU-simulated small-N run. Panel
(c) model-free bootstrap peak-ramp distributions at CONC=20, the paper's reserve-procurement
operating point.

Usage:
    python scripts/pesim/gen_overview_v3_data.py   # once, writes overview_real_data_v3.npz
    python scripts/pesim/plot_paper_overview_schematic_v3.py
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

MONO = "#4c72b0"
CHUNK = "#dd8452"
GRAY = "#999999"
INK = "#333333"

LABEL_BBOX = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=1.0)


def rolling_mean(x, win):
    """Same box-filter convention as gen_overview_real_data.py's panel (a) smoothing --
    edge-padded so the output stays the same length as the input."""
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, win - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def panel_a(ax, d):
    mono_t, mono_p = d["mono_t"], d["mono_p"]
    chunk_t, chunk_p = d["chunk_t"], d["chunk_p"]

    ax.plot(mono_t, mono_p, color=MONO, linewidth=1.1)
    ax.plot(chunk_t, chunk_p, color=CHUNK, linewidth=1.1)

    peak = max(mono_p.max(), chunk_p.max())
    ax.plot([mono_t.min(), mono_t.max()], [peak, peak], color=GRAY, linestyle=":", linewidth=0.9)

    i_mono_peak = np.argmax(mono_p)
    ax.annotate("mono", xy=(mono_t[i_mono_peak], mono_p[i_mono_peak]),
                xytext=(0.04, 0.60), textcoords="axes fraction", color=MONO, fontsize=8.5,
                weight="bold", arrowprops=dict(arrowstyle="-", color=MONO, lw=0.7),
                bbox=LABEL_BBOX)
    ax.text(0.62, 0.62, "chunk=512", color=CHUNK, fontsize=8.5, weight="bold",
            transform=ax.transAxes, bbox=LABEL_BBOX)

    ax.text(0.62, 0.94, "peak power ~unchanged,\nramp rate falls", color=INK, fontsize=7.3,
            style="italic", ha="center", transform=ax.transAxes, bbox=LABEL_BBOX)

    ax.set_xlabel("time relative to whale start (s)", fontsize=8.3)
    ax.set_ylabel("GPU power (W), measured", fontsize=8.5)
    ax.set_title("(a) Single GPU, measured\nsame whale request, same peak, slower ramp",
                 fontsize=9.3)
    ax.tick_params(labelsize=7.5)


def panel_b(ax, d):
    t = d["b_t"]
    mono_mat, chunk_mat = d["b_mono"], d["b_chunk"]
    N = int(d["b_N"])

    # D(t) = sum_i P_i(t) -- the same aggregate quantity Section VI's method and panel (c)'s
    # ramp statistics are computed from, not a per-server mean. Shown in kW since N=100 pushes
    # the raw-watt sum into the tens of thousands.
    mono_agg = rolling_mean(mono_mat.sum(axis=0), 6) / 1000.0
    chunk_agg = rolling_mean(chunk_mat.sum(axis=0), 6) / 1000.0
    ax.plot(t, mono_agg, color=MONO, linewidth=1.6, zorder=5)
    ax.plot(t, chunk_agg, color=CHUNK, linewidth=1.6, zorder=5)

    # This panel is illustrative (fleet-aggregation concept, not a quantitative peak-power
    # comparison -- that claim is panel (a)'s, and is genuinely ~unchanged). The two curves'
    # real absolute levels differ by only ~3% (chunk's own trace has a modestly higher duty
    # cycle -- see fig_datacenter_power_profile.png), but a tightly-autoscaled axis exaggerates
    # that into what reads as a large, constant gap; pad the y-range so it doesn't visually
    # compete with panel (a)'s "peak power ~unchanged" framing. Lighter smoothing than the first
    # cut (window 6 vs 20) restores more of the underlying oscillation so mono's sharper/chunk's
    # gentler ramp behavior -- panel (b)'s actual point -- reads clearly.
    ymin = min(mono_agg.min(), chunk_agg.min())
    ymax = max(mono_agg.max(), chunk_agg.max())
    span = ymax - ymin
    ax.set_ylim(ymin - 2.2 * span, ymax + 0.9 * span)

    i_mono = np.argmax(mono_agg)
    i_chunk = np.argmax(chunk_agg)
    ax.annotate("mono aggregate\n(sharper peaks)", xy=(t[i_mono], mono_agg[i_mono]),
                xytext=(0.03, 0.92), textcoords="axes fraction",
                color=MONO, fontsize=7.6, ha="left",
                arrowprops=dict(arrowstyle="-", color=MONO, lw=0.7), bbox=LABEL_BBOX)
    ax.annotate("chunk aggregate\n(gentler peaks)", xy=(t[i_chunk], chunk_agg[i_chunk]),
                xytext=(0.58, 0.90), textcoords="axes fraction",
                color=CHUNK, fontsize=7.6, ha="left",
                arrowprops=dict(arrowstyle="-", color=CHUNK, lw=0.7), bbox=LABEL_BBOX)

    ax.set_xlabel("time (s)", fontsize=8.3)
    ax.set_ylabel(f"aggregate power (kW),\n$D(t)=\\sum_i P_i(t)$, $N{{=}}{N}$", fontsize=8.1)
    ax.set_title(f"(b) Data-center-scale power\nprofile ($N{{=}}{N}$ shown; $N{{=}}10{{,}}000$\n"
                  "for panel c), model-free resample",
                 fontsize=9.0)
    ax.tick_params(labelsize=7.5)


def panel_c(ax, d):
    mono_pool, chunk_pool = d["c_mono_pool"], d["c_chunk_pool"]
    r99_mono, r99_chunk = float(d["c_r99_mono"]), float(d["c_r99_chunk"])

    lo = min(mono_pool.min(), chunk_pool.min())
    hi = max(mono_pool.max(), chunk_pool.max())
    r = np.linspace(lo, hi, 500)
    mono_kde = gaussian_kde(mono_pool)(r)
    chunk_kde = gaussian_kde(chunk_pool)(r)

    ax.fill_between(r, mono_kde, color=MONO, alpha=0.25)
    ax.plot(r, mono_kde, color=MONO, linewidth=1.4)
    ax.fill_between(r, chunk_kde, color=CHUNK, alpha=0.25)
    ax.plot(r, chunk_kde, color=CHUNK, linewidth=1.4)

    ax.axvline(r99_mono, color=MONO, linestyle="--", linewidth=1.1)
    ax.axvline(r99_chunk, color=CHUNK, linestyle="--", linewidth=1.1)

    peak_h = max(mono_kde.max(), chunk_kde.max())
    arrow_y = peak_h * 0.72
    ax.annotate("", xy=(r99_chunk, arrow_y), xytext=(r99_mono, arrow_y),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    reduction = (1 - r99_chunk / r99_mono) * 100
    ax.text((r99_mono + r99_chunk) / 2, peak_h * 0.86, f"-{reduction:.0f}%\nreserve",
            fontsize=7.6, color=INK, ha="center", bbox=LABEL_BBOX)

    ax.text(r99_mono, peak_h * 0.40, "mono\n$R^*_{99}$", color=MONO, fontsize=8, ha="center",
            bbox=LABEL_BBOX)
    ax.text(r99_chunk, peak_h * 0.40, "chunk=512\n$R^*_{99}$", color=CHUNK, fontsize=8,
            ha="center", bbox=LABEL_BBOX)

    ax.text((lo + hi) / 2, peak_h * 1.55,
            r"$R^*(\epsilon) = \min_R R$   s.t.   $P(|D'(t)| > R) \leq \epsilon$",
            fontsize=8.3, color=INK, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.7))

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, peak_h * 1.85)
    ax.set_xlabel("ramp-following requirement $R$ (MW/min)", fontsize=8.3)
    ax.set_ylabel("density (model-free bootstrap,\n$N{=}10{,}000$, CONC=20,\none representative seed)",
                 fontsize=8.0)
    ax.set_title("(c) Chance-constrained reserve\nprocurement (model-free)", fontsize=9.3)
    ax.tick_params(labelsize=7.5)


def add_flow_arrow(fig, x0, x1, y, label):
    arrow = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                             arrowstyle="-|>", mutation_scale=16, color=INK, linewidth=1.4,
                             clip_on=False)
    fig.add_artist(arrow)
    fig.text((x0 + x1) / 2, y + 0.07, label, ha="center", va="bottom", fontsize=7.8,
              color=INK, style="italic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scripts/pesim/overview_real_data_v3.npz")
    ap.add_argument("--out", default="paper-pes-im/figs/fig_overview_schematic.png")
    args = ap.parse_args()

    d = np.load(args.data)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    plt.subplots_adjust(wspace=0.70, top=0.66, bottom=0.19, left=0.06, right=0.98)

    panel_a(axes[0], d)
    panel_b(axes[1], d)
    panel_c(axes[2], d)

    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()
    pos_c = axes[2].get_position()
    y_mid = (pos_a.y0 + pos_a.y1) / 2

    margin_ab = (pos_b.x0 - pos_a.x1) * 0.32
    margin_bc = (pos_c.x0 - pos_b.x1) * 0.32
    shift_ab = (pos_b.x0 - pos_a.x1) * 0.12
    shift_bc = (pos_c.x0 - pos_b.x1) * 0.12
    add_flow_arrow(fig, pos_a.x1 + margin_ab - shift_ab, pos_b.x0 - margin_ab - shift_ab, y_mid,
                   "aggregate\nover fleet")
    add_flow_arrow(fig, pos_b.x1 + margin_bc - shift_bc, pos_c.x0 - margin_bc - shift_bc, y_mid,
                   "solve for $R^*$")

    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
