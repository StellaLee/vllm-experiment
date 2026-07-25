#!/usr/bin/env python3
"""Plot request arrival/prefill-size timeline against the GPU power trace, one column per
arm (mono/16384, chunk/2048, chunk/512), for the PES-IM gate experiment. Same trial number
across arms replays the identical whale-injection seed (--pad-seed 1000+trial in
orchestrate_pesim_gate.sh is keyed on trial only, not arm), so the three columns show the
same request sequence and sizes under different scheduling budgets -- this is the direct
visual evidence for the ramp-rate/duty-cycle finding in prepare.md Sec 3 / paper.md Sec 5.

Usage:
    python scripts/plot_pesim_timeline.py --dir <dir-with-jsonl-and-power-csv> \
        --arms b16384 b2048 b512 --trial 1 --labels mono/16384 chunk/2048 chunk/512 \
        --out pesim_timeline.png
"""
import argparse
import json

import matplotlib.pyplot as plt


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["start_ts"] = r["ts"] - r["latency"]
            recs.append(r)
    return recs


def load_power(path):
    ts, pw = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            ts.append(float(parts[0]))
            pw.append(float(parts[2]))
    return ts, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--date", default="2026-07-24")
    ap.add_argument("--name", default="pesim_gate", help="matches orchestrate_pesim_gate.sh's NAME (pesim_gate or pesim_gate_$TAG)")
    ap.add_argument("--arms", nargs="+", default=["b16384", "b2048", "b512"])
    ap.add_argument("--labels", nargs="+", default=["mono (budget=16384)", "chunk (budget=2048)", "chunk (budget=512)"])
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--whale-thresh", type=int, default=40000, help="pad_chars cutoff for the whale marker; lower for a widened whale distribution")
    ap.add_argument("--out", default="pesim_timeline.png")
    args = ap.parse_args()

    n = len(args.arms)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 6.5), sharex="col",
                              gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.08})

    for col, (arm, label) in enumerate(zip(args.arms, args.labels)):
        base = f"{args.dir}/{args.date}-{args.name}-{arm}-t{args.trial}"
        recs = load_records(f"{base}.jsonl")
        pts, ppw = load_power(f"{base}-power.csv")

        t0 = min(r["start_ts"] for r in recs)
        t0 = min(t0, pts[0])

        rel_start = [r["start_ts"] - t0 for r in recs]
        pad_chars = [r["pad_chars"] for r in recs]
        is_whale = [pc >= args.whale_thresh for pc in pad_chars]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        short_x = [x for x, w in zip(rel_start, is_whale) if not w]
        short_y = [y for y, w in zip(pad_chars, is_whale) if not w]
        whale_x = [x for x, w in zip(rel_start, is_whale) if w]
        whale_y = [y for y, w in zip(pad_chars, is_whale) if w]

        ax_top.scatter(short_x, short_y, s=8, c="#4c72b0", alpha=0.6, label="short")
        ax_top.scatter(whale_x, whale_y, s=40, c="#c44e52", marker="^", label="whale")
        ax_top.set_title(label)
        ax_top.set_ylabel("prompt size\n(pad_chars)" if col == 0 else "")
        if col == 0:
            ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9)

        rel_pt = [t - t0 for t in pts]
        ax_bot.plot(rel_pt, ppw, color="#333333", linewidth=0.6)
        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylabel("power (W)" if col == 0 else "")
        ax_bot.axhline(457, color="#c44e52", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_bot.set_ylim(280, 475)

        xmax = max(rel_pt[-1], rel_start[-1] if rel_start else 0)
        ax_top.set_xlim(0, xmax)
        ax_bot.set_xlim(0, xmax)

    fig.suptitle(f"PES-IM gate experiment: request timeline vs. power profile (trial {args.trial})", y=1.00)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
