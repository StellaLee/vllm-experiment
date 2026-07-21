#!/usr/bin/env python3
"""Summarize/plot a slocvar chunk-budget trace (DYNAMIC_CHUNK_TRACE csv:
step,wall_s,depth,signal_ms,chunk). Answers "how does the budget move over time?"
- prints time(steps) spent at each budget level, # transitions, min/max/last;
- prints a downsampled ASCII trajectory of chunk-vs-step (log2 budget);
- writes a PNG (chunk & signal vs step) if matplotlib is available.

Usage: python scripts/plot_chunk_trace.py logs/<...>-ours-chunktrace.csv [more.csv ...]
"""
import csv, sys, math, os


def load(path):
    steps, sig, chunk = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row["step"])); sig.append(float(row["signal_ms"]))
                chunk.append(int(row["chunk"]))
            except (KeyError, ValueError):
                continue
    return steps, sig, chunk


def summarize(path):
    steps, sig, chunk = load(path)
    n = len(chunk)
    print(f"\n=== {os.path.basename(path)}  (n={n} steps) ===")
    if not n:
        print("  (empty)"); return
    # dwell time at each level
    from collections import Counter
    dwell = Counter(chunk)
    print("  budget dwell (steps @ level):")
    for b in sorted(dwell):
        print(f"    {b:6d} : {dwell[b]:5d}  ({100*dwell[b]/n:4.1f}%)")
    trans = sum(1 for a, b in zip(chunk, chunk[1:]) if a != b)
    print(f"  transitions={trans}  min={min(chunk)} max={max(chunk)} last={chunk[-1]}")
    print(f"  signal_ms: p50={_pct(sig,50):.1f} p95={_pct(sig,95):.1f} max={max(sig):.1f}")
    _ascii(chunk)
    _png(path, steps, sig, chunk)


def _pct(xs, p):
    ys = sorted(xs); return ys[min(len(ys) - 1, int(p / 100 * len(ys)))] if ys else float("nan")


def _ascii(chunk, width=100, height=12):
    lo = math.log2(min(chunk)); hi = math.log2(max(chunk)) or lo + 1
    span = (hi - lo) or 1
    # downsample to width columns (max budget in each bucket)
    step = max(1, len(chunk) // width)
    cols = [max(chunk[i:i + step]) for i in range(0, len(chunk), step)]
    grid = [[" "] * len(cols) for _ in range(height)]
    for x, c in enumerate(cols):
        y = int((math.log2(c) - lo) / span * (height - 1))
        grid[height - 1 - y][x] = "#"
    print("  chunk-vs-step (log2 budget, downsampled):")
    for r, row in enumerate(grid):
        lab = f"{int(2**(hi - r/(height-1)*span)):6d}" if r in (0, height - 1) else " " * 6
        print(f"    {lab} |{''.join(row)}")


def _png(path, steps, sig, chunk):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.step(steps, chunk, where="post", color="C0", label="chunk budget")
    ax1.set_yscale("log", base=2); ax1.set_xlabel("scheduler step"); ax1.set_ylabel("chunk budget (tok)")
    ax2 = ax1.twinx()
    ax2.plot(steps, sig, color="C3", alpha=0.5, lw=0.8, label="signal (cvar ms)")
    ax2.set_ylabel("signal_ms")
    fig.tight_layout()
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=110); print(f"  wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        summarize(p)
