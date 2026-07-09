#!/usr/bin/env python3
"""General-purpose ablation analysis for multi-turn ShareGPT replay logs.

Each JSONL file contains records written by replay_sharegpt.py with fields:
  turn (int, 1-indexed), ttft (float, seconds), tpot (float, seconds/token)

Usage examples:

  # Soft aging vs ratio-based vs baseline
  python3 src/analyze_ablation.py \
    --base "baseline:logs/2026-07-08-mt-mt_base_c15.jsonl" \
    "ratio_t1:logs/2026-07-08-mt-mt_ratio_reorder_c15_t1.jsonl" \
    "ratio_t2:logs/2026-07-08-mt-mt_ratio_reorder_c15_t2.jsonl" \
    "soft_t1:logs/2026-07-09-mt-mt_soft_aging_c15_t1.jsonl" \
    "soft_t2:logs/2026-07-09-mt-mt_soft_aging_c15_t2.jsonl" \
    "combined:logs/2026-07-08-mt-mt_comb_c15.jsonl"

  # Chunk hysteresis vs no-hysteresis
  python3 src/analyze_ablation.py \
    --base "baseline:logs/2026-07-08-mt-mt_base_c15.jsonl" \
    "chunk_t1:logs/2026-07-08-mt-mt_chunk_c15.jsonl" \
    "chunk_t2:logs/2026-07-08-mt-mt_chunk_c15_t2.jsonl" \
    "hyst_t1:logs/2026-07-09-mt-mt_chunk_hyst_c15_t1.jsonl" \
    "hyst_t2:logs/2026-07-09-mt-mt_chunk_hyst_c15_t2.jsonl" \
    "combined:logs/2026-07-08-mt-mt_comb_c15.jsonl"

  # Average pairs (append _avg suffix to paired labels)
  python3 src/analyze_ablation.py --avg-pairs \
    --base "baseline:logs/2026-07-08-mt-mt_base_c15.jsonl" \
    ...

Positional args: one or more "label:glob_pattern" strings.
--base: designate one label as the baseline for % deltas (default: first arg).
--avg-pairs: for labels ending in _t1/_t2, also emit a _avg column.
--p: percentile to report (default 95).
--field: ttft or tpot (default ttft).
"""
import argparse
import glob
import json
import os
import sys


def load(pattern):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not paths:
        return []
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return recs


def by_turn(recs):
    d = {}
    for r in recs:
        t = r.get("turn")
        if t is not None:
            d.setdefault(t, []).append(r)
    return d


def pct(recs, field, p):
    vals = sorted(r[field] for r in recs if field in r and r[field] is not None)
    if not vals:
        return None
    idx = min(int(len(vals) * p / 100), len(vals) - 1)
    return vals[idx] * 1000  # seconds → ms


def fmt_cell(v, base, width=16):
    if v is None:
        return "N/A".center(width)
    if base is None or base == 0:
        return ("%6.1fms" % v).center(width)
    delta = (v - base) / base * 100
    sign = "+" if delta >= 0 else ""
    return ("%6.1fms (%s%.0f%%)" % (v, sign, delta)).center(width)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ablation conditions turn-by-turn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "conditions",
        nargs="+",
        metavar="label:glob",
        help='Each arg is "label:glob_pattern"',
    )
    parser.add_argument(
        "--base",
        metavar="label:glob",
        help="Baseline condition for %% delta (default: first positional arg)",
    )
    parser.add_argument("--p", type=int, default=95, help="Percentile (default 95)")
    parser.add_argument(
        "--field",
        choices=["ttft", "tpot"],
        default="ttft",
        help="Metric field (default ttft)",
    )
    parser.add_argument(
        "--avg-pairs",
        action="store_true",
        help="For labels ending in _t1/_t2, emit a _avg column",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Prepend this directory to relative glob patterns (default: logs)",
    )
    args = parser.parse_args()

    # Parse label:glob pairs
    def parse_pair(s):
        if ":" not in s:
            print(f"ERROR: expected 'label:glob', got {s!r}", file=sys.stderr)
            sys.exit(1)
        label, pattern = s.split(":", 1)
        if not os.path.isabs(pattern) and not pattern.startswith("logs/"):
            pattern = os.path.join(args.log_dir, pattern)
        return label, pattern

    all_pairs = []
    if args.base:
        all_pairs.append(parse_pair(args.base))
    for c in args.conditions:
        pair = parse_pair(c)
        if not all_pairs or pair[0] != all_pairs[0][0]:
            all_pairs.append(pair)

    base_label = all_pairs[0][0]

    # Load data
    print("Loading logs...")
    data = {}
    for label, pattern in all_pairs:
        recs = load(pattern)
        data[label] = by_turn(recs)
        status = "%d records" % sum(len(v) for v in data[label].values())
        if not recs:
            status = "NOT FOUND (%s)" % pattern
        print("  %-16s : %s" % (label, status))

    # Build ordered list of columns (with optional _avg columns)
    # _avg is the mean of the two individual per-turn p-values, not the
    # combined-record percentile. This matches what findings documents report.
    labels = [label for label, _ in all_pairs]
    avg_pairs_map = {}  # avg_lbl -> (t1_lbl, t2_lbl)
    if args.avg_pairs:
        paired, seen = [], set()
        for lbl in labels:
            if lbl in seen:
                continue
            paired.append(lbl)
            seen.add(lbl)
            if lbl.endswith("_t1"):
                partner = lbl[:-3] + "_t2"
                if partner in labels:
                    paired.append(partner)
                    seen.add(partner)
                    avg_lbl = lbl[:-3] + "_avg"
                    paired.append(avg_lbl)
                    avg_pairs_map[avg_lbl] = (lbl, partner)
        labels = paired

    field = args.field
    field_label = "TTFT" if field == "ttft" else "TPOT"
    p = args.p
    turns = sorted(set().union(*[data[lbl].keys() for lbl in labels if lbl in data]))

    # Column width
    cw = 18

    print()
    print("%s p%d (ms)" % (field_label, p))
    header = "%-5s" % "Turn"
    for lbl in labels:
        header += lbl.center(cw)
    print(header)
    print("-" * (5 + cw * len(labels)))

    base_vals = {}
    for t in turns:
        base_recs = data.get(base_label, {}).get(t, [])
        base_vals[t] = pct(base_recs, field, p)

    for t in turns:
        row = "T%-4d" % t
        for lbl in labels:
            if lbl in avg_pairs_map:
                t1_lbl, t2_lbl = avg_pairs_map[lbl]
                v1 = pct(data.get(t1_lbl, {}).get(t, []), field, p)
                v2 = pct(data.get(t2_lbl, {}).get(t, []), field, p)
                v = (v1 + v2) / 2 if v1 is not None and v2 is not None else (v1 or v2)
            else:
                v = pct(data.get(lbl, {}).get(t, []), field, p)
            row += fmt_cell(v, base_vals[t], width=cw)
        print(row)

    # Per-condition variance summary (range across t1/t2 if avg-pairs)
    if args.avg_pairs and avg_pairs_map:
        print()
        print("Trial-to-trial spread (p%d range across _t1/_t2 pairs):" % p)
        for avg_lbl, (t1_lbl, t2_lbl) in avg_pairs_map.items():
            base_lbl = avg_lbl[:-4]
            spreads = []
            for t in turns:
                v1 = pct(data.get(t1_lbl, {}).get(t, []), field, p)
                v2 = pct(data.get(t2_lbl, {}).get(t, []), field, p)
                if v1 is not None and v2 is not None:
                    spreads.append((t, abs(v1 - v2)))
            if spreads:
                print("  %s:" % base_lbl)
                for t, spread in spreads:
                    print("    T%d: %.1fms" % (t, spread))


if __name__ == "__main__":
    main()
