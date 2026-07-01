#!/usr/bin/env python3
"""
Generate comparison summary tables from sweep logs.

Usage:
  python3 summarize.py --sweep qps [--date 2026-07-01]
  python3 summarize.py --sweep kvcache [--date 2026-07-01]
  python3 summarize.py --sweep all [--date 2026-07-01]
"""
import argparse, json, math, os
from datetime import datetime

EXPERIMENT_DIR = "/root/vllm-experiment"


# ── shared helpers ────────────────────────────────────────────────────────────

def pct(data, p):
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def find_field(rec, candidates):
    for c in candidates:
        if c in rec:
            return c
    return None


def latency_stats(path):
    """Return (n, ttfts, lats) from a BurstGPT or ShareGPT detail JSONL."""
    recs = load_jsonl(path)
    if not recs:
        return 0, [], []
    s = recs[0]
    lat_f = find_field(s, ["total_chunk_time", "latency", "elapsed_time",
                            "e2e_latency", "total_time", "end2end_latency", "response_time"])
    ttft_f = find_field(s, ["first_chunk_time", "ttft", "first_token_latency",
                             "time_to_first_token", "first_token_time", "TTFT"])
    lats  = [r[lat_f]  for r in recs if lat_f  and lat_f  in r]
    ttfts = [r[ttft_f] for r in recs if ttft_f and ttft_f in r]
    return len(recs), ttfts, lats


def gpu_energy_wh(path):
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        g = json.load(f)
    recs = g.get("records", [])
    if not recs:
        return float("nan")
    return sum(r["power_w"] for r in recs) * g["interval_s"] / 3600


def fmt_s(v):
    return f"{v:.3f}s" if not math.isnan(v) else "N/A"


def fmt_wh(v):
    return f"{v:.3f} Wh" if not math.isnan(v) else "N/A"


def write_summary(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    content = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(content)
    print(f"[summarize] Written: {path}\n")
    print(content)


# ── QPS sweep ─────────────────────────────────────────────────────────────────

def qps_summary(date, log_dir, findings_dir):
    rows = []

    # Prior 1x baseline (different date — include if present)
    bl_detail = f"{log_dir}/2026-06-30-burstgpt-detail.jsonl"
    bl_gpu    = f"{log_dir}/2026-06-30-gpu.json"
    if os.path.exists(bl_detail):
        n, ttfts, lats = latency_stats(bl_detail)
        rows.append(("1x (2026-06-30)", n, ttfts, lats, gpu_energy_wh(bl_gpu)))

    for label in ["0.5x", "2x", "4x"]:
        detail  = f"{log_dir}/{date}-qps-{label}-burstgpt-detail.jsonl"
        gpu_log = f"{log_dir}/{date}-qps-{label}-gpu.json"
        n, ttfts, lats = latency_stats(detail)
        rows.append((label, n, ttfts, lats, gpu_energy_wh(gpu_log)))

    lines = [
        f"# BurstGPT QPS Sweep Summary — {date}",
        "",
        "**Model:** Qwen2.5-0.5B-Instruct  |  **gpu-memory-utilization:** 0.9",
        "**Scales:** relative to calibrated base (1.2344×). Lower scale = fewer req/s.",
        "",
        "| Scale | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy |",
        "|-------|---|----------|----------|---------|---------|--------|",
    ]
    for label, n, ttfts, lats, energy in rows:
        lines.append(
            f"| {label} | {n} | {fmt_s(pct(ttfts,50))} | {fmt_s(pct(ttfts,95))} "
            f"| {fmt_s(pct(lats,50))} | {fmt_s(pct(lats,95))} | {fmt_wh(energy)} |"
        )

    lines += [
        "",
        "## Per-scale findings",
        f"- [0.5x]({date}-qps-0.5x.md)",
        f"- [2x]({date}-qps-2x.md)",
        f"- [4x]({date}-qps-4x.md)",
    ]

    write_summary(f"{findings_dir}/{date}-qps-sweep-summary.md", lines)


# ── KV cache size sweep ───────────────────────────────────────────────────────

def kvcache_summary(date, log_dir, findings_dir):
    rows = []
    for label in ["0.3", "0.5", "0.7", "0.9"]:
        detail  = f"{log_dir}/{date}-kvcache-{label}-sharegpt-detail.jsonl"
        gpu_log = f"{log_dir}/{date}-kvcache-{label}-gpu.json"
        n, ttfts, lats = latency_stats(detail)
        rows.append((f"gpu-util={label}", n, ttfts, lats, gpu_energy_wh(gpu_log)))

    lines = [
        f"# ShareGPT KV Cache Size Sweep Summary — {date}",
        "",
        "**Model:** Qwen2.5-0.5B-Instruct  |  **Trace:** ShareGPT multi-turn",
        "**Signal:** rising P95 TTFT on later turns = KV cache pressure forcing evictions.",
        "",
        "| gpu-memory-utilization | N | P50 TTFT | P95 TTFT | P50 Lat | P95 Lat | Energy |",
        "|------------------------|---|----------|----------|---------|---------|--------|",
    ]
    for label, n, ttfts, lats, energy in rows:
        lines.append(
            f"| {label} | {n} | {fmt_s(pct(ttfts,50))} | {fmt_s(pct(ttfts,95))} "
            f"| {fmt_s(pct(lats,50))} | {fmt_s(pct(lats,95))} | {fmt_wh(energy)} |"
        )

    lines += [
        "",
        "## Per-utilization findings",
    ] + [f"- [gpu-util={l}]({date}-kvcache-{l}.md)" for l in ["0.3", "0.5", "0.7", "0.9"]]

    write_summary(f"{findings_dir}/{date}-kvcache-sweep-summary.md", lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate sweep summary tables.")
    parser.add_argument("--sweep", choices=["qps", "kvcache", "all"], required=True)
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date prefix for log files (default: today)")
    parser.add_argument("--log-dir", default=f"{EXPERIMENT_DIR}/logs")
    parser.add_argument("--findings-dir", default=f"{EXPERIMENT_DIR}/findings")
    args = parser.parse_args()

    if args.sweep in ("qps", "all"):
        qps_summary(args.date, args.log_dir, args.findings_dir)
    if args.sweep in ("kvcache", "all"):
        kvcache_summary(args.date, args.log_dir, args.findings_dir)


if __name__ == "__main__":
    main()
