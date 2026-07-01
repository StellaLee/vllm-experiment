#!/usr/bin/env python3
"""Merge GPU timeseries + request detail log -> dated markdown findings report.

Supports two trace types:
  burstgpt  -- BurstGPT JSONL/JSON logs (original format)
  sharegpt  -- replay_sharegpt.py JSONL logs (multi-turn, has conv_id/turn fields)
"""
import argparse, json, os
from datetime import datetime


def percentile(data, p):
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def find_field(record, candidates):
    for c in candidates:
        if c in record:
            return c
    return None


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not records:
        try:
            raw = json.loads(content)
            if isinstance(raw, list):
                records = raw
            elif isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, list) and v:
                        records = v
                        break
        except json.JSONDecodeError:
            pass
    return records


def load_gpu(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_energy_wh(records, interval_s):
    total_ws = sum(r["power_w"] for r in records) * interval_s
    return total_ws / 3600


def render_gpu_section(gpu):
    lines = ["## GPU Metrics"]
    if gpu and gpu.get("records"):
        recs = gpu["records"]
        powers = [r["power_w"] for r in recs]
        utils  = [r["util_pct"] for r in recs]
        mems   = [r["mem_mib"] for r in recs]
        temps  = [r["temp_c"] for r in recs]
        energy = compute_energy_wh(recs, gpu["interval_s"])
        duration_s = len(recs) * gpu["interval_s"]
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Duration | {duration_s:.1f} s |",
            f"| Total Energy | {energy*1000:.1f} mWh ({energy:.4f} Wh) |",
            f"| Avg Power | {sum(powers)/len(powers):.1f} W |",
            f"| Peak Power | {max(powers):.1f} W |",
            f"| Idle Power (first 5s) | {sum(powers[:10])/max(1,len(powers[:10])):.1f} W |",
            f"| Avg GPU Util | {sum(utils)/len(utils):.1f}% |",
            f"| Peak GPU Util | {max(utils)}% |",
            f"| Avg Mem Used | {sum(mems)/len(mems):.0f} MiB |",
            f"| Peak Mem Used | {max(mems)} MiB |",
            f"| Avg Temp | {sum(temps)/len(temps):.1f} C |",
            f"| Peak Temp | {max(temps)} C |",
        ]
    else:
        lines.append("_GPU log not found or empty._")
    return lines


def render_burstgpt_section(requests):
    lines = ["## Request Latency & TTFT"]
    if not requests:
        lines.append("_BurstGPT detail log not found or empty._")
        return lines

    sample = requests[0]
    print(f"[analyze] {len(requests)} BurstGPT records, keys: {list(sample.keys())}")
    ttft_field = find_field(sample, ["first_chunk_time", "ttft", "first_token_latency",
                                      "time_to_first_token", "first_token_time", "TTFT"])
    lat_field  = find_field(sample, ["total_chunk_time", "latency", "elapsed_time",
                                      "e2e_latency", "total_time", "end2end_latency",
                                      "response_time"])
    if lat_field:
        lats = [r[lat_field] for r in requests if lat_field in r]
        lines += [
            "| Metric | P50 | P95 | P99 | Min | Max |",
            "|--------|-----|-----|-----|-----|-----|",
            f"| Latency (s) | {percentile(lats,50):.3f} | {percentile(lats,95):.3f} | {percentile(lats,99):.3f} | {min(lats):.3f} | {max(lats):.3f} |",
        ]
        if ttft_field:
            ttfts = [r[ttft_field] for r in requests if ttft_field in r]
            lines.append(
                f"| TTFT (s)    | {percentile(ttfts,50):.3f} | {percentile(ttfts,95):.3f} | {percentile(ttfts,99):.3f} | {min(ttfts):.3f} | {max(ttfts):.3f} |"
            )
        else:
            lines.append(f"_TTFT field not found. Keys: {list(sample.keys())}_")
    elif requests:
        lines.append(f"_Latency field not detected. Keys: {list(sample.keys())}_")
        lines.append(f"_Raw first record:_\n```json\n{json.dumps(requests[0], indent=2)}\n```")
    return lines


def render_sharegpt_section(requests):
    lines = ["## Request Latency & TTFT"]
    if not requests:
        lines.append("_ShareGPT detail log not found or empty._")
        return lines

    print(f"[analyze] {len(requests)} ShareGPT records")

    lats  = [r["latency"] for r in requests if r.get("latency") is not None]
    ttfts = [r["ttft"]    for r in requests if r.get("ttft")    is not None]

    lines += [
        "| Metric | P50 | P95 | P99 | Min | Max |",
        "|--------|-----|-----|-----|-----|-----|",
        f"| Latency (s) | {percentile(lats,50):.3f} | {percentile(lats,95):.3f} | {percentile(lats,99):.3f} | {min(lats):.3f} | {max(lats):.3f} |",
    ]
    if ttfts:
        lines.append(
            f"| TTFT (s)    | {percentile(ttfts,50):.3f} | {percentile(ttfts,95):.3f} | {percentile(ttfts,99):.3f} | {min(ttfts):.3f} | {max(ttfts):.3f} |"
        )
    lines.append("")

    # Per-turn breakdown: key metric for KV cache evaluation
    lines.append("## Per-Turn Breakdown (KV Cache Signal)")
    lines.append("_Lower latency on later turns indicates KV prefix cache hits._")
    lines.append("")
    lines += [
        "| Turn | Requests | Avg Latency (s) | Avg TTFT (s) | Avg Prompt Words |",
        "|------|----------|-----------------|--------------|------------------|",
    ]
    by_turn = {}
    for r in requests:
        t = r.get("turn", 1)
        by_turn.setdefault(t, []).append(r)
    for t in sorted(by_turn):
        recs = by_turn[t]
        avg_lat  = sum(r["latency"] for r in recs) / len(recs)
        ttft_recs = [r["ttft"] for r in recs if r.get("ttft") is not None]
        avg_ttft = sum(ttft_recs) / len(ttft_recs) if ttft_recs else float("nan")
        avg_words = sum(r.get("prompt_tokens_approx", 0) for r in recs) / len(recs)
        lines.append(
            f"| {t} | {len(recs)} | {avg_lat:.3f} | {avg_ttft:.3f} | {avg_words:.0f} |"
        )

    # Conversation count
    conv_ids = set(r.get("conv_id") for r in requests)
    lines.append("")
    lines.append(f"**Conversations replayed:** {len(conv_ids)}  |  "
                 f"**Total requests:** {len(requests)}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--trace-log", "--burstgpt-log", dest="trace_log", required=True)
    parser.add_argument("--trace-type", choices=["burstgpt", "sharegpt"], default="burstgpt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gpu = load_gpu(args.gpu_log)
    requests = load_jsonl(args.trace_log)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace_label = "ShareGPT multi-turn" if args.trace_type == "sharegpt" else "BurstGPT"

    lines = [
        f"# Baseline Profile - Qwen2.5-0.5B-Instruct ({trace_label})",
        f"**Date:** {date_str}",
        "**Model:** /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct",
        "**GPU:** NVIDIA GeForce RTX 4090 (450W TDP)",
        f"**vLLM:** 0.23.0  |  **Trace:** {trace_label}  |  **Requests:** {len(requests)}",
        "",
    ]

    lines += render_gpu_section(gpu)
    lines.append("")

    if args.trace_type == "sharegpt":
        lines += render_sharegpt_section(requests)
    else:
        lines += render_burstgpt_section(requests)
    lines.append("")

    lines.append("## Raw Log Paths")
    lines += [
        f"- GPU timeseries: `{args.gpu_log}`",
        f"- Trace detail: `{args.trace_log}`",
    ]

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"[analyze] Report written to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
