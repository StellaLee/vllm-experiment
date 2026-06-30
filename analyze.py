#!/usr/bin/env python3
"""Merge GPU timeseries + BurstGPT detail log -> dated markdown findings report."""
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

def load_burstgpt(path):
    """Load JSONL or JSON detail log. Returns (records_list, field_map)."""
    if not os.path.exists(path):
        return [], {}
    records = []
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return [], {}
    # Try JSONL first (one JSON object per line)
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # Fall back to a single JSON array/object if JSONL parsing yielded nothing
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
    if not records:
        return [], {}
    sample = records[0]
    print(f"[analyze] {len(records)} BurstGPT records, keys: {list(sample.keys())}")
    ttft_field = find_field(sample, ["first_chunk_time", "ttft", "first_token_latency",
                                      "time_to_first_token", "first_token_time", "TTFT"])
    lat_field  = find_field(sample, ["total_chunk_time", "latency", "elapsed_time",
                                      "e2e_latency", "total_time", "end2end_latency",
                                      "response_time"])
    return records, {"ttft": ttft_field, "lat": lat_field}

def load_gpu(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def compute_energy_wh(records, interval_s):
    total_ws = sum(r["power_w"] for r in records) * interval_s
    return total_ws / 3600

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--burstgpt-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gpu = load_gpu(args.gpu_log)
    requests, fields = load_burstgpt(args.burstgpt_log)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Baseline Profile - Qwen2.5-0.5B-Instruct",
        f"**Date:** {date_str}",
        "**Model:** /model/ModelScope/Qwen/Qwen2.5-0.5B-Instruct",
        "**GPU:** NVIDIA GeForce RTX 4090 (450W TDP)",
        f"**vLLM:** 0.23.0  |  **Requests:** {len(requests)}",
        "",
    ]

    lines.append("## GPU Metrics")
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
    lines.append("")

    lines.append("## Request Latency & TTFT")
    if requests and fields.get("lat"):
        lats = [r[fields["lat"]] for r in requests if fields["lat"] in r]
        lines += [
            "| Metric | P50 | P95 | P99 | Min | Max |",
            "|--------|-----|-----|-----|-----|-----|",
            f"| Latency (s) | {percentile(lats,50):.3f} | {percentile(lats,95):.3f} | {percentile(lats,99):.3f} | {min(lats):.3f} | {max(lats):.3f} |",
        ]
        if fields.get("ttft"):
            ttfts = [r[fields["ttft"]] for r in requests if fields["ttft"] in r]
            lines.append(
                f"| TTFT (s)    | {percentile(ttfts,50):.3f} | {percentile(ttfts,95):.3f} | {percentile(ttfts,99):.3f} | {min(ttfts):.3f} | {max(ttfts):.3f} |"
            )
        else:
            lines.append(f"_TTFT field not found. Keys: {list(requests[0].keys())}_")
    elif requests:
        lines.append(f"_Latency field not detected. Keys: {list(requests[0].keys())}_")
        lines.append(f"_Raw first record:_\n```json\n{json.dumps(requests[0], indent=2)}\n```")
    else:
        lines.append("_BurstGPT detail log not found or empty._")
    lines.append("")

    lines.append("## Raw Log Paths")
    lines += [
        f"- GPU timeseries: `{args.gpu_log}`",
        f"- BurstGPT detail: `{args.burstgpt_log}`",
    ]

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"[analyze] Report written to {args.output}")
    print(report)

if __name__ == "__main__":
    main()
