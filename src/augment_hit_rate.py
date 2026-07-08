#!/usr/bin/env python3
"""Augment a benchmark result JSON with KV prefix-cache hit rate.

Called from run_near_sat_bench.sh and run_cf_factorial.sh after each benchmark
run.  Computes a delta hit rate from metric snapshots taken before and after the
run (so multiple benchmarks on the same server accumulate correctly).

Usage:
    python3 src/augment_hit_rate.py <tag> <log_dir> <hit_before> <query_before> <hit_after> <query_after>
"""
import glob
import json
import os
import sys

# vLLM has used several metric names across versions; try all of them.
HIT_NAMES = [
    "vllm:prefix_cache_hits_total",           # current (no gpu_ prefix)
    "vllm:gpu_prefix_cache_hit_count_total",  # older
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:gpu_cache_hit_count_total",
]
QUERY_NAMES = [
    "vllm:prefix_cache_queries_total",          # current (no gpu_ prefix)
    "vllm:gpu_prefix_cache_query_count_total",  # older
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:gpu_cache_query_count_total",
]


def scrape(port, names):
    """Sum a Prometheus counter across all label combos; return 0.0 on failure."""
    import re
    import urllib.request
    try:
        body = urllib.request.urlopen(
            f"http://localhost:{port}/metrics", timeout=5
        ).read().decode()
    except Exception as e:
        print(f"  WARNING: could not reach metrics endpoint: {e}")
        return None, None

    for name in names:
        pattern = re.compile(
            r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([\d.e+\-]+)",
            re.MULTILINE,
        )
        vals = [float(m) for m in pattern.findall(body)]
        if vals:
            return sum(vals), name
    return None, None


def main():
    if len(sys.argv) == 7:
        # called with pre-scraped snapshots
        tag, log_dir, hb, qb, ha, qa = sys.argv[1:]
        hb, qb, ha, qa = float(hb), float(qb), float(ha), float(qa)
        dq = qa - qb
        hit_rate = (ha - hb) / dq if dq > 0 else None
        hit_delta = ha - hb
        query_delta = dq
    elif len(sys.argv) == 4:
        # called with tag, log_dir, port; scrapes live
        tag, log_dir, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
        ha, hit_name = scrape(port, HIT_NAMES)
        qa, _ = scrape(port, QUERY_NAMES)
        if ha is None or qa is None:
            print(f"  WARNING: no prefix-cache metrics found for tag={tag}")
            return
        hit_rate = ha / qa if qa > 0 else 0.0
        hit_delta, query_delta = ha, qa
    else:
        print("Usage: augment_hit_rate.py <tag> <log_dir> <hit_before> <query_before> <hit_after> <query_after>")
        sys.exit(1)

    for f in sorted(
        glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime, reverse=True
    ):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("tag") == tag:
            if hit_rate is not None:
                d["kv_hit_rate"] = hit_rate
            d["kv_hit_delta"] = hit_delta
            d["kv_query_delta"] = query_delta
            json.dump(d, open(f, "w"), indent=2)
            rate_str = f"{hit_rate:.1%}" if hit_rate is not None else "N/A"
            print(f"  KV hit rate: {rate_str}  ({hit_delta:.0f}/{query_delta:.0f} blocks)")
            return
    print(f"  WARNING: no JSON found for tag={tag} in {log_dir}")


if __name__ == "__main__":
    main()
