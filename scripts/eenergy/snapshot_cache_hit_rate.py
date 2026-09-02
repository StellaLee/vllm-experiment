#!/usr/bin/env python3
"""Standalone end-of-trial snapshot of the REAL (server-side) prefix-cache hit rate, read
directly from each replica's vLLM /metrics endpoint -- ground truth, unlike the router's own
cache_mirror.py estimate (a client-side hash-chain mirror that assumes infinite cache
capacity and never sees real eviction).

Every trial in this project's convention launches fresh replica processes, so
vllm:prefix_cache_{hits,queries}_total (cumulative counters since process start) already ARE
that trial's totals -- call this once, right before teardown, no polling/delta needed.

Usage:
  snapshot_cache_hit_rate.py --replicas 127.0.0.1:8001,127.0.0.1:8002 \\
    --model /data/pli/models/Qwen2.5-Coder-7B-Instruct --output logs/real_cache_hit_rate.csv

--replicas accepts the same REPLICA_SPECS shell variable launch_router_experiment.sh already
builds (colon-separated fields per replica, comma-separated across replicas) -- only the
host:port prefix of each entry is used, the rest is ignored.
"""
import argparse
import asyncio
import csv
import os
import sys

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "router"))
from cache_telemetry import fetch_prefix_cache  # noqa: E402


def parse_replica_hostports(spec: str) -> list:
    """Extract (host, port) pairs from a REPLICA_SPECS-format string -- only the first two
    colon-separated fields of each comma-separated entry are used."""
    hostports = []
    for part in spec.split(","):
        fields = part.split(":")
        hostports.append((fields[0], int(fields[1])))
    return hostports


def summarize(per_replica: list) -> dict:
    """per_replica: list of (host, port, hits, queries). Returns fleet totals plus the
    real hit rate (0.0 if there were no queries at all, not NaN -- an all-zero trial is a
    real, reportable result, not a missing one)."""
    total_hits = sum(h for _, _, h, _ in per_replica)
    total_queries = sum(q for _, _, _, q in per_replica)
    hit_rate = (total_hits / total_queries) if total_queries > 0 else 0.0
    return dict(total_hits=total_hits, total_queries=total_queries, hit_rate=hit_rate)


async def snapshot(hostports: list, model_name: str) -> list:
    async with aiohttp.ClientSession() as session:
        async def one(host, port):
            hits, queries = await fetch_prefix_cache(session, host, port, model_name)
            return (host, port, hits, queries)
        return await asyncio.gather(*(one(h, p) for h, p in hostports))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", required=True, help="REPLICA_SPECS-format string")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    hostports = parse_replica_hostports(args.replicas)
    per_replica = asyncio.run(snapshot(hostports, args.model))
    totals = summarize(per_replica)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["host", "port", "hits_tokens", "queries_tokens", "hit_rate"])
        for host, port, hits, queries in per_replica:
            rate = (hits / queries) if queries > 0 else 0.0
            writer.writerow([host, port, hits, queries, f"{rate:.4f}"])
        writer.writerow(["FLEET_TOTAL", "", totals["total_hits"], totals["total_queries"],
                          f"{totals['hit_rate']:.4f}"])

    print(f"real cache hit rate: {totals['hit_rate']:.4f} "
          f"({totals['total_hits']:.0f}/{totals['total_queries']:.0f} tokens) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
