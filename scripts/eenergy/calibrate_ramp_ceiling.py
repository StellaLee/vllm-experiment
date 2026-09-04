#!/usr/bin/env python3
"""Per-GPU ramp-ceiling calibration -- replicates the exact methodology documented in
scripts/eenergy/README.md for the original global 450 W/s constant (burst of 24 concurrent
45k-char prefills against one idle replica, power sampled at the router's actual poll
cadence, ROUTER_POWER_INTERVAL_S=0.5s default), run independently per GPU instead of
assuming GPU 0's measurement generalizes. Motivated by round_robin-trace evidence that
per-GPU p99 ramp rate varies ~30% across 6 nominally-identical 4090s (see findings).

Usage:
  python3 calibrate_ramp_ceiling.py --gpus 2,3,4,5,6,7 --model /data/pli/models/Qwen2.5-Coder-7B-Instruct \
      --output ramp_ceiling_per_gpu.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
import urllib.request

import pynvml

PROMPT_CHARS = 45_000
BASE_PORT = 8091


def wait_health(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def fire_one(port, model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:
        print(f"  request error: {e}", file=sys.stderr)


def calibrate_one_gpu(gpu, model, port, interval_s=0.5, settle_s=8, headroom_w=1.0):
    print(f"=== calibrating GPU {gpu} (port {port}) ===")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    replica_log = open(f"logs/calibrate_replica_gpu{gpu}.log", "w")
    proc = subprocess.Popen(
        ["python3", "-m", "vllm.entrypoints.openai.api_server",
         "--model", model, "--port", str(port), "--dtype", "auto",
         "--max-num-batched-tokens", "16384", "--max-num-seqs", "64"],
        env=env, stdout=replica_log, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_health(port):
            raise RuntimeError(f"replica on GPU {gpu} never became healthy "
                                f"(see logs/calibrate_replica_gpu{gpu}.log)")
        print(f"  replica healthy, settling {settle_s}s before burst")
        time.sleep(settle_s)

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

        samples = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                samples.append((time.time(), p))
                time.sleep(interval_s)

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        time.sleep(interval_s * 3)  # a few idle samples before the burst

        prompt = "The quick brown fox jumps over the lazy dog. " * (PROMPT_CHARS // 48)
        print(f"  firing 24 concurrent {len(prompt)}-char prefills")
        threads = [threading.Thread(target=fire_one, args=(port, model, prompt)) for _ in range(24)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        time.sleep(interval_s * 4)  # settle after burst
        stop.set()
        t.join(timeout=5)
        pynvml.nvmlShutdown()

        ramps = []
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if dt > 0:
                ramps.append(abs(p1 - p0) / dt)
        max_ramp = max(ramps) if ramps else 0.0
        ceiling = max_ramp + headroom_w
        print(f"  n_samples={len(samples)} max_ramp={max_ramp:.1f} W/s -> ceiling={ceiling:.1f} W/s")
        return ceiling, max_ramp, len(samples)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-f", f"port {port}"], check=False)
        replica_log.close()
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    results = {}
    for i, gpu in enumerate(gpus):
        port = BASE_PORT + i
        ceiling, max_ramp, n = calibrate_one_gpu(gpu, args.model, port)
        results[gpu] = {"ceiling_w_per_s": round(ceiling, 1), "max_ramp_observed": round(max_ramp, 1), "n_samples": n}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== summary ===")
    for gpu, r in results.items():
        print(f"  GPU {gpu}: ceiling={r['ceiling_w_per_s']} W/s (max observed {r['max_ramp_observed']})")


if __name__ == "__main__":
    main()
