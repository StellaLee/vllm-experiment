#!/usr/bin/env python3
"""Slice a window of the real BurstGPT_1.csv trace into arrival_s,prompt_tokens,response_tokens
-- consumed by src/replay_sharegpt.py's --trace-csv mode.

Reuses the windowing approach already proven in orchestrate/mlsys/orchestrate_burstgpt_one.sh:
slice NROWS conversation-log rows starting at OFF, cap inter-arrival gaps at GAPCAP seconds (so
a real trace's sparse periods don't waste wall-clock replaying them literally), rebase to t=0,
and rescale the whole span to TARGET_S seconds. Unlike orchestrate_burstgpt_one.sh (which drives
requests through the old burstgpt.cli tool and its own prompt-matcher, previously diagnosed to
silently censor prompts at 1024 tokens), this script only extracts (arrival_s, prompt_tokens,
response_tokens) -- the replay client generates its own filler-text prompt of the target token
length directly, so the matcher/censoring bug never enters the picture. The raw CSV itself is
uncapped (real max ~29,665 tokens, p99~=3,387 -- already confirmed in earlier project work).

Usage: make_burstgpt_trace.py SRC_CSV OUT_CSV [--off N] [--nrows N] [--gapcap S] [--target-s S]
"""
import argparse
import csv
import statistics as st
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--off", type=int, default=75000)
    ap.add_argument("--nrows", type=int, default=500)
    ap.add_argument("--gapcap", type=float, default=3.0)
    ap.add_argument("--target-s", type=float, default=600.0)
    args = ap.parse_args()

    rows = list(csv.reader(open(args.src_csv)))
    hdr = rows[0]
    # Conversation-log rows with a non-zero response (max_tokens=0 -> vLLM errors)
    conv = [r for r in rows[1:] if r[5] == "Conversation log" and int(r[3]) >= 1]
    window = conv[args.off:args.off + args.nrows]
    if not window:
        sys.exit(f"ERROR: empty window at off={args.off} (pool size {len(conv)})")

    ts = [int(r[0]) for r in window]
    capped = [0.0]
    for i in range(1, len(ts)):
        capped.append(capped[-1] + min(ts[i] - ts[i - 1], args.gapcap))
    span = capped[-1] or 1.0
    scale = span / args.target_s

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_s", "prompt_tokens", "response_tokens"])
        for r, t in zip(window, capped):
            w.writerow([round(t / scale, 3), int(r[2]), int(r[3])])

    req = [int(r[2]) for r in window]
    resp = [int(r[3]) for r in window]
    m = st.mean(req)
    cs2 = st.pvariance(req) / (m * m) if m > 0 else 0.0
    n_whale = sum(1 for x in req if x > 4000)
    print(f"[trace] n={len(window)} span={span:.0f}s -> scaled {args.target_s:.0f}s (scale={scale:.3f})")
    print(f"[trace] req_tokens: mean={m:.0f} max={max(req)} p99={sorted(req)[int(0.99*len(req))]} "
          f"realized_Cs2={cs2:.3f} n_over4000={n_whale} ({100*n_whale/len(req):.1f}%)")
    print(f"[trace] resp_tokens: mean={st.mean(resp):.0f} p50={st.median(resp)} max={max(resp)}")
    print(f"[trace] -> {args.out_csv}")


if __name__ == "__main__":
    main()
