#!/usr/bin/env python3
"""
Replay ShareGPT multi-turn conversations against vLLM /generate (streaming).
Sends each turn with accumulated conversation history as prefix so vLLM's
KV cache sees the shared prefix across turns.

Output: JSONL with one record per request, including conv_id, turn number,
prompt length, TTFT, and total latency.
"""
import argparse, json, os, time, urllib.request, urllib.error
from datetime import datetime


def stream_request(host, port, prompt, max_tokens):
    """POST to /generate with stream=True. Returns (ttft_s, total_s)."""
    url = f"http://{host}:{port}/generate"
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    chunk = line[6:]
                    try:
                        obj = json.loads(chunk)
                        text = obj.get("text", "")
                        if ttft is None and text and text != prompt:
                            ttft = time.monotonic() - t0
                    except json.JSONDecodeError:
                        pass
        total = time.monotonic() - t0
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e}")
    return ttft, total


def build_prompt(history, new_human):
    """Concatenate prior (human, gpt) pairs + new human turn into a single prompt."""
    parts = []
    for h, g in history:
        parts.append(f"Human: {h}\nAssistant: {g}")
    parts.append(f"Human: {new_human}\nAssistant:")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Replay ShareGPT conversations against vLLM")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dataset", required=True, help="ShareGPT JSON file (list of conversations)")
    ap.add_argument("--num-convs", type=int, default=50, help="Conversations to replay")
    ap.add_argument("--max-turns", type=int, default=4, help="Max turns per conversation")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", required=True, help="JSONL output path for per-request records")
    args = ap.parse_args()

    with open(args.dataset) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        data = list(raw.values())
    else:
        data = raw

    convs = [
        c for c in data
        if isinstance(c.get("conversations"), list)
        and len(c["conversations"]) >= 2
    ]
    if not convs:
        raise SystemExit("ERROR: no valid multi-turn conversations found in dataset.")

    convs = convs[:args.num_convs]
    print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | max_tokens={args.max_tokens}")

    records = []
    for ci, conv in enumerate(convs):
        turns = conv["conversations"]
        history = []
        turn_num = 0
        i = 0

        while i < len(turns) and turn_num < args.max_turns:
            if turns[i].get("from") != "human":
                i += 1
                continue

            human_msg = turns[i]["value"].strip()
            prompt = build_prompt(history, human_msg)

            try:
                ttft, total = stream_request(args.host, args.port, prompt, args.max_tokens)
                gpt_placeholder = f"[turn {turn_num+1} response]"
                if i + 1 < len(turns) and turns[i + 1].get("from") == "gpt":
                    gpt_placeholder = turns[i + 1]["value"].strip()[:256]

                record = {
                    "conv_id": conv.get("id", ci),
                    "turn": turn_num + 1,
                    "history_turns": len(history),
                    "prompt_tokens_approx": len(prompt.split()),
                    "ttft": round(ttft, 4) if ttft is not None else None,
                    "latency": round(total, 4),
                    "ts": time.time(),
                }
                records.append(record)
                history.append((human_msg, gpt_placeholder))
                print(f"  [{ci+1}/{len(convs)}] turn {turn_num+1}: ttft={ttft:.3f if ttft else 0:.3f}s  lat={total:.3f}s  prompt_words={len(prompt.split())}")

            except RuntimeError as e:
                print(f"  [{ci+1}/{len(convs)}] turn {turn_num+1} SKIP: {e}")

            turn_num += 1
            i += 2 if (i + 1 < len(turns) and turns[i+1].get("from") == "gpt") else 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[replay] {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
