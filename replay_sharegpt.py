#!/usr/bin/env python3
"""
Replay ShareGPT multi-turn conversations against vLLM /generate (streaming).
Sends each turn with accumulated conversation history as prefix so vLLM's
KV cache sees the shared prefix across turns.

With --concurrency N, N conversations run in parallel (turns within each
conversation remain sequential so accumulated prefixes stay consistent).

Output: JSONL with one record per request, including conv_id, turn number,
prompt length, TTFT, and total latency.
"""
import argparse, json, os, time, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def stream_request(host, port, prompt, max_tokens):
    """POST to /generate with stream=True. Returns (ttft_s, total_s).

    vLLM native API server streams raw JSONL (no SSE prefix):
      {"text": ["prompt + generated tokens so far"]}
    TTFT = time until the first chunk whose text is longer than the prompt.
    """
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
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    texts = obj.get("text", [])
                    full_text = texts[0] if isinstance(texts, list) and texts else (texts or "")
                    if ttft is None and len(full_text) > len(prompt):
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


def replay_conversation(ci, conv, args, records, records_lock, print_lock):
    """Replay one conversation sequentially. Safe to call from multiple threads."""
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
            with records_lock:
                records.append(record)

            history.append((human_msg, gpt_placeholder))
            ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"
            with print_lock:
                print(f"  [conv {ci+1}] turn {turn_num+1}: ttft={ttft_str}s  lat={total:.3f}s  words={len(prompt.split())}")

        except RuntimeError as e:
            with print_lock:
                print(f"  [conv {ci+1}] turn {turn_num+1} SKIP: {e}")

        turn_num += 1
        i += 2 if (i + 1 < len(turns) and turns[i + 1].get("from") == "gpt") else 1


def main():
    ap = argparse.ArgumentParser(description="Replay ShareGPT conversations against vLLM")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dataset", required=True, help="ShareGPT JSON file (list of conversations)")
    ap.add_argument("--num-convs", type=int, default=50, help="Conversations to replay")
    ap.add_argument("--max-turns", type=int, default=4, help="Max turns per conversation")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel conversations (default 1 = sequential). "
                         "Higher values create KV cache pressure for eviction studies.")
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
    print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | "
          f"max_tokens={args.max_tokens} | concurrency={args.concurrency}")

    records = []
    records_lock = threading.Lock()
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(replay_conversation, ci, conv, args, records, records_lock, print_lock): ci
            for ci, conv in enumerate(convs)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                ci = futures[future]
                with print_lock:
                    print(f"  [conv {ci+1}] ERROR: {e}")

    records.sort(key=lambda r: (str(r["conv_id"]), r["turn"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[replay] {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
