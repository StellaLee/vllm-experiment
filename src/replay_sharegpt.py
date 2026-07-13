#!/usr/bin/env python3
"""
Replay ShareGPT multi-turn conversations against vLLM OpenAI-compatible server.
Sends each turn with accumulated conversation history as prefix so vLLM's
KV cache sees the shared prefix across turns.

With --concurrency N, N conversations run in parallel (turns within each
conversation remain sequential so accumulated prefixes stay consistent).

With --rate R, conversations are submitted as a Poisson process at R conv/s
(open-loop). --concurrency is ignored when --rate is set. This models
realistic arrival patterns rather than closed-loop thundering-herd.

Uses /v1/completions with a flat text prompt built from conversation history.
This guarantees the same token prefix is presented on each successive turn,
which is what makes prefix caching effective for multi-turn workloads.

Output: JSONL with one record per request, including conv_id, turn number,
prompt length, TTFT, and total latency.
"""
import argparse, json, math, os, random, time, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def sample_pad_len(ci, turn_num, args):
    """Per-request pad length (chars). Deterministic in (pad_seed, ci, turn_num) so
    the SAME length sequence is replayed against every policy arm (paired comparison).

    Two modes:
      * fixed  (--pad-chars N)                 -> uniform, Cs^2 = 0  (legacy)
      * varied (--pad-mean-chars M --pad-cv2 C)-> lognormal with mean M and squared
                                                  coefficient of variation C, so we can
                                                  sweep prefill-size dispersion directly.
    Lognormal: sigma^2 = ln(1+C), mu = ln(M) - sigma^2/2  => E[L]=M, Var/E^2 = C.
    C=0 collapses to the constant M. Draw is clamped to [pad_min, pad_max]; clamping
    slightly reduces the achieved Cs^2 at large C, so we log the realized pad_chars and
    recompute Cs^2 empirically in analysis rather than trusting the nominal value.
    """
    if args.pad_mean_chars and args.pad_mean_chars > 0:
        rng = random.Random(f"{args.pad_seed}-{ci}-{turn_num}")
        m, c = float(args.pad_mean_chars), float(args.pad_cv2)
        if c <= 0:
            L = m
        else:
            s2 = math.log(1.0 + c)
            mu = math.log(m) - 0.5 * s2
            L = math.exp(rng.gauss(mu, math.sqrt(s2)))
        return int(max(args.pad_min, min(args.pad_max, L)))
    return int(getattr(args, "pad_chars", 0) or 0)


def stream_request(host, port, prompt, max_tokens, model):
    """POST to /v1/completions with stream=True. Returns (ttft_s, total_s, output_tokens).

    OpenAI SSE format:
      data: {"choices": [{"text": "..."}]}
    TTFT = time until first non-empty text chunk arrives.
    output_tokens = real completion_tokens from the server usage chunk
    (stream_options.include_usage); falls back to a word-count proxy if absent.
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # Ask the server to append a final usage chunk with real token counts,
        # so TPOT uses actual completion_tokens, not a word-count proxy.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    output_parts = []
    completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    obj = json.loads(line)
                    usage = obj.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        completion_tokens = usage["completion_tokens"]
                    text = ""
                    choices = obj.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "") or ""
                    if text:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        output_parts.append(text)
                except json.JSONDecodeError:
                    pass
        total = time.monotonic() - t0
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e}")
    output_text = "".join(output_parts)
    # Prefer the server's real token count; fall back to a word-count proxy
    # only if the usage chunk was absent.
    word_tokens = max(1, len(output_text.split()))
    tokens_exact = completion_tokens is not None and completion_tokens > 0
    output_tokens = completion_tokens if tokens_exact else word_tokens
    return ttft, total, output_tokens, tokens_exact


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
        pad_len = sample_pad_len(ci, turn_num, args)
        if pad_len > 0:
            # Unique filler -> un-cacheable large prefill. Tag is deterministic and
            # unique per (conv,turn): identical across arms (paired), distinct across
            # requests (no cross-request prefix sharing).
            uniq = f"[req {ci}.{turn_num}] "
            unit = uniq + "The quick brown fox jumps over the lazy dog. "
            pad = (unit * (pad_len // len(unit) + 1))[:pad_len]
            prompt = pad + "\n\n" + prompt

        try:
            ttft, total, output_tokens, tokens_exact = stream_request(args.host, args.port, prompt, args.max_tokens, args.model)
            gpt_placeholder = f"[turn {turn_num+1} response]"
            if i + 1 < len(turns) and turns[i + 1].get("from") == "gpt":
                gpt_placeholder = turns[i + 1]["value"].strip()[:256]

            decode_s = total - (ttft or 0)
            tpot = round(decode_s / output_tokens, 5) if output_tokens > 0 else None

            record = {
                "conv_id": conv.get("id", ci),
                "turn": turn_num + 1,
                "history_turns": len(history),
                "prompt_tokens_approx": len(prompt.split()),
                "pad_chars": pad_len,
                "output_tokens": output_tokens,
                "tokens_exact": tokens_exact,
                "ttft": round(ttft, 4) if ttft is not None else None,
                "tpot": tpot,
                "latency": round(total, 4),
                "ts": time.time(),
            }
            with records_lock:
                records.append(record)

            history.append((human_msg, gpt_placeholder))
            ttft_str = f"{ttft:.3f}" if ttft is not None else "N/A"
            tpot_str = f"{tpot*1000:.1f}ms" if tpot is not None else "N/A"
            with print_lock:
                print(f"  [conv {ci+1}] turn {turn_num+1}: ttft={ttft_str}s  tpot={tpot_str}/tok  lat={total:.3f}s")

        except RuntimeError as e:
            with print_lock:
                print(f"  [conv {ci+1}] turn {turn_num+1} SKIP: {e}")

        turn_num += 1
        i += 2 if (i + 1 < len(turns) and turns[i + 1].get("from") == "gpt") else 1


def main():
    ap = argparse.ArgumentParser(description="Replay ShareGPT conversations against vLLM")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="/model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="Model name as registered with the server")
    ap.add_argument("--dataset", required=True, help="ShareGPT JSON file (list of conversations)")
    ap.add_argument("--num-convs", type=int, default=50, help="Conversations to replay")
    ap.add_argument("--max-turns", type=int, default=4, help="Max turns per conversation")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel conversations (closed-loop). Ignored when --rate is set.")
    ap.add_argument("--rate", type=float, default=None,
                    help="Conversation arrival rate in conv/s (open-loop Poisson). "
                         "When set, conversations are submitted at Poisson-distributed "
                         "inter-arrival times instead of closed-loop concurrency.")
    ap.add_argument("--stagger-window", type=float, default=None,
                    help="Staggered closed-loop: with --concurrency N, spread the "
                         "initial N worker starts over this many seconds instead of "
                         "all at t=0, then maintain concurrency N. Removes the t=0 "
                         "thundering herd. Ignored when --rate is set.")
    ap.add_argument("--min-turns", type=int, default=1,
                    help="Only include conversations with at least this many human turns. "
                         "Use --min-turns 4 to guarantee all conversations reach turn 4.")
    ap.add_argument("--output", required=True, help="JSONL output path for per-request records")
    ap.add_argument("--pad-chars", type=int, default=0,
                    help="Prepend N chars of UNIQUE filler to each prompt to force a "
                         "large, un-cacheable prefill (fixed => Cs^2=0). 0=off.")
    ap.add_argument("--pad-mean-chars", type=int, default=0,
                    help="Variable-padding mode: target MEAN pad chars. When set, "
                         "per-request pad length is lognormal(mean, cv2), overriding "
                         "--pad-chars. Lets us sweep prefill-size dispersion (Cs^2).")
    ap.add_argument("--pad-cv2", type=float, default=0.0,
                    help="Target squared coefficient of variation of pad length "
                         "(0=uniform; >1=heavy-tailed). Used with --pad-mean-chars.")
    ap.add_argument("--pad-seed", type=int, default=12345,
                    help="Seed for the deterministic per-request pad-length draw "
                         "(same lengths across policy arms for a paired comparison).")
    ap.add_argument("--pad-min", type=int, default=200,
                    help="Clamp floor for variable pad length (chars).")
    ap.add_argument("--pad-max", type=int, default=40000,
                    help="Clamp ceiling for variable pad length (chars).")
    args = ap.parse_args()

    with open(args.dataset) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        data = list(raw.values())
    else:
        data = raw

    def human_turn_count(c):
        return sum(1 for t in c.get("conversations", []) if t.get("from") == "human")

    convs = [
        c for c in data
        if isinstance(c.get("conversations"), list)
        and human_turn_count(c) >= max(2, args.min_turns)
    ]
    if not convs:
        raise SystemExit("ERROR: no valid conversations found matching --min-turns filter.")

    convs = convs[:args.num_convs]
    print(f"[replay] filtered to {len(convs)} conversations (min_turns={args.min_turns})")

    if args.rate:
        print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | "
              f"max_tokens={args.max_tokens} | rate={args.rate} conv/s (open-loop Poisson)")
    elif args.stagger_window is not None:
        print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | "
              f"max_tokens={args.max_tokens} | concurrency={args.concurrency} "
              f"(staggered closed-loop, window={args.stagger_window}s)")
    else:
        print(f"[replay] {len(convs)} conversations | max_turns={args.max_turns} | "
              f"max_tokens={args.max_tokens} | concurrency={args.concurrency} (closed-loop, herd)")

    records = []
    records_lock = threading.Lock()
    print_lock = threading.Lock()

    if args.rate:
        # Open-loop Poisson arrival: spawn each conversation thread after an
        # exponentially-distributed inter-arrival delay. Max workers is capped
        # high (num_convs) so no conversation is ever blocked waiting for a slot.
        threads = []
        for ci, conv in enumerate(convs):
            t = threading.Thread(
                target=replay_conversation,
                args=(ci, conv, args, records, records_lock, print_lock),
                daemon=True,
            )
            threads.append(t)
            t.start()
            if ci < len(convs) - 1:
                # Exponential inter-arrival time; mean = 1/rate
                time.sleep(random.expovariate(args.rate))
        for t in threads:
            t.join()
    elif args.stagger_window is not None:
        # Staggered closed-loop: N worker threads pull conversations from a shared
        # queue (maintaining concurrency N), but their initial starts are spread
        # over the window instead of all firing at t=0 -- removes the thundering
        # herd while preserving the steady-state c=N population.
        from queue import Queue, Empty
        work = Queue()
        for ci, conv in enumerate(convs):
            work.put((ci, conv))
        offset = args.stagger_window / max(1, args.concurrency)

        def _worker(widx):
            time.sleep(widx * offset)
            while True:
                try:
                    ci, conv = work.get_nowait()
                except Empty:
                    return
                replay_conversation(ci, conv, args, records, records_lock, print_lock)

        threads = [threading.Thread(target=_worker, args=(w,), daemon=True)
                   for w in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        # Closed-loop: fixed concurrency via ThreadPoolExecutor
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
