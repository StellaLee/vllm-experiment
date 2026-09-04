import json, sys

def analyze(path, label):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    whales = [r for r in recs if r.get("prompt_tokens_approx", 0) > 4000]
    whales.sort(key=lambda r: r["ts"])
    print(f"\n=== {label}: n_whales={len(whales)} ===")

    # finish-of-prefill time proxy = ts + ttft (ttft assumed seconds)
    for w in whales:
        w["prefill_done"] = w["ts"] + w["ttft"]

    # FCFS-order violation check: does a later-arriving whale finish prefill
    # before an earlier-arriving whale that's still within its own prefill window?
    violations = 0
    total_pairs = 0
    for i in range(len(whales)):
        for j in range(i + 1, len(whales)):
            a, b = whales[i], whales[j]
            # b arrived after a; if a hasn't finished prefill by the time b arrives,
            # both are concurrently "in the system" -- check who finishes prefill first
            if b["ts"] < a["prefill_done"]:
                total_pairs += 1
                if b["prefill_done"] < a["prefill_done"]:
                    violations += 1
    print(f"  concurrent whale pairs: {total_pairs}, order violations (later arrival finishes prefill first): {violations} ({100*violations/total_pairs:.1f}%)" if total_pairs else "  no concurrent whale pairs")

    # TTFT vs concurrency-at-arrival: for each whale, count how many OTHER whales
    # were still mid-prefill (arrived earlier, not yet prefill_done) at its arrival ts.
    print(f"  {'ttft_s':>8} {'prompt_tok':>10} {'concurrent_whales_ahead':>24} {'ttft_per_1k_tok':>16}")
    rows = []
    for w in whales:
        ahead = sum(1 for o in whales if o is not w and o["ts"] < w["ts"] < o["prefill_done"])
        rows.append((w["ttft"], w["prompt_tokens_approx"], ahead, w["ttft"] / (w["prompt_tokens_approx"] / 1000)))
    rows.sort(key=lambda r: -r[2])
    for r in rows[:15]:
        print(f"  {r[0]:>8.2f} {r[1]:>10} {r[2]:>24} {r[3]:>16.2f}")

    # correlation-ish: bucket by concurrency-ahead, show mean ttft_per_1k_tok
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[2]].append(r[3])
    print("  concurrency_ahead -> mean ttft_per_1k_tok (n)")
    for k in sorted(buckets):
        v = buckets[k]
        print(f"    {k:>3} -> {sum(v)/len(v):>8.2f}  (n={len(v)})")

for path, label in [
    ("logs/2026-08-27-lgate-b512fsb05-t1.jsonl", "512fsb05"),
    ("logs/2026-08-27-lgate-b512fsb10-t1.jsonl", "512fsb10"),
]:
    analyze(path, label)
