import json

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnswhaleawarev2-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)

# find whales that arrived before t=270.9 and check when THEY finished prefill (arrival + ttft)
whales = [r for r in recs if (r.get("prompt_tokens_approx") or 0) > 4000
          and r.get("ts") is not None and r.get("latency") is not None]
for r in whales:
    arr = r["ts"] - r["latency"] - t0
    ttft = r.get("ttft")
    if ttft is None: continue
    prefill_done = arr + ttft
    if 200 <= arr <= 271 and abs(prefill_done - 270.9) < 5:
        print(f"  whale arrival_t={arr:.2f}s  prompt_tok={r['prompt_tokens_approx']}  ttft={ttft:.2f}s  prefill_done_t={prefill_done:.2f}s  <-- near t=270.9")

print("\nall whales arriving in [180,271]s with their prefill-done time:")
for r in whales:
    arr = r["ts"] - r["latency"] - t0
    ttft = r.get("ttft")
    if ttft is None or not (180 <= arr <= 271): continue
    print(f"  arrival_t={arr:.2f}s  prompt_tok={r['prompt_tokens_approx']}  ttft={ttft:.2f}s  prefill_done_t={arr+ttft:.2f}s")
