import json
recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnsstatic512-t1.jsonl") if l.strip()]
starts = [r["ts"] - r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)
window = [r for r in recs if r.get("ts") is not None and r.get("latency") is not None and 631 <= (r["ts"]-r["latency"]-t0) <= 661]
print(f"n arrived in [631,661]s window: {len(window)}")
whales = [r for r in window if (r.get("prompt_tokens_approx") or 0) > 4000]
print(f"whales arrived in that window: {len(whales)}")
for w in whales:
    print(f"  {w['conv_id']}  prompt_tok={w['prompt_tokens_approx']}")
# also: whales that arrived BEFORE the window but might still be mid-processing
earlier_whales = [r for r in recs if (r.get("prompt_tokens_approx") or 0) > 4000
                   and r.get("ts") is not None and r.get("latency") is not None
                   and 560 <= (r["ts"]-r["latency"]-t0) < 631]
print(f"\nwhales arrived in [560,631)s (could still be mid-processing at 646.7s): {len(earlier_whales)}")
for w in earlier_whales:
    print(f"  {w['conv_id']}  prompt_tok={w['prompt_tokens_approx']}  arrival_t={w['ts']-w['latency']-t0:.1f}s  own_ttft={w.get('ttft')}")
