import json

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnswhaleawarev2-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)

window = [r for r in recs if r.get("ts") is not None and r.get("latency") is not None
          and 265 <= (r["ts"]-r["latency"]-t0) <= 271]
print(f"requests arriving in [265,271]s window: {len(window)}")
window.sort(key=lambda r: r["ts"]-r["latency"]-t0)
for r in window:
    pt = r.get("prompt_tokens_approx")
    print(f"  arrival_t={r['ts']-r['latency']-t0:.2f}s  prompt_tok={pt}  {'WHALE' if pt and pt>4000 else 'sub-whale'}")
total_sub_whale_tokens = sum(r.get("prompt_tokens_approx") or 0 for r in window if (r.get("prompt_tokens_approx") or 0) <= 4000)
print(f"\nsum of sub-whale prompt tokens in window: {total_sub_whale_tokens}")
