import json

recs = [json.loads(l) for l in open("logs/2026-08-27-lgate-bfsbudget05-t1.jsonl") if l.strip()]
worst = None
for r in recs:
    tbt = r.get("tbt_ms") or []
    if tbt:
        m = max(tbt)
        if worst is None or m > worst[0]:
            worst = (m, r)

print("worst gap:", worst[0], "ms")
print("prompt_tokens_approx:", worst[1].get("prompt_tokens_approx"))
print("output_tokens:", worst[1].get("output_tokens"))
print("ttft:", worst[1].get("ttft"))
print("latency:", worst[1].get("latency"))

# how many of the top-10 worst-gap records are whales (prompt_tokens_approx > 4000)?
gaps_with_rec = []
for r in recs:
    tbt = r.get("tbt_ms") or []
    if tbt:
        gaps_with_rec.append((max(tbt), r.get("prompt_tokens_approx")))
gaps_with_rec.sort(key=lambda x: -x[0])
print("\ntop 10 worst per-record gaps -> (gap_ms, prompt_tokens_approx):")
for g, pt in gaps_with_rec[:10]:
    print(f"  {g:.1f}ms  prompt_tokens={pt}  {'WHALE' if pt and pt > 4000 else 'short'}")
