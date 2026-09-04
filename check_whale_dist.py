import json, statistics as st

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnswhaleawarev2-t1.jsonl") if l.strip()]
whales = [r["prompt_tokens_approx"] for r in recs if (r.get("prompt_tokens_approx") or 0) > 2500]
whales.sort()
print(f"n candidate-whale-range requests (>2500 tok): {len(whales)}")
print(f"min={whales[0]} p10={whales[int(0.1*len(whales))]} p50={whales[len(whales)//2]} p90={whales[int(0.9*len(whales))]} max={whales[-1]}")
below_4000 = [w for w in whales if w <= 4000]
print(f"\nrequests with prompt_tokens in (2500,4000] -- i.e. workload-scale whales the WHALE_AWARE_WHALE_TOK=4000 controller does NOT detect:")
print(f"  n={len(below_4000)}: {below_4000}")

all_whales = [r["prompt_tokens_approx"] for r in recs if (r.get("prompt_tokens_approx") or 0) > 4000]
print(f"\nfull >4000 (controller-detected) whale distribution: n={len(all_whales)}")
all_whales.sort()
print(f"  min={all_whales[0]} p10={all_whales[int(0.1*len(all_whales))]} p50={all_whales[len(all_whales)//2]} p90={all_whales[int(0.9*len(all_whales))]} max={all_whales[-1]}")
