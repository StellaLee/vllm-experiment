import json, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnsoraclebudget-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)

gaps = []
for r in recs:
    ts, lat = r.get("ts"), r.get("latency")
    if ts is None or lat is None: continue
    start = ts - lat
    tt = token_times(r)
    for j, ms in enumerate(r.get("tbt_ms") or []):
        emit = tt[j+1] if (j+1) < len(tt) else ts
        gaps.append((ms, emit - t0, r.get("prompt_tokens_approx"), r.get("conv_id")))
gaps.sort(key=lambda x: -x[0])
print("top 15 worst gaps -> (gap_ms, emit_t_rel, prompt_tok, conv_id):")
for g in gaps[:15]:
    print(f"  {g[0]:>8.1f}ms  t={g[1]:>7.1f}s  prompt_tok={g[2]:>6}  conv={g[3]}")

print(f"\nn_gaps={len(gaps)}  mean={sum(g[0] for g in gaps)/len(gaps):.2f}ms")
big = [g for g in gaps if g[0] > 500]
print(f"gaps >500ms: {len(big)} ({100*len(big)/len(gaps):.3f}%)")
