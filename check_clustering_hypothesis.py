import json, sys, os
sys.path.insert(0, "scripts")
from replay_timing import token_times

WHALE_TOK = 4000
CLUSTER_WINDOW_S = 60.0

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def analyze(path, label):
    recs = load(path)
    starts = [(r["ts"] - r["latency"]) for r in recs if r.get("ts") is not None and r.get("latency") is not None]
    if not starts:
        print(f"{label}: no data"); return
    t0 = min(starts)
    whale_arrivals = sorted(
        (r["ts"] - r["latency"] - t0) for r in recs
        if r.get("ts") is not None and r.get("latency") is not None
        and (r.get("prompt_tokens_approx") or 0) > WHALE_TOK
    )

    def whales_before(t_rel, window=CLUSTER_WINDOW_S):
        return sum(1 for w in whale_arrivals if t_rel - window < w <= t_rel)

    gaps = []
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None:
            continue
        start_abs = ts - lat
        tt = token_times(r)  # absolute timestamps
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit_abs = tt[j + 1] if (j + 1) < len(tt) else ts
            emit_rel = emit_abs - t0
            gaps.append((ms, emit_rel))

    if not gaps:
        print(f"{label}: no tbt gaps"); return
    gaps.sort(key=lambda x: -x[0])
    n = len(gaps)
    top_n = max(1, int(0.001 * n))
    top_gaps = gaps[:top_n]
    baseline_avg = sum(whales_before(g[1]) for g in gaps) / n
    top_avg = sum(whales_before(g[1]) for g in top_gaps) / len(top_gaps)
    ratio = (top_avg / baseline_avg) if baseline_avg > 0 else float('inf') if top_avg > 0 else float('nan')
    print(f"{label}: n_gaps={n} n_whales={len(whale_arrivals)} "
          f"baseline_whales_in_{int(CLUSTER_WINDOW_S)}s={baseline_avg:.3f} "
          f"top0.1%worst_whales_in_{int(CLUSTER_WINDOW_S)}s={top_avg:.3f} ratio={ratio}")
    print(f"   top 5 worst gaps' local whale-cluster counts: {[(round(g[0],1), whales_before(g[1])) for g in gaps[:5]]}")

candidates = {
    "nsstatic512 (oracle-probe workload)": "logs/2026-08-28-lgate-bnsstatic512-t1.jsonl",
    "512fsb05": "logs/2026-08-27-lgate-b512fsb05-t1.jsonl",
    "512fsb10": "logs/2026-08-27-lgate-b512fsb10-t1.jsonl",
}
for label, path in candidates.items():
    if os.path.exists(path):
        analyze(path, label)
    else:
        print(f"{label}: file not found ({path})")
