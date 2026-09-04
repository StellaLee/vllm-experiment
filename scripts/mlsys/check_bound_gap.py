import json, csv, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnswhaleawarev2-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)

gaps = []
for r in recs:
    ts, lat = r.get("ts"), r.get("latency")
    if ts is None or lat is None: continue
    tt = token_times(r)
    for j, ms in enumerate(r.get("tbt_ms") or []):
        emit = tt[j+1] if (j+1) < len(tt) else ts
        gaps.append((ms, emit - t0))
gaps.sort(key=lambda x: -x[0])

# load whale trace: wall_s, n_whales_active, token_budget
rows = list(csv.reader(open("logs/2026-08-28-lgate-bnswhaleawarev2-whaletrace.csv")))
trace = [(float(r[0]), int(r[1]), int(r[2])) for r in rows if len(r) > 2]
trace.sort(key=lambda x: x[0])
trace_t0 = trace[0][0]
trace_rel = [(t-trace_t0, nw, tb) for t, nw, tb in trace]

def nearest_state(t_rel, window=0.5):
    # find trace entries within +-window seconds of t_rel (client clock vs server clock offset unknown, use relative)
    cands = [x for x in trace_rel if abs(x[0]-t_rel) < window]
    return cands

print("top 10 worst gaps -> (gap_ms, t_rel, n_whales_active range, token_budget range nearby)")
for g, t in gaps[:10]:
    near = nearest_state(t, window=1.0)
    if near:
        nws = sorted(set(x[1] for x in near))
        tbs = sorted(set(x[2] for x in near))
        print(f"  {g:>8.1f}ms  t={t:>7.1f}s  n_whales~{nws}  token_budget~{tbs}  (n_trace_pts={len(near)})")
    else:
        print(f"  {g:>8.1f}ms  t={t:>7.1f}s  NO TRACE MATCH NEARBY")
