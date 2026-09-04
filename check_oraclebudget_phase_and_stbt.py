import json, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times, parse_phase_schedule, phase_type_at

SCHED = "1.0:0.0@120,1.0:0.3@120"
sched = parse_phase_schedule(SCHED)
def ptype(t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"

recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnsoraclebudget-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)

# check phase of the worst gaps
worst_times = [270.9, 264.6, 254.4]
for t in worst_times:
    print(f"t={t}s -> phase={ptype(t)}")

s_gaps = []
w_gaps = []
for r in recs:
    ts, lat = r.get("ts"), r.get("latency")
    if ts is None or lat is None: continue
    tt = token_times(r)
    for j, ms in enumerate(r.get("tbt_ms") or []):
        emit = tt[j+1] if (j+1) < len(tt) else ts
        emit_rel = emit - t0
        (s_gaps if ptype(emit_rel) == "S" else w_gaps).append(ms)

s_gaps.sort(); w_gaps.sort()
def pctl(x, p):
    return x[min(len(x)-1, int(p/100*len(x)))]
print(f"\nS-phase (LO) TBT: n={len(s_gaps)} mean={sum(s_gaps)/len(s_gaps):.1f} p99={pctl(s_gaps,99):.1f} max={s_gaps[-1]:.1f}")
print(f"W-phase (HI) TBT: n={len(w_gaps)} mean={sum(w_gaps)/len(w_gaps):.1f} p99={pctl(w_gaps,99):.1f} max={w_gaps[-1]:.1f}")
