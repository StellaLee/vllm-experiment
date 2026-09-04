import json, sys, os
sys.path.insert(0, "scripts")
from replay_timing import parse_phase_schedule, phase_type_at, token_times

SCHED = "1.0:0.0@120,1.0:0.3@120"
sched = parse_phase_schedule(SCHED)

def ptype(t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"

def analyze(path, label):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    starts = [r["ts"] - r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
    t0 = min(starts)

    gaps = []  # (gap_ms, arrival_phase, emit_phase, emit_t_rel, prompt_tokens, conv_id, idx_in_record)
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None:
            continue
        start = ts - lat
        arr_phase = ptype(start - t0)
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit = tt[j + 1] if (j + 1) < len(tt) else start
            emit_rel = emit - t0
            em_phase = ptype(emit_rel)
            if em_phase == "W":
                gaps.append((ms, arr_phase, em_phase, emit_rel, r.get("prompt_tokens_approx"), r.get("conv_id"), j))

    gaps.sort(key=lambda x: -x[0])
    n_w = len(gaps)
    n_crossing = sum(1 for g in gaps if g[1] != g[2])
    print(f"\n=== {label}: W-bucket gaps={n_w}, arrival!=emission phase (crossing)={n_crossing} ({100*n_crossing/n_w:.2f}%)")
    print(f"  top 15 worst W-bucket gaps -> (gap_ms, arrival_phase, emit_t_rel, prompt_tok, conv_id, tok_idx):")
    for g in gaps[:15]:
        print(f"    {g[0]:>9.1f}ms  arr={g[1]}  emit_t={g[3]:>7.1f}s  prompt_tok={g[4]:>6}  conv={g[5]}  tok_idx={g[6]}")

for path, label in [
    ("logs/2026-08-28-lgate-bnsmono-t1.jsonl", "nsmono"),
    ("logs/2026-08-28-lgate-bnsstatic512-t1.jsonl", "nsstatic512"),
]:
    analyze(path, label)
