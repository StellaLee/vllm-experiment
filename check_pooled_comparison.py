import json, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times

def pooled_stats(path, label):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    gaps = []
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None: continue
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            gaps.append(ms)
    gaps.sort()
    n = len(gaps)
    def pctl(p): return gaps[min(n-1, int(p/100*n))]
    print(f"{label}: n={n}  mean={sum(gaps)/n:.1f}  p99={pctl(99):.1f}  p99.9={pctl(99.9):.1f}  max={gaps[-1]:.1f}")

for label, path in [
    ("mono          ", "logs/2026-08-28-lgate-bnsmono-t1.jsonl"),
    ("static512     ", "logs/2026-08-28-lgate-bnsstatic512-t1.jsonl"),
    ("oracle-budget ", "logs/2026-08-28-lgate-bnsoraclebudget-t1.jsonl"),
]:
    pooled_stats(path, label)
