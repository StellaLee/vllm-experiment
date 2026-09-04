import json, csv

recs = [json.loads(l) for l in open("logs/2026-08-27-lgate-bfsreserve05-t1.jsonl") if l.strip()]
gaps_with_rec = []
for r in recs:
    tbt = r.get("tbt_ms") or []
    if tbt:
        gaps_with_rec.append((max(tbt), r.get("prompt_tokens_approx"), r.get("output_tokens")))
gaps_with_rec.sort(key=lambda x: -x[0])
print("top 10 worst per-record gaps -> (gap_ms, prompt_tokens_approx, output_tokens):")
for g, pt, ot in gaps_with_rec[:10]:
    print(f"  {g:.1f}ms  prompt_tokens={pt}  output_tokens={ot}  {'WHALE' if pt and pt > 4000 else 'short'}")

trace = list(csv.reader(open("logs/2026-08-27-lgate-bfsreserve05-chunktrace.csv")))
rows = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in trace if len(r) > 4]
# columns: wall_s, running, reserve, share, token_budget
dts = [(rows[i][0]-rows[i-1][0], i, rows[i-1][1], rows[i-1][2], rows[i-1][3], rows[i-1][4]) for i in range(1, len(rows))]
dts.sort(key=lambda x: -x[0])
print("\ntop 10 slowest individual rounds (dt_s, idx, running, reserve, share, budget):")
for d in dts[:10]:
    print(f"  dt={d[0]:.3f}s  running={d[2]}  reserve={d[3]}  share={d[4]}  budget={d[5]}")
