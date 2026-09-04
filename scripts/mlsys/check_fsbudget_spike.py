import csv, json

trace = list(csv.reader(open("logs/2026-08-27-lgate-bfsbudget05-chunktrace.csv")))
rows = [(float(r[0]), int(r[1]), int(r[2])) for r in trace if len(r) > 2]
print("trace rows:", len(rows))

recs = [json.loads(l) for l in open("logs/2026-08-27-lgate-bfsbudget05-t1.jsonl") if l.strip()]
# find the single worst TBT gap across all records
worst = None
for r in recs:
    tbt = r.get("tbt_ms") or []
    if tbt:
        m = max(tbt)
        if worst is None or m > worst[0]:
            worst = (m, r)
print("worst single TBT gap (ms):", worst[0] if worst else None)

# correlate: budget distribution right before large TBT events (top 5 worst gaps overall)
all_gaps = []
for r in recs:
    for ms in (r.get("tbt_ms") or []):
        all_gaps.append(ms)
all_gaps.sort(reverse=True)
print("top 10 TBT gaps (ms):", all_gaps[:10])

# budget value distribution just to sanity check ceiling events
budgets = [b for _, _, b in rows]
ceiling_idx = [i for i, b in enumerate(budgets) if b >= 16384]
print("ceiling steps:", len(ceiling_idx), "of", len(budgets))
for i in ceiling_idx[:5]:
    lo = max(0, i - 3)
    hi = min(len(rows), i + 4)
    print("context around ceiling step", i, ":", rows[lo:hi])
