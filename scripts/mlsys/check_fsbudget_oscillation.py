import csv

trace = list(csv.reader(open("logs/2026-08-27-lgate-bfsbudget05-chunktrace.csv")))
rows = [(float(r[0]), int(r[1]), int(r[2])) for r in trace if len(r) > 2]
budgets = [b for _, _, b in rows]

jumps = []
for i in range(1, len(budgets)):
    prev, cur = budgets[i-1], budgets[i]
    if prev > 0:
        ratio = cur / prev
        jumps.append((ratio, i, prev, cur, rows[i][0] - rows[i-1][0]))

jumps.sort(key=lambda x: -x[0])
print("top 10 biggest step-to-step JUMPS UP (ratio, idx, prev_budget, cur_budget, wall_dt_s):")
for j in jumps[:10]:
    print(f"  ratio={j[0]:.1f}x  prev={j[2]}  cur={j[3]}  dt={j[4]*1000:.1f}ms  idx={j[1]}")

# how many steps have >2x jump either direction
big_jumps = sum(1 for r,_,_,_,_ in jumps if r > 2 or r < 0.5)
print(f"\nsteps with >2x budget jump (either direction): {big_jumps} / {len(jumps)} ({100*big_jumps/len(jumps):.1f}%)")

# also: longest single-step wall-clock duration (dt between consecutive trace rows = proxy for round time)
by_dt = sorted(rows_dt := [(rows[i][0]-rows[i-1][0], i, budgets[i-1], budgets[i]) for i in range(1,len(rows))], key=lambda x: -x[0])
print("\ntop 10 SLOWEST individual rounds (wall_dt_s, idx, prev_budget, cur_budget):")
for dt, i, pb, cb in by_dt[:10]:
    print(f"  dt={dt:.3f}s  idx={i}  prev_budget={pb}  cur_budget={cb}  running_before={rows[i-1][1]}")
