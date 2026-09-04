import csv

rows = list(csv.reader(open("logs/2026-08-28-lgate-bnsstatic512rt-roundtrace.csv")))
rounds = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in rows if len(r) > 4]
rounds.sort(key=lambda x: x[0])
t_start = rounds[0][0]
rel = [(r[0]-t_start, r[1], r[2], r[3], r[4]) for r in rounds]  # rel_s, n_running, tokens, n_skipped, n_whales

# worst gap was ~11.2s ending at client-relative emit_t=650.8s (conv Fy9JAep_79)
# approximate server-clock window: look at rounds in [635, 655]s (server-relative), a window
# wide enough to catch alignment slop between client epoch clock and server monotonic clock
window = [r for r in rel if 630 <= r[0] <= 656]
print(f"rounds in server-relative window [630,656]s: {len(window)}")
if window:
    dts = []
    for i in range(1, len(window)):
        dt = window[i][0] - window[i-1][0]
        dts.append(dt)
    tot_dt = sum(dts)
    print(f"sum of dt across this window = {tot_dt:.2f}s over {len(window)} rounds")
    print(f"mean tokens/round in window = {sum(w[2] for w in window)/len(window):.0f}")
    print(f"mean n_whales_active in window = {sum(w[4] for w in window)/len(window):.2f}")
    print(f"max n_whales_active in window = {max(w[4] for w in window)}")
    n_elevated = sum(1 for w in window if w[2] > 512)
    print(f"rounds with >512 tokens admitted: {n_elevated} / {len(window)}")
    print(f"\nfirst 40 rounds in window (rel_s, n_running, tokens, n_skipped, n_whales_active):")
    for w in window[:40]:
        print(f"  {w[0]:>8.2f}s  n_run={w[1]:>3}  tok={w[2]:>5}  skip={w[3]}  whales={w[4]}")
