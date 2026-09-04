import json, csv, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times

# --- load round trace ---
rows = list(csv.reader(open("logs/2026-08-28-lgate-bnsstatic512rt-roundtrace.csv")))
rounds = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in rows if len(r) > 4]
# wall_s, n_running, total_tokens, n_skipped, n_whales_active
rounds.sort(key=lambda x: x[0])
print(f"n_rounds={len(rounds)}")

# --- Mechanism 1: does total_tokens_this_round predict round latency (dt)? ---
dts = []
for i in range(1, len(rounds)):
    dt = rounds[i][0] - rounds[i-1][0]
    tok = rounds[i-1][2]  # tokens admitted in the round that JUST finished (producing this dt)
    dts.append((dt, tok, rounds[i-1][3], rounds[i-1][4]))  # dt, tokens, n_skipped, n_whales_active

# simple correlation + binned means
import statistics as st
n = len(dts)
mean_dt = sum(d[0] for d in dts) / n
mean_tok = sum(d[1] for d in dts) / n
cov = sum((d[0]-mean_dt)*(d[1]-mean_tok) for d in dts) / n
sd_dt = (sum((d[0]-mean_dt)**2 for d in dts)/n) ** 0.5
sd_tok = (sum((d[1]-mean_tok)**2 for d in dts)/n) ** 0.5
corr = cov / (sd_dt*sd_tok) if sd_dt>0 and sd_tok>0 else float('nan')
print(f"\n=== Mechanism 1: dt vs tokens_this_round ===")
print(f"pearson corr(dt, tokens) = {corr:.3f}  (n={n})")

# bin by token volume, show mean dt per bin
bins = [(0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,20000)]
for lo,hi in bins:
    sel = [d[0] for d in dts if lo <= d[1] < hi]
    if sel:
        print(f"  tokens in [{lo},{hi}): n={len(sel)} mean_dt={sum(sel)/len(sel)*1000:.1f}ms max_dt={max(sel)*1000:.1f}ms")

# worst dt rounds
dts_sorted = sorted(dts, key=lambda x: -x[0])
print(f"\ntop 10 slowest rounds -> (dt_ms, tokens_admitted, n_skipped, n_whales_active):")
for d in dts_sorted[:10]:
    print(f"  dt={d[0]*1000:>8.1f}ms  tokens={d[1]:>6}  n_skipped={d[2]:>3}  n_whales_active={d[3]:>2}")

# --- Mechanism 2: does n_skipped correlate with n_whales_active? ---
print(f"\n=== Mechanism 2: n_skipped vs n_whales_active ===")
skip_vals = [r[3] for r in rounds]
whale_vals = [r[4] for r in rounds]
mean_skip = sum(skip_vals)/len(skip_vals)
mean_whale = sum(whale_vals)/len(whale_vals)
cov2 = sum((s-mean_skip)*(w-mean_whale) for s,w in zip(skip_vals,whale_vals))/len(rounds)
sd_skip = (sum((s-mean_skip)**2 for s in skip_vals)/len(rounds))**0.5
sd_whale = (sum((w-mean_whale)**2 for w in whale_vals)/len(rounds))**0.5
corr2 = cov2/(sd_skip*sd_whale) if sd_skip>0 and sd_whale>0 else float('nan')
print(f"pearson corr(n_skipped, n_whales_active) = {corr2:.3f}")
print(f"mean n_skipped overall = {mean_skip:.3f}, mean when n_whales_active>=5: ", end="")
hi_whale = [r[3] for r in rounds if r[4] >= 5]
print(f"{sum(hi_whale)/len(hi_whale):.3f}" if hi_whale else "n/a (no such rounds)")
print(f"rounds with n_whales_active>=5: {len(hi_whale)} / {len(rounds)}")
print(f"max n_skipped in any round: {max(skip_vals)}  (at n_whales_active={rounds[skip_vals.index(max(skip_vals))][4]})")

# --- Cross-check against the actual worst TBT gap event from t1.jsonl (t~646.7s) ---
recs = [json.loads(l) for l in open("logs/2026-08-28-lgate-bnsstatic512rt-t1.jsonl") if l.strip()]
starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
t0 = min(starts)
gaps = []
for r in recs:
    ts, lat = r.get("ts"), r.get("latency")
    if ts is None or lat is None: continue
    tt = token_times(r)
    for j, ms in enumerate(r.get("tbt_ms") or []):
        emit = tt[j+1] if (j+1) < len(tt) else ts
        gaps.append((ms, emit - t0, r.get("prompt_tokens_approx"), r.get("conv_id")))
gaps.sort(key=lambda x: -x[0])
print(f"\n=== Worst TBT gaps this replicate run, cross-referenced against round trace ===")
round_t0 = rounds[0][0]
for g in gaps[:5]:
    ms, emit_rel, ptok, cid = g
    emit_wall = emit_rel + t0  # this is in `ts` epoch units; need to map to round trace's monotonic clock
    print(f"  gap={ms:.1f}ms  emit_t_rel={emit_rel:.1f}s  prompt_tok={ptok}  conv={cid}")
