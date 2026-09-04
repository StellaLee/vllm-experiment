import json, csv, sys
sys.path.insert(0, "scripts")
from replay_timing import token_times

def roundtrace_stats(path):
    rows = list(csv.reader(open(path)))
    rounds = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in rows if len(r) > 4]
    rounds.sort(key=lambda x: x[0])
    dts = [(rounds[i][0]-rounds[i-1][0], rounds[i-1][2], rounds[i-1][3], rounds[i-1][4]) for i in range(1, len(rounds))]
    n = len(dts)
    mean_dt = sum(d[0] for d in dts)/n
    mean_tok = sum(d[1] for d in dts)/n
    cov = sum((d[0]-mean_dt)*(d[1]-mean_tok) for d in dts)/n
    sd_dt = (sum((d[0]-mean_dt)**2 for d in dts)/n)**0.5
    sd_tok = (sum((d[1]-mean_tok)**2 for d in dts)/n)**0.5
    r1 = cov/(sd_dt*sd_tok) if sd_dt>0 and sd_tok>0 else float('nan')

    skip_vals = [r[3] for r in rounds]
    whale_vals = [r[4] for r in rounds]
    mean_skip = sum(skip_vals)/len(skip_vals)
    mean_whale = sum(whale_vals)/len(whale_vals)
    cov2 = sum((s-mean_skip)*(w-mean_whale) for s,w in zip(skip_vals,whale_vals))/len(rounds)
    sd_skip = (sum((s-mean_skip)**2 for s in skip_vals)/len(rounds))**0.5
    sd_whale = (sum((w-mean_whale)**2 for w in whale_vals)/len(rounds))**0.5
    r2 = cov2/(sd_skip*sd_whale) if sd_skip>0 and sd_whale>0 else float('nan')

    return dict(n_rounds=len(rounds), r_dt_tokens=r1, r_skip_whale=r2,
                mean_skip=mean_skip, max_skip=max(skip_vals))

def clustering_ratio(t1_path, window=60.0, whale_tok=4000):
    recs = [json.loads(l) for l in open(t1_path) if l.strip()]
    starts = [r["ts"]-r["latency"] for r in recs if r.get("ts") is not None and r.get("latency") is not None]
    t0 = min(starts)
    whale_arrivals = sorted(
        (r["ts"]-r["latency"]-t0) for r in recs
        if r.get("ts") is not None and r.get("latency") is not None
        and (r.get("prompt_tokens_approx") or 0) > whale_tok
    )
    def whales_before(t_rel):
        return sum(1 for w in whale_arrivals if t_rel - window < w <= t_rel)
    gaps = []
    for r in recs:
        ts, lat = r.get("ts"), r.get("latency")
        if ts is None or lat is None: continue
        tt = token_times(r)
        for j, ms in enumerate(r.get("tbt_ms") or []):
            emit_abs = tt[j+1] if (j+1) < len(tt) else ts
            gaps.append((ms, emit_abs - t0))
    gaps.sort(key=lambda x: -x[0])
    n = len(gaps)
    top_n = max(1, int(0.001*n))
    baseline = sum(whales_before(g[1]) for g in gaps)/n
    top = sum(whales_before(g[1]) for g in gaps[:top_n])/top_n
    return dict(n_whales=len(whale_arrivals), baseline=baseline, top=top,
                ratio=(top/baseline if baseline>0 else float('nan')))

trials = ["rt", "rt2", "rt3"]
rt_results = []
cl_results = []
for t in trials:
    rtp = f"logs/2026-08-28-lgate-bnsstatic512{t}-roundtrace.csv"
    t1p = f"logs/2026-08-28-lgate-bnsstatic512{t}-t1.jsonl"
    rt_results.append(roundtrace_stats(rtp))
    cl_results.append(clustering_ratio(t1p))

print(f"{'trial':>6} {'n_rounds':>9} {'r(dt,tok)':>11} {'r(skip,whale)':>14} {'mean_skip':>10} {'max_skip':>9}")
for t, r in zip(trials, rt_results):
    print(f"{t:>6} {r['n_rounds']:>9} {r['r_dt_tokens']:>11.3f} {r['r_skip_whale']:>14.4f} {r['mean_skip']:>10.4f} {r['max_skip']:>9}")

import statistics as st
r1s = [r['r_dt_tokens'] for r in rt_results]
r2s = [r['r_skip_whale'] for r in rt_results]
print(f"\nr(dt,tokens):     mean={st.mean(r1s):.3f}  std={st.pstdev(r1s):.4f}")
print(f"r(skip,whale):    mean={st.mean(r2s):.4f}  std={st.pstdev(r2s):.4f}")

print(f"\n{'trial':>6} {'n_whales':>9} {'baseline':>9} {'top0.1%':>9} {'ratio':>7}")
for t, c in zip(trials, cl_results):
    print(f"{t:>6} {c['n_whales']:>9} {c['baseline']:>9.2f} {c['top']:>9.2f} {c['ratio']:>7.2f}")
ratios = [c['ratio'] for c in cl_results]
print(f"\nclustering ratio: mean={st.mean(ratios):.2f}  std={st.pstdev(ratios):.2f}")
