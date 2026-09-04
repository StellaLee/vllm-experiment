import csv, sys

def analyze(path, label):
    rows = list(csv.reader(open(path)))
    rounds = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in rows if len(r) > 4]
    rounds.sort(key=lambda x: x[0])
    dts = []
    for i in range(1, len(rounds)):
        dt = rounds[i][0] - rounds[i-1][0]
        tok = rounds[i-1][2]
        dts.append((dt, tok, rounds[i-1][3], rounds[i-1][4]))
    n = len(dts)
    mean_dt = sum(d[0] for d in dts)/n
    mean_tok = sum(d[1] for d in dts)/n
    cov = sum((d[0]-mean_dt)*(d[1]-mean_tok) for d in dts)/n
    sd_dt = (sum((d[0]-mean_dt)**2 for d in dts)/n)**0.5
    sd_tok = (sum((d[1]-mean_tok)**2 for d in dts)/n)**0.5
    corr = cov/(sd_dt*sd_tok) if sd_dt>0 and sd_tok>0 else float('nan')

    skip_vals = [r[3] for r in rounds]
    whale_vals = [r[4] for r in rounds]
    mean_skip = sum(skip_vals)/len(skip_vals)
    mean_whale = sum(whale_vals)/len(whale_vals)
    cov2 = sum((s-mean_skip)*(w-mean_whale) for s,w in zip(skip_vals,whale_vals))/len(rounds)
    sd_skip = (sum((s-mean_skip)**2 for s in skip_vals)/len(rounds))**0.5
    sd_whale = (sum((w-mean_whale)**2 for w in whale_vals)/len(rounds))**0.5
    corr2 = cov2/(sd_skip*sd_whale) if sd_skip>0 and sd_whale>0 else float('nan')

    print(f"{label}: n_rounds={len(rounds)}  r(dt,tokens)={corr:.3f}  r(n_skipped,n_whales)={corr2:.4f}  mean_skip={mean_skip:.4f}  max_skip={max(skip_vals)}")

for tag, path in [("trial1 (rt)", "logs/2026-08-28-lgate-bnsstatic512rt-roundtrace.csv"),
                   ("trial2 (rt2)", "logs/2026-08-28-lgate-bnsstatic512rt2-roundtrace.csv")]:
    analyze(path, tag)
