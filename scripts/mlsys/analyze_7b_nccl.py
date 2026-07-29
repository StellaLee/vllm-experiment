#!/usr/bin/env python3
"""7B NCCL control analyzer. Compares mono-vs-chunk chunk Δ% between:
  SINGLE  = 7B single-GPU (TP=1, no NCCL)
  TP2     = 7B TP=2 (host-staged NCCL)
Files tagged cs2replSINGLE-* / cs2replTP2-*. Per Cs2, paired per-trial chunk Δ%
(mean/p50/p95), mean±std over trials, plus the NCCL effect = TP2 − SINGLE
(more positive => NCCL taxes chunk). Also raw chunk-arm latency to expose per-step
comms overhead directly."""
import json, glob, statistics as st, sys

R_SINGLE = sys.argv[1] if len(sys.argv) > 1 else "?"
R_TP2    = sys.argv[2] if len(sys.argv) > 2 else "?"
GRID = [("0","~0.01"),("2","~2.0"),("4","~4.0")]
TRIALS = ["1","2","3"]

def load(p):
    return sorted(r["ttft"]*1000 for r in (json.loads(l) for l in open(p) if l.strip())
                  if r.get("ttft") is not None)
def pct(xs,p):
    return xs[min(len(xs)-1, int(p/100*len(xs)))] if xs else float("nan")
def m_of(xs, metric):
    return {"mean":st.mean, "p50":st.median, "p95":lambda v:pct(v,95)}[metric](xs)

def deltas(tag, cv, metric):
    ds=[]
    for tr in TRIALS:
        mp=glob.glob(f"logs/*cs2repl{tag}-mono-cv{cv}-t{tr}.jsonl")
        cp=glob.glob(f"logs/*cs2repl{tag}-chunk-cv{cv}-t{tr}.jsonl")
        if not mp or not cp: continue
        m,c=load(mp[0]),load(cp[0])
        if not m or not c: continue
        mv,cvv=m_of(m,metric),m_of(c,metric)
        ds.append((cvv-mv)/mv*100)
    return ds
def fmt(ds):
    return f"{st.mean(ds):+6.1f}±{(st.pstdev(ds) if len(ds)>1 else 0):>4.1f}" if ds else " n/a"

print("="*92)
print(f"7B NCCL CONTROL (matched ρ≈0.8) — chunk Δ%: SINGLE-GPU rate {R_SINGLE} (no NCCL) vs TP=2 rate {R_TP2} (host-staged NCCL)")
print("NEG = chunk wins.  NCCL effect = TP2 − SINGLE (POSITIVE => host-staged NCCL taxes chunk)")
print("="*92)
for metric in ("mean","p50","p95"):
    print(f"\n--- metric: {metric} ---")
    print(f"{'Cs2':>6} | {'SINGLE Δ%':>13} | {'TP2 Δ%':>13} | {'NCCL effect':>14}")
    for cv,cs in GRID:
        s=deltas("SINGLE",cv,metric); t=deltas("TP2",cv,metric)
        eff = f"{st.mean(t)-st.mean(s):+.1f} pts" if s and t else "n/a"
        print(f"{cs:>6} | {fmt(s):>13} | {fmt(t):>13} | {eff:>14}")

print("\n--- raw arm latency, pooled trials, cv0 (exposes per-step/comms overhead directly) ---")
for tag,label in (("SINGLE","single-GPU"),("TP2","TP=2")):
    for arm in ("mono","chunk"):
        vals=[]
        for tr in TRIALS:
            for f in glob.glob(f"logs/*cs2repl{tag}-{arm}-cv0-t{tr}.jsonl"): vals+=load(f)
        if vals:
            vals=sorted(vals)
            print(f"  {label:11s} {arm:5s}: p50={st.median(vals):.0f}ms  p95={pct(vals,95):.0f}ms  n={len(vals)}")

print("\n--- saturation sanity (cv0 mono, first50->last50 median; flat => sub-sat, confirms matched-rho) ---")
for tag,label in (("SINGLE","single-GPU"),("TP2","TP=2")):
    rows=[]
    for tr in TRIALS:
        for f in glob.glob(f"logs/*cs2repl{tag}-mono-cv0-t{tr}.jsonl"):
            rows+=[(r["ts"], r["ttft"]*1000) for r in (json.loads(l) for l in open(f) if l.strip())
                   if r.get("ttft") is not None and r.get("ts") is not None]
    rows.sort()
    v=[t for _,t in rows]
    if len(v)>=100:
        f50=st.median(v[:50]); l50=st.median(v[-50:])
        flag="SATURATED!" if l50>2*f50 else "sub-sat OK"
        print(f"  {label:11s}: first50={f50:.0f}ms  last50={l50:.0f}ms  ratio={l50/f50:.1f}x  [{flag}]")

print("\nInterpretation: if TP2 chunk Δ% (esp. p50/median) is MORE POSITIVE than SINGLE,")
print("host-staged NCCL is inflating chunk's per-step overhead -> the 14B null is partly a")
print("comms artifact, and NVLink would tilt the tradeoff back toward chunk. If SINGLE≈TP2,")
print("the NCCL penalty is not the driver and the null is more trustworthy.")
