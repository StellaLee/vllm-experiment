#!/usr/bin/env python3
"""Sensitive p99-controller experiment. Per arm: TPOT mean/p99/max, TTFT mean/p95.
For the adaptive arms, extract the controller budget trajectory from the server log
(engaged = budget drops off the ceiling). Question: does slotail (p99 signal) engage and
reach the static512 frontier's TPOT tail-capping, where slo (mean signal) stays pinned?
"""
import glob, json, os, re, statistics as st, sys

def load(fp): return [json.loads(l) for l in open(fp) if l.strip()]
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else None

def budget_traj(prefix, date, arm):
    fp = f"logs/{date}-{prefix}-{arm}-server.log"
    if not os.path.exists(fp): return None
    vals = [int(m) for m in re.findall(r"chunk=(\d+)", open(fp, errors='ignore').read())]
    if not vals: return None
    ceil = max(vals)
    return dict(n=len(vals), lo=min(vals), med=int(st.median(vals)), hi=ceil,
               at_ceiling=100.0*sum(1 for v in vals if v == ceil)/len(vals))

def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-13"
    trial = sys.argv[2] if len(sys.argv) > 2 else "1"
    prefix = "sp99"
    print(f"{'arm':>12} | {'TPOT mean':>9} {'p99':>7} {'max':>7} | {'TTFT mean':>9} {'p95':>7} | budget(lo/med/hi %ceil)")
    print("-"*92)
    for arm in ["static8192", "static512", "slo", "slotail"]:
        fps = glob.glob(f"logs/{date}-{prefix}-{arm}_t{trial}.jsonl")
        if not fps: continue
        rows = load(fps[0])
        tpot = [r["tpot"]*1000 for r in rows if r.get("tpot")]
        ttft = [r["ttft"]*1000 for r in rows if r.get("ttft")]
        bt = budget_traj(prefix, date, arm)
        bts = f"{bt['lo']}/{bt['med']}/{bt['hi']} {bt['at_ceiling']:.0f}%" if bt else "(static)"
        print(f"{arm:>12} | {st.mean(tpot):>9.1f} {pct(tpot,0.99):>7.1f} {max(tpot):>7.1f} | "
              f"{st.mean(ttft):>9.0f} {pct(ttft,0.95):>7.0f} | {bts}")
    print("\n(TPOT/TTFT ms. Engaged controller = budget lo < hi and %ceil < 100.")
    print(" Success: slotail engages + TPOT p99/max approaches static512, unlike slo.)")

if __name__ == "__main__":
    main()
