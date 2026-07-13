#!/usr/bin/env python3
"""Analyze the replicated Cs^2 crossover (paired, 3 trials).

Within each trial, mono & chunk saw identical prompts (same pad seed), so we form the
PAIRED per-trial delta  d_t = (chunk_mean_t - mono_mean_t)/mono_mean_t, then report the
mean +- std across trials and how many trials chunk won. A robust crossover = d flips
from clearly >0 (chunk loses) below Cs^2=1 to clearly <0 (chunk wins) above it, with the
sign consistent across trials.
"""
import glob, json, os, re, statistics as st, sys

def load(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]

def cs2(xs):
    m = st.mean(xs); return st.pvariance(xs)/m**2 if m else 0.0

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "logs/*cs2repl-*.jsonl"
    rx = re.compile(r"cs2repl-(mono|chunk)-cv([0-9p]+)-t([0-9]+)")
    # (arm,cv,trial) -> mean ttft ; (cv) -> cs2_total samples
    means, cs2tot = {}, {}
    for fp in sorted(glob.glob(pat)):
        m = rx.search(os.path.basename(fp))
        if not m: continue
        arm, cv, tr = m.group(1), m.group(2).replace('p', '.'), m.group(3)
        rows = load(fp)
        ttft = [r["ttft"]*1000 for r in rows if r.get("ttft") is not None]
        toks = [r.get("prompt_tokens_approx", 0) for r in rows if r.get("prompt_tokens_approx")]
        if ttft: means[(arm, cv, tr)] = st.mean(ttft)
        if toks: cs2tot.setdefault(cv, []).append(cs2(toks))
    cvs = sorted({cv for (_, cv, _) in means}, key=float)
    trials = sorted({tr for (_, _, tr) in means}, key=int)
    print(f"{'Cs2_tot':>8} | {'mono(ms)':>18} | {'chunk(ms)':>18} | {'paired chunk d%':>22}")
    print(f"{'':>8} | {'mean  (per-trial)':>18} | {'mean  (per-trial)':>18} | {'mean+-std   wins/N':>22}")
    print("-"*78)
    for cv in cvs:
        mo = [means.get(("mono", cv, t)) for t in trials]
        ch = [means.get(("chunk", cv, t)) for t in trials]
        pairs = [(m, c) for m, c in zip(mo, ch) if m and c]
        if not pairs: continue
        deltas = [(c-m)/m*100 for m, c in pairs]
        mo_mean = st.mean([m for m, _ in pairs]); ch_mean = st.mean([c for _, c in pairs])
        dmean = st.mean(deltas); dstd = st.stdev(deltas) if len(deltas) > 1 else 0.0
        wins = sum(1 for d in deltas if d < 0)
        cbar = st.mean(cs2tot.get(cv, [0]))
        mo_s = "/".join(f"{m:.0f}" for m in mo if m)
        ch_s = "/".join(f"{c:.0f}" for c in ch if c)
        flag = "  <== CHUNK WINS" if dmean < 0 else ""
        print(f"{cbar:>8.2f} | {mo_mean:>7.0f} ({mo_s:>9}) | {ch_mean:>7.0f} ({ch_s:>9}) | "
              f"{dmean:>+6.1f}+-{dstd:>4.1f}  {wins}/{len(deltas)}{flag}")
    print("\n(paired per-trial delta = (chunk-mono)/mono; negative => chunking beats")
    print(" run-to-completion. Robust crossover = sign flips + consistent across trials.)")

if __name__ == "__main__":
    main()
