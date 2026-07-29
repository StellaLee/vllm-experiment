#!/usr/bin/env python3
"""Analyze the Cs^2 genuine-term sweep.

For each (arm, cv2) log it reports: n, achieved Cs^2 of pad_chars (the realized
prefill-size dispersion, which is what the theory keys on -- nominal cv2 is only the
knob), and TTFT mean / median / p95. Then, per Cs^2 point, the chunk-vs-mono delta:
  delta% = (chunk_mean_ttft - mono_mean_ttft) / mono_mean_ttft * 100
Prediction: delta% > 0 (chunk worse) at low Cs^2, crossing < 0 (chunk better) as the
realized Cs^2 passes ~1 -- the genuine head-of-line term switching on.
"""
import glob, json, os, re, statistics as st, sys

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
    return rows

def pctl(xs, q):
    if not xs: return None
    xs = sorted(xs); i = min(len(xs)-1, int(q*len(xs)))
    return xs[i]

def cs2(xs):
    if len(xs) < 2: return 0.0
    m = st.mean(xs)
    return st.pvariance(xs)/m**2 if m else 0.0

def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "logs/*cs2sweep-*.jsonl"
    files = sorted(glob.glob(pat))
    # key: (arm, cvtag) -> stats
    stats = {}
    rx = re.compile(r"cs2sweep-(mono|chunk|ours)-cv([0-9p]+)-")
    for fp in files:
        mobj = rx.search(os.path.basename(fp))
        if not mobj: continue
        arm, cvtag = mobj.group(1), mobj.group(2).replace('p', '.')
        rows = load(fp)
        ttft = [r["ttft"]*1000 for r in rows if r.get("ttft") is not None]
        pads = [r.get("pad_chars", 0) for r in rows if r.get("pad_chars") is not None]
        # Cs^2 of the TOTAL prompt (pad + variable base ShareGPT message), the real
        # prefill-size dispersion the theory keys on -- not just the pad.
        toks = [r.get("prompt_tokens_approx", 0) for r in rows if r.get("prompt_tokens_approx")]
        stats[(arm, cvtag)] = dict(
            n=len(ttft), cv2_nom=cvtag, cs2_pad=cs2(pads), cs2_tot=cs2(toks),
            pad_mean=st.mean(pads) if pads else 0,
            mean=st.mean(ttft) if ttft else None,
            med=st.median(ttft) if ttft else None,
            p95=pctl(ttft, 0.95),
        )
    cvs = sorted({k[1] for k in stats}, key=float)
    have_ours = any(k[0] == "ours" for k in stats)
    # mean TTFT (ms) per arm + delta vs mono (negative => beats run-to-completion)
    hdr = f"{'cv2_nom':>7} {'Cs2_tot':>8} {'Cs2_pad':>8} | {'mono':>7} {'chunk':>7} {'chunk_d%':>8}"
    if have_ours: hdr += f" {'ours':>7} {'ours_d%':>8}"
    print(hdr); print("-"*len(hdr))
    for cv in cvs:
        mo, ch = stats.get(("mono", cv)), stats.get(("chunk", cv))
        if not (mo and ch and mo["mean"] and ch["mean"]): continue
        dch = (ch["mean"]-mo["mean"])/mo["mean"]*100
        row = (f"{cv:>7} {mo['cs2_tot']:>8.2f} {ch['cs2_pad']:>8.2f} | "
               f"{mo['mean']:>7.1f} {ch['mean']:>7.1f} {dch:>7.1f}%")
        if have_ours:
            ou = stats.get(("ours", cv))
            if ou and ou["mean"]:
                dou = (ou["mean"]-mo["mean"])/mo["mean"]*100
                row += f" {ou['mean']:>7.1f} {dou:>7.1f}%"
            else:
                row += f" {'--':>7} {'--':>8}"
        row += "  <-- chunk WINS" if dch < 0 else ""
        print(row)
    print("\n(mean TTFT in ms; d% = (arm-mono)/mono; negative => beats run-to-completion.")
    print(" mono=run-to-completion floor, chunk=PS frontier, ours=adaptive controller.)")

if __name__ == "__main__":
    main()
