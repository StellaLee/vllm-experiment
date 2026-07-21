#!/usr/bin/env python3
"""3-arm controller analysis (14B/TP=2, mt=1024, Cs2=0): mono vs chunk(512) vs ours(slocvar).
Reports TTFT + TPOT (p50/p95) per arm and vs mono. The win condition for slocvar: get
chunk's TPOT-tail protection (negative vs mono, like chunk) WITHOUT chunk's TTFT cost
(TTFT closer to mono than to chunk). Also checks controller ENGAGEMENT (did the budget
actually move, or pin at ceiling = 'wins nothing because never engaged')."""
import json, glob, re, statistics as st

ARMS = ["mono", "chunk", "ours"]
TRIALS = ["1", "2", "3"]
def rows(p): return [r for r in (json.loads(l) for l in open(p) if l.strip())]
def pct(xs, p): return sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")
def vals(arm, field):
    xs = []
    for tr in TRIALS:
        for f in glob.glob(f"logs/*cs23arm-{arm}-cv0-t{tr}.jsonl"):
            xs += [r[field] * 1000 for r in rows(f) if r.get(field) is not None]
    return sorted(xs)

print("3-ARM CONTROLLER — 14B/TP=2, mt=1024, Cs2=0.  mono vs chunk(512) vs ours(slocvar 512..16384)")
print("Win for slocvar: TPOT tail like CHUNK (protected) but TTFT like MONO (no throughput cost).\n")
for field, lab in (("ttft", "TTFT (prefill wait)"), ("tpot", "TPOT (decode step latency)")):
    print(f"=== {lab} (ms) ===")
    print(f"{'arm':>6} | {'p50':>7} | {'p95':>7} | {'vs mono p50%':>12} | {'vs mono p95%':>12}")
    mv = vals("mono", field); m50 = st.median(mv); m95 = pct(mv, 95)
    for arm in ARMS:
        v = vals(arm, field)
        if not v:
            print(f"{arm:>6} | (no data)"); continue
        p50, p95 = st.median(v), pct(v, 95)
        print(f"{arm:>6} | {p50:7.0f} | {p95:7.0f} | {(p50-m50)/m50*100:+11.1f} | {(p95-m95)/m95*100:+11.1f}")
    print()

print("=== slocvar controller engagement (did the chunk budget move?) ===")
found = False
for f in sorted(glob.glob("logs/*cs23arm-ours-server.log")):
    buds = [int(m) for m in re.findall(r"ChunkCtrl\[slocvar\].*?chunk=(\d+)", open(f).read())]
    if buds:
        found = True
        print(f"  {f.split('/')[-1]}: budget min={min(buds)} max={max(buds)} last={buds[-1]} n={len(buds)}")
if not found:
    print("  NO slocvar log lines — controller may not have engaged (check server log / DYNAMIC_CHUNK).")
print("\nRead: if ours-budget stayed ~16384 (=mono ceiling) -> pinned, never engaged (tuning/mechanism);")
print("if it dropped toward 512 -> engaged. Engaged + TPOT protected + TTFT>mono-ish = slocvar works.")
