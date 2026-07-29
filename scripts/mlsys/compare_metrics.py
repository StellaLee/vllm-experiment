#!/usr/bin/env python3
"""Cross-check the custom client-side measurements against vLLM's /metrics.

Usage: compare_metrics.py <client.jsonl> <metrics.txt>

Client side (per-request JSONL from replay_sharegpt.py): ttft, tpot, latency(=E2EL).
Server side (Prometheus text from /metrics): vllm:*_seconds histograms.

Compares EXACT means (sum/count on both sides -- the reliable comparison) and,
secondarily, p95 (client empirical vs server coarse bucket upper-bound).
Also dumps every vllm:*_seconds family found so we can adjust substring matches.
"""
import json, sys, re
from collections import defaultdict

client_path, metrics_path = sys.argv[1], sys.argv[2]

# ---- client side ----
recs = [json.loads(l) for l in open(client_path)]
def col(k): return [r[k] for r in recs if r.get(k) is not None]
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float("nan")
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

ttft = col("ttft"); tpot = col("tpot"); e2e = col("latency")
decode = [r["latency"] - (r["ttft"] or 0) for r in recs if r.get("latency") is not None]

# ---- server side: parse Prometheus histograms ----
sums, counts, buckets = defaultdict(float), defaultdict(float), defaultdict(lambda: defaultdict(float))
line_re = re.compile(r'^(vllm:[a-zA-Z_]+_seconds)_(bucket|sum|count)(\{[^}]*\})?\s+([0-9.eEnN+-]+)$')
le_re = re.compile(r'le="([^"]+)"')
for raw in open(metrics_path):
    m = line_re.match(raw.strip())
    if not m: continue
    fam, kind, labels, val = m.group(1), m.group(2), m.group(3) or "", m.group(4)
    try: val = float(val)
    except ValueError: continue
    if kind == "sum": sums[fam] += val
    elif kind == "count": counts[fam] += val
    elif kind == "bucket":
        le = le_re.search(labels)
        if le: buckets[fam][le.group(1)] += val

def srv_mean(fam):
    return sums[fam]/counts[fam] if counts.get(fam) else float("nan")
def srv_pct(fam, q):
    b = buckets.get(fam); tot = counts.get(fam)
    if not b or not tot: return float("nan")
    def key(le): return float("inf") if le in ("+Inf","Inf") else float(le)
    for le in sorted(b, key=key):
        if b[le] >= q*tot: return key(le)
    return float("inf")

def find(*subs):
    for fam in list(sums)+list(counts):
        if any(s in fam for s in subs): return fam
    return None

fam_ttft   = find("time_to_first_token")
fam_tpot   = find("time_per_output_token", "inter_token")
fam_e2e    = find("e2e_request_latency", "e2e")
fam_queue  = find("request_queue_time", "queue_time")
fam_prefill= find("request_prefill_time", "prefill_time")
fam_decode = find("request_decode_time", "decode_time")

print("=== vllm:*_seconds families found ===")
for fam in sorted(set(list(sums)+list(counts))):
    print(f"  {fam:<45} count={counts.get(fam,0):.0f} mean={srv_mean(fam)*1000:.1f}ms")

print(f"\nclient records: {len(recs)}   server e2e count: {counts.get(fam_e2e,0):.0f}\n")

def row(label, client_vals, fam, client_is_ms=False):
    cm = mean(client_vals)*1000; cp = pct(client_vals,0.95)*1000
    sm = srv_mean(fam)*1000 if fam else float("nan")
    sp = srv_pct(fam,0.95)*1000 if fam else float("nan")
    dm = (cm-sm)/sm*100 if fam and sm==sm and sm else float("nan")
    fam_s = fam or "(not found)"
    print(f"{label:<14} client mean={cm:8.1f}ms  server mean={sm:8.1f}ms  Δ={dm:+6.1f}%   "
          f"| client p95={cp:8.1f}  server p95(bkt)={sp:8.1f}   [{fam_s}]")

print("=== means are exact (sum/count); p95 server side is coarse bucket upper-bound ===")
row("TTFT",        ttft,   fam_ttft)
row("E2EL",        e2e,    fam_e2e)
row("decode(=lat-ttft)", decode, fam_decode)
row("TPOT",        tpot,   fam_tpot)
print("\nNote: TPOT denominator is client WORD count vs server real-token count -> "
      "expect client TPOT ~1.3x server. TTFT/E2EL should match within a few % "
      "(client adds only localhost network/connect). queue/prefill families: "
      f"queue={fam_queue}  prefill={fam_prefill}")
