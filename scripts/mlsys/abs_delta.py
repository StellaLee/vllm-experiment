import json, statistics as st

SOURCES = {
    (0, "20"): "logs/2026-07-23-cs2wt-t0-w05-c20-t1.jsonl",
    (512, "20"): "logs/2026-07-23-cs2wt-t512-w05-c20-t1.jsonl",
}
for c in ("26", "32", "38"):
    SOURCES[(0, c)] = f"logs/2026-07-24-cs2wt-t0-w05-c{c}-t1.jsonl"
    SOURCES[(512, c)] = f"logs/2026-07-24-cs2wt-t512-w05-c{c}-t1.jsonl"


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


print(f"{'conc':>5} | {'mono S:TTFTmean':>16} {'thr512 mean':>12} {'ABSdelta mean':>14} | "
      f"{'mono p95':>9} {'thr512 p95':>11} {'ABSdelta p95':>13}")
for c in ("20", "26", "32", "38"):
    mono = [r for r in load(SOURCES[(0, c)]) if (r.get("pad_chars") or 0) < 40000]
    chunk = [r for r in load(SOURCES[(512, c)]) if (r.get("pad_chars") or 0) < 40000]
    m_tt = [r["ttft"] * 1000 for r in mono if r.get("ttft") is not None]
    c_tt = [r["ttft"] * 1000 for r in chunk if r.get("ttft") is not None]
    mtt, ctt = st.mean(m_tt), st.mean(c_tt)
    mp95, cp95 = pctl(m_tt, 95), pctl(c_tt, 95)
    print(f"{c:>5} | {mtt:16.0f} {ctt:12.0f} {mtt-ctt:14.0f} | {mp95:9.0f} {cp95:11.0f} {mp95-cp95:13.0f}")
