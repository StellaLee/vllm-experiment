#!/usr/bin/env python3
"""Split analyze_lengthgate.py's Phase-W bucket further into whale vs non-whale
individual requests (by prompt_tokens_approx), so we can tell whether a Phase-W
delta is driven by the whale's own request or spread across the short requests
sharing that phase with it.

Env: SCHEDULE (required), ARMS (default from analyze_lengthgate), SLO_TBT_MS
(default 500), WHALE_TOK_THRESHOLD (default 4000 -- clear gap above short-request
p99 and below whale floor at the calibrated pad/whale-char settings).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_timing import parse_phase_schedule, phase_type_at, token_times


def _ptype(sched, t):
    return "S" if phase_type_at(sched, t)[2] == 0.0 else "W"


def pctl(x, p):
    y = sorted(x)
    return y[min(len(y) - 1, int(p / 100 * len(y)))] if y else float("nan")


def main():
    import statistics as st
    SCHEDULE = os.environ["SCHEDULE"]
    ARMS = os.environ.get("ARMS", "16384 512 lpt512 adaptivelpt").split()
    SLO = float(os.environ.get("SLO_TBT_MS", "500"))
    WTHRESH = float(os.environ.get("WHALE_TOK_THRESHOLD", "4000"))
    sched = parse_phase_schedule(SCHEDULE)
    print(f"whale-split analysis  schedule={SCHEDULE}  arms={ARMS}  "
          f"SLO_TBT={SLO:.0f}ms  whale_threshold={WTHRESH:.0f}tok\n")

    def load(arm):
        R = []
        for f in glob.glob(f"logs/*-lgate-b{arm}-t1.jsonl"):
            R += [json.loads(l) for l in open(f) if l.strip()]
        return R

    hdr = (f"{'arm':>14} | {'grp':>8} {'n':>5} | {'TTFTmean':>9} {'TTFTp95':>8} | "
           f"{'TBTp99':>7} {'TBTmax':>8} {'gpReq%':>7}")
    print(hdr); print("-" * len(hdr))
    for arm in ARMS:
        recs = load(arm)
        starts = [r["ts"] - r["latency"] for r in recs
                  if r.get("ts") is not None and r.get("latency") is not None]
        if not starts:
            print(f"{arm:>14} | (no records)")
            continue
        t0 = min(starts)
        groups = {"W-whale": [], "W-short": []}
        for r in recs:
            ts, lat = r.get("ts"), r.get("latency")
            if ts is None or lat is None:
                continue
            start = ts - lat
            if _ptype(sched, start - t0) != "W":
                continue
            is_whale = (r.get("prompt_tokens_approx") or 0) > WTHRESH
            groups["W-whale" if is_whale else "W-short"].append(r)

        for gname, grecs in groups.items():
            ttft = [r["ttft"] for r in grecs if r.get("ttft") is not None]
            tbt = [ms for r in grecs for ms in (r.get("tbt_ms") or [])]
            ttm = 1000.0 * st.mean(ttft) if ttft else float("nan")
            tp95 = 1000.0 * pctl(ttft, 95) if ttft else float("nan")
            tp99 = pctl(tbt, 99)
            tmax = max(tbt) if tbt else float("nan")
            ok = sum(1 for r in grecs if r.get("tbt_ms") and max(r["tbt_ms"]) <= SLO)
            gp = 100.0 * ok / len(grecs) if grecs else float("nan")
            print(f"{arm:>14} | {gname:>8} {len(grecs):>5} | {ttm:>9.0f} {tp95:>8.0f} | "
                  f"{tp99:>7.1f} {tmax:>8.1f} {gp:>6.1f}%")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
