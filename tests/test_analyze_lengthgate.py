#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from replay_timing import parse_phase_schedule
import analyze_lengthgate as AL


def _rec(start, ttft, tbt_ms):
    lat = ttft + sum(tbt_ms) / 1000.0
    return {"ts": start + lat, "latency": lat, "ttft": ttft, "tbt_ms": tbt_ms}


def test_bucket_splits_S_and_W():
    sched = parse_phase_schedule("100:0.0@10,100:0.5@10")   # S = [0,10), W = [10,20)
    recs = [
        _rec(1.0, 0.20, [30.0, 30.0]),   # arrives in S
        _rec(12.0, 0.40, [400.0, 30.0]), # arrives in W (a frozen decoder tail)
    ]
    b = AL.bucket_lengthgate(recs, sched)
    assert b["S"]["n"] == 1 and b["W"]["n"] == 1
    assert abs(b["S"]["ttft"][0] - 0.20) < 1e-9
    assert 400.0 in b["W"]["tbt"]        # the big gap lands in W


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
