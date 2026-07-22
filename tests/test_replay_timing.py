import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import replay_timing as rt


def test_parse_schedule():
    assert rt.parse_schedule("8@50,40@50") == [(8, 50.0), (40, 50.0)]
    assert rt.parse_schedule(" 8@50 , 40@50 ") == [(8, 50.0), (40, 50.0)]


def test_phase_at_cycles():
    s = rt.parse_schedule("8@50,40@50")
    assert rt.phase_at(s, 0.0)   == (0, 8, 0)
    assert rt.phase_at(s, 49.9)  == (0, 8, 0)
    assert rt.phase_at(s, 50.0)  == (1, 40, 0)
    assert rt.phase_at(s, 99.9)  == (1, 40, 0)
    assert rt.phase_at(s, 100.0) == (0, 8, 1)     # second cycle, low phase
    assert rt.phase_at(s, 150.0) == (1, 40, 1)


def test_token_times():
    # start = ts - latency = 100.0; ttft = 2.0; two ITL gaps of 30ms, 50ms => 3 tokens
    rec = {"ts": 105.0, "latency": 5.0, "ttft": 2.0, "tbt_ms": [30.0, 50.0]}
    tt = rt.token_times(rec)
    assert len(tt) == 3
    assert abs(tt[0] - 102.0) < 1e-9      # 100 + ttft
    assert abs(tt[1] - 102.03) < 1e-9     # + 30ms
    assert abs(tt[2] - 102.08) < 1e-9     # + 50ms
    assert rt.token_times({"ts": None}) == []


def test_realized_concurrency_and_throughput():
    # three overlapping requests: [0,10], [5,15], [12,20]
    recs = [
        {"ts": 10.0, "latency": 10.0},
        {"ts": 15.0, "latency": 10.0},
        {"ts": 20.0, "latency": 8.0},
    ]
    mean, mx = rt.realized_concurrency(recs)
    assert mx == 2                        # never 3 at once (3rd starts at 12, 1st ends at 10)
    assert 0.0 < mean < 2.0
    tp = rt.realized_throughput(recs)     # 3 completions over span [0,20]
    assert abs(tp - 3 / 20.0) < 1e-9


def test_realized_concurrency_tie():
    # req A [0,10], req B starts EXACTLY when A ends at t=10 -> [10,20].
    # opens-before-closes tie rule => peak counts them as briefly overlapping at t=10 => max == 2.
    recs = [
        {"ts": 10.0, "latency": 10.0},   # A: [0, 10]
        {"ts": 20.0, "latency": 10.0},   # B: [10, 20]
    ]
    mean, mx = rt.realized_concurrency(recs)
    assert mx == 2, f"expected tie peak 2, got {mx}"


def test_bucket_by_phase():
    s = rt.parse_schedule("8@50,40@50")
    # one request: start_wall=0 (ttft in phase 0), tokens stretch to t=60 (phase 1)
    # ts - latency = 0 => ts=latency. ttft=1s. many 500ms ITLs walking across the boundary.
    rec = {"ts": 61.0, "latency": 61.0, "ttft": 1.0, "tbt_ms": [500.0] * 118}
    b = rt.bucket_by_phase([rec], s)
    assert b[0]["conc"] == 8 and b[1]["conc"] == 40
    assert len(b[0]["ttft"]) == 1 and len(b[1]["ttft"]) == 0   # ttft at start (phase 0)
    assert len(b[0]["tbt"]) > 0 and len(b[1]["tbt"]) > 0        # tokens span both phases


def test_bucket_trace_rows():
    # emulate chunktrace rows: wall_s crossing the 50s boundary; depth>0 marks activity start
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import analyze_nonstationary as an
    s = rt.parse_schedule("8@50,40@50")
    rows = [
        {"wall_s": "1000.0", "depth": "0",  "chunk": "16384"},  # idle, ignored for t0
        {"wall_s": "1002.0", "depth": "5",  "chunk": "2000"},   # t0 here -> phase 0
        {"wall_s": "1055.0", "depth": "30", "chunk": "900"},    # +53s -> phase 1
    ]
    b = an.bucket_trace(rows, s)
    assert 2000 in b[0] and 900 in b[1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
