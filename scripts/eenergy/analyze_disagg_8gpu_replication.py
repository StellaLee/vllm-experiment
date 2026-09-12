"""Aggregates n=3 replication trials per condition (trial 1 = base outname, trials 2-3 =
base outname + _t2/_t3) for the paper's two flagship comparisons: closed-loop and Poisson,
each no-gate vs. 92.5%-of-mean-cap gated. Reports mean +/- std of max windowed-average power
(per pool) and mean TTFT across the 3 trials, plus per-trial values so a single outlier
trial is visible rather than hidden inside the aggregate."""
import statistics
import sys

from analyze_disagg_8gpu_power import load_pool_power, max_windowed_avg, load_ttft, WINDOW_S

CONDITIONS = [
    ("disagg_8gpu_baseline", ["", "_t2", "_t3"]),
    ("disagg_8gpu_gated", ["", "_t2", "_t3"]),
    ("disagg_8gpu_poisson_baseline", ["", "_t2", "_t3"]),
    ("disagg_8gpu_poisson_gated", ["", "_t2", "_t3"]),
]


def summarize(vals):
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def main():
    for base, suffixes in CONDITIONS:
        print(f"=== {base} (n={len(suffixes)}) ===")
        prefill_max, decode_max, mean_ttfts = [], [], []
        for suf in suffixes:
            outname = base + suf
            try:
                t0, p0 = load_pool_power(f"logs/{outname}_power_prefill.csv")
                t1, p1 = load_pool_power(f"logs/{outname}_power_decode.csv")
            except FileNotFoundError as e:
                print(f"  MISSING: {outname} ({e})")
                continue
            mw_prefill = max_windowed_avg(t0, p0, WINDOW_S)
            mw_decode = max_windowed_avg(t1, p1, WINDOW_S)
            n_records, n_failed, ttfts, tpots = load_ttft(f"logs/{outname}_records.jsonl")
            mean_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
            prefill_max.append(mw_prefill)
            decode_max.append(mw_decode)
            mean_ttfts.append(mean_ttft)
            print(f"  trial {outname}: prefill_max={mw_prefill:.1f}W decode_max={mw_decode:.1f}W "
                  f"mean_ttft={mean_ttft:.3f}s n_failed={n_failed}/{n_records}")
        if prefill_max:
            pm, ps = summarize(prefill_max)
            dm, ds = summarize(decode_max)
            tm, ts = summarize(mean_ttfts)
            print(f"  AGGREGATE: prefill_max={pm:.1f}+/-{ps:.1f}W  decode_max={dm:.1f}+/-{ds:.1f}W  "
                  f"mean_ttft={tm:.3f}+/-{ts:.3f}s")
        print()


if __name__ == "__main__":
    sys.exit(main())
