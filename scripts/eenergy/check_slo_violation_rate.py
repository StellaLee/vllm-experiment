import json

LOGDIR = "/root/pli/vllm-experiment/logs"

TTFT_THRESHOLD_S = 1.0
TBT_MAX_THRESHOLD_MS = 200.0


def slo_violation_rate(prefix, arm, trial):
    path = f"{LOGDIR}/{prefix}_records_{arm}_t{trial}.jsonl"
    total = 0
    ttft_viol = 0
    tbt_viol = 0
    either_viol = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            ttft = r.get("ttft")
            is_ttft = ttft is None or ttft > TTFT_THRESHOLD_S
            tbt_list = r.get("tbt_ms") or []
            is_tbt = max(tbt_list) > TBT_MAX_THRESHOLD_MS if tbt_list else False
            if is_ttft:
                ttft_viol += 1
            if is_tbt:
                tbt_viol += 1
            if is_ttft or is_tbt:
                either_viol += 1
    if total == 0:
        return 0.0, 0.0, 0.0
    return ttft_viol / total, tbt_viol / total, either_viol / total


if __name__ == "__main__":
    import sys
    PREFIX = sys.argv[1] if len(sys.argv) > 1 else "closedloopheavylongpergpu"
    ARM = sys.argv[2] if len(sys.argv) > 2 else "lmetric_power_coincidence_ceiling"
    TRIALS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5, 6]

    tv, bv, ev = [], [], []
    print(f"{'trial':<8}{'ttft_viol_rate':>16}{'tbt_viol_rate':>16}{'either_viol_rate':>18}")
    for t in TRIALS:
        try:
            a, b, e = slo_violation_rate(PREFIX, ARM, t)
        except FileNotFoundError:
            continue
        tv.append(a); bv.append(b); ev.append(e)
        print(f"{t:<8}{a:>16.4f}{b:>16.4f}{e:>18.4f}")

    def stats(vals):
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        return mu, sd

    for name, vals in [("ttft_viol_rate", tv), ("tbt_viol_rate", bv), ("either_viol_rate", ev)]:
        mu, sd = stats(vals)
        print(f"{name}: mean={mu:.4f} std={sd:.4f} rel_std={sd/mu*100 if mu else 0:.2f}%")
