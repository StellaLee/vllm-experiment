import csv
import json
import glob

LOGDIR = "/root/pli/vllm-experiment/logs"
TTFT_THRESHOLD_S = 1.0
TBT_MAX_THRESHOLD_MS = 200.0


def energy_per_token(prefix, arm, trial):
    pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    first, last = {}, {}
    with open(pow_path) as f:
        for row in csv.DictReader(f):
            gpu = row["gpu_index"]
            e = float(row["energy_mj"])
            if gpu not in first:
                first[gpu] = e
            last[gpu] = e
    total_energy_j = sum((last[g] - first[g]) / 1000.0 for g in first)

    rec_path = f"{LOGDIR}/{prefix}_records_{arm}_t{trial}.jsonl"
    total_tokens = 0
    ttft_viol = 0
    tbt_viol = 0
    total = 0
    with open(rec_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total_tokens += r.get("output_tokens", 0)
            total += 1
            ttft = r.get("ttft")
            if ttft is None or ttft > TTFT_THRESHOLD_S:
                ttft_viol += 1
            tbt_list = r.get("tbt_ms") or []
            if tbt_list and max(tbt_list) > TBT_MAX_THRESHOLD_MS:
                tbt_viol += 1
    ept = total_energy_j / total_tokens if total_tokens else None
    tv_rate = ttft_viol / total if total else None
    bv_rate = tbt_viol / total if total else None
    return ept, tv_rate, bv_rate


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return mu, sd, len(vals)


CONDITIONS = {
    "Heavy/Closed-Loop (short)": "closedloopheavypergpu",
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
    "BurstGPT": "burstgptpergpu",
    "Ramp & Route": "rampandroutepergpu",
    "WildChat": "wildchatnaturalpergpu",
}

ARMS = [
    "round_robin", "lmetric", "compute_only", "load_only",
    "coincidence_ceiling", "weighted_sum_coincidence_ceiling",
    "lmetric_power_coincidence_ceiling", "lmetric_power_pareto_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_all_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_small_coincidence_ceiling",
]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*95}\n{label}: {prefix}\n{'='*95}")
    rows = {}
    for arm in ARMS:
        trial_files = sorted(int(p.split("_t")[-1].split(".")[0])
                              for p in glob.glob(f"{LOGDIR}/{prefix}_records_{arm}_t*.jsonl"))
        epts, tvs, bvs = [], [], []
        for t in trial_files:
            try:
                ept, tv, bv = energy_per_token(prefix, arm, t)
            except (FileNotFoundError, KeyError):
                continue
            epts.append(ept); tvs.append(tv); bvs.append(bv)
        if not epts:
            continue
        me, se, ne = stats(epts)
        mt, st, nt = stats(tvs)
        mb, sb, nb = stats(bvs)
        rows[arm] = (me, se, mt, st, mb, sb, ne)

    print(f"{'arm':<58}{'J/token':>14}{'ttft_viol':>16}{'tbt_viol':>16}{'n':>4}")
    for arm, (me, se, mt, st, mb, sb, n) in sorted(rows.items(), key=lambda kv: kv[1][0]):
        print(f"{arm:<58}{me:>8.4f}±{se:<5.4f}{mt:>8.4f}±{st:<5.4f}{mb:>8.4f}±{sb:<5.4f}{n:>4}")
