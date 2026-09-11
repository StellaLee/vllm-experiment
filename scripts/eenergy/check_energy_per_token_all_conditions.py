import csv
import json
import glob

LOGDIR = "/root/pli/vllm-experiment/logs"


def energy_and_tokens(prefix, arm, trial):
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
    with open(rec_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total_tokens += r.get("output_tokens", 0)
    return total_energy_j, total_tokens, total_energy_j / total_tokens


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
    print(f"\n{'='*80}\n{label}: {prefix}\n{'='*80}")
    results = {}
    for arm in ARMS:
        trial_files = sorted(int(p.split("_t")[-1].split(".")[0])
                              for p in glob.glob(f"{LOGDIR}/{prefix}_records_{arm}_t*.jsonl"))
        epts = []
        for t in trial_files:
            try:
                _, _, ept = energy_and_tokens(prefix, arm, t)
                epts.append(ept)
            except (FileNotFoundError, KeyError):
                continue
        if not epts:
            continue
        mu = sum(epts) / len(epts)
        sd = (sum((v - mu) ** 2 for v in epts) / (len(epts) - 1)) ** 0.5 if len(epts) > 1 else 0.0
        results[arm] = (mu, sd, len(epts))

    if not results:
        print("  no data")
        continue
    print(f"{'arm':<58}{'J/token':>12}{'std':>10}{'rel_std':>10}{'n':>5}")
    for arm, (mu, sd, n) in sorted(results.items(), key=lambda kv: kv[1][0]):
        rs = f"{sd/mu*100:.2f}%" if mu else "n/a"
        print(f"{arm:<58}{mu:>12.4f}{sd:>10.4f}{rs:>10}{n:>5}")
    vals = [v[0] for v in results.values()]
    print(f"  range across {len(vals)} arms: {max(vals)-min(vals):.4f} J/token "
          f"({(max(vals)-min(vals))/min(vals)*100:.1f}% of min)")
