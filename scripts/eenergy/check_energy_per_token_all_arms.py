import csv
import json

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


PREFIX = "closedloopheavylongpergpu"
ARMS = [
    "round_robin", "lmetric", "compute_only", "load_only",
    "coincidence_ceiling", "weighted_sum_coincidence_ceiling",
    "lmetric_power_coincidence_ceiling", "lmetric_power_pareto_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_all_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_small_coincidence_ceiling",
]
SIX = {"round_robin", "compute_only", "load_only", "lmetric_power_coincidence_ceiling"}

results = {}
for arm in ARMS:
    trials = range(1, 7) if arm in SIX else range(1, 4)
    epts = []
    for t in trials:
        try:
            _, _, ept = energy_and_tokens(PREFIX, arm, t)
            epts.append(ept)
        except FileNotFoundError:
            continue
    if not epts:
        continue
    mu = sum(epts) / len(epts)
    sd = (sum((v - mu) ** 2 for v in epts) / (len(epts) - 1)) ** 0.5 if len(epts) > 1 else 0.0
    results[arm] = (mu, sd, len(epts))

print(f"{'arm':<58}{'J/token':>12}{'std':>10}{'rel_std':>10}{'n':>5}")
for arm, (mu, sd, n) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"{arm:<58}{mu:>12.4f}{sd:>10.4f}{sd/mu*100:>9.2f}%{n:>5}")

vals = [v[0] for v in results.values()]
print(f"\nrange across {len(vals)} arms: {max(vals)-min(vals):.4f} J/token "
      f"({(max(vals)-min(vals))/min(vals)*100:.1f}% of min)")
