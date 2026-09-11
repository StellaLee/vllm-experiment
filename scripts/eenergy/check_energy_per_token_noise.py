import csv
import json
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"


def energy_and_tokens(prefix, arm, trial):
    # energy: sum over GPUs of (last energy_mj - first energy_mj) for that GPU in this trial
    pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    first = {}
    last = {}
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
ARM = "lmetric_power_coincidence_ceiling"

vals = []
print(f"{'trial':<8}{'energy_J':>14}{'tokens':>10}{'J/token':>12}")
for t in range(1, 7):
    e, tok, ept = energy_and_tokens(PREFIX, ARM, t)
    vals.append(ept)
    print(f"{t:<8}{e:>14.1f}{tok:>10}{ept:>12.4f}")

mu = sum(vals) / len(vals)
sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
print(f"\nmean={mu:.4f}  std={sd:.4f}  relative_std={sd/mu*100:.2f}%")
