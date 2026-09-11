import csv
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"


def cv(values):
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    sd = var ** 0.5
    return sd / mu if mu else 0.0


def energy_cv(prefix, arm, trial):
    """Coefficient of variation of total energy consumed (energy_mj delta) across the 6 GPUs."""
    path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    first, last = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gpu = row["gpu_index"]
            e = float(row["energy_mj"])
            if gpu not in first:
                first[gpu] = e
            last[gpu] = e
    per_gpu_energy = [last[g] - first[g] for g in first]
    return cv(per_gpu_energy)


def request_count_cv(prefix, arm, trial):
    """Coefficient of variation of total dispatched-request counts across replicas."""
    path = f"{LOGDIR}/{prefix}_assignment_{arm}_t{trial}.csv"
    counts = defaultdict(int)
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[row["replica_id"]] += 1
    return cv(list(counts.values()))


if __name__ == "__main__":
    import sys
    PREFIX = sys.argv[1] if len(sys.argv) > 1 else "closedloopheavylongpergpu"
    ARM = sys.argv[2] if len(sys.argv) > 2 else "lmetric_power_coincidence_ceiling"
    TRIALS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5, 6]

    e_vals, r_vals = [], []
    print(f"{'trial':<8}{'energy_cv':>14}{'request_count_cv':>20}")
    for t in TRIALS:
        try:
            ecv = energy_cv(PREFIX, ARM, t)
            rcv = request_count_cv(PREFIX, ARM, t)
        except FileNotFoundError:
            continue
        e_vals.append(ecv)
        r_vals.append(rcv)
        print(f"{t:<8}{ecv:>14.4f}{rcv:>20.4f}")

    def stats(vals):
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        return mu, sd

    mu_e, sd_e = stats(e_vals)
    mu_r, sd_r = stats(r_vals)
    print(f"\nenergy_cv: mean={mu_e:.4f} std={sd_e:.4f} rel_std={sd_e/mu_e*100 if mu_e else 0:.2f}%")
    print(f"request_count_cv: mean={mu_r:.4f} std={sd_r:.4f} rel_std={sd_r/mu_r*100 if mu_r else 0:.2f}%")
