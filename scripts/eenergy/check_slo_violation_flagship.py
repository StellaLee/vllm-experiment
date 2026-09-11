from check_slo_violation_rate import slo_violation_rate

CONDITIONS = {
    "Heavy/Closed-Loop (short)": "closedloopheavypergpu",
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
}
ARMS = ["coincidence_ceiling", "lmetric", "round_robin"]


def stats(vals):
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return mu, sd


for label, prefix in CONDITIONS.items():
    print(f"\n{'='*80}\n{label} (n=6 each)\n{'='*80}")
    print(f"{'arm':<24}{'ttft_viol':>16}{'tbt_viol':>16}{'either_viol':>16}")
    results = {}
    for arm in ARMS:
        tv, bv, ev = [], [], []
        for t in range(1, 7):
            a, b, e = slo_violation_rate(prefix, arm, t)
            tv.append(a); bv.append(b); ev.append(e)
        mt, st = stats(tv)
        mb, sb = stats(bv)
        me, se = stats(ev)
        results[arm] = (mt, st, mb, sb, me, se)
        print(f"{arm:<24}{mt:>8.4f}±{st:<6.4f}{mb:>8.4f}±{sb:<6.4f}{me:>8.4f}±{se:<6.4f}")

    print("\n  z-score of coincidence_ceiling vs baselines (using coincidence_ceiling's own std):")
    cc = results["coincidence_ceiling"]
    for other in ["round_robin", "lmetric"]:
        ov = results[other]
        for name, ci, oi, sdi in [("ttft_viol", 0, 0, 1), ("tbt_viol", 2, 2, 3), ("either_viol", 4, 4, 5)]:
            diff = cc[ci] - ov[oi]
            scale = cc[sdi] if cc[sdi] > 0 else 1e-9
            z = diff / scale
            print(f"    vs {other} {name}: cc={cc[ci]:.4f} other={ov[oi]:.4f} diff={diff:+.4f} z={z:+.2f}")
