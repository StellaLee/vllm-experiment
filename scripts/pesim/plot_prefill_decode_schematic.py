#!/usr/bin/env python3
"""Illustrative schematic of per-iteration batch composition (prefill vs. decode tokens)
for mono (budget=16384) vs chunk (budget=512), processing one representative whale prompt.

NOT measured telemetry -- vLLM's scheduler doesn't log per-iteration token composition in
our setup, so this is a worked-example diagram using the actual experiment parameters
(whale size, max-num-seqs, chunked-prefill's decode-maximal batching rule: decode tokens for
already-in-flight requests are scheduled first each iteration, the remaining budget goes to
the prefill chunk) rather than raw measurement. Purpose: help a reader unfamiliar with chunked
prefill see mechanically why chunking stretches wall-clock time (many small iterations)
without reducing total compute (same total prefill tokens processed either way).
"""
import matplotlib.pyplot as plt

WHALE_TOKENS = 13000       # representative whale size, matches the paper's 12-15k range
DECODE_TOKENS_PER_ITER = 48  # ~max-num-seqs concurrent decode requests, 1 token/iter each

PREFILL_COLOR = "#c44e52"
DECODE_COLOR = "#4c72b0"


def mono_iterations(budget):
    # Whole prompt fits in one iteration if budget allows; decode tokens piggyback.
    return [(WHALE_TOKENS, DECODE_TOKENS_PER_ITER)]


def chunk_iterations(budget):
    prefill_budget_per_iter = budget - DECODE_TOKENS_PER_ITER
    iters = []
    remaining = WHALE_TOKENS
    while remaining > 0:
        chunk = min(prefill_budget_per_iter, remaining)
        iters.append((chunk, DECODE_TOKENS_PER_ITER))
        remaining -= chunk
    return iters


def main():
    mono = mono_iterations(16384)
    chunk = chunk_iterations(512)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)

    for ax, iters, title, budget in [
        (axes[0], mono, "mono (budget=16384)", 16384),
        (axes[1], chunk, "chunk (budget=512)", 512),
    ]:
        idx = list(range(1, len(iters) + 1))
        prefill = [p for p, d in iters]
        decode = [d for p, d in iters]
        ax.bar(idx, prefill, color=PREFILL_COLOR, label="prefill tokens (this whale)")
        ax.bar(idx, decode, bottom=prefill, color=DECODE_COLOR, label="decode tokens (other reqs)")
        ax.axhline(budget, color="#333333", linestyle=":", linewidth=1, label=f"budget={budget}")
        ax.set_xlabel("scheduler iteration")
        ax.set_xlim(0, max(len(iters), 2) + 1)
        ax.set_xticks(idx if len(idx) <= 10 else range(0, len(idx) + 1, 5))
        ax.set_title(f"{title}\n{len(iters)} iteration(s) to finish this whale's prefill", fontsize=9)
        # The decode segment (48 tok, ~max-num-seqs concurrent decode reqs) is real and really
        # stacked on top of prefill -- but at only 0.3% of the 16384-token shared y-axis, it is
        # sub-pixel and would be invisible without this callout (annotated rather than exaggerated,
        # since faking its height would misrepresent the real batch-composition ratio).
        callout_x = idx[0]
        callout_y = prefill[0] + decode[0]
        ax.annotate(f"+{decode[0]} decode tok/iter\n(too thin to see at this scale)",
                    xy=(callout_x, callout_y), xytext=(0.97, 0.55), textcoords="axes fraction",
                    ha="right", va="center", fontsize=7, color=DECODE_COLOR,
                    arrowprops=dict(arrowstyle="->", color=DECODE_COLOR, lw=0.8))

    axes[0].set_ylabel("tokens in batch")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.suptitle(f"Scheduling a {WHALE_TOKENS}-token prompt: same total prefill work,\n"
                 f"different wall-clock stretch (illustrative, from experiment parameters)", y=1.14, fontsize=10)
    fig.savefig("fig_prefill_decode_mono_chunk.png", dpi=200, bbox_inches="tight")
    print("wrote fig_prefill_decode_mono_chunk.png")
    print(f"mono: {len(mono)} iteration(s)")
    print(f"chunk=512: {len(chunk)} iterations, last chunk size = {chunk[-1][0]}")


if __name__ == "__main__":
    main()
