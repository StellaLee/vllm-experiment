# Boundary condition: chunking's ramp-rate benefit needs whale-scale, bursty prefills — not just prompt-size variance, and not just prefill duration alone

**Date:** 2026-07-27
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-7B-Instruct, single GPU (no TP)
**Harness:** `orchestrate_pesim_gate.sh` (mono/16384, chunk/2048, chunk/512; conc=20, max-num-seqs=48,
max-tokens=256, nconv=200, single-turn). Both runs below set `WHALE_FRAC=0` (no bimodal
whale population at all) and instead push the ordinary "body" prompt-size distribution up by
itself, to test what's actually necessary for the ramp-rate reduction found in the whale
conditions (see `2026-07-21-longprompt-tbt-win.md` and the widened/Poisson/Pareto-tail
fleet-scale checks in the PES-IM paper work). Analysis: `scripts/analyze_power_shape.py`
(whole-trace ramp stats; no whale-windowing needed since there are no whales in either run).
Single trial each — these are boundary/mechanism probes, not replicated headline numbers.

## Motivating question

Across every whale-containing condition tested so far, the fleet-scale ramp-rate reduction
from chunking tracked the whale population's mean size: narrow uniform (mean ≈44-50k chars)
→ 34.4%; widened uniform (mean ≈34k chars) → 36.9%±4.7; Pareto-tail (mean ≈26k chars) →
22.8%±5.9. This raised the question: does the benefit require whales (a rare long-prefill
outlier against a mostly-short background) at all, or would it show up with any elevated
prompt-size variance/duration, whale or not?

## Experiment 1: moderate prefill, no whale

`PAD_MEAN=4000 PAD_CV2=1.0 PAD_MIN=200 PAD_MAX=10000 WHALE_FRAC=0` — a single lognormal
population, meaningfully longer than the paper's default short-prompt body (mean 800 chars)
but nowhere near whale scale.

Realized: pad_chars mean=3212, max=10000 (≈993/3091 tokens at the measured 3.235 chars/token
ratio). **Max TTFT across all 200 requests, in every arm including mono (which never
chunks), was only 1.5-1.6s** — barely above ordinary decode-step timescales, nothing like the
multi-second-to-tens-of-seconds prefill a real whale produces.

| Arm | mean_ramp (W/s) | frac_above 420W | n_events |
|---|---|---|---|
| mono (16384) | 38.6 | 0.172 | 10 |
| chunk (2048) | 34.0 | 0.156 | 7 |
| chunk (512) | 37.9 | 0.282 | 14 |

Chunk=512 vs mono: **1.8% ramp reduction** — noise, not an effect. Not even monotonic
(chunk=2048 sits below chunk=512). Chunk=512 spends *more* time near the power ceiling than
mono (frac_above 0.282 vs 0.172).

**Caveat this experiment can't rule out on its own:** because nothing in this population ever
took long enough to prefill (max TTFT 1.6s), this result conflates "no whales" with "no
individually-long prefills at all" — it doesn't yet distinguish whether burstiness or raw
prefill duration is the necessary ingredient. That's what Experiment 2 was designed to
separate.

## Experiment 2: uniformly long prefill, no whale, no bimodality

`PAD_MEAN=12000 PAD_CV2=0.08 PAD_MIN=7000 PAD_MAX=18000 WHALE_FRAC=0` — every request is
individually long (right up to, but never crossing, the 18,000-char whale floor used
elsewhere), tightly clustered (no short population mixed in, no rare-outlier structure).

Realized: pad_chars mean=11,344, range [7000, 18000] (≈2164-5564 tokens, mean ≈3507 tokens).
TTFT now genuinely multi-second: mono mean=0.9s, p95=2.1s, max=4.5s.

| Arm | mean_ramp (W/s) | frac_above 420W | n_events |
|---|---|---|---|
| mono (16384) | 36.7 | 0.581 | 32 |
| chunk (2048) | 36.3 | 0.659 | 30 |
| chunk (512) | **17.3** | **0.809** | 14 |

**Chunk=2048 vs mono: essentially no change (−1%)** — the paper's primary studied budget
does nothing here. Consistent with the burstiness story: at mean ≈3507 tokens, a 2048-token
budget only buys ~1.8 rounds per request, not enough to meaningfully spread the work out.

**Chunk=512 vs mono: a real 53% ramp reduction (36.7→17.3 W/s)** — but via a *different*
mechanism than the whale case. Its `frac_above` is *higher* than mono's (0.809 vs 0.581), not
lower, and its event count is lower (14 vs 32) — the opposite signature from spike-smoothing
(which lowers both frac_above and peak height while breaking up a small number of large
excursions). Here, every request needs ~7 rounds at budget=512, and with conc=20 keeping the
batch constantly full of interleaved prefill-chunks and decode work, the GPU saturates into a
continuous high-power steady state instead of oscillating between idle/low and spike. Fewer
transitions happen not because spikes got smoothed, but because there's no longer a
spike-vs-baseline contrast to smooth — the system is just pinned near ceiling most of the
time.

## Interpretation

1. **Burstiness (rare-long-vs-short) is necessary for the spike-smoothing mechanism the paper
   actually studies (chunk=2048).** Removing the whale population — even while individually
   lengthening every request to just under the whale floor — kills the effect for chunk=2048
   entirely (−1%, same as the no-effect moderate-prefill case). Prompt-size variance or
   absolute duration alone, without a short-vs-long contrast, does not reproduce the
   whale-driven result.
2. **Sufficiently fine-grained chunking (512) can still reduce ramp in a uniformly-long,
   whale-free population — but through saturation, not smoothing.** This is a genuinely
   different mechanism and arguably a different (not obviously better) outcome from a grid
   perspective: continuous near-ceiling draw vs. intermittent draw is a duty-cycle trade, not
   a straightforward win. Conflating this with the whale-driven spike-smoothing result would
   overstate what the paper's primary studied configuration (chunk=2048) actually does.
3. This cleanly brackets the effect's operating regime: the ramp-rate benefit reported
   throughout this project requires (a) a population with a long right tail reaching
   whale-scale duration, and (b) that tail being rare relative to a much shorter background —
   not just "more variance" or "everything a bit longer."

## Caveats

- Both experiments are single-trial; the magnitudes (1.8%, −1%, 53%) should be treated as
  indicative, not final, especially the noisier moderate-prefill numbers (n_events as low as
  7).
- Experiment 2's chunk=512 saturation-driven reduction is a new, only partially-understood
  mechanism — no fleet-scale (data-center) simulation has been run for it, and the existing
  `coincidence_factor_model.py` calibration (built around an alternating-renewal ON/OFF
  whale-triggered process) doesn't cleanly apply to a saturated, non-bursty trace. A different
  analysis would be needed before this number could feed the reserve-procurement pipeline.
- Only chunk=512 and chunk=2048 were tested; the transition between "no effect" (2048) and
  "saturation effect" (512) was not swept, so the exact budget where saturation kicks in for
  this population is unknown.

## Next steps

1. If this boundary condition goes in the paper, it belongs as a short paragraph establishing
   scope (chunking's grid benefit is specific to whale/bursty workloads, not a general
   property of prompt-size variance) — likely alongside or right after the Pareto-tail
   robustness result, since together they trace out how the effect shrinks as the whale
   population becomes smaller/less extreme and finally vanishes (for 2048) once whales are
   removed entirely.
2. Optional follow-up if useful: sweep chunk budget (e.g. 1024, 768, 512, 384) at the
   long-uniform/no-whale population to locate where the saturation effect turns on, and
   whether it's monotonic.
3. Not planned unless requested: replicate either experiment 3×, or extend the saturation
   effect into a fleet-scale reserve-procurement number.
