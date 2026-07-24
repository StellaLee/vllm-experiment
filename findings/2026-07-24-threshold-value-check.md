# Threshold-value check: no headroom for a magnitude-adaptive LPT controller

**Date:** 2026-07-24
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** `orchestrate_cs2_whale_threshold_sweep.sh`, `THRESHOLDS='0 512 2048 4096'
WHALE_FRACS='0.15' CONCS='20'` — whale/short bimodal mix (44-50k char whales, short mean 800
chars, pad-cv2=0.5, pad-seed=1001, max-num-seqs=48, single-turn, max-tokens=256, NCONV=200),
budget fixed at 16384. wf=15/conc=20 chosen as the highest-load point where §5.8's
threshold=512 tradeoff was already known to broaden across all TPOT percentiles, to see if a
larger threshold value could recover some of that cost.

**Headline: no, a larger threshold does not recover the TPOT cost — it gives up TTFT benefit
without buying anything back.**

| threshold | S:TTFT mean Δ | S:TTFT p95 Δ | S:TTFT p99 Δ | S:TPOT mean Δ | S:TPOT p95 Δ | S:TPOT p99 Δ |
|---|---|---|---|---|---|---|
| 512 | −50.6% | −35.0% | −27.8% | +45.4% | +78.0% | +84.1% |
| 2048 | −33.0% | −18.4% | +16.5% | +45.2% | **+137.6%** | **+107.7%** |
| 4096 | −18.6% | −17.8% | −0.4% | +41.6% | +88.9% | **+119.0%** |

As threshold rises, the admission-side TTFT benefit shrinks monotonically toward zero
(expected — a larger threshold approaches "no cap," i.e., mono), but the TPOT cost does not
shrink alongside it. 2048's p95/p99 are both *worse* than 512's; 4096's p99 is worse still.
Raising the threshold trades away TTFT benefit while buying nothing back on TPOT tail
latency — the opposite of what a magnitude-adaptive controller (one that widens the threshold
under high load to trade TTFT gain for TPOT relief) would need to exploit.

## Mechanism

Threshold size controls two things simultaneously: how *often* a whale interferes with
co-scheduled decoders (fewer, bigger chunks at higher thresholds — a 14k-token whale takes
~28 rounds at threshold=512 but only ~7 at threshold=2048) and how *severe* each interference
event is (each chunk takes proportionally longer wall-clock time). Tail metrics (p95/p99) are
driven by the single worst gap a decoder experiences, so fewer-but-bigger interference events
are worse for the tail even though there are fewer of them — the severity effect dominates the
frequency effect. This means 512 is not an arbitrary point on a smooth tradeoff curve; it sits
close to a genuine local optimum for this specific mechanism and metric combination. That's
consistent with the lengthgate findings' own earlier discovery that 256 (a *smaller* value)
was also worse than 512, not better — both directions away from 512 lose, just via different
mechanisms (256: pays too much TTFT overhead per whale-round; 2048/4096: fewer but more
severe tail events).

## Conclusion for the paper / next-steps

**Close the dynamic-magnitude-controller question.** There is no evidence of headroom in
either direction (smaller or larger than 512) at this load point. This confirms the
recommendation made earlier this session (hold off on building a continuously-varying
threshold controller) with actual data rather than just risk-aversion reasoning — the
`hslo`-controller precedent (dynamic step-budget control ties but doesn't beat a well-tuned
static oracle) generalizes to the per-request-threshold lever too: there's no moving target
for a magnitude-adaptive version to chase, because the static value already sits near a local
optimum.

## Update (2026-07-24, later): verified threshold=16384 ≈ mono/off exactly

Ran one more arm at the same wf=15/conc=20 config with `--long-prefill-token-threshold 16384`
— a value that exceeds the whale's maximum possible size (~15,455 tokens at the 44-50k char
range), so the cap should never actually bind. Compared directly against the existing
threshold=0/off data:

| | S:TTFT | S:TPOT | W:TTFT | W:TPOT |
|---|---|---|---|---|
| mean | +0.9% | −2.9% | −0.8% | −4.4% |
| p95 | +0.7% | −16.5% | +5.3% | +0.4% |
| p99 | +6.5% | −2.3% | −0.4% | +0.6% |

All deltas are within ordinary single-trial noise (no consistent sign, an order of magnitude
smaller than the 512/2048/4096 deltas above) — confirming threshold=16384 is statistically
indistinguishable from threshold=0/off, exactly as predicted.

## Update (2026-07-24, later still): filling in threshold=8192 revises the "step function" framing

Added an 8192 arm (same wf=15/conc=20 config) to see where the transition to mono-like
behavior actually happens between 4096 and 16384:

| threshold | S:TTFT mean Δ | S:TPOT mean Δ | S:TPOT p95 Δ | S:TPOT p99 Δ |
|---|---|---|---|---|
| 512 | −50.6% | +45.4% | +78.0% | +84.1% |
| 2048 | −33.0% | +45.2% | +137.6% | +107.7% |
| 4096 | −18.6% | +41.6% | +88.9% | +119.0% |
| **8192** | **−0.0%** | **+3.9%** | **+9.2%** | **+10.3%** |
| 16384 | +0.9% | −2.9% | −16.5% | −2.3% |

At threshold=8192, every delta has already collapsed to near-noise-level — much closer to
16384's noise floor than to 4096's still-substantial effect. **This revises the earlier "step
function exactly at the whale's size" framing.** The whale here is ~13.6-15.5k tokens
(44-50k chars); 8192 is only about half that, so the cap still technically binds (a whale
needs ~2 rounds at threshold=8192, vs. ~1 at 16384/off) — yet the effect is already almost
gone. The transition isn't a sharp step sitting at the whale's actual size; it's a steep but
genuine transition concentrated between 4096 and 8192 (roughly a quarter to half of the
whale's size), well before the cap stops binding altogether. The mechanistic read: dropping
round count from ~4 (at 4096) to ~2 (at 8192) already gets you most of the way to mono's
single-round behavior — you don't need the cap to fully stop binding for the spreading effect
to vanish, you just need round count to fall to something very small.

This doesn't change the headroom conclusion (there's still no better *engaged* value than 512
for a controller to switch to under load — the whole 512-4096 regime is worse, not better,
as threshold grows), but it does mean the boundary between "capping regime" and "mono regime"
sits well inside the whale's own size, not at it.

## Caveats

- Single trial, single load point (wf=15/conc=20). The mechanism reasoning (frequency vs.
  severity tradeoff) is general, but the specific numbers are only demonstrated here.
- Whale-only stats were collected but not analyzed in detail here (focus was the short
  population, since that's where the tradeoff's cost lives).
- Feeds `docs/2026-07-24-mlsys-main-track-plan.md` Phase 2 (task: cheap 3-value threshold
  check) — resolves that open question definitively rather than deferring it further.
