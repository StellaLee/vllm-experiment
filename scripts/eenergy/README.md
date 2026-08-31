# scripts/eenergy

Scripts (router, hotpatches, analysis) for the ACM e-Energy routing/coincidence-factor
experiment — see `../../paper-eenergy/README.md`.

Router implemented in `router/` (DRF/LMETRIC/round-robin policies, see
`docs/superpowers/specs/2026-08-30-eenergy-routing-design.md`, local-only). Live-verified
against a real vLLM replica on the 8x4090 server 2026-08-31 (Task 9 Step 6 smoke test),
which caught and fixed a real bug: replicas must run
`vllm.entrypoints.openai.api_server`, not the legacy `vllm.entrypoints.api_server` — the
router proxies `/v1/completions`, which only the OpenAI-compatible server exposes.

**Ramp-ceiling calibration (2026-08-31):** `RAMP_CEILING_W_PER_S` defaults to 450.0 in
`orchestrate/eenergy/launch_router_experiment.sh`, calibrated by bursting 24 concurrent
45k-char prefills at one idle 7B replica on GPU 0 and sampling power at the router's actual
500ms poll cadence. Steady-state ramp is near-zero (p50=0.1 W/s); nearly all signal is in
the idle->loaded transition itself (p99=212 W/s, p99.9/max=433 W/s — observed 62W -> 235W
-> 449W across two consecutive 500ms polls). 450 is the observed max with slight headroom.
Single-GPU measurement, assumed to generalize across the other 7 (same hardware/host,
4090s don't share a per-GPU power budget). Re-calibrate if replica count, model size, or
`ROUTER_POWER_INTERVAL_S` changes materially — the transition sharpness depends on how
fast a batch actually saturates the GPU, which scales with model/batch size.
