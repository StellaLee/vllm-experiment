# Threat to Validity: TP=2 Host-Staged NCCL on RTX 4090

**Date:** 2026-07-20
**Purpose:** Document a communication-regime confound in the multi-chip (14B / TP=2)
crossover experiments, its direction of bias, how it should gate our conclusions, and
the control run required to bound it. Doubles as a draft for the paper's §8
(Limitations & Threats to Validity).

---

## Summary

The 14B mono-vs-chunk crossover experiments run **tensor-parallel size 2 (TP=2) on a
pair of RTX 4090s that have no NVLink, with PCIe peer-to-peer disabled**
(`torch.cuda.can_device_access_peer(0,1) == False`, verified 2026-07-20). Consequently
every NCCL all-reduce — which happens ~twice per layer, every forward pass — is
**staged through host memory** (GPU→CPU→GPU over PCIe) rather than over a direct
GPU-to-GPU link.

This is a threat to validity for the mono-vs-chunk comparison and for any claim we make
about "multi-chip" generalization.

## Why it biases the comparison — and in which direction

An all-reduce cost has two components:

- a **fixed per-call latency** (launch + synchronization), roughly independent of size;
- a **bandwidth term** proportional to the number of tokens in the step.

Chunk (budget 512) does **more, smaller steps** than mono (budget 16384), so it pays the
**fixed all-reduce latency more times**. Host-staging *inflates* that fixed latency
(the CPU bounce adds a size-independent penalty). The extra cost therefore lands
disproportionately on chunk — it is part of exactly the per-step overhead that makes
chunk lose at low Cs².

**Net direction: the 4090 host-staged NCCL penalty biases the comparison _against_
chunk.** (There is a smaller counter-effect — mono's larger all-reduces are more
bandwidth-bound, lengthening its blocking step, which favors chunk — but the extra
fixed-latency-per-step term on chunk most plausibly dominates.)

## How this must gate our conclusions

| Outcome of the crossover run | Interpretation under this confound |
|---|---|
| **Chunk wins at Cs²>1 (crossover present)** | **Conservative / robust.** Chunk won *despite* an inflated overhead; on NVLink hardware the effect would be *stronger*. Safe to report. |
| **No chunk win (null)** | **Confounded.** Cannot separate "decode-bound regime" from "4090 host-staged comms penalty suppressing chunk." A null here does **not** cleanly support "no genuine chunking win generalizes to multi-chip." |

There is also an **external-validity** issue independent of the result: production
multi-GPU serving typically uses NVLink/NVSwitch (A100/H100), where TP all-reduce is far
cheaper. So "multi-chip" as tested here is really **no-NVLink PCIe multi-chip**, an
atypically expensive communication regime. Generalizing to production multi-chip is not
automatic.

This matters specifically because the multi-chip move exists to answer the reviewer
"n=1 / single-chip scope" objection. A confounded null weakens exactly that
scope-hardening argument.

## Required control run (to bound the confound)

1. **7B single-GPU (no NCCL) vs 7B TP=2 (NCCL), identical padded Cs²-sweep workload.**
   14B does not fit on one 4090; 7B does, so this isolates and quantifies how much
   host-staged NCCL adds per step, and whether it moves the mono-vs-chunk delta.
2. **Quantify the NCCL fraction directly** — profile all-reduce time, or compare
   `--enforce-eager` vs cudagraph — and report "comms = X% of step latency."
3. **Write §8 with the direction-of-bias framing**: positive crossover = conservative,
   null = confounded, production NVLink would have lower overhead.
4. **(Stretch) validate the boundary on an NVLink box** (A100/H100 pair) if one becomes
   available.

## Draft limitations paragraph (for paper §8)

> Our multi-GPU results use tensor parallelism across two RTX 4090s, which lack NVLink
> and have PCIe peer-to-peer disabled; NCCL all-reduces are therefore host-staged. This
> inflates the per-step overhead paid by the chunked policy, which issues more, smaller
> steps. The bias is against chunking: a measured crossover is conservative and would be
> stronger under NVLink, whereas the absence of a crossover cannot be cleanly attributed
> to the serving regime rather than to the communication penalty. We bound this effect
> with a single-GPU-vs-TP control on a 7B model (App. X) and note that production
> multi-GPU deployments with NVLink/NVSwitch would exhibit lower communication overhead.

---

*Cross-reference: memory `project_tp_nccl_caveat`, `project_cs2_2x4090_plan`,
`project_reviewer_punchlist`. Server/topology details in `project_8x4090_server`
(GPUs 0,1 share a PCIe switch — the fastest available pair for TP on this box).*
