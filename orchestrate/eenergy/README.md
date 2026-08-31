# orchestrate/eenergy

Orchestration scripts for the ACM e-Energy routing/coincidence-factor experiment (see
`../../paper-eenergy/README.md`). Empty for now — nothing built yet.

First thing needed here: a minimal request router in front of N vLLM replicas on the
8×4090 server, supporting at least two routing policies (whale-clustering /
whale-spreading) so their effect on per-GPU power correlation can be compared.
