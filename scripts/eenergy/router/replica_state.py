"""Per-replica configuration and mutable runtime state for the e-Energy router. Kept as
plain dataclasses with no I/O so every other module in this package can be unit tested
without a real vLLM replica or GPU present."""
from dataclasses import dataclass, field


@dataclass
class ReplicaConfig:
    replica_id: str
    host: str
    port: int
    gpu_index: int
    token_budget: int           # vLLM's max_num_scheduled_tokens for this replica
    max_num_seqs: int           # vLLM's max_num_seqs for this replica
    ramp_ceiling_w_per_s: float  # calibrated ramp-rate ceiling for Share_power normalization


@dataclass
class ReplicaState:
    config: ReplicaConfig
    in_flight: int = 0                                  # BS proxy: dispatched, not yet complete
    cached_block_hashes: set = field(default_factory=set)
    last_power_w: float = 0.0
    last_power_ts: float = 0.0
    ramp_rate_w_per_s: float = 0.0
