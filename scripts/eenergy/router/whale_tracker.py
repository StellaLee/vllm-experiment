"""Tracks active (in-flight) whale-request counts per replica, updated from the same
dispatch/complete lifecycle events LoadTracker uses -- no polling. Feeds pick_p2c_whale's
"fewer currently-active whales" comparison; ignored by every other policy, same as how
ramp_rate_w_per_s is always computed but only pick_drf reads it."""


class WhaleTracker:
    def __init__(self):
        self._counts: dict = {}

    def on_dispatch(self, replica_id: str, is_whale: bool) -> None:
        if not is_whale:
            return
        self._counts[replica_id] = self._counts.get(replica_id, 0) + 1

    def on_complete(self, replica_id: str, is_whale: bool) -> None:
        if not is_whale:
            return
        self._counts[replica_id] = max(0, self._counts.get(replica_id, 0) - 1)

    def active_whale_count(self, replica_id: str) -> int:
        return self._counts.get(replica_id, 0)

    def active_whale_count_if_dispatched(self, replica_id: str) -> int:
        """What-if: active whale count if one more whale were dispatched here. Does not
        mutate tracker state -- used to score candidates before a decision is made."""
        return self.active_whale_count(replica_id) + 1
