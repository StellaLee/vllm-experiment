"""Tracks in-flight (running+queued, from the router's own vantage point) request counts
per replica, updated purely from the request lifecycle events the router already observes
(dispatch when it picks a replica, complete when that request's response finishes) -- no
polling of the replica needed. This is LMETRIC's BS term."""


class LoadTracker:
    def __init__(self):
        self._counts: dict = {}

    def on_dispatch(self, replica_id: str) -> None:
        self._counts[replica_id] = self._counts.get(replica_id, 0) + 1

    def on_complete(self, replica_id: str) -> None:
        self._counts[replica_id] = max(0, self._counts.get(replica_id, 0) - 1)

    def in_flight(self, replica_id: str) -> int:
        return self._counts.get(replica_id, 0)

    def in_flight_if_dispatched(self, replica_id: str) -> int:
        """What-if: in-flight count if one more request were dispatched here. Does not
        mutate tracker state -- used to score candidates before a decision is made."""
        return self.in_flight(replica_id) + 1
