"""Thin NVML wrapper -- the only module in this package that imports pynvml, so every
other module (and its tests) stays hardware-free. Mirrors scripts/pesim/power_logger.py's
sampling approach but exposes live per-GPU reads in-process for routing decisions, rather
than only writing a CSV trace. power_logger.py itself stays unmodified and keeps running as
an independent sidecar for the ground-truth experiment trace (see spec S3.1)."""
import time

import pynvml


class NvmlPowerReader:
    def __init__(self, gpu_indices: list):
        pynvml.nvmlInit()
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}

    def read(self, gpu_index: int) -> tuple:
        """Returns (power_w, timestamp)."""
        power_w = pynvml.nvmlDeviceGetPowerUsage(self._handles[gpu_index]) / 1000.0
        return power_w, time.time()

    def shutdown(self) -> None:
        pynvml.nvmlShutdown()
