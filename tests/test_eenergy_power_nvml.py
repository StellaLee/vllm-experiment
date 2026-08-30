import importlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))


def _install_fake_pynvml():
    fake = types.ModuleType("pynvml")
    calls = {"init": 0, "shutdown": 0}

    def nvmlInit():
        calls["init"] += 1

    def nvmlDeviceGetHandleByIndex(i):
        return f"handle-{i}"

    def nvmlDeviceGetPowerUsage(handle):
        return 250000  # milliwatts

    def nvmlShutdown():
        calls["shutdown"] += 1

    fake.nvmlInit = nvmlInit
    fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    fake.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    fake.nvmlShutdown = nvmlShutdown
    sys.modules["pynvml"] = fake
    return calls


def test_read_converts_milliwatts_to_watts_and_inits_once():
    calls = _install_fake_pynvml()
    import power_nvml
    importlib.reload(power_nvml)
    reader = power_nvml.NvmlPowerReader([0, 1])
    power_w, ts = reader.read(0)
    assert power_w == 250.0
    assert ts > 0
    assert calls["init"] == 1


def test_shutdown_calls_nvml_shutdown():
    calls = _install_fake_pynvml()
    import power_nvml
    importlib.reload(power_nvml)
    reader = power_nvml.NvmlPowerReader([0])
    reader.shutdown()
    assert calls["shutdown"] == 1
