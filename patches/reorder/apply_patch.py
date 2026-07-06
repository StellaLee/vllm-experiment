#!/usr/bin/env python3
"""Apply prefix-aware request reordering patch to vLLM scheduler.

Makes two insertions to vllm/v1/core/sched/scheduler.py:
  1. __init__: read PREFIX_REORDER env var, set self._prefix_reorder
  2. schedule(): sort self.waiting by cached prefix length before the loop
"""
import os
import sys
import shutil
import difflib

INIT_ANCHOR = "        else:\n            self._chunk_ctrl = None\n"
INIT_INSERT = (
    "        _prefix_reorder = os.getenv(\"PREFIX_REORDER\", \"0\").strip() == \"1\"\n"
    "        self._prefix_reorder = _prefix_reorder\n"
    "        if _prefix_reorder:\n"
    "            logger.info(\"Prefix-aware request reordering enabled\")\n"
)

SCHED_ANCHOR = "            step_skipped_waiting = create_request_queue(self.policy)\n"
SCHED_INSERT = (
    "            if self._prefix_reorder and self.waiting and hasattr(self.waiting, 'extendleft'):\n"
    "                def _cached_tokens(req):\n"
    "                    if req.num_computed_tokens > 0:\n"
    "                        return req.num_computed_tokens\n"
    "                    _, n = self.kv_cache_manager.get_computed_blocks(req)\n"
    "                    return n\n"
    "                _sorted = sorted(self.waiting, key=_cached_tokens, reverse=True)\n"
    "                self.waiting.clear()\n"
    "                self.waiting.extend(_sorted)\n"
)

ALREADY_APPLIED = "self._prefix_reorder"


def patch_file(path):
    with open(path) as f:
        original = f.read()

    if ALREADY_APPLIED in original:
        print(f"[reorder] Already applied — skipping {path}")
        return False

    if INIT_ANCHOR not in original:
        print(f"ERROR: init anchor not found in {path}")
        sys.exit(1)
    if SCHED_ANCHOR not in original:
        print(f"ERROR: schedule anchor not found in {path}")
        sys.exit(1)

    # Back up
    shutil.copy2(path, path + ".bak.reorder")

    patched = original.replace(INIT_ANCHOR, INIT_ANCHOR + INIT_INSERT, 1)
    patched = patched.replace(SCHED_ANCHOR, SCHED_ANCHOR + SCHED_INSERT, 1)

    with open(path, "w") as f:
        f.write(patched)

    print(f"[reorder] Patched {path}")
    return True


def make_diff(path):
    bak = path + ".bak.reorder"
    with open(bak) as f:
        orig_lines = f.readlines()
    with open(path) as f:
        new_lines = f.readlines()
    diff = difflib.unified_diff(
        orig_lines, new_lines,
        fromfile="a/vllm/v1/core/sched/scheduler.py",
        tofile="b/vllm/v1/core/sched/scheduler.py",
    )
    return "".join(diff)


def main():
    python = sys.executable
    import subprocess
    site = subprocess.check_output(
        [python, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
    ).strip()

    install_path = os.path.join(site, "vllm/v1/core/sched/scheduler.py")
    src_path = "/root/vllm/vllm/v1/core/sched/scheduler.py"

    patched = False
    for p in [install_path, src_path]:
        if os.path.exists(p):
            patched = patch_file(p) or patched
        else:
            print(f"[reorder] Skipping (not found): {p}")

    if patched:
        diff = make_diff(install_path if os.path.exists(install_path) else src_path)
        diff_path = os.path.join(os.path.dirname(__file__), "../patches/reorder/scheduler.patch")
        os.makedirs(os.path.dirname(diff_path), exist_ok=True)
        with open(diff_path, "w") as f:
            f.write(diff)
        print(f"[reorder] Patch saved to {diff_path}")


if __name__ == "__main__":
    main()
