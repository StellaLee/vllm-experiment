import re
p = "/root/miniconda3/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py"
s = open(p).read()
if '.info(\n                "ChunkCtrl[slo]' in s or '.info(\n                    "ChunkCtrl[slo]' in s:
    print("already INFO-level")
else:
    new, n = re.subn(r'\.debug\((\s*"ChunkCtrl\[slo\])', r'.info(\1', s)
    if n:
        open(p, "w").write(new)
        print(f"flipped ChunkCtrl[slo] log debug->info ({n} site)")
    else:
        print("WARN: ChunkCtrl[slo] debug log not found -- budget trajectory won't be visible")
