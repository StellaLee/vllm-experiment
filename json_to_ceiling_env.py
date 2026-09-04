import json, sys
d = json.load(open(sys.argv[1]))
print(",".join(f"{gpu}:{v['ceiling_w_per_s']}" for gpu, v in d.items()))
