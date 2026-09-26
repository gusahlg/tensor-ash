import json, sys, statistics
d = json.load(sys.stdin)
for r in d:
    s = r["samples_ts"]
    print("pp512 samples t/s:", " ".join(f"{x:.0f}" for x in s))
    print(f"median {statistics.median(s):.0f} mean {statistics.mean(s):.0f} min {min(s):.0f} max {max(s):.0f} build {r['build_commit']} gpu {r['gpu_info']}")
