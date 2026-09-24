#!/usr/bin/env python3
"""Per-rank kernel time by category for one profile (traces gathered into DIR/rankN/)."""
import collections, glob, gzip, json, re, sys
d, steps = sys.argv[1], int(sys.argv[2])
cats = [("roce", r"roce|Roce"), ("nccl", r"nccl"), ("moe", r"marlin|Marlin|GroupProblemShape"), ("bf16", r"wmma|nvjet|gemv|splitK"),
        ("fp8", r"w8a8|fp8"), ("attn", r"_fwd_kernel|attn|flash")]
print("%-6s" % "rank" + "".join("%9s" % c for c, _ in cats) + "%9s%9s" % ("other", "busy"))
for rd in sorted(glob.glob(d + "/rank*")):
    f = sorted(glob.glob(rd + "/*.gz"))[-1]
    ev = [e for e in json.load(gzip.open(f))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    t = collections.defaultdict(float)
    for e in ev:
        for c, rx in cats:
            if re.search(rx, e["name"]): t[c] += e["dur"]; break
        else: t["other"] += e["dur"]
    tot = sum(t.values())
    print("%-6s" % rd.rsplit("/", 1)[1] + "".join("%9.2f" % (t[c] / 1e3 / steps) for c, _ in cats) + "%9.2f%9.2f" % (t["other"] / 1e3 / steps, (tot - t["roce"] - t["nccl"]) / 1e3 / steps))
