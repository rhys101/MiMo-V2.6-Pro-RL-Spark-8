#!/usr/bin/env python3
"""GPU idle-gap analysis of a rank-0 torch-profiler trace.

Merges kernel intervals from all streams, lists the largest idle gaps with the
kernels on either side, and sums idle time by (before -> after) kernel pair.
usage: trace_gaps.py TRACE_DIR STEPS [MIN_GAP_US]
"""
import collections, glob, gzip, json, sys

d, steps = sys.argv[1], int(sys.argv[2])
min_gap = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
f = sorted(glob.glob(d + "/*TP-0*.gz"))[-1]
ev = [e for e in json.load(gzip.open(f))["traceEvents"]
      if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])
span = ev[-1]["ts"] + ev[-1]["dur"] - ev[0]["ts"]
busy_union = 0.0
gaps = []
cur_s, cur_e, last = ev[0]["ts"], ev[0]["ts"] + ev[0]["dur"], ev[0]
for e in ev[1:]:
    s, t = e["ts"], e["ts"] + e["dur"]
    if s > cur_e:
        busy_union += cur_e - cur_s
        if s - cur_e >= min_gap:
            gaps.append((s - cur_e, last["name"][:70], e["name"][:70]))
        cur_s, cur_e = s, t
    else:
        cur_e = max(cur_e, t)
    if t >= cur_e:
        last = e
busy_union += cur_e - cur_s
idle = span - busy_union
print("span %.1f ms (%.2f/step)  GPU busy(union) %.1f ms  idle %.1f ms (%.1f%%, %.2f ms/step)  gaps>=%.0fus: %d"
      % (span / 1e3, span / 1e3 / steps, busy_union / 1e3, idle / 1e3, 100 * idle / span, idle / 1e3 / steps, min_gap, len(gaps)))
pair = collections.defaultdict(lambda: [0.0, 0])
for g, a, b in gaps:
    pair[(a, b)][0] += g; pair[(a, b)][1] += 1
print("idle by boundary (ms/step, count):")
for (a, b), (g, c) in sorted(pair.items(), key=lambda x: -x[1][0])[:15]:
    print("  %6.2f ms/step %4d  %s  ->  %s" % (g / 1e3 / steps, c, a, b))
