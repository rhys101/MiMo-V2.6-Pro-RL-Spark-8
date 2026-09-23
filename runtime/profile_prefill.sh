#!/usr/bin/env bash
# Profile a cold prefill of ~N tokens on rank 0 and group GPU kernel time.
# Usage: profile_prefill.sh LABEL [tokens]
set -euo pipefail
label=${1:?label}; tokens=${2:-20000}
dir=$HOME/mimo-v2.6-pro/cache/prof-$label; rm -rf "$dir"
curl -s -XPOST localhost:30000/flush_cache >/dev/null
python3 - "$tokens" > /tmp/prefill-req.json <<'PY'
import json, random, sys
n = int(sys.argv[1]); r = random.Random(n)
words = "amber basin cedar delta ember fjord garnet harbor iris juniper".split()
text = " ".join(f"{r.choice(words)}{r.randint(0,999)}" for _ in range(int(n / 4.6)))
print(json.dumps({"model": "mimo-v2.6-pro", "messages": [{"role": "user", "content": text + "\nSummarise in one word."}],
                  "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))
PY
curl -s -XPOST localhost:30000/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"/root/.cache/prof-$label\",\"num_steps\":$(( tokens / 8192 + 2 )),\"activities\":[\"GPU\"]}" >/dev/null
t0=$(date +%s.%N); curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/prefill-req.json | python3 -c "import json,sys; print('prompt_tokens', json.load(sys.stdin)['usage']['prompt_tokens'])"
echo "wall $(echo "$(date +%s.%N) - $t0" | bc) s"
curl -s -XPOST localhost:30000/stop_profile >/dev/null || true
for _ in $(seq 90); do ls "$dir"/*TP-0*.gz >/dev/null 2>&1 && break; sleep 1; done; sleep 5
python3 - "$dir" <<'PY'
import gzip, json, collections, glob, sys, re
f = sorted(glob.glob(sys.argv[1] + "/*TP-0*.gz"))[-1]
ev = [e for e in json.load(gzip.open(f))["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
span = max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev); busy = sum(e["dur"] for e in ev)
print("span %.1f ms  gpu busy %.1f ms (%.0f%%)" % (span / 1e3, busy / 1e3, 100 * busy / span))
top = collections.defaultdict(lambda: [0, 0])
for e in ev: top[e["name"][:100]][0] += e["dur"]; top[e["name"][:100]][1] += 1
for n, (d, c) in sorted(top.items(), key=lambda x: -x[1][0])[:16]: print("  %8.1f ms %6d  %s" % (d / 1e3, c, n))
PY
