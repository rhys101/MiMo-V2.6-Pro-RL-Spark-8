#!/usr/bin/env bash
# Profile N steady-state decode steps on rank 0 (mid-generation, no prefill) and
# group GPU kernel time by category. Usage: profile_decode.sh LABEL [steps] [prompt]
set -euo pipefail
label=${1:?label}; steps=${2:-10}; prompt=${3:-Write a Python function implementing an LRU cache class with get/put, with type hints and docstrings. Code only.}
dir=$HOME/mimo-v2.6-pro/cache/prof-$label; rm -rf "$dir"
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"mimo-v2.6-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":600,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null &
sleep 2
curl -s -XPOST localhost:30000/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"/root/.cache/prof-$label\",\"num_steps\":$steps,\"activities\":[\"GPU\"]}" >/dev/null
wait
for _ in $(seq 60); do ls "$dir"/*TP-0*.gz >/dev/null 2>&1 && break; sleep 1; done; sleep 3
python3 - "$dir" "$steps" <<'PY'
import gzip, json, collections, glob, sys, re
f = sorted(glob.glob(sys.argv[1] + "/*TP-0*.gz"))[-1]; steps = int(sys.argv[2])
ev = [e for e in json.load(gzip.open(f))["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
span = max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)
busy = sum(e["dur"] for e in ev)
rules = [("allreduce-roce", r"roce|Roce"), ("allreduce/gather-nccl", r"nccl"), ("moe-gemm", r"GroupProblemShape|fused_moe|marlin|Marlin"),
         ("moe-misc", r"MoeRouting|expandInputRows|doActivation|ExpertMaps|computeStrides|router|quantize_with_block|topk"),
         ("bf16-gemm (o_proj/lm_head/draft)", r"wmma|nvjet|gemv|splitKreduce|cublas|gemm_bf16"),
         ("fp8-gemm (qkv/dense)", r"w8a8|fp8"), ("attention", r"_fwd_kernel|attn|attention|decode_grouped|flash"),
         ("norm/rope/kv-store", r"rmsnorm|RMSNorm|rotary|kvcache|store_kv"), ]
cat = collections.defaultdict(float); top = collections.defaultdict(lambda: [0, 0])
for e in ev:
    n = e["name"]; top[n[:90]][0] += e["dur"]; top[n[:90]][1] += 1
    for c, rx in rules:
        if re.search(rx, n): cat[c] += e["dur"]; break
    else: cat["other"] += e["dur"]
print("steps %d  span %.1f ms (%.1f ms/step)  gpu busy %.1f ms (%.0f%%)" % (steps, span/1e3, span/1e3/steps, busy/1e3, 100*busy/span))
for c, d in sorted(cat.items(), key=lambda x: -x[1]): print("  %-34s %7.2f ms/step  %4.1f%%" % (c, d/1e3/steps, 100*d/busy))
print("top kernels:")
for n, (d, c) in sorted(top.items(), key=lambda x: -x[1][0])[:12]: print("  %7.2f ms/step %5d  %s" % (d/1e3/steps, c, n))
PY
