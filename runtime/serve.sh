#!/usr/bin/env bash
# Per-rank SGLang launcher for MiMo-V2.6-Pro-RL; runs inside the container.
# Every knob comes from the environment (see configs/cluster.env).
set -Eeuo pipefail

: "${NODE_RANK:?}" "${NNODES:=8}" "${TP_SIZE:=8}" "${EP_SIZE:=8}" "${DIST_INIT_ADDR:?}"
MODEL_PATH=${MODEL_PATH:-/models/MiMo-V2.6-Pro-RL}

args=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --served-model-name "${SERVED_MODEL_NAME:-mimo-v2.6-pro}"
  --tp "$TP_SIZE" --ep "$EP_SIZE"
  --nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR"
  --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-30000}"
  --attention-backend "${ATTN_BACKEND:-triton}"
  --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
  --moe-a2a-backend "${MOE_A2A_BACKEND:-none}"
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.85}"
  --context-length "${CONTEXT_LENGTH:-262144}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-8}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}"
  --page-size "${PAGE_SIZE:-64}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.1}"
  --reasoning-parser mimo --tool-call-parser mimo
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-1800}"
  --enable-metrics --enable-cache-report
)

if [ "${ENABLE_VISION:-1}" = 1 ]; then
  args+=(--enable-multimodal --mm-attention-backend "${MM_ATTN_BACKEND:-triton_attn}")
fi

case "${SPEC_ALGO:-DFLASH}" in
  DFLASH)
    args+=(--speculative-algorithm DFLASH
           --speculative-draft-model-path "$MODEL_PATH/dflash"
           --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-8}")
    [ -n "${SPEC_DRAFT_KV_RATIO:-}" ] && args+=(--speculative-draft-kv-ratio "$SPEC_DRAFT_KV_RATIO")
    ;;
  EAGLE)
    args+=(--speculative-algorithm EAGLE --enable-multi-layer-eagle
           --speculative-num-steps "${SPEC_STEPS:-3}" --speculative-eagle-topk 1
           --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-4}")
    ;;
  none|"") ;;
  *) echo "unknown SPEC_ALGO=$SPEC_ALGO" >&2; exit 2 ;;
esac

if [ "${CUDA_GRAPH:-1}" = 0 ]; then
  args+=(--disable-decode-cuda-graph)
else
  args+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS:-8}")
fi
[ "${CUDA_GRAPH_PREFILL:-0}" = 1 ] || args+=(--disable-prefill-cuda-graph)
# Unified memory: page cache from reading the shards competes with the KV pool.
[ "${DROP_CACHE_AFTER_LOAD:-1}" = 1 ] && args+=(--weight-loader-drop-cache-after-load)
[ -n "${FP8_GEMM_BACKEND:-}" ] && args+=(--fp8-gemm-backend "$FP8_GEMM_BACKEND")
[ -n "${PREFILL_ATTN_BACKEND:-}" ] && args+=(--prefill-attention-backend "$PREFILL_ATTN_BACKEND")
[ -n "${KV_CACHE_DTYPE:-}" ] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "${MAX_TOTAL_TOKENS:-}" ] && args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
# shellcheck disable=SC2206
[ -n "${EXTRA_SGLANG_ARGS:-}" ] && args+=(${EXTRA_SGLANG_ARGS})

echo "mimo26 rank $NODE_RANK: sglang $(cd /sgl-workspace/sglang && git rev-parse --short HEAD) + PRs $(tr '\n' ' ' < /opt/mimo26/applied-prs.txt)"
printf 'mimo26 args:'; printf ' %q' "${args[@]}"; echo
exec python3 -m sglang.launch_server "${args[@]}"
