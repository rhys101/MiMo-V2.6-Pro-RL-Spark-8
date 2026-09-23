# SGLang patches

Applied in this order by `docker/Dockerfile` on top of sglang `06008c17` (the
2026-09-23 `dev-cu13` nightly). Each must apply cleanly or the build fails.

| Patch | Source | Why |
|---|---|---|
| `40722.diff` | [sglang#40722](https://github.com/sgl-project/sglang/pull/40722) (open) | Picks a packed-MXFP4 MoE runner on SM12x instead of the FP8 Triton runner that cannot read the experts. We set the runner explicitly anyway. |
| `40888.diff` | [sglang#40888](https://github.com/sgl-project/sglang/pull/40888) (open) | With DFlash/MTP enabled, MiMo dropped image/audio embeddings when capturing aux hidden states. Needed for vision + speculation. |
| `38142.diff` | [sglang#38142](https://github.com/sgl-project/sglang/pull/38142) (open) | Sliding window and per-head sinks in Triton varlen prefill, used by the MiMo ViT (`--mm-attention-backend triton_attn`). |
| `40633.diff` | [sglang#40633](https://github.com/sgl-project/sglang/pull/40633) (open) | Strict (XGrammar) MiMo tool calling. |
| `40665.diff` | [sglang#40665](https://github.com/sgl-project/sglang/pull/40665) (open) | `--speculative-draft-kv-ratio` for the all-SWA DFlash draft pool (not enabled by default). |
| `29858.diff` | [sglang#29858](https://github.com/sgl-project/sglang/pull/29858) (open), **ported** | SWA window buffers for the EAGLE draft-extend CUDA graph. The PR predates a signature change; unported it crashes EAGLE-MTP startup (`'Tensor' object has no attribute 'fill_packed_read_stream'`). |
| `rocenante.diff` | DeepSeek V4.1 SG17 build, ported | Routes graph-captured TP8 BF16/FP32 all-reduces of at most 512 KiB through the b12x RoCEnante one-shot RDMA all-reduce (`SGLANG8_ROCE_ALLREDUCE=1`). |
| `mimo26-fast-expert-load.diff` | this repo | Skip routed experts owned by other EP ranks before converting their UE8M0 scales. Only matters with EP > 1. |
| `mimo26-marlin-skip-zero-fill.diff` | this repo | Marlin MoE zero-filled an `M*topk*K` scratch per layer (0.8 GB per MoE layer for an 8K prefill chunk). Without an expert map every row is written, so the fill is skipped. The MXFP4 Marlin path never uses atomic-add. |
