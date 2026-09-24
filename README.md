# MiMo-V2.6-Pro-RL on 8 × DGX Spark

Run Xiaomi's [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)
(1.02T total / 42B active MoE, MXFP4 experts, hybrid SWA attention, omni
encoders) across eight NVIDIA DGX Sparks with SGLang: TP8 over RoCE, packed
MXFP4 experts on Marlin, DFlash speculative decoding, vision input and an
OpenAI-compatible API.

> **Attribution:** this project builds on the earlier 8 × DGX Spark MiMo-V2.6-Pro-RL
> work by [Aevonix](https://github.com/Aevonix/mimo-2.6-dgx-spark), which uses a
> patched vLLM tuned for numerical stability. Their findings shaped choices here
> (Marlin W4A16 experts, DFlash, the mixed-workload crash check). This repository takes a different route: patched SGLang tuned for throughput,
> on the same checkpoint and hardware.

## Performance

Measured **24 September 2026** on build `a13` (TP8 · EP1 · Marlin W4A16 MoE ·
DFlash-8 with **per-request adaptive verify width** · RoCEnante all-reduce · FA4
prefill and verify · FP8 `o_proj` and DFlash draft · GB10-tuned FP8 GEMM tiles).
Benchmark prompts and metrics are identical to the
[DeepSeek V4.1 Flash deployment](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8)
(`bench/mimobench.py`, temperature 0, thinking off).

| Per-stream decode tok/s | C1 | C8 | a9 C1 | a9 C8 |
|---|---:|---:|---:|---:|
| format | **99.5** | 46.3 | 111.2 | 45.8 |
| json | 91.2 | 33.6 | 67.1 | 29.0 |
| coding | 91.1 | 42.3 | 88.9 | 41.3 |
| math | 86.6 | 37.7 | 87.9 | 33.8 |
| reasoning | 56.9 | 20.4 | 58.8 | 19.2 |
| summary | 51.3 | 16.9 | 41.8 | 15.4 |
| prose | 40.3 | 16.9 | 32.3 | 13.2 |
| narrative | 36.0 | 14.1 | 28.9 | 10.2 |
| **mean of 8 categories** | **69.1** | 28.5 | 64.6 | 26.0 |
| **all 8 categories, aggregate** | 59.9 | **190.8** | 55.5 | 176.5 |

Single benchmark requests move ±20–30% between builds because greedy output, and
with it DFlash acceptance, changes with small numeric differences. Compare the
means, and the repeatable long-extraction measurement:

Long structured extraction (four CSV→JSON tasks, cold cache, end to end,
prefill included, 3 repeats, 12/12 correct): **104.7 tok/s** on a13, 101.7 on
a9, 95.0 on a6.

Cold prefill (unique prefix, TTFT-based, a6): **1,899 tok/s** at 5K tokens, 1,629
at 20K, 1,720 at 81K, 1,515 at 163K.

**Adaptive verify width.** DFlash drafts 8 tokens per step, but on free prose
only ~2 of them are accepted, so most of the verify step's expert reads are
wasted. With `MIMO26_DFLASH_ADAPTIVE=1` (the default profile) each request keeps
a running average of accepted drafts per step; while every request in the batch
is below 1.8 the target verifies only the first 4 draft tokens, and it returns to
8 once any request climbs above 2.6. Against a9, prose, narrative and summaries
gain 23–25% at C1 and the C8 aggregate rises 8%. Code and math mostly keep the full
window; format is 11% lower at C1, so some high-acceptance steps still narrow.
a13 also keeps the draft's fused `qkv` in BF16, which re-enables DFlash's fused KV
materialization; part of the json gain likely comes from that. DFlash now beats
the EAGLE-MTP profile (`SPEC_ALGO=EAGLE`, 3 steps × 4 tokens: prose 35.0,
narrative 36.5, coding 63.1) on prose and roughly matches it on narrative, without
its loss on code.
[All builds and the experiment trail](docs/progress.md).

## Validation

| Check | Result |
|---|---|
| Smoke: arithmetic, reasoning split, tool call, image (red/blue halves) | pass on every build |
| GSM8K, first 200 test questions, greedy, thinking off | **98.0%** (a13), 97.5% (a9, a7), 96.5% (a6), 97.0% (a4-marlin, a5-eagle) |
| Needle retrieval, 3 facts | pass at 127,803 tokens (a5, a6) and 244,168 tokens (a5) |
| Mixed workload: 300-integer JSON-schema generation + 3 short requests + forced tool call admitted mid-decode, ×3 | 3/3 pass (a5 EAGLE, a6 DFlash) |
| Model copy | identical sha256 manifest (155 files) on all 8 nodes |

GSM8K here is a regression check between kernel/back-end choices, not a
leaderboard score. Needle and mixed checks are finite; they do not establish
long unattended reliability. Audio/video input is not validated.

## Where the time goes

Decode, one full-width DFlash verify step (8 tokens), rank 0, a9 (`runtime/profile_decode.sh`).
The real step is ~70 ms; GPU idle time is under 1% (`runtime/trace_gaps.py`).

| | ms/step | share |
|---|---:|---:|
| Marlin MXFP4 MoE | 35.7 | 52% — memory bound (~59 distinct experts/layer, ~10 GB/rank/step) |
| FP8 block GEMMs (`qkv`, `o_proj`, draft) | 13.6 | 20% |
| RoCEnante all-reduce (151/step) | 9.8 | 14% |
| BF16 GEMMs (lm_head, remaining draft and dense) | 5.4 | 8% |
| attention, norms, other | 3.1 | 5% |

Against a6, FP8 `o_proj` and FP8 draft weights take the BF16/FP8 GEMM share from
24.3 to 19.0 ms per step. MoE only gets cheaper with fewer verified tokens or
fewer bits per expert; adaptive verify width (a13) takes the first route on
low-acceptance steps.

Prefill is bound by the fabric (ring all-reduce tops out at ~23.5 GB/s bus
bandwidth over 2 × 200G; ~32%) and Marlin at large M (~34%).

## Eight Sparks and a switch

- 8 × DGX Spark (GB10, sm_121, 128 GB unified memory), two ConnectX-7 200G fabric
  ports each on two switched subnets (MTU 9000, RoCE v2).
- The head node hosts the API and rank 0. All ranks hold a full local model copy
  (535 GB); about 66–69 GB of weights per rank.
- DeepSeek V4.1 Flash and MiMo cannot be resident at the same time.

## Quick start

On the head node, with the model at `~/models/MiMo-V2.6-Pro-RL` on every node
(`scripts/copy-model.sh` fans it out over the fabric only, with a route guard
that refuses the management NIC):

```bash
git clone https://github.com/rhys101/MiMo-V2.6-Pro-RL-Spark-8.git ~/mimo-v2.6-pro
cd ~/mimo-v2.6-pro
cp configs/cluster.env configs/cluster.local.env
$EDITOR configs/cluster.local.env    # user, nodes, fabric IPs, paths (git-ignored)
scripts/cluster.sh build             # builds on BUILD_NODE from the pinned base
scripts/cluster.sh distribute        # docker save | load over the fabric
scripts/cluster.sh serve             # workers first, then rank 0; waits for /health
python3 runtime/smoke.py             # text, reasoning, tool call, image, speed
```

API: `http://<head-node>:30000/v1`, model `mimo-v2.6-pro`. Recommended sampling from
the model card is `temperature=1.0, top_p=0.95`. `scripts/cluster.sh status|stop|logs N`.

Profiles are environment overrides, e.g.:

```bash
EXTRA_ENV="SPEC_ALGO=EAGLE SPEC_STEPS=3 SPEC_DRAFT_TOKENS=4" scripts/cluster.sh serve   # prose-leaning
EXTRA_ENV="MOE_RUNNER_BACKEND=flashinfer_mxfp4" scripts/cluster.sh serve                # CUTLASS W4A8 MoE
```

Boot takes 12–14 minutes, mostly weight loading.

## What is in the image

Pinned base `lmsysorg/sglang` dev-cu13 (sglang `06008c17`, which carries MiMo-V2.6
day-0 support, sglang#40448) plus eleven patches: six open upstream PRs, the
RoCEnante all-reduce hook ported from the DeepSeek build, and four local changes
(fast expert load, Marlin zero-fill skip, optional FP8 serving of BF16 linears,
adaptive DFlash verify width).
It also carries Triton block-FP8 tile tables tuned on GB10 for MiMo's per-rank
shapes (`patches/fp8-configs/`).
See [patches/README.md](patches/README.md) and [versions.lock.json](versions.lock.json).

## Credits and licenses

Built on [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0) and the
[DeepSeek V4.1 Flash 8-Spark work](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8),
with RoCEnante from [b12x](https://github.com/local-inference-lab/b12x) by Luke
Alonso (Apache-2.0, frozen snapshot in `third_party/b12x`). The benchmark is
adapted from [tonyd2wild's](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
(MIT). Configuration choices were informed by
[MiaAI-Lab's 2×Spark MiMo-V2.6-Flash kit](https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks)
and [Aevonix's 8×Spark vLLM recipe](https://github.com/Aevonix/mimo-2.6-dgx-spark), whose
benchmark suite supplied the long structured-extraction tasks; no code is copied
from either. Model weights remain under XiaomiMiMo's license.
