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

Measured **23 September 2026** on build `a6` (TP8 · EP1 · Marlin W4A16 MoE ·
DFlash-8 · RoCEnante all-reduce · FA4 prefill). Benchmark prompts and metrics are
identical to the [DeepSeek V4.1 Flash deployment](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8)
(`bench/mimobench.py`, temperature 0, thinking off).

| Per-stream decode tok/s | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| coding | **91.4** | 67.0 | 46.3 | 40.1 |
| json | 68.2 | 61.7 | 42.9 | 30.2 |
| math | 78.8 | 66.4 | 53.3 | 36.0 |
| format | 107.0 | 82.0 | 60.7 | 43.2 |
| reasoning | 49.2 | 36.7 | 23.4 | 18.0 |
| summary | 34.3 | 29.2 | 21.2 | 14.9 |
| prose | 32.3 | 23.4 | 16.9 | 11.9 |
| narrative | 26.3 | 19.3 | 14.6 | 9.4 |
| **all 8 categories, aggregate** | 52.9 | 82.3 | 119.0 | **171.1** |

Cold prefill (unique prefix, TTFT-based): **1,899 tok/s** at 5K tokens, 1,629 at 20K,
1,720 at 81K, 1,515 at 163K.

Long structured extraction (four CSV→JSON tasks, end to end): **93.3 tok/s**.

Free-form prose is where DFlash's draft is weakest. The EAGLE-MTP profile
(`SPEC_ALGO=EAGLE`, 3 steps × 4 tokens) trades structured speed for prose:
prose 35.0 / narrative 36.5 tok/s at C1, but coding 63.1.
[All builds and the experiment trail](docs/progress.md).

## Validation

| Check | Result |
|---|---|
| Smoke: arithmetic, reasoning split, tool call, image (red/blue halves) | pass on every build |
| GSM8K, first 200 test questions, greedy, thinking off | **96.5%** (a6), 97.0% (a4-marlin, a5-eagle) |
| Needle retrieval, 3 facts | pass at 127,803 tokens (a5, a6) and 244,168 tokens (a5) |
| Mixed workload: 300-integer JSON-schema generation + 3 short requests + forced tool call admitted mid-decode, ×3 | 3/3 pass (a5 EAGLE, a6 DFlash) |
| Model copy | identical sha256 manifest (155 files) on all 8 nodes |

GSM8K here is a regression check between kernel/back-end choices, not a
leaderboard score. Needle and mixed checks are finite; they do not establish
long unattended reliability. Audio/video input is not validated.

## Where the time goes

Decode, one DFlash verify step (8 tokens), rank 0, Marlin EP1 (`runtime/profile_decode.sh`):

| | ms/step | share |
|---|---:|---:|
| Marlin MXFP4 MoE | 35.5 | 42% — memory bound (~59 distinct experts/layer, ~10 GB/rank/step) |
| BF16 GEMMs (`o_proj`, lm_head, draft) | 17.1 | 20% |
| RoCEnante all-reduce (151/step) | 9.5 | 11% |
| FP8 block `qkv` (Triton) | 7.6 | 9% |
| attention | 2.9 | 3% |
| host gaps (GPU idle) | ~8 | 10% |

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
day-0 support, sglang#40448) plus nine patches: six open upstream PRs, the
RoCEnante all-reduce hook ported from the DeepSeek build, and two local fixes.
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
