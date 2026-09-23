# Progress and experiment trail

All runs on 23 September 2026, same eight Sparks, same checkpoint revision.
"Smoke" speeds use `runtime/smoke.py`'s own prompts (LRU-cache code, lighthouse
story, 256 tokens) and only compare builds with each other. Benchmark figures are
`bench/mimobench.py` per-stream C1 decode.

| Build | Change | smoke code / prose | bench C1 coding | C1 mean (8 cat.) | C8 aggregate | Notes |
|---|---|---:|---:|---:|---:|---|
| a2 | Baseline: dev-cu13 + 6 PRs, TP8/EP8, CUTLASS W4A8 MoE, DFlash-8, NCCL | 40.8 / 19.8 | — | — | — | functional checks pass |
| a3 | + RoCEnante TP8 all-reduce | 52.4 / 20.3 | 62.2 | 43.1 | 148.8 | NCCL all-reduce was 38% of a verify step |
| a4 | + fast expert load; **EP1** (TP-sharded experts) | 58.3 / 23.4 | 72.2 | 48.9 | **171.6** | balanced expert load, less all-reduce wait |
| a4-marlin | **Marlin W4A16 MoE** | 67.2 / 27.0 | 78.0 | 57.6 | 161.8 | GSM8K-200 97.0%; json 50.8 → 68.6 |
| a5-eagle | EAGLE-MTP 3×4 instead of DFlash-8 (+ ported #29858) | 56.9 / 38.1 | 63.1 | 51.0 | 161.1 | prose 35.0, narrative 36.5; GSM8K 97.0% |
| **a6** | DFlash-8 + Marlin zero-fill skip + **FA4 prefill** | 72.2 / 27.4 | **91.4** | **60.9** | 171.1 | GSM8K 96.5%; prefill 1.5–1.9K tok/s |

## Findings

- **EP1 beat EP8.** Synchronous all-reduces wait for the slowest rank; with EP8,
  whichever rank received the most routed experts set the pace. Pure TP expert
  sharding reads the same bytes per rank but evenly.
- **Marlin W4A16 beat FlashInfer CUTLASS W4A8 at small M**, and draft acceptance
  rose (json +35%), consistent with BF16 activations tracking the draft better than
  MXFP8-quantized ones. CUTLASS is still ahead at C8 aggregate (171.6 vs 161.8).
  Both weight layouts cannot be resident together.
- **DFlash vs EAGLE-MTP** splits by workload: DFlash for code/structured output,
  EAGLE for free prose. `ceiling_count` shows DFlash's ceiling (112.6 vs 65.6).
- **NCCL protocol does not matter for prefill.** Standalone 8-node sweep
  (`runtime/nccl_sweep.sh`): default, `Simple` and 16 channels all reach ~23.5 GB/s
  bus bandwidth at 50–100 MB; forcing `LL` drops to 6.6 GB/s.
- **FA4 prefill on sm121 works** (SGLang's `flash_attention_v4_sm120`), keeps SWA
  block skipping (per-chunk SWA call time constant at 0.47 ms through 58K tokens),
  and cut attention in an 18K prefill from 1.27 s to 0.30 s.
- **Weight loading** is 8–13 minutes and CPU-bound in the loader, not disk bound
  (direct reads run at 10.4 GB/s). Not yet fixed for EP1.

## Invalid or excluded measurements

- a4-ep1 prefill at 2,000/8,000 targets: radix-cache hits from an interrupted earlier run.
- a5 needle at 128K time (55.9 s): an interrupted identical request had partly
  filled the radix cache. Retrieval correctness stands; the time does not.
- First decode profile per build sometimes included a lagging-rank stall (27 ms
  eager NCCL call); profiles quoted are clean repeats.

## Not done / next

- A Triton small-M BF16 kernel for `o_proj`: microbenchmark says ~3 ms/step (3.5%).
- Newer b12x RoCEnante (prepared-plan API, traffic class) in place of the SG17 snapshot.
- Loader speed for EP1, prefill MoE at large M, audio/video validation, long soak.
