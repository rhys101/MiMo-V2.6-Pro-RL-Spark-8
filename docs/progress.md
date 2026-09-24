# Progress and experiment trail

All runs on 23–24 September 2026, same eight Sparks, same checkpoint revision.
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
| a6 | DFlash-8 + Marlin zero-fill skip + **FA4 prefill** | 72.2 / 27.4 | **91.4** | **60.9** | 171.1 | GSM8K 96.5%; prefill 1.5–1.9K tok/s |
| a6-b4 | DFlash verify window 4 instead of 8 | — | 65.8 | 50.7 | 174.2 | prose 32.3 → 41.3, summary +21%; code −28%, format −42% |
| a7 | + **FP8 `o_proj`** (block 128×128, fp32 scales) | 74.5 / 28.5 | 86.2 | 62.2 | 170.7 | GSM8K 97.5%; BF16+FP8 GEMMs 24.3 → 21.1 ms/step |
| a9 | + FP8 DFlash draft linears + GB10-tuned FP8 tiles | 70.9 / 29.1 | 88.9 | 64.6 | 176.5 | GSM8K 97.5%; long extraction 101.7; GEMMs 19.0 ms/step |
| a11 | draft `qkv_proj` back to BF16 (re-enables fused KV materialization) | 76.5 / 30.3 | — | — | — | smoke only |
| **a13** | + **adaptive DFlash verify width** (4 or 8 per step), FA4 verify width-aware | 63.8 / 38.8 | 91.1 | **69.1** | **190.8** | GSM8K 98.0%; long extraction **104.7**; prose 40.3, narrative 36.0; C8 long extraction 234.6 |
| **a14** | + chunked-prefill admission fix (scheduler) | 67.9 / 37.7 | — | — | — | C8 long extraction **306.7** (+31%), 24/24 |

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
- **GPU idle is under 1%.** The real verify step (~70 ms on a9) matches kernel plus
  all-reduce time; apparent 10–25 ms eager NCCL "waits" in single-rank profiles
  were profiler skew between ranks, not a slow node.
- **FP8 GEMM backends other than Triton do not fit the checkpoint.** FlashInfer
  CUTLASS / CUTLASS groupwise need N multiples of 128; the fused qkv is N=3392.
- **GB10 had no Triton block-FP8 tables.** Tuning N=3392/6144/4096 for M 1–64
  (`patches/fp8-configs/`) changes little on its own; the gain is from halving bytes.
- **Verify window is workload-dependent.** 4 tokens wins prose and summaries by
  20–28% and loses code/format by 28–42%. a13 picks it per step from each
  request's recent acceptance (EMA, hysteresis 1.8 / 2.6): +23–25% on prose,
  narrative and summaries at C1, +8% C8 aggregate, format −11% at C1.
- **Target verify must stay on FA4.** SGLang's Triton target-verify path
  (`--speculative-attention-mode decode`) produces wrong output on MiMo even at
  full width. The FA backend hard-coded the 8-token width and rebuilt its graph
  metadata on a second capture (which corrupted the full-width graphs); a13
  keeps separate metadata per width.
- **FP8 on the draft's fused `qkv` disables fused KV materialization.** Keeping
  it BF16 is faster overall than the FP8 bytes it saves.
- **An 8th request could wait out a whole decode.** When a prefill batch needed
  chunking, SGLang's admission check counted the request finishing its last chunk
  (which already holds a request slot) against the free slots, set
  `batch_is_full`, and kept it set until a request finished. MiMo's SWA pool is not
  SGLang's hybrid-SWA mode, so nothing reset the flag earlier. Found with
  rate-limited admission logging; fixed by counting only requests that still
  need a slot (a14). Upstream SGLang has the same check.
- Startup occasionally fails with `CUDA error: operation not permitted` while
  capturing draft CUDA graphs on one rank (a10, a13); a restart succeeds.
- A per-step thinking/answer split (`eval/phase_accept.py`, `results/phase/`):
  easy prompts spend ~3% of output in (English) reasoning; hard ones 61–70%, with
  reasoning accept length ~3.5 and Welsh prose answers the lowest (~1.8).

## Invalid or excluded measurements

- a4-ep1 prefill at 2,000/8,000 targets: radix-cache hits from an interrupted earlier run.
- a5 needle at 128K time (55.9 s): an interrupted identical request had partly
  filled the radix cache. Retrieval correctness stands; the time does not.
- First decode profile per build sometimes included a lagging-rank stall (27 ms
  eager NCCL call); profiles quoted are clean repeats.

## Not done / next

- Tune the adaptive thresholds (format still narrows on some steps); a width-2/6 tier.
- A faster small-M block-FP8 GEMM (Triton reaches ~205 GB/s of ~260).
- Newer b12x RoCEnante (prepared-plan API, traffic class) in place of the SG17 snapshot.
- Loader speed for EP1, prefill MoE at large M, audio/video validation, long soak.
