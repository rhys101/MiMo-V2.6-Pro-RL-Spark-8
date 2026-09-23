## a4-dflash8-ep1-marlin (2026-09-23T14:54:48Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 50.16 | 57.62 | 0.339 |
| C8 | 161.81 | 24.74 | 0.836 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 77.96 | 36.68 |
| json | 68.55 | 31.95 |
| narrative | 24.95 | 9.54 |
| prose | 28.48 | 11.26 |
| math | 79.51 | 34.29 |
| reasoning | 47.6 | 17.49 |
| summary | 38.87 | 14.63 |
| format | 95.04 | 42.07 |
| ceiling_count | 110.68 | 51.07 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
