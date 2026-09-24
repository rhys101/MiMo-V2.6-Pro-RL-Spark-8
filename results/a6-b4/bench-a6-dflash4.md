## a6-dflash4 (2026-09-24T08:46:30Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 45.45 | 50.67 | 0.33 |
| C8 | 174.18 | 25.9 | 0.746 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 65.76 | 39.7 |
| json | 60.73 | 28.27 |
| narrative | 28.84 | 12.29 |
| prose | 41.25 | 15.04 |
| math | 58.78 | 33.08 |
| reasoning | 46.28 | 21.32 |
| summary | 41.72 | 20.31 |
| format | 62.03 | 37.21 |
| ceiling_count | 71.0 | 41.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
