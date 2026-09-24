## a7-fp8oproj (2026-09-24T09:07:47Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 54.48 | 62.17 | 0.322 |
| C8 | 170.66 | 26.51 | 0.835 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 86.23 | 39.4 |
| json | 81.37 | 32.57 |
| narrative | 28.21 | 10.13 |
| prose | 36.18 | 12.24 |
| math | 83.1 | 37.53 |
| reasoning | 51.1 | 19.77 |
| summary | 44.92 | 15.98 |
| format | 86.28 | 44.43 |
| ceiling_count | 120.35 | 52.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
