## a13-prefill (2026-09-24T14:16:27Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.6 | 64.33 | 0.301 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 92.39 |
| json | 71.3 |
| narrative | 33.05 |
| prose | 41.29 |
| math | 80.05 |
| reasoning | 53.98 |
| summary | 43.47 |
| format | 99.11 |
| ceiling_count | 124.69 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 3.008 | 1700.3 |
| 8000 | 20264 | 14.642 | 1384.0 |
| 32000 | 81292 | 47.818 | 1700.0 |
| 64000 | 162620 | 108.387 | 1500.4 |
