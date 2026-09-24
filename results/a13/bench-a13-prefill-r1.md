## a13-prefill-r1 (2026-09-24T14:22:33Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.66 | 68.91 | 0.304 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 91.06 |
| json | 81.01 |
| narrative | 35.51 |
| prose | 39.74 |
| math | 89.07 |
| reasoning | 55.32 |
| summary | 47.7 |
| format | 111.87 |
| ceiling_count | 121.74 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 2.7 | 1894.4 |
| 8000 | 20264 | 12.748 | 1589.6 |
| 32000 | 81292 | 48.406 | 1679.4 |
| 64000 | 162620 | 108.486 | 1499.0 |
