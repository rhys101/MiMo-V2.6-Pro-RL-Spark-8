## a13-prefill-r2 (2026-09-24T14:25:55Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 56.69 | 66.4 | 0.327 |

### Per-stream tok/s by category

| category | C1 |
|---|---|
| coding | 82.82 |
| json | 74.39 |
| narrative | 34.72 |
| prose | 39.32 |
| math | 86.55 |
| reasoning | 57.07 |
| summary | 48.67 |
| format | 107.65 |
| ceiling_count | 125.18 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 2.721 | 1879.8 |
| 8000 | 20264 | 13.415 | 1510.5 |
| 32000 | 81292 | 49.817 | 1631.8 |
| 64000 | 162620 | 108.995 | 1492.0 |
