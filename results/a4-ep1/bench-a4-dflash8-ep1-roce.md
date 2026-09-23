## a4-dflash8-ep1-roce (2026-09-23T14:33:38Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 44.63 | 48.87 | 0.264 |
| C4 | 113.76 | 31.68 | 0.309 |
| C8 | 171.6 | 24.36 | 0.349 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 72.23 | 46.81 | 37.62 |
| json | 50.8 | 34.64 | 30.1 |
| narrative | 21.68 | 12.59 | 8.82 |
| prose | 26.74 | 15.71 | 10.97 |
| math | 68.26 | 46.84 | 33.6 |
| reasoning | 42.44 | 23.08 | 17.11 |
| summary | 31.44 | 17.63 | 14.97 |
| format | 77.38 | 56.15 | 41.71 |
| ceiling_count | 101.75 | 68.56 | 50.32 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 0.324 | 15777.9 |
| 8000 | 20264 | 0.299 | 67733.3 |
| 32000 | 81292 | 58.084 | 1399.5 |
