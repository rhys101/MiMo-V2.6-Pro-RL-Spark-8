## a3-dflash8-ep8-roce (2026-09-23T13:53:17Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 38.49 | 43.1 | 0.378 |
| C4 | 88.46 | 27.32 | 0.733 |
| C8 | 148.83 | 21.78 | 0.657 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 |
|---|---|---|---|
| coding | 62.24 | 41.0 | 30.0 |
| json | 50.46 | 29.14 | 26.85 |
| narrative | 19.12 | 11.02 | 8.54 |
| prose | 23.37 | 12.99 | 10.25 |
| math | 60.1 | 38.72 | 30.93 |
| reasoning | 36.27 | 19.93 | 16.78 |
| summary | 26.75 | 16.99 | 12.8 |
| format | 66.5 | 48.75 | 38.12 |
| ceiling_count | 83.81 | 59.42 | 46.97 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 3.16 | 1618.6 |
| 8000 | 20264 | 10.394 | 1949.7 |
| 32000 | 81292 | 48.957 | 1660.5 |
