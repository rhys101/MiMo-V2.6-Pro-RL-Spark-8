## a5-eagle3x4-ep1-marlin (2026-09-23T15:37:21Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 45.39 | 50.98 | 0.364 |
| C8 | 161.14 | 25.39 | 1.225 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 63.1 | 33.79 |
| json | 57.7 | 28.6 |
| narrative | 36.48 | 15.05 |
| prose | 34.96 | 16.78 |
| math | 61.76 | 31.95 |
| reasoning | 49.51 | 23.55 |
| summary | 41.25 | 19.62 |
| format | 63.12 | 33.79 |
| ceiling_count | 65.6 | 36.88 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
