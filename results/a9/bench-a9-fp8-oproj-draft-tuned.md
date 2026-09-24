## a9-fp8-oproj-draft-tuned (2026-09-24T10:19:41Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 55.49 | 64.61 | 0.348 |
| C8 | 176.49 | 25.98 | 0.753 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 88.9 | 41.29 |
| json | 67.11 | 29.0 |
| narrative | 28.85 | 10.22 |
| prose | 32.28 | 13.18 |
| math | 87.89 | 33.77 |
| reasoning | 58.83 | 19.21 |
| summary | 41.77 | 15.41 |
| format | 111.23 | 45.77 |
| ceiling_count | 124.94 | 52.47 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
