## a13-adaptive-verify (2026-09-24T13:42:49Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 59.92 | 69.11 | 0.304 |
| C8 | 190.81 | 28.52 | 0.611 |

### Per-stream tok/s by category

| category | C1 | C8 |
|---|---|---|
| coding | 91.13 | 42.27 |
| json | 91.19 | 33.62 |
| narrative | 36.0 | 14.09 |
| prose | 40.26 | 16.93 |
| math | 86.63 | 37.68 |
| reasoning | 56.85 | 20.4 |
| summary | 51.34 | 16.86 |
| format | 99.45 | 46.27 |
| ceiling_count | 125.6 | 55.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
