## a6-dflash8-ep1-marlin-fa4pf (2026-09-23T17:00:11Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 52.92 | 60.93 | 0.343 |
| C2 | 82.31 | 48.21 | 0.378 |
| C4 | 118.98 | 34.9 | 0.488 |
| C8 | 171.12 | 25.44 | 0.635 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| coding | 91.39 | 66.98 | 46.29 | 40.05 |
| json | 68.15 | 61.67 | 42.85 | 30.19 |
| narrative | 26.31 | 19.34 | 14.55 | 9.37 |
| prose | 32.25 | 23.37 | 16.89 | 11.89 |
| math | 78.8 | 66.41 | 53.31 | 35.95 |
| reasoning | 49.21 | 36.74 | 23.37 | 18.02 |
| summary | 34.34 | 29.22 | 21.24 | 14.87 |
| format | 107.01 | 81.97 | 60.72 | 43.18 |
| ceiling_count | 112.62 | 97.64 | 77.22 | 51.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 5115 | 2.694 | 1898.9 |
| 8000 | 20264 | 12.439 | 1629.1 |
| 32000 | 81292 | 47.273 | 1719.6 |
| 64000 | 162620 | 107.318 | 1515.3 |
