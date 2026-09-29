# Load test

Run against a live API. `DETECTOR=mock` measures the process overhead. GPU latency and saturation numbers come from `DETECTOR=vllm` on the 24GB host.

```bash
k6 run policylora/loadtest/validate.js
```

Raise `rate` until requests fail or p95 stops climbing, and record p50, p95, p99, achieved RPS, and the tenant mix. Write that log under `policylora/loadtest/results/`. This directory ships without fabricated timing numbers.
