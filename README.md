# PolicyLoRA

PolicyLoRA is a reference implementation for risk reduction. It checks an AI-written client message against FINRA Rule 2210 and Rule 2220, plus one customer policy, and returns pass, rewrite, block, or escalate.

It is not legal, regulatory, or compliance advice. It does not guarantee compliance with FINRA, the SEC, or any other regime, and it does not cover the broader set of regimes a production enforcement product would carry. Outputs can be wrong. A qualified person has to review anything you would rely on.

The measured numbers in [policylora/evals/results/report.md](policylora/evals/results/report.md) are the rules engine on the curator gold set. The LoRA weights, frontier quality, GPU latency, and dollar cost are not measured in this tree. `DETECTOR=mock` applies the written tenant policies with explicit heuristics so the endpoint can be exercised before a GPU is available. Those heuristics are not the model.

## What is here

- Curator gold (232), held-out OOD themes (24), and three fictional tenant policies.
- About 6,000 synthetic train/val/test rows, split by whole scenario cell. Gold, OOD, and tenant-eval ids cannot enter the train file.
- SFT LoRA configs for a shared adapter and three tenant adapters on frozen `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0). vLLM loads one adapter per request.
- `POST /v1/validate`: rules and the detector run together, the harsher verdict wins, a rewrite is checked again, and any failed stage blocks.
- An eval harness and a GitHub Actions gate. The gate fails when the gold set, taxonomy, or adapter registry changes without a refreshed eval artifact.

## Setup

```bash
make setup
make generate
make test
make eval
make serve
```

`make train` needs a CUDA GPU and the Qwen weights. `make train-dry-run` checks leakage, renders the SFT text, and writes a manifest on a laptop.

GPU serving:

```bash
DETECTOR=vllm docker compose --profile gpu up --build
```

The vLLM service loads four LoRA modules (`base`, `northline`, `harbor`, `cedar`) at rank 16. Point `active` in [policylora/serve/adapters/registry.json](policylora/serve/adapters/registry.json) at an older version to roll back without changing the base model.

## Validate

```bash
curl -s localhost:8000/v1/validate \
  -H 'content-type: application/json' \
  -d '{"body":"The Harbor Index Fund is a safe way to beat the market.","tenant_id":"base","author_type":"ai","rulepack":"financial_services_client_communications"}'
```

The same return-number sentence passes for `base` and rewrites for `northline`. Recommending the Apex Growth Fund blocks from the restricted-product list. The scripted calls are in [policylora/data/fixtures/demo.json](policylora/data/fixtures/demo.json).

Unknown tenants, an unknown rulepack, a model timeout, malformed JSON, and a model error all return `verdict: block` with `stage: fail_closed` and an evidence row.

## Layout

```
policylora/data     taxonomy, gold, generation, splits
policylora/train    SFT configs and the dry-run / CUDA trainer
policylora/serve    FastAPI, rules, vLLM client, audit log
policylora/evals    harness, thresholds, results
policylora/ci       eval gate
policylora/loadtest k6 script
policylora/docs     write-up, architecture, limitations, shot list
```

## Docs

- [Write-up](policylora/docs/writeup.md)
- [Architecture](policylora/docs/architecture.md)
- [Limitations](policylora/docs/limitations.md)
- [Demo shot list](policylora/docs/demo_shotlist.md)
