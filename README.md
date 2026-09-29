# PolicyLoRA

`POST /v1/validate` checks a client message against FINRA Rule 2210, Rule 2220, and one customer policy. Verdicts are pass, rewrite, block, or escalate.

One rulepack. Not a compliance determination.

The numbers in [policylora/evals/results/report.md](policylora/evals/results/report.md) are the rules engine on the gold set. Adapter weights are not in this repo. `DETECTOR=mock` applies the tenant policies in code so the API runs without a GPU. The eval report does not score that path.

## Run

```bash
make setup
make test
make eval
make serve
```

`make train` needs CUDA and the Qwen weights. `make train-dry-run` checks leakage and writes a manifest.

```bash
DETECTOR=vllm docker compose --profile gpu up --build
```

vLLM loads `base`, `northline`, `harbor`, and `cedar` at rank 16. Rollback is the `active` field in [policylora/serve/adapters/registry.json](policylora/serve/adapters/registry.json).

## Example

```bash
curl -s localhost:8000/v1/validate \
  -H 'content-type: application/json' \
  -d '{"body":"The Harbor Index Fund is a safe way to beat the market.","tenant_id":"base","author_type":"ai","rulepack":"financial_services_client_communications"}'
```

`returned 11% last year` with a past-performance sentence passes for `base` and rewrites for `northline`. Recommending the Apex Growth Fund blocks. Calls are in [policylora/data/fixtures/demo.json](policylora/data/fixtures/demo.json).

Unknown tenant, unknown rulepack, timeout, malformed model output, and model errors return `block` with `stage` `fail_closed`.

Notes on the split and the measured gap: [policylora/docs/writeup.md](policylora/docs/writeup.md).
