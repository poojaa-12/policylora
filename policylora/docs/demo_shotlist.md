# Demo shot list

About 90 seconds, against `make serve` with `DETECTOR=mock`, or against vLLM when the adapters exist. Say on camera which detector is running.

1. Health check. `GET /health` shows `MockDetector` or `VLLMDetector`.
2. Promissory pitch. Send "The Harbor Index Fund is a safe way to beat the market." as tenant `base`, author `ai`. Show verdict `rewrite`, the FINRA citation, and a rewrite that no longer contains "safe way to beat."
3. Tenant swap. Send "The Horizon Equity Fund returned 11% last year. Past performance does not guarantee future results." to `base` (pass) and then to `northline` (rewrite). Point at `adapter_version`.
4. Restricted product. Send a recommendation of the Apex Growth Fund. Show verdict `block`, no rewritten text, and the restricted-product policy match.
5. Evidence. `GET /v1/evidence/{evidence_id}` from one of the calls. Mention the input hash, the rule, the adapter version, and the latency.
6. Close on the eval report: rules catch the obvious phrasing, miss the held-out paraphrases, and the LoRA numbers are whatever the GPU run actually wrote.
