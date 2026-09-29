# PolicyLoRA eval report

Rules-only numbers below were measured by running the deterministic engine on the curator gold set.
PolicyLoRA, the untuned base model, frontier quality, GPU latency, and cost stay unmeasured until a GPU run writes a prediction cache.

- Fingerprint: `029eb8b514cd80141b01926256f126b3f883f7300082331fd2018b306e79ff1e`
- Suite wall clock (seconds): 0.013
- Gold rows: 232
- OOD rows: 24
- Tenant-eval rows: 24

## Rules baseline

- Obvious-violation recall: 1.000
- Hard-negative false-positive rate: 0.000
- Span accuracy on obvious violations: 1.000
- Tenant-policy accuracy: 0.542
- Rules latency p50/p95 ms on this CPU: 0.021 / 0.034
- OOD macro recall: 0.000

| Category | Support | Precision | Recall |
| --- | --- | --- | --- |
| exaggerated_unwarranted | 16 | 1.000 | 0.750 |
| false_or_misleading | 28 | 1.000 | 0.857 |
| material_omission | 17 | 1.000 | 0.706 |
| misleading_comparison | 16 | 1.000 | 0.750 |
| options_risk_minimized | 16 | 1.000 | 0.750 |
| options_without_risk_disclosure | 14 | 1.000 | 0.857 |
| performance_claim_without_basis | 16 | 1.000 | 0.750 |
| promissory_claim | 53 | 1.000 | 0.906 |
| unbalanced_presentation | 16 | 1.000 | 0.750 |

## Model slots

- Untuned base: not_measured
- PolicyLoRA: not_measured
- Frontier: not_measured
- Cost per 1k: not_measured

Cost method when measured: SLM rental dollars per hour divided by measured messages per hour, times 1,000. Frontier published token price times measured tokens, times 1,000. Detection p95 is reported both as model time and as end-to-end time.
