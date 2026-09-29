# PolicyLoRA eval report

Rules engine on the gold set. Model, frontier, GPU latency, and cost are unmeasured until a prediction cache is present.

- Fingerprint: `029eb8b514cd80141b01926256f126b3f883f7300082331fd2018b306e79ff1e`
- Suite wall clock (seconds): 0.012
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

## Unmeasured

- Untuned base: not_measured
- PolicyLoRA: not_measured
- Frontier: not_measured
- Cost per 1k: not_measured
