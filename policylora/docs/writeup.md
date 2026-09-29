# Notes

One frozen base, `Qwen/Qwen2.5-1.5B-Instruct`. FINRA 2210 and 2220 are the shared task. Northline, Harbor, and Cedar are separate LoRAs because vLLM applies one adapter per request and does not stack them. Each tenant run starts from the shared adapter and is saved on the original base. The registry pins the version.

Rules and the detector run in parallel. Block beats escalate, then rewrite, then pass. A restricted product (Apex Growth Fund) blocks. Human authors get block instead of rewrite. An AI rewrite is checked again and dropped if it still violates. Timeout, bad JSON, model errors, and an unknown tenant block and write an evidence row.

The obvious gold sentences use the same phrases as the regex, so obvious-violation recall of 1.0 is expected. Subtle gold and the OOD paraphrases do not, and the rules miss them. OOD macro recall is 0. Tenant-policy accuracy for the rules is 0.542: they miss return numbers, Cedar's banned words, and options mentions that are not `buy calls` or `buy puts`.

| Slice | Rules |
| --- | --- |
| Obvious-violation recall | 1.000 |
| Hard-negative false-positive rate | 0.000 |
| Span accuracy, obvious | 1.000 |
| Per-category recall | 0.706 to 0.906 |
| Tenant-policy accuracy | 0.542 |
| OOD macro recall | 0.000 |
| Suite time | 0.012 s |

PolicyLoRA, the untuned base, a frontier model, GPU latency, and cost per 1k messages are not measured. No weights are checked in. Gold is template text with hand-assigned labels, not a second reviewer's set and not production mail. Training labels use confidence 0.95. The escalate cutoff is 0.6 and is not calibrated.

Synthetic split is 6,000 rows (4,758 train, 677 val, 565 test), plus 232 gold, 24 OOD, and 24 tenant-eval. Holdout ids are refused by the train loader. One scenario cell is reserved for test. A Northline dry run rendered 4,638 examples at seed 2210.
