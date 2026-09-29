# Write-up

PolicyLoRA is a reference implementation of a small enforcement loop for FINRA client communications. A frozen open model is meant to carry the regulation. Each customer adds one LoRA. A rules list and that detector vote together. The API returns one verdict before anything is sent: pass, rewrite, block, or escalate. This tree is a work sample of that loop on one rulepack. It is a risk-reduction tool, not a compliance guarantee, and it is not a copy of a vendor product.

## Design

The request is the message, a rulepack, a tenant id, and whether a human or a model wrote it. High-precision patterns cover promissory language, absolute claims, bare options trade talk, and a restricted-product list. The Apex Growth Fund is on that list because recommending it should block even when the wording is otherwise fixable. Patterns honor a short negation window, so "not risk-free" and "you can lose" stay clean.

Fuzzy tenant rules are separate, because a keyword pass gets them wrong. Northline Wealth flags a return such as "returned 11%" and ignores an expense ratio. Harbor Options requires the sentence "You can lose more than you invest." Cedar Credit bans "guaranteed," "safe," "risk-free," and "FDIC," including negated uses. The shared rules engine does not apply those policies. The tenant adapter is supposed to.

vLLM can hot-swap one adapter onto a frozen base. It does not stack a shared adapter under a tenant adapter. Training therefore starts from the shared LoRA and saves each tenant as a single adapter. The registry pins the version. Changing `active` rolls back without reloading Qwen2.5-1.5B-Instruct.

Detection is a short JSON object with a 160-token cap. The rewrite runs only when the merged verdict is rewrite and the author is AI, and the rewrite is scored again before it is returned. Timeout, malformed JSON, a model error, an unknown tenant, or an unknown rulepack all block and write an evidence row. The eval gate on GitHub-hosted runners replays the committed artifact and fails if the gold set or the registry moved without a new run. The GPU job is manual and stays off unless a self-hosted runner is configured.

Gold, OOD, and tenant-eval rows are held out by id. Synthetic rows are split by scenario cell, and one cell is reserved for test before generation. Near-duplicates of holdout text are dropped. The default duplicate check is token Jaccard; `POLICLORA_DEDUP=embeddings` switches it to `all-MiniLM-L6-v2`.

## Results

Measured on the curator gold set with the rules engine, on a laptop, in 0.013 seconds for the full suite. This is not a GPU latency number.

| Slice | Result |
| --- | --- |
| Obvious-violation recall | 1.000 |
| Hard-negative false-positive rate | 0.000 |
| Span accuracy on obvious violations | 1.000 |
| Per-category precision | 1.000 |
| Per-category recall | 0.706 to 0.906 |
| Tenant-policy accuracy | 0.542 |
| OOD macro recall | 0.000 |
| Rules latency p50 / p95 | 0.02 ms / 0.03 ms |
| PolicyLoRA, untuned base, frontier | not measured |
| Cost per 1k messages | not measured |
| GPU detection p95 | not measured |

Category recall sits between 0.706 (material omission) and 0.906 (promissory claims) because each category includes subtle wording the patterns do not contain. Every obvious row is caught, and none of the 40 hard negatives fire. The 24 OOD paraphrases, which avoid those phrases, are all missed. Rules agree with the tenant label on 13 of 24 tenant-eval rows (0.542). They miss Northline's return numbers, Cedar's banned words, and Harbor messages that say "options" without "buy calls."

The synthetic pool is 6,000 rows after the cap (4,758 train, 677 val, 565 test), plus 232 gold, 24 OOD, and 24 tenant-eval rows. A dry run of the Northline config rendered 4,638 training strings from the general set plus Northline, with seed 2210, and refused holdout ids. No weights were saved. There is no CUDA device in this environment.

The cost helpers are unit-tested as formulas only. An SLM figure is rental dollars per hour divided by measured messages per hour. A frontier figure is published token price times measured tokens. Neither input has been measured, so the report leaves both blank. The hypothesis, still unchecked, is detection p95 in the tens of milliseconds on one GPU, cost per 1k at least 10x under a frontier baseline, hard-negative false positives at or under that baseline, and recall within a stated margin or an explicit list of the categories where the small model loses.

## Failure analysis

The rules baseline is a high-precision filter, and the gold set was written so that fact is visible. Treating its 1.000 obvious recall as the quality of the system would be circular: the obvious sentences were built from the same phrases the regex looks for. The number that matters next to it is the OOD recall of 0.000.

Subtle gold is the same lesson inside the training distribution. "Smooth ride higher" and "tidy overlay" are labeled violations and the rules pass them. A LoRA that only copies the regex will look strong on the obvious slice and fail here. The training file includes those subtle frames so the adapter has something to learn that the rules do not already solve. Whether it learns them is unmeasured.

Tenant policy is the other miss. A shared regulatory pattern cannot encode "any percent that is a return, but not a fee" or "this word is banned even when the sentence is true." That is why the adapter is per tenant. The mock detector implements those three policies so a laptop can show the swap. It is not evidence that a LoRA can.

Rewrite re-check is tested with a scripted model that hands back the original violation. The API drops the rewrite and blocks. That path is only as good as the second detection. If both passes share a blind spot, a bad rewrite goes out. The re-check does not create a new signal.

## What this does not show

No frontier model was called. No adapter was trained. Load-test p95 on a GPU was not recorded. Confidence is not calibrated, the gold set has not had a second reviewer, and the OOD file is a few dozen paraphrases rather than production traffic. Details are in [limitations.md](limitations.md).
