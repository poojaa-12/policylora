# Limitations

PolicyLoRA is a reference implementation on one rulepack, FINRA Rule 2210 and Rule 2220. FCA, MiFID II, insurance, and healthcare are out of scope.

The gold set is curator-authored. Many obvious rows are the same trigger phrase dropped into different fund names and channels. A compliance reviewer has not signed them, and they are not a sample of real advisor mail. The OOD file is short public-rule language plus paraphrases of public AWC themes, with links to the source PDFs. It is not a scrape of those PDFs and it is not customer traffic.

The rules engine's perfect score is on messages that contain its own trigger phrases. On the subtle gold slice and on the OOD paraphrases, that score goes to zero. That gap is the job of the adapter, and the adapter has not been trained in this environment.

Tenant policies are fictional. Northline bans return numbers, Harbor requires one max-loss sentence, and Cedar bans four words including in negations and in "safe harbor." Cedar will flag a true sentence such as "not FDIC insured." That is the policy as written, and it is a sharp edge.

Training labels use a fixed confidence of 0.95. The escalate threshold of 0.6 is not calibrated.

`DETECTOR=mock` is a demo stand-in. Eval numbers in the report do not include it.

No GPU run is checked in: no LoRA weights, no frontier cache, no load-test log, no dollar cost. Quantization, preference tuning on rewrites, and Grafana were left out on purpose. The third tenant was not cut.

Do not describe these results as compliance-grade.
