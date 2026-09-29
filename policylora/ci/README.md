# Eval gate

GitHub Actions only loads workflows from `.github/workflows`. The required check is [.github/workflows/adapter-eval.yml](../../.github/workflows/adapter-eval.yml). It installs the package, runs `python -m policylora.data.generate --check`, the unit tests, and `python -m policylora.ci.eval_gate`.

The gate compares [policylora/evals/results/latest.json](../evals/results/latest.json) with [policylora/evals/thresholds.yaml](../evals/thresholds.yaml). It fails when the fingerprint of the gold set, OOD set, tenant eval, taxonomy, restricted-product list, or adapter registry does not match the artifact. PolicyLoRA recall thresholds stay off until a GPU cache is scored and `slm.required` is set to true. A missing cache then fails the check.

The workflow's GPU job runs only on `workflow_dispatch` when the repository variable `GPU_RUNNER` is `true`, on a self-hosted runner labeled `gpu`.
