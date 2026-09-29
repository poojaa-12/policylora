PY ?= python3

.PHONY: setup generate check train train-dry-run train-ablation eval serve test gate

setup:
	$(PY) -m pip install -e ".[dev]"

generate:
	$(PY) -m policylora.data.generate

check:
	$(PY) -m policylora.data.generate --check

train:
	$(PY) -m policylora.train.sft --config policylora/train/configs/shared.yaml

train-dry-run:
	$(PY) -m policylora.train.sft --config policylora/train/configs/shared.yaml --dry-run
	$(PY) -m policylora.train.sft --config policylora/train/configs/northline.yaml --dry-run
	$(PY) -m policylora.train.sft --config policylora/train/configs/harbor.yaml --dry-run
	$(PY) -m policylora.train.sft --config policylora/train/configs/cedar.yaml --dry-run

train-ablation:
	$(PY) -m policylora.train.sft --ablation --dry-run

eval:
	$(PY) -m policylora.evals.harness

serve:
	DETECTOR=$${DETECTOR:-mock} $(PY) -m uvicorn "policylora.serve.app:create_app" --factory --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest

gate:
	$(PY) -m policylora.ci.eval_gate
