"""SFT LoRA training. Dry-run validates data and writes a manifest. GPU fit is separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from policylora.data.load import LeakageError, assert_trainable, read_jsonl
from policylora.paths import repo_root
from policylora.serve.prompts import render_training_text

ROOT = repo_root()
CONFIG_DIR = ROOT / "policylora/train/configs"


def load_config(path: Path) -> dict:
    document = yaml.safe_load(path.read_text())
    document["config_path"] = str(path)
    return document


def policy_text(tenant_id: str | None) -> str:
    name = tenant_id or "base"
    path = ROOT / "policylora/data/tenants" / f"{name}.yaml"
    document = yaml.safe_load(path.read_text())
    return document["policy"].strip()


def assistant_json(row: dict) -> str:
    violation = row["label"] != "compliant"
    payload = {
        "violation": violation,
        "category": row["label"],
        "rule": row.get("rule"),
        "policy_match": row.get("policy_match"),
        "span": row.get("span"),
        "confidence": 0.95 if violation else 0.9,
        "explanation": row.get("policy_match") or "No violation found.",
        "suggested_fix": row.get("compliant_rewrite"),
    }
    return json.dumps(payload, ensure_ascii=True)


def render_example(row: dict) -> str:
    tenant = row.get("tenant_id") or "base"
    return render_training_text(row["message"], tenant, policy_text(tenant), assistant_json(row))


def select_rows(rows: list[dict], config: dict) -> list[dict]:
    data = config["data"]
    chosen = rows
    if not data.get("include_hard_negatives", True):
        chosen = [row for row in chosen if not row.get("hard_negative")]
    general = [row for row in chosen if row.get("tenant_id") in (None, "base")]
    tenant_id = data.get("tenant_id")
    if tenant_id:
        specific = [row for row in chosen if row.get("tenant_id") == tenant_id]
        chosen = general + specific if data.get("mix_general", True) else specific
    else:
        chosen = general
    return chosen


def prepare(config: dict, rows: list[dict], blocked_ids: set[str]) -> list[str]:
    chosen = [row for row in select_rows(rows, config) if row.get("split", "train") == "train"]
    try:
        assert_trainable(chosen, blocked_ids)
    except LeakageError:
        raise
    return [render_example(row) for row in chosen]


def manifest(config: dict, texts: list[str], blocked_ids: set[str]) -> dict:
    lora = config["lora"]
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode())
    return {
        "adapter_name": config["adapter_name"],
        "init_adapter": config.get("init_adapter"),
        "model_name": config["model_name"],
        "seed": config["seed"],
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "include_hard_negatives": config["data"].get("include_hard_negatives", True),
        "tenant_id": config["data"].get("tenant_id"),
        "examples": len(texts),
        "text_sha256": digest.hexdigest(),
        "holdout_ids": len(blocked_ids),
    }


def ablation_configs() -> list[Path]:
    names = [
        "shared.yaml",
        "shared_rank8.yaml",
        "shared_rank32.yaml",
        "shared_no_hard_negatives.yaml",
    ]
    return [CONFIG_DIR / name for name in names]


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def fit(config: dict, texts: list[str], output_dir: Path) -> None:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_name = config["model_name"]
    init = config.get("init_adapter")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    trainer_model = model
    trainer_peft = peft_config
    if init:
        trainer_model = PeftModel.from_pretrained(model, init, is_trainable=True)
        trainer_peft = None
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        seed=config["seed"],
        max_length=config["max_seq_length"],
        dataset_text_field="text",
        logging_steps=10,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=trainer_model,
        args=training_args,
        train_dataset=Dataset.from_dict({"text": texts}),
        peft_config=trainer_peft,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    try:
        import mlflow

        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment"])
        with mlflow.start_run(run_name=config["adapter_name"]):
            mlflow.log_params(
                {
                    "seed": config["seed"],
                    "lora_r": lora["r"],
                    "adapter_name": config["adapter_name"],
                    "examples": len(texts),
                }
            )
    except Exception as exc:
        print(f"mlflow logging skipped: {exc}", file=sys.stderr)


def run_config(config: dict, *, dry_run: bool, blocked_ids: set[str] | None = None) -> dict:
    rows = read_jsonl(ROOT / config["data"]["train_path"])
    holdouts: set[str] = set()
    for relative in config["data"]["holdout_paths"]:
        holdouts.update(row["id"] for row in read_jsonl(ROOT / relative))
    if blocked_ids:
        holdouts.update(blocked_ids)
    texts = prepare(config, rows, holdouts)
    summary = manifest(config, texts, holdouts)
    output = ROOT / config["output_dir"]
    if dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    if not cuda_available():
        raise SystemExit("CUDA is required for make train. Use make train-dry-run on a machine without a GPU.")
    output.mkdir(parents=True, exist_ok=True)
    fit(config, texts, output)
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PolicyLoRA adapters")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args(argv)
    if args.ablation:
        grid = []
        for path in ablation_configs():
            config = load_config(path)
            grid.append(
                {
                    "adapter_name": config["adapter_name"],
                    "lora_r": config["lora"]["r"],
                    "include_hard_negatives": config["data"].get("include_hard_negatives", True),
                    "seed": config["seed"],
                }
            )
        destination = ROOT / "artifacts/runs/ablation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(grid, indent=2) + "\n")
        print(destination.read_text())
        return
    if args.config is None:
        raise SystemExit("--config is required")
    summary = run_config(load_config(args.config), dry_run=args.dry_run)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
