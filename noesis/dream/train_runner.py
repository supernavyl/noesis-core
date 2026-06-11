"""LoRA training subprocess — uses Unsloth + TRL.

This runs out-of-process from the dream orchestrator so a training crash doesn't kill the
service. Import-guarded so non-train deployments don't have to install Unsloth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", default="huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq", type=int, default=8192)
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"[train_runner] missing dependency: {exc}", file=sys.stderr)
        return 2

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    ds = load_dataset("json", data_files=str(args.dataset), split="train")

    def fmt(ex: dict) -> dict:
        msgs = ex["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    ds = ds.map(fmt)

    cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        optim="adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)
    trainer.train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
