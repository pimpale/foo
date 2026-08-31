"""LoRA SFT of Qwen3-4B on DeepSeek-R1 stratos traces.

Run:  accelerate launch train.py            (or: uv run train.py — single GPU)
Smoke test:  uv run train.py --max-steps 5 --max-rows 64
"""

import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

MODEL_ID = "Qwen/Qwen3-4B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for smoke tests")
    ap.add_argument("--max-rows", type=int, default=0, help="cap train rows for smoke tests")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--output-dir", default="out/qwen3-4b-stratos-r1-lora")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "eval": "data/eval.jsonl"})
    ds = ds.select_columns(["prompt", "completion"])
    if args.max_rows:
        ds["train"] = ds["train"].shuffle(seed=0).select(range(args.max_rows))

    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        max_length=args.max_length,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        num_train_epochs=2,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=cfg,
        peft_config=peft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
