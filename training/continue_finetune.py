#!/usr/bin/env python3
"""
Continue fine-tuning the aura-qwen-merged model on the balanced dataset.
Loads the already fine-tuned safetensors as base, then trains with LoRA.

Because aura-qwen-merged is a full Unsloth-saved model, we can:
1. Load it as FastVisionModel
2. Apply new LoRA adapters
3. Train on the balanced dataset (more 3-5 concept examples)
4. Save the new LoRA adapter
5. Merge into a new full model

Usage:
    python3 continue_finetune.py

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes datasets
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "aura-qwen"
DATA_DIR = BASE_DIR.parent / "training_data"

# ------------------------------------------------------------------
# 1. Load the fine-tuned model as base (NOT the original Qwen!)
# ------------------------------------------------------------------
max_seq_length = 2048

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=str(MODEL_PATH),
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# ------------------------------------------------------------------
# 2. New LoRA adapters on top
# ------------------------------------------------------------------
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,                    # higher rank for harder task
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    target_modules="all-linear",
)

FastVisionModel.for_training(model)

# ------------------------------------------------------------------
# 3. Format
# ------------------------------------------------------------------
def format_record(example):
    user_text = example["instruction"]
    if example.get("input"):
        user_text += "\n\n" + example["input"]
    assistant_text = str(example.get("output", "[]")).strip()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    return {"messages": messages}


train_ds = load_dataset("json", data_files=str(DATA_DIR / "okf_train_pairs_v5.jsonl"), split="train")
test_ds = load_dataset("json", data_files=str(DATA_DIR / "okf_test_pairs_v5.jsonl"), split="train")

train_conv = train_ds.map(format_record, remove_columns=train_ds.column_names)
test_conv = test_ds.map(format_record, remove_columns=test_ds.column_names)

print(f"Train: {len(train_conv)}  |  Test: {len(test_conv)}")

# ------------------------------------------------------------------
# 4. Train — MORE steps since dataset is larger and task harder
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(
        model, 
        tokenizer,
        train_on_responses_only=True,
        instruction_part="user",
        response_part="assistant",
    ),
    train_dataset=train_conv,
    eval_dataset=test_conv,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=180,          # ~477 train records v5 (~3 epochs at batch 2×accum 4)
        learning_rate=1e-4,     # lower LR since continuing on already-fine-tuned model
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_okf_v5",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_seq_length,
        eval_strategy="steps",
        eval_steps=30,
    ),
)

trainer.train()

# ------------------------------------------------------------------
# 5. Save LoRA adapter
# ------------------------------------------------------------------
OUT_DIR = BASE_DIR.parent.parent / "okf_qwen35_lora_v5"
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

# Merge LoRA into full model and save
merged_path = BASE_DIR.parent.parent / "lib-qwen"
FastVisionModel.for_inference(model)

model.save_pretrained_merged(
    str(merged_path),
    tokenizer,
    save_method="merged_16bit",
)
print(f"\nMerged model saved to: {merged_path}")
print(f"LoRA adapter saved to: {OUT_DIR}")