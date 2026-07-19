#!/usr/bin/env python3
"""
Fine-tune Qwen3.5-0.8B (multimodal) on the OKF extraction dataset.

Because Qwen3.5 is a vision-language model, we use FastVisionModel and
UnslothVisionDataCollator, but we freeze the vision layers and train only the
language side since OKF extraction is text-only.

Output:
  - LoRA adapter saved to outputs_okf/
  - Optionally a merged GGUF for Ollama (okf_qwen35_q4_k_m.gguf)

Requirements (run in a fresh Colab / Kaggle / local CUDA env):
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes datasets
"""

import json
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

# ------------------------------------------------------------------
# 1. Load base model (Qwen3.5 is multimodal on HF)
# ------------------------------------------------------------------
max_seq_length = 2048

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3.5-0.8B",
    max_seq_length=max_seq_length,
    dtype=None,          # auto-detect: fp16 on T4, bf16 on Ampere+
    load_in_4bit=True,
)

# ------------------------------------------------------------------
# 2) LoRA adapters — train language side only
# ------------------------------------------------------------------
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,     # keep vision frozen
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    target_modules="all-linear",
)

FastVisionModel.for_training(model)

# ------------------------------------------------------------------
# 3) Format Alpaca instruction/input/output -> Qwen3.5 chat messages
# ------------------------------------------------------------------
def format_record(example):
    # Combine instruction + optional input as the user prompt
    user_text = example["instruction"]
    if example.get("input"):
        user_text += "\n\n" + example["input"]

    # The JSON output the model must learn to emit
    assistant_text = str(example.get("output", "[]")).strip()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]
    return {"messages": messages}


train_ds = load_dataset(
    "json",
    data_files="training_data/okf_train_pairs_v3.jsonl",
    split="train",
)
test_ds = load_dataset(
    "json",
    data_files="training_data/okf_test_pairs_v3.jsonl",
    split="train",
)

train_converted = train_ds.map(format_record, remove_columns=train_ds.column_names)
test_converted = test_ds.map(format_record, remove_columns=test_ds.column_names)

print(f"Train: {len(train_converted)}  |  Test: {len(test_converted)}")

# ------------------------------------------------------------------
# 4) Train
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
    train_dataset=train_converted,
    eval_dataset=test_converted,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=120,                       # ~1.5 epochs on 585 train records
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_okf",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",               # messages are processed by the collator
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_seq_length,
        eval_strategy="steps",
        eval_steps=20,
    ),
)

trainer.train()

# ------------------------------------------------------------------
# 5) Save
# ------------------------------------------------------------------
model.save_pretrained("okf_qwen35_lora")
tokenizer.save_pretrained("okf_qwen35_lora")

# Optional: export merged GGUF. If this fails for Qwen3.5's vision architecture,
# switch the base model above to Qwen/Qwen2.5-1.5B-Instruct and re-run.
try:
    model.save_pretrained_gguf("okf_qwen35", tokenizer, quantization_method="q4_k_m")
    print("GGUF saved: okf_qwen35/okf_qwen35-q4_k_m.gguf")
except Exception as exc:
    print(f"GGUF export failed (common for VLMs): {exc}")
    print("LoRA adapter is still saved under okf_qwen35_lora/")
