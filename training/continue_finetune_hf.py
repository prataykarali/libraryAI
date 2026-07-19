#!/usr/bin/env python3
"""
Standard Hugging Face SFT fine-tuning script.
Does not require Unsloth. Uses system python SFT packages (peft, trl, datasets, transformers).
"""
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "aura-qwen"
DATA_DIR = BASE_DIR.parent / "training_data"
OUTPUT_DIR = BASE_DIR.parent.parent / "outputs_okf_v5"
MERGED_DIR = BASE_DIR.parent.parent / "lib-qwen"

print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Loading model on {device} (dtype={dtype}) ...")

if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quantization_config,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=dtype,
    trust_remote_code=True
)

if device == "cuda":
    model = prepare_model_for_kbit_training(model)

# Apply LoRA Config
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Format record to HF chat format
def format_record(example):
    user_text = example["instruction"]
    if example.get("input"):
        user_text += "\n\n" + example["input"]
    assistant_text = str(example.get("output", "[]")).strip()
    
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text}
    ]
    templated = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": templated}

# Load datasets
print(f"Loading datasets from: {DATA_DIR}")
train_ds = load_dataset("json", data_files=str(DATA_DIR / "okf_train_pairs_v5.jsonl"), split="train")
test_ds = load_dataset("json", data_files=str(DATA_DIR / "okf_test_pairs_v5.jsonl"), split="train")

train_conv = train_ds.map(format_record, remove_columns=train_ds.column_names)
test_conv = test_ds.map(format_record, remove_columns=test_ds.column_names)

print(f"Train size: {len(train_conv)} | Test size: {len(test_conv)}")

# Configure SFT
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=120,
    learning_rate=1e-4,
    logging_steps=5,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    dataset_text_field="text",
    max_length=1536,
    eval_strategy="steps",
    eval_steps=30,
    gradient_checkpointing=True,
    report_to="none"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_conv,
    eval_dataset=test_conv,
    args=training_args
)

# Train!
print("Starting training...")
trainer.train()

# Save LoRA adapter
adapter_dir = BASE_DIR.parent.parent / "okf_qwen35_lora_v5"
print(f"Saving LoRA adapter to: {adapter_dir}")
trainer.model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))

# Clean up memory before merging
del model
del trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Merge LoRA and save base model
print("Merging LoRA adapter and saving full model...")
base_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    device_map="cpu",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
from peft import PeftModel
peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
merged_model = peft_model.merge_and_unload()

# Save merged model directly to lib-qwen
merged_model.save_pretrained(str(MERGED_DIR))
tokenizer.save_pretrained(str(MERGED_DIR))
print(f"Merged model successfully saved to: {MERGED_DIR}")
