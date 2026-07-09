# Fine-Tuning Qwen 3.5 0.8B for Open Knowledge Format (OKF) Extraction

To maximize extraction accuracy, reduce malformed JSON outputs, and avoid common model errors (like self-loops or blank summaries), you can fine-tune Qwen 3.5 0.8B (or Qwen 2.5 0.5B/1.5B/3B) using **LoRA (Low-Rank Adaptation)** on synthetic instruction data.

---

## 1. Where to Fine-Tune

For small models (0.5B to 3B parameters), you do not need expensive hardware. You can train a LoRA adapter in **15 to 30 minutes** for free:

| Platform | Recommended Setup | VRAM Needed | Cost |
|----------|-------------------|-------------|------|
| **Google Colab** (Free Tier) | T4 GPU instance | ~14-16 GB | Free |
| **Kaggle** (Free Tier) | Dual T4 or P100 GPU | ~16 GB | Free (30 hrs/week) |
| **Local Machine** | PyTorch + CUDA GPU | >= 8 GB (with 4-bit loading) | Free |
| **RunPod / Vast.ai** | RTX 3090 or RTX 4090 | 24 GB | ~$0.20 - $0.40/hour |

The easiest framework to use is **Unsloth**, which makes training 2-5x faster and uses 60% less VRAM than vanilla Hugging Face TRL.

---

## 2. Generating the Training Dataset (Self-Distillation)

The pipeline itself produces raw OKF files. You can bootstrap your training data using your current pipeline outputs:

1. Run the pipeline on your target PDFs:
   ```bash
   python okf_pipeline.py
   ```
2. Open `okf_results.json` and extract the text chunks along with the generated JSON.
3. **Manually clean up the errors** in the JSON:
   - Trim long concept names to short noun phrases.
   - Fill in missing `summary` or `prerequisites` fields.
   - Remove self-loops where `concept_name` matches an item in `prerequisites`.
   - Correct misaligned `concept_type` tags.
4. Save these cleaned examples into a JSONL file (`training_data.jsonl`) with the following format:

```json
{"instruction": "Extract OKF from this text. Return JSON with concept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags.", "input": "[TEXT CHUNK FROM PDF]", "output": "{\"concept_name\": \"...\", \"concept_type\": \"...\", \"difficulty\": \"...\", \"summary\": \"...\", \"prerequisites\": [...], \"unlocks\": [...], \"related_to\": [...], \"tags\": [...]}"}
```

Aim for **200 to 500 high-quality training pairs**.

---

## 3. Training Script (Unsloth Notebook/Script Template)

Create a notebook on Google Colab or Kaggle and run the following script:

```python
# 1. Install Unsloth & dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 2. Initialize Model & Fast LoRA configuration
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Supports context window size
dtype = None # None for auto-detection. Float16 for Tesla T4, Bfloat16 for Ampere+
load_in_4bit = True # Use 4-bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B-Instruct", # or "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA Rank (16 or 32 recommended)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05, # Good for general regularization
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Format the datasets using Chat Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 4. Set up the SFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Adjust depending on data size (usually 60-120 steps is enough)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Start Training!
trainer_stats = trainer.train()

# 6. Save LoRA weights & Export to GGUF format
# This converts the fine-tuned model back to GGUF format for use in your models folder
model.save_pretrained_gguf("qwen-okf-adapter", tokenizer, quantization_method = "q4_k_m")
```

---

## 4. Deploying the Fine-Tuned Model

Once training is complete, Unsloth exports a `.gguf` file (e.g., `qwen-okf-adapter-q4_k_m.gguf`):

1. Copy the exported `.gguf` file back to your project directory:
   `c:\Users\AIML-IEDC\Desktop\libraryAI\models/qwen3.5-0.8b.gguf` (replace the old model).
2. Start/Restart Ollama or reload your local `Llama` script.
3. Run the pipeline:
   ```bash
   python okf_pipeline.py
   ```
4. Check the `accuracy.json` overall score. The structured evaluation rate and relation consistency will rise significantly.
