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
2. Generate and split the cleaned training files:
   ```bash
   python prepare_okf_training_data.py
   python split_okf_dataset.py
   ```
3. Review `training_data/okf_dataset_report_v3.json` and spot-check `training_data/okf_training_pairs_v3.jsonl` before training.

The JSONL rows use the same prompt shape the model will see at extraction time: `instruction` contains the full extraction instruction plus `TEXT:`, `input` is blank, and `output` is a JSON array string. Source fields are top-level metadata for audit/splitting only; the model should not emit them.

```json
{"instruction": "You are an OKF extraction engine... TEXT:\n[TEXT CHUNK]\n\nReturn ONLY the JSON array, no other text:", "input": "", "output": "[{\"concept_name\":\"...\",\"concept_type\":\"definition\",\"difficulty\":\"intermediate\",\"summary\":\"...\",\"prerequisites\":[],\"unlocks\":[],\"related_to\":[],\"tags\":[]}]", "doc_id": "papers/example.pdf", "chunk_id": "chunk_001", "page_number": 1, "section_title": "Introduction"}
```

This repo currently generates:

- `training_data/okf_training_pairs_v3.jsonl` — 999 total cleaned examples
- `training_data/okf_train_pairs_v3.jsonl` — training split
- `training_data/okf_test_pairs_v3.jsonl` — held-out test split
- `training_data/okf_dataset_report_v3.json` — generation counts and discarded examples

Current split: 852 train rows and 149 test rows. The split is chunk-held-out, so exact source chunks do not leak; the same source document may still appear in both train and test. Use the test split for JSON validity and schema-completeness checks, not as a true held-out-document benchmark.

---

## 3. Ready-to-run scripts

- `finetune_qwen35_okf.py` — SFT recipe for the default Ollama model `qwen3.5:0.8b` using its HF checkpoint `Qwen/Qwen3.5-0.8B`. Because Qwen3.5 is a VLM, the script freezes vision layers and trains only the language side.
- The generic Unsloth recipe below uses `Qwen/Qwen2.5-0.5B-Instruct` / `1.5B-Instruct` as a pure-text alternative (easiest GGUF export).

## 4. Post-Mortem: Why Previous Fine-Tuning Failed (Colab/Kaggle Errors)

Before running the training script below, it is crucial to understand why previous fine-tuning attempts on Colab/Kaggle failed, to avoid reproducing those errors:

### Category A: Training Pipeline & Loss Masking Failures
- **Label Masking Collapse via `train_on_responses_only` on Vision Processor**: Applying `train_on_responses_only` to a model loaded via `FastVisionModel` (like Qwen2-VL / Qwen3.5-Vision) causes response-masking logic to fail. It masks out all target JSON response tokens as `-100` (ignored loss tokens), leaving only the end bracket `]` or empty array `[]` as the unmasked target.
- **Artificial Loss Drop (0.7056 → 0.0066)**: Because all valid concept tokens are masked out, the model learns a degenerate shortcut: outputting `[]` for every prompt minimizes training loss instantly to near zero.
- **Hyperparameter Over-Training (48 Epochs)**: Running 48 epochs causes catastrophic forgetting and over-indexing on noise. Limit training to **3.0 epochs** on a clean dataset.
- **Excessive Learning Rate (1e-4 on Instruct Base)**: A learning rate of `1e-4` destroys pre-trained instruction alignment. Use a safe, lower learning rate like **2e-5**.

### Category B: Data Structure & Collator Errors
- **Nested Column Tensorization Crash (DataLoader ValueError)**: Leaving raw, nested columns like `messages` (lists of dicts) in `converted_dataset` causes the PyTorch DataLoader worker to crash when trying to convert raw dicts into tensors. You **must** drop these columns using `remove_columns=dataset.column_names` when mapping.
- **Data Collator Incompatibility**: Leaving the collator choice to defaults when dealing with a vision processor (which automatically assigns `UnslothVisionDataCollator` even in text-only tasks) causes tensor size mismatch errors during the forward pass. Use an explicit **`DataCollatorForSeq2Seq`** with `pad_to_multiple_of=8`.
- **Jinja Template String Concatenation Failure (TypeError)**: Concatenating strings natively using custom tokenizer templates raises a `TypeError` when calling `format_clean_chatml`. Use explicit ChatML formatting.

### Category C: Inference & Evaluation Errors
- **Vision Processor Tokenizer Inverted Signature**: In Qwen2-VL / Qwen3.5-Vision, the processor's tokenizer signature expects `text=...` as a keyword argument (not a positional argument), whereas standard text-only tokenizers accept `messages` as positional arguments. Passing `messages` positionally to the processor tokenizer causes the engine to interpret the structure as image inputs, resulting in crashes.
- **Prompt Format Mismatch**: Training on standard Alpaca tags but querying the model with native ChatML prompts at inference time results in format mismatch and high perplexity. Always train and run inference using the exact same ChatML formatting.

### Category D: Historical Dataset Flaws (Resolved in v3)
- **Chunk Split Duplication**: Overlapping sliding windows caused duplicate chunks across train/test splits.
- **Ghost Prerequisites**: Empty arrays and missing concepts in prerequisites lists.
- **Schema Key Dropping**: Fields like `doc_id` and `chunk_id` were dropped.

---

## 5. Corrected Training Script (Unsloth Notebook/Script Template)

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

# 3. Format the datasets using ChatML natively
def format_clean_chatml(example):
    # Construct exact ChatML string natively to prevent Jinja templates issues
    system_text = "<|im_start|>system\n" + example["instruction"] + "<|im_end|>\n"
    user_text   = "<|im_start|>user\n" + example["input"] + "<|im_end|>\n"
    assist_text = "<|im_start|>assistant\n" + example["output"] + "<|im_end|>"
    return {"text": system_text + user_text + assist_text}

from datasets import load_dataset
# Load the high-quality v3 dataset split
dataset = load_dataset("json", data_files="okf_train_pairs_v3.jsonl", split="train")

# CRITICAL: remove_columns drops raw dicts/lists to prevent DataLoader worker crashes!
formatted_dataset = dataset.map(
    format_clean_chatml,
    remove_columns=dataset.column_names
)

# 4. Set up SFTTrainer with explicit DataCollatorForSeq2Seq
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# CRITICAL: Explicitly use DataCollatorForSeq2Seq to prevent vision processor collator mismatch
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    data_collator = data_collator,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective Batch Size = 8
        warmup_steps = 5,
        num_train_epochs = 3.0, # Safe epoch count to prevent over-training
        learning_rate = 2e-5, # Low learning rate to preserve instruction alignment
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
# NOTE: No problematic response-masking wrappers are applied here to avoid label masking collapse!
trainer_stats = trainer.train()

# 6. Save LoRA weights & Export to GGUF format
model.save_pretrained_gguf("qwen-okf-adapter", tokenizer, quantization_method = "q4_k_m")
```

---

## 6. Deploying the Fine-Tuned Model

Once training is complete, Unsloth exports a `.gguf` file (e.g., `qwen-okf-adapter-q4_k_m.gguf`).

> **Important:** `okf_pipeline.py` does **not** load a GGUF file from disk. It talks to Ollama and
> selects the model purely by tag: it calls `ollama.chat(model=MODEL_NAME, ...)`, where
> `MODEL_NAME = os.environ.get("OKF_MODEL_NAME", "qwen3.5:0.8b")` (see the `Config` block near the
> top of `okf_pipeline.py`). Dropping a new `.gguf` into a folder does nothing on its own — you must
> first **register the GGUF as an Ollama model**, then point the pipeline at that model tag.

### 4.1 Import the fine-tuned GGUF into Ollama

Create a `Modelfile` next to your exported GGUF (no extension):

```dockerfile
# Modelfile
FROM ./qwen-okf-adapter-q4_k_m.gguf
```

Then register it with Ollama under a tag of your choice:

```bash
ollama create qwen-okf:0.8b -f Modelfile
```

Confirm it is now a known Ollama model:

```bash
ollama list        # you should see qwen-okf:0.8b in the list
```

### 4.2 Point the pipeline at your new Ollama model

The pipeline reads its model tag from the `OKF_MODEL_NAME` environment variable (falling back to
`qwen3.5:0.8b`). Choose **one** of the following:

- **Env var (recommended, no code change):**
  ```bash
  OKF_MODEL_NAME=qwen-okf:0.8b python okf_pipeline.py
  ```
  or export it for the whole shell session:
  ```bash
  export OKF_MODEL_NAME=qwen-okf:0.8b
  python okf_pipeline.py
  ```

- **Edit the constant in `okf_pipeline.py`** (in the `Config` block) if you want a permanent default:
  ```python
  MODEL_NAME = os.environ.get("OKF_MODEL_NAME", "qwen-okf:0.8b")
  ```

### 4.3 Run and verify

1. Make sure the Ollama server is running (`ollama serve`, or the desktop app).
2. Run the pipeline with the tag selected above:
   ```bash
   OKF_MODEL_NAME=qwen-okf:0.8b python okf_pipeline.py
   ```
   The Stage 2 banner prints the active model (`STAGE 2: OKF v1.5 Extraction via <MODEL_NAME>`) —
   confirm it shows `qwen-okf:0.8b` so you know the fine-tuned model is actually in use.
3. Check the `accuracy.json` overall score. The structured evaluation rate and relation consistency
   should rise relative to the base model.
