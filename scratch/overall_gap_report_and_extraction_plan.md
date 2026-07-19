# Overall Gap Report & Re-Extraction Implementation Plan

This report synthesizes the findings of our multi-agent audits, assesses the correctness of the Ontological Knowledge Framework (OKF) schema, details remaining pipeline issues, and outlines the step-by-step procedure to re-run the model extraction to ingest the full corpus into the graph.

---

## 1. What the Graph and Model Missed

Our audits of the 7 papers and the *Mathematics for Machine Learning* (MML) textbook revealed that the current graph only contains **92 concepts** out of a potential **600+ key concepts** in the corpus. The base model failed to extract 92% of the content due to citation-filtering bugs (now fixed) and model capability limits.

Here is a summary of what was missed per document:

### 🧠 BERT (`Devlin2018_BERT.pdf`) — 23% Extraction Rate
* **Core Concepts Missed**: WordPiece Embeddings, the special `[CLS]` sequence representation token, the `[SEP]` sentence separator token, and Segment Embeddings.
* **Ablations & Parameters**: The dual-sequence-length training schedule (128 and 512 tokens), the 80/10/10 masking schedule, and the ablation results showing the impact of removing the Next Sentence Prediction (NSP) task.

### ⚡ Attention (`Vaswani2017_Attention_Is_All_You_Need.pdf`) — 65% Extraction Rate
* **Core Concepts Missed**: The structural definitions of the Encoder and Decoder stacks ($N=6$), Sub-layer Residual Connections, Layer Normalization (`LayerNorm(x + Sublayer(x))`), and BPE Tokenization.
* **Regularization & Evaluation**: Residual Dropout, Label Smoothing, Checkpoint Averaging, and Beam Search decoding settings.

### 🔧 LoRA (`Hu2021_LoRA.pdf`) — 24% Extraction Rate
* **Core Concepts Missed**: The low-rank update matrix representation ($W_0 + \frac{\alpha}{r}BA$), Frozen Pre-trained Weight tensors, Kaiming Gaussian Initialization (for matrix $A$), Zero Initialization (for matrix $B$), and the rank $r$ hyperparameter.
* **Theory**: Subspace Similarity Analysis using Grassmann distance/projection metrics.

### 📚 RAG (`Lewis2020_RAG.pdf`) — 13% Extraction Rate
* **Core Concepts Missed**: Dense Passage Retrieval (DPR) bi-encoders, the Query and Document Encoders, the BART generator seq2seq component, Token-level vs. Sequence-level marginalization formulas, and Index Hot-Swapping.

### 🎲 Probable (`probable.pdf`) — 9.8% Extraction Rate
* **Core Concepts Missed**: Soft-target fine-tuning, knowledge distillation metrics, logit-matching objectives, and dataset-specific benchmarks.

### 📐 Math for ML (`Deisenroth_Math_For_ML.pdf`) — 1.1% Extraction Rate
* **Core Concepts Missed**: Out of 812 chunks, only **9 concepts** were extracted (all from the Foreword). The entire core mathematical foundation is missing:
  * **Chapter 2 (Linear Algebra)**: Vector Subspaces, Bases, Span, Linear Independence, Rank-Nullity Theorem.
  * **Chapter 3 (Analytic Geometry)**: Metrics, Inner Products, Symmetric Positive Definite Matrices, Orthogonal Matrices.
  * **Chapter 4 (Matrix Decompositions)**: Eigenvalues, Eigenvectors, Singular Value Decomposition (SVD), Cholesky Decomposition.
  * **Chapter 5 (Vector Calculus)**: Jacobians, Gradients, Partial Derivatives, Backpropagation math.
  * **Chapter 6 (Probability)**: CDFs, PDFs, Expected Values, Covariance Matrices, Multivariate Gaussians.
  * **Chapter 7 (Optimization)**: Gradient Descent with Momentum, Lagrange Multipliers, Primal/Dual formulations.
  * **Chapter 8-12 (ML Problems)**: Empirical Risk Minimization, MLE/MAP, PCA, EM Algorithms, SVM Margins, Kernels.

---

## 2. Is the OKF Schema Correct?

Yes, the **Ontological Knowledge Framework (OKF) version 1.5 specification is correct and highly effective**. The JSON schema defined in `OKF_SPEC.md` provides all the fields needed to represent a concept:

```json
{
  "concept_name": "Masked Language Model",
  "concept_type": "method",
  "difficulty": "foundational",
  "summary": "A pre-training objective...",
  "prerequisites": ["Transformer architecture"],
  "unlocks": ["BERT pre-training"],
  "related_to": [{"concept": "Next Sentence Prediction", "relation": "related_to"}],
  "tags": ["pre-training", "BERT"],
  "source_text_span": "Unlike left-to-right..."
}
```

### Why it failed to populate correctly using the default model:
1. **Model Parameter Limit**: The default model (Qwen 0.8B) lacks the reasoning capacity to output valid structured JSON matching a complex schema while analyzing long academic paragraphs.
2. **Missing Relations**: The model frequently returned empty arrays `[]` for `prerequisites` and `unlocks` because it could not infer structural learning paths from a single paragraph. 
3. **Flipped Edge Direction**: Prerequisites and unlocks were often swapped (e.g. listing a complex method as a prerequisite for a basic definition). Fine-tuning on our training pairs (`okf_training_pairs_v3.jsonl`) is specifically designed to fix this.

---

## 3. What Else to Fix in the Ingestion Pipeline

To prepare for a full model run, we recommend addressing the following remaining pipeline issues:

### Issue A: Page Number Resolution in Chunks
* **Problem**: In `pdf_ingestion.py`, the page number of a chunk is set to the start page of the section. If Section 1 spans pages 1 to 8, all chunks in it are marked as page 1.
* **Fix**: tag each chunk with the **actual** page number of the PyMuPDF blocks making up that chunk.

### Issue B: Math / LaTeX Block Handling
* **Problem**: Chunks from the Math for ML textbook contain raw LaTeX equations (e.g., `\sum_{i=1}^n x_i` or `W_0 + BA`). Standard JSON serialization often breaks when these backslashes are printed directly inside LLM JSON outputs, leading to silent JSON parsing errors.
* **Fix**: Ensure the extraction parser cleans out or escapes double-backslashes `\\` and wraps LLM output decoding in a robust JSON parser that handles mathematical formatting gracefully.

### Issue C: Multi-Column Heading Parsing
* **Problem**: Many PDFs (especially BERT and LoRA) use multi-column formats. The heading detector in `pdf_ingestion.py` often misses headings because text spans are split, resulting in empty section titles (`""`).
* **Fix**: Improve the heading heuristic in `pdf_ingestion.py` to identify multi-column headings and section numbers (e.g., `5 EMPIRICAL EVALUATION`) more reliably.

---

## 4. How to Re-Run the Ingestion with the Model

To extract all concepts from scratch and generate a complete, high-density graph, follow these steps:

### Step 1: Prepare the Ingestion Settings
Ensure your fine-tuned model (or a larger model like Qwen 7B/14B) is running in your Ollama or API server. Set the model name in `okf_pipeline.py` by editing:
```python
# In okf_pipeline.py
MODEL_NAME = "your_finetuned_model_name"
```

### Step 2: Clear the Cache
Since the pipeline will resume if `okf_results.json` exists, you must rename or delete it to force a full re-extraction:
```bash
mv okf_results.json okf_results_backup.json
```

### Step 3: Run the Ingestion Command
Run the pipeline. If you are using a local GPU, run:
```bash
.venv/bin/python okf_pipeline.py
```
If you wish to enforce a uniform page cap per PDF to save cost/time (e.g., maximum 20 pages per PDF), run:
```bash
.venv/bin/python okf_pipeline.py --max-pages 20
```

### Step 4: Export to UI and Restart the Server
Once the pipeline finishes:
1. Copy the generated visual graph to the UI:
   ```bash
   cp okf_graph.json graph_ui/okf_graph.json
   ```
2. Open [http://localhost:5050](http://localhost:5050) to view the complete graph in your browser!
