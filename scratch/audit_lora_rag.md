# Extraction Coverage Audit: LoRA & RAG Papers

## 1. Executive Summary

An audit of the extraction coverage was performed on `Hu2021_LoRA.pdf` and `Lewis2020_RAG.pdf` by comparing their source chunks in `pdf_chunks.json` against the concepts extracted by the Qwen 3.5 0.8B SLM in `okf_results.json`.

Across both documents, the extraction pipeline exhibits severe coverage gaps, particularly in mathematical formalisms, experimental results, and appendices.

### Comparative Extraction Metrics

| Metric | LoRA Paper (`Hu2021_LoRA.pdf`) | RAG Paper (`Lewis2020_RAG.pdf`) |
| :--- | :--- | :--- |
| **Total Chunks** | 62 | 62 |
| **Total Extracted Concepts** | 15 | 8 |
| **Chunks with at least 1 Extraction** | 7 (11.3% of total) | 3 (4.8% of total) |
| **Chunks with NO Extractions** | 55 (88.7% of total) | 59 (95.2% of total) |
| **Prose Chunks** | 47 | 29 |
| **Prose Chunks with Extractions** | 7 (14.9% of prose) | 3 (10.3% of prose) |
| **Prose Chunks with NO Extractions** | 40 (85.1% of prose) | 26 (89.7% of prose) |
| **Reference Chunks (Expected Empty)** | 10 (0% extracted) | 32 (0% extracted) |
| **Table/Math Chunks (Expected Empty)** | 5 (0% extracted) | 1 (0% extracted) |
| **Primary Extraction Source** | Page 1 only (100% of extractions) | Pages 1-2 only (100% of extractions) |

---

## 2. LoRA Paper Coverage Audit (`Hu2021_LoRA.pdf`)

### 2.1 ✅ Extracted Concepts (15)

The following 15 concepts were successfully extracted, all stemming from page 1 source chunks:

| Concept Name | Concept Type | Reported Page | Source Chunk |
| :--- | :--- | :--- | :--- |
| Full Fine-Tuning | method | 1 | `chunk_003` |
| Parameter-Efficient Fine-Tuning | method | 1 | `chunk_003` |
| Low-Rank Adaptation (LoRA) | method | 4 | `chunk_011` |
| Intrinsic Rank Hypothesis | principle | 4 | `chunk_011` |
| Weight Merging for Inference | technique | 4 | `chunk_012` |
| LoRA Scaling Factor | technique | 4 | `chunk_011` |
| Adapter Tuning | method | 3 | `chunk_010` |
| Prefix Tuning | method | 3 | `chunk_010` |
| LoRA Applied to Attention Weights | technique | 5 | `chunk_013` |
| VRAM Reduction via LoRA | metric | 5 | `chunk_014` |
| Task Switching with LoRA | technique | 5 | `chunk_014` |
| LoRA Training Speedup | metric | 5 | `chunk_014` |
| LoRA Batching Limitation | principle | 5 | `chunk_014` |
| LoRA Trainable Parameter Count | metric | 6 | `chunk_019` |
| LoRA as Generalization of Fine-Tuning | principle | 4 | `chunk_012` |

> [!NOTE]
> Even though these concepts are marked as pages 3, 4, 5, or 6 in `okf_results.json`, their source chunks (`chunk_010` to `chunk_019`) are actually labeled with `page_number: 1` in `pdf_chunks.json`. This indicates that the extraction pipeline correctly identified the page numbers from the text, but the ingestor incorrectly labeled the source chunks.

### 2.2 ❌ Unextracted Sections & Chunks

The following sections had **zero** concepts extracted. We map the unextracted chunks back to their actual content and page numbers:

1. **Title / Abstract Block (Page 1)**: `chunk_001` and `chunk_002` were missed. They contain author names, affiliations, and the initial abstract text.
2. **Section 1: Introduction (Pages 1-2)**: `chunk_004` to `chunk_009` were missed. They discuss adapter latency issues, parameter-efficient methods, and transfer learning history.
3. **Section 5: Empirical Evaluation (Pages 5-8)**: Chunks `chunk_015` (RoBERTa evaluation), `chunk_016` (deployment details), `chunk_020` (RoBERTa setup), and `chunk_021` (DeBERTa setup) were missed. Chunks `chunk_022` and `chunk_023` (representing GPT-3 results) were also missed.
4. **Section 6: Related Work (Page 8)**: Chunks `chunk_024` (Prompt engineering), `chunk_025` (Parameter-efficient adaptation), and `chunk_026` (Low-rank structures in DL) were missed.
5. **Section 7: Understanding the Low-Rank Updates (Pages 9-10)**: Chunks `chunk_027` and `chunk_028` (Section 7 header and introductory paragraphs) and chunks `chunk_029` to `chunk_034` (subspace similarity analysis, projections, and ablation studies) were missed.
6. **Appendix E: Combining LoRA with Prefix Tuning (Page 20)**: Chunks `chunk_051` to `chunk_055` were completely missed.
7. **Appendix H: Additional Experiments on Low-Rank Matrices (Page 24)**: Chunks `chunk_059` to `chunk_062` were completely missed.

### 2.3 🔍 Missing Key Concepts for LoRA

The following key concepts were completely omitted and should have been extracted:

| Concept | Actual Section/Page | Why it is Important |
| :--- | :--- | :--- |
| **Low-Rank Matrix Factorization** | §4.1, p. 4 | The fundamental mathematical concept behind LoRA: representing weight updates as $\Delta W = B \times A$. |
| **Frozen Pre-trained Weights** | §4.1, p. 4 | Critical parameter management strategy: keeping the original weights $W_0$ static to ensure no gradient updates. |
| **Kaiming Initialization** | §4.1, p. 4 | Random Gaussian initialization applied to matrix $A$ to break symmetry at the start of training. |
| **Zero Initialization** | §4.1, p. 4 | Initializing matrix $B$ to zero so that $\Delta W = B \times A = 0$ initially, ensuring the starting model behavior is unaltered. |
| **Rank r Hyperparameter** | §4.1, p. 4 | The key scaling parameter controlling the rank of adaptation; analyzed extensively via ablation in Section 7. |
| **Adapter Latency** | §2, p. 3 | The major drawback of sequential adapters that LoRA resolves by adding updates in parallel. |
| **Subspace Similarity Analysis** | §7.2, p. 10 | Quantitative analysis (using Grassmann distance/projection) showing that LoRA learns updates similar to full fine-tuning. |
| **Low-Rank Update Direction** | §7.3, p. 10 | The discovery that the update direction remains stable across different ranks and seeds. |
| **LoRA+PE / LoRA+PL** | Appendix E, p. 20 | Architectural variants combining low-rank adaptation with prefix-embedding or prefix-layer tuning. |
| **Singular Value Decomposition (SVD)** | Appendix H, p. 24 | Used to analyze the low-rank updates and inspect the spectrum of the weight matrices. |
| **Frobenius Norm** | Appendix H, p. 24 | Metric used to measure the magnitude of weight changes $\Delta W$ relative to the pretrained weights. |
| **Grassmann Distance** | §7.2, p. 10 | Mathematical metric used to measure distance between low-rank update subspaces. |
| **Inference Slowdown** | §3, p. 3 | The latency increase caused by adding sequential adapters during deployment. |
| **AdamW Optimizer** | §5, p. 6 | The optimization algorithm used for training the model parameters. |
| **GPT-3 175B Evaluation** | §5, p. 8 | Flagship validation demonstrating LoRA's performance on very large scale models. |

---

## 3. RAG Paper Coverage Audit (`Lewis2020_RAG.pdf`)

### 3.1 ✅ Extracted Concepts (8)

The following 8 concepts were extracted from the RAG paper, limited to the first two pages:

| Concept Name | Concept Type | Reported Page | Source Chunk |
| :--- | :--- | :--- | :--- |
| Retrieval-Augmented Generation (RAG) | architecture | 1 | `chunk_001` |
| Parametric Memory | model | 1 | `chunk_002` |
| Non-Parametric Memory | model | 1 | `chunk_002` |
| Maximum Inner Product Search (MIPS) | algorithm | 1 | `chunk_003` |
| RAG Marginalization | technique | 1 | `chunk_003` |
| RAG Sequence-Level Conditioning | architecture | 1 | `chunk_001` |
| RAG Token-Level Conditioning | architecture | 1 | `chunk_001` |
| End-to-End RAG Fine-Tuning | method | 1 | `chunk_003` |

### 3.2 ❌ Unextracted Sections & Chunks

The following sections had **zero** concepts extracted:

1. **Title / Abstract Block (Page 1)**: Abstract chunk `chunk_001` had extractions, but `chunk_002` was partially unextracted (though the concepts themselves were derived from it).
2. **Section 2: Methods (Pages 2-4)**: Chunks `chunk_004` and `chunk_005` (representing RAG probability distributions) and chunks `chunk_006` to `chunk_010` (detailing the retriever and generator components) were missed.
3. **Section 3: Decoding/Approximations (Page 4)**: Chunks `chunk_011` and `chunk_012` were missed. They discuss RAG-Token vs RAG-Sequence decoding and beam search.
4. **Section 3.1 - 3.4: Tasks (Pages 4-5)**: Chunks `chunk_013` (Open-domain QA), `chunk_014` (Abstractive QA), and `chunk_015` & `chunk_016` (Jeopardy question gen) were missed.
5. **Section 4: Results (Pages 5-7)**: Chunks `chunk_018` (Open-domain QA results), `chunk_019` (results table), `chunk_020` (factual accuracy analysis), `chunk_021` (Abstractive QA results), `chunk_022` & `chunk_023` (Jeopardy results), and `chunk_024` (FEVER results) were missed.
6. **Section 4.4: Qualitative Analysis (Page 7)**: Chunks `chunk_025` & `chunk_026` (document posterior visualization) and `chunk_027` (generation diversity) were missed.
7. **Section 4.5: Non-parametric Memory Updates (Page 7)**: Chunk `chunk_028` (index hot-swapping) was missed.
8. **Appendices (Pages 17-19)**: Chunks `chunk_054` (Implementation), `chunk_055` (Human eval), `chunk_056` (Training setup), `chunk_057` & `chunk_058` (Open-domain QA details), `chunk_059` (FEVER details), `chunk_060` (Null document probabilities), `chunk_061` (Parameter counts), and `chunk_062` (Retrieval collapse) were missed.

### 3.3 🔍 Missing Key Concepts for RAG

The following key concepts were completely omitted and should have been extracted:

| Concept | Actual Section/Page | Why it is Important |
| :--- | :--- | :--- |
| **Dense Passage Retrieval (DPR)** | §3.1, p. 3 | The bi-encoder retrieval backbone that encodes query and document vectors. |
| **BERT-base Document Encoder** | §3.1, p. 3 | The component of DPR that creates dense representations of documents. |
| **BERT-base Query Encoder** | §3.1, p. 3 | The component of DPR that creates a dense representation of the query. |
| **BART Generator** | §3.1, p. 3 | The seq2seq model used to generate output sequences conditioned on retrieved documents. |
| **Thorough Decoding** | §3, p. 3-4 | Modified beam search algorithm that marginalizes over the top-k retrieved documents for RAG-Sequence. |
| **Fast Decoding Approximation** | §3, p. 4 | A computationally efficient alternative to thorough decoding that decodes token-by-token. |
| **Joint Training** | §3.1, p. 3 | Optimizing retriever and generator together without direct supervision on document relevance. |
| **Stale Index Problem** | §3.1, p. 3 | The problem where the document index becomes outdated as document encoder weights are updated during training. |
| **Asynchronous Index Refresh** | §3.1, p. 3 | The solution to the stale index problem, refreshing document embeddings periodically. |
| **Wikipedia Knowledge Source** | §4, p. 4 | The 21M-document corpus utilized as RAG's external non-parametric memory source. |
| **Index Hot-Swapping** | §4.5, p. 7 | The ability to update the non-parametric knowledge base at test time without retraining the parametric generator. |
| **Retrieval Collapse** | Appendix H, p. 19 | A failure mode where the retriever learns to retrieve the same documents regardless of query. |
| **Null Document Probability** | Appendix F, p. 18 | Adding a dummy document to the retrieved list to allow RAG to generate using parametric knowledge only. |
| **Fact Verification (FEVER)** | §3.4, p. 5 | The claim verification task used to test RAG's ability to cross-reference multiple documents. |
| **MS-MARCO NLG Dataset** | §3.2, p. 4 | The abstractive QA dataset used to evaluate generative response quality. |

---

## 4. Root Cause Analysis

We identify the following systematic failure modes in the extraction pipeline:

1. **Ingestor Page Number Locking**: A major logical bug in `pdf_ingestion.py` maps the page number of every chunk in a section to the *start page* of that section. For example, because the "Introduction" section spans pages 1-8, all chunks within it get labeled as `page_number: 1`. This leads to downstream provenance errors.
2. **Small-Model Size Limits**: The `qwen3.5:0.8b` SLM lacks the reasoning capacity to digest dense, multi-layered text. When confronted with paragraphs containing multiple definitions, it extracts the first obvious one and ignores the rest.
3. **Mathematical Representation Failure**: Chunks containing mathematical notation or equations (e.g., probability definitions in RAG §2 or low-rank factorization in LoRA §4.1) cause the model to output empty arrays `[]`. The model struggles to translate equations into conceptual summaries.
4. **Tabular Results and Figures**: Chunks dominated by table numbers or figure captions (such as experimental results tables) are either misclassified as prose and return no concepts, or are parsed into low-quality, formatting-derived concepts (e.g., `Best Model Without Gold Access Underlined`).
5. **No-Section-Title Chunks**: The heading parser often misses headings in multi-column PDF layouts. Chunks labeled with `section_title: ""` lack structural context, which decreases extraction quality and complicates audit mapping.

---

## 5. Technical Recommendations

1. **Fix the Ingestor Page Numbering**: Modify `pdf_ingestion.py` to tag each chunk with the *actual* page number of the blocks making up that chunk, rather than the section start page.
2. **Update the Extraction Prompt**: Inject the generic peer-review example from `OKF_SPEC.md` §3 and pin the Ollama temperature to `0.1`. Low temperature prevents high-temperature drift and reduces JSON syntax errors.
3. **Improve Section Heading Detection**: Refine the heading heuristic in `pdf_ingestion.py` to identify multi-column headings and section numbers (e.g., `5 EMPIRICAL EVALUATION`) more reliably, reducing `(No Section Title)` chunks.
4. **Apply Fine-Tuning**: Leverage the prepared `okf_training_pairs.jsonl` dataset to fine-tune the model, teaching it to suppress metadata extraction, handle math/tables gracefully, and maintain edge directionality.
