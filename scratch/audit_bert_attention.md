# Extraction Audit: BERT & Attention Papers

## BERT (Devlin et al., 2018) — 60 chunks, 14 extracted (23.3% concept-to-chunk ratio, 15.0% chunk coverage)

### ✅ Extracted Concepts (14)
| Concept | Type | Page | Chunk ID |
|---------|------|------|----------|
| Masked Language Model | method | 2 | chunk_004 |
| Bidirectional vs Unidirectional Pre-training | principle | 2 | chunk_004 |
| Pre-trained Representations Reduce Task-Specific Engineering | principle | 2 | chunk_004 |
| Unsupervised Fine-tuning Approaches | method | 2 | chunk_007 |
| BERT Pre-training and Fine-tuning Framework | framework | 3 | chunk_010 |
| Unified Architecture Across Tasks | principle | 3 | chunk_010 |
| Bidirectional Self-Attention | technique | 3 | chunk_011 |
| BERT Model Sizes (BASE and LARGE) | model | 3 | chunk_011 |
| Next Sentence Prediction | method | 4 | chunk_015 |
| BERT Input Representation | architecture | 4 | chunk_015 |
| Fine-tuning with Task-Specific Inputs/Outputs | method | 5 | chunk_017 |
| GLUE Benchmark Fine-tuning Protocol | method | 6 | chunk_020 |
| SQuAD Span Prediction | method | 6 | chunk_022 |
| SQuAD 2.0 No-Answer Extension | technique | 7 | chunk_025 |

### ❌ Missing Concepts (~20+ should have been extracted)
| Missing Concept | Section/Page | Why Important |
|----------------|--------------|---------------|
| Deep Bidirectional Representation | §Abstract, p.1 | BERT's core design conditions jointly on both left and right context in all layers. |
| Unidirectional Model Limitations | §1, p.1 | Standard language models are unidirectional, restricting pre-training and downstream architectures. |
| WordPiece Embeddings | §3.2, p.3 | The input representation uses a 30,000 token vocabulary. |
| Special `[CLS]` Token | §3.2, p.3 | The first token of every sequence; its final hidden state $C$ is used as the aggregate sequence representation. |
| Special `[SEP]` Token | §3.2, p.3 | Token used to separate non-consecutive token sequences (sentence pairs) in the input. |
| Sentence A/B segment embeddings | §3.2, p.3 | Learnable segment embeddings added to indicate whether a token belongs to sentence A or B. |
| Cloze Task Foundation | §3.1, p.4 | The literature context (Taylor, 1953) for MLM, solving the bidirectional "see itself" problem. |
| Pre-training Parameter Transfer | §3.1, p.4 | Unlike sentence-embedding-only transfer, BERT transfers all parameters to initialize downstream models. |
| Fine-Tuning Parameter Efficiency | §4, p.5 | Replicable in <1 hour on Cloud TPU or a few hours on GPU, introducing minimal task-specific weights. |
| Classification Layer Weights ($W$) | §4.1, p.5 | The only new parameters introduced during GLUE fine-tuning: $W \in \mathbb{R}^{K \times H}$. |
| SQuAD 2.0 Prediction Threshold | §4.2, p.7 | Decision boundary threshold $\tau$ selected on dev set to maximize F1 for predicting non-null answers. |
| SWAG Formatting | §4.3, p.7 | Constructing four input sequences of A + B for classification to choose the most plausible continuation. |
| Ablation: No NSP Model | §5.1, p.8 | Evaluating MLM without NSP, showing drops in QNLI, MNLI, and SQuAD 1.1. |
| Ablation: Left-to-Right & No NSP | §5.1, p.8 | LTR Transformer LM (comparable to GPT) demonstrating that bidirectionality drives the improvements. |
| Criticism of Shallow Bi-directionality | §5.1, p.8 | Explains why ELMo's concatenated LTR/RTL approach is twice as expensive and less powerful. |
| Feature-Based Approach Advantages | §5.3, p.9 | Fixed feature extraction method allows training cheaper task-specific models and pre-computing representations. |
| Top Four Layer Concatenation | §5.3, p.9 | Concatenating top four hidden layers for feature-based classification yields results close to full fine-tuning (96.1 F1). |
| MLM Masking Schedule (80/10/10) | §C.2, p.16 | Replaces 15% of selected tokens with `[MASK]` (80%), a random word (10%), or keeping them unchanged (10%). |
| Dual Sequence Length Schedule | §A.2, p.13 | Pre-training 90% of steps on sequence length 128, and 10% on 512 to learn positional embeddings efficiently. |
| Pre-training Optimization details | §A.2, p.13 | Batch size (128k tokens), 1M steps, Adam optimizer ($LR=1e-4$), GELU activation, and compute resources. |
| Fine-tuning Hyperparameter Ranges | §A.3, p.13 | Standard range of batch sizes (16, 32), learning rates (5e-5, 3e-5, 2e-5), and epochs (2, 3, 4). |

**31 prose chunks were missed.** Additionally, **12 reference-kind chunks containing core prose were missed** due to parser misclassification.

---

## Attention (Vaswani et al., 2017) — 37 chunks, 24 extracted (64.9% concept-to-chunk ratio, 29.7% chunk coverage)

### ✅ Extracted Concepts (24)
| Concept | Type | Page | Chunk ID |
|---------|------|------|----------|
| Attention Function | definition | 3 | chunk_009 |
| Scaled Dot-Product Attention | algorithm | 3 | chunk_010 |
| Attention Scaling Factor | technique | 3 | chunk_010 |
| Additive Attention | algorithm | 3 | chunk_010 |
| Dot-Product Attention | algorithm | 3 | chunk_010 |
| Multi-Head Attention | architecture | 4 | chunk_011 |
| Multi-Head Attention Dimensions | definition | 4 | chunk_011 |
| Encoder Self-Attention | architecture | 4 | chunk_012 |
| Decoder Self-Attention | architecture | 4 | chunk_012 |
| Masked Self-Attention | technique | 4 | chunk_012 |
| Encoder-Decoder Attention | architecture | 4 | chunk_012 |
| Position-wise Feed-Forward Network | architecture | 5 | chunk_013 |
| FFN Dimensions | definition | 5 | chunk_013 |
| Embeddings and Softmax | architecture | 5 | chunk_014 |
| Layer Type Complexity Comparison | principle | 5 | chunk_014 |
| Sinusoidal Positional Encoding | algorithm | 5 | chunk_015 |
| Positional Encoding | definition | 5 | chunk_015 |
| Sinusoidal vs Learned Positional Embeddings | principle | 5 | chunk_015 |
| Self-Attention Path Length | principle | 6 | chunk_016 |
| Self-Attention vs Recurrent vs Convolutional | principle | 6 | chunk_016 |
| Self-Attention Interpretability | principle | 7 | chunk_018 |
| Attention Head Specialization | definition | 7 | chunk_018 |
| Transformer Training Hardware Schedule | definition | 7 | chunk_020 |
| Learning Rate Schedule | technique | 7 | chunk_021 |

### ❌ Missing Concepts (~15+ should have been extracted)
| Missing Concept | Section/Page | Why Important |
|----------------|--------------|---------------|
| Dispensing with Recurrence/Convolutions | §Abstract, p.1 | Core Transformer proposal: architecture based solely on self-attention, removing recurrence/convolutions. |
| Training Speed and Cost Metrics | §Abstract, p.1 | Outperforming prior SOTA translation models at a fraction of the cost (12 hours / 3.5 days on 8 P100 GPUs). |
| Encoder Stack Layer Definition ($N=6$) | §3.1, p.3 | Stack of $N=6$ identical layers, each with 2 sub-layers (multi-head self-attention and position-wise FFN). |
| Decoder Stack Layer Definition ($N=6$) | §3.1, p.3 | Stack of $N=6$ identical layers, with 3 sub-layers (adding encoder-decoder attention) and masked self-attention. |
| Residual & LayerNorm Implementation | §3.1, p.3 | Structure of sub-layer outputs: `LayerNorm(x + Sublayer(x))`. |
| Residual Output Dimension ($d_{model}=512$) | §3.1, p.3 | All sub-layers and embedding layers produce outputs of dimension 512 to enable residual addition. |
| BPE Tokenization and Vocabularies | §5.1, p.7 | BPE shared vocab of 37,000 tokens (EN-DE) and word-piece vocab of 32,000 tokens (EN-FR). |
| Token-based Batching | §5.1, p.7 | Sentence pairs are batched by approximate sequence length to contain ~25,000 source and target tokens. |
| Residual Dropout Regularization | §5.4, p.8 | Applying $P_{drop} = 0.1$ dropout to outputs of each sub-layer and to the sum of embeddings. |
| Label Smoothing Regularization | §5.4, p.8 | Using label smoothing ($\epsilon_{ls} = 0.1$) to prevent model overconfidence, helping BLEU. |
| Checkpoint Averaging | §6, p.8 | Averaging the last 5 checkpoints (base) or 20 checkpoints (big) to construct the final single model. |
| Beam Search Decoding settings | §6, p.8 | Beam size 4, length penalty $\alpha = 0.6$ decoding parameters. |
| Ablation: Attention Head Count ($h$) | §6.2, p.8 | Row (A) variations showing that single-head attention is 0.9 BLEU worse, and too many heads also degrade quality. |
| English Constituency Parsing Generalization | §6.3, p.9 | Generalizing Transformer to structural constituency parsing under WSJ-only and semi-supervised settings. |
| Attention Head Syntactic Specialization | §Appendix, p.13 | Different heads learning to perform specialized structure-related tasks like long-distance dependency resolution. |
| Anaphora Resolution Visualization | §Appendix, p.13 | Visualizing specialized attention heads resolving pronoun reference (e.g. 'its'). |

**10 prose chunks and 1 table chunk were missed.** Additionally, **10 reference-kind chunks containing core prose were missed** due to parser misclassification.

---

## Root Cause Analysis

1. **Ingestion-level Reference Classification Bug (Major)**:
   - **Mechanism**: The parser `pdf_ingestion.py` processes PDF blocks and joins them with `\n` to build chunks. This means a chunk in `pdf_chunks.json` contains only a few lines (typically 1 to 3), where each line is a full paragraph.
   - **Defect**: The classification function `classify_chunk_kind` uses a citation line ratio check: `cite_lines / len(lines) > 0.45` to filter out bibliographies. Because `lines` represent paragraphs, any paragraph containing a single citation (like `[1]`, `Peters et al. (2018)`) is flagged as a citation line. In a 1 or 2 paragraph chunk, this makes the ratio 100% or 50%, which exceeds the 45% threshold.
   - **Impact**: Crucial prose sections containing background reviews, related work, or layer references (such as the Abstract, Section 2 Literature Reviews, and Section 3.1 Encoder/Decoder Stack definition) are flagged as `reference` and are completely ignored by the concept extraction pipeline.

2. **Page and Section Bias**:
   - The extraction pipeline heavily favors introductory sections (Pages 1-3) and fails to extract concepts from experimental sections, ablation studies, or appendices. 
   - This occurs because experimental data is often packed in dense tables, which are classified as `table` (ratio of numeric tokens > 0.28) and dropped, or because the SLM is not prompted/trained to extract empirical/ablation conclusions.

3. **Mathematical Notation Blindness**:
   - Core mathematical formulations (such as residual definitions, classification loss formulas, and parsing heuristics) are ignored. The pipeline categorizes math-dense blocks as `math` or `table` and filters them out before concept extraction.
