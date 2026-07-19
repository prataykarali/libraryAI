#!/usr/bin/env python3
"""
Programmatic OKF dataset generator - creates 500-600 training cases
using text analysis + existing overrides + pattern-based extraction.
No Ollama needed.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

from pdf_ingestion import ingest_folder

# ─── Config ──────────────────────────────────────────────────────
PDF_DIR = Path(__file__).resolve().parent / "pdfs"
CORE_DOCS = {
    "papers/Devlin2018_BERT.pdf",
    "papers/Edge2024_GraphRAG.pdf",
    "papers/Hu2021_LoRA.pdf",
    "papers/Lewis2020_RAG.pdf",
    "papers/Vaswani2017_Attention_Is_All_You_Need.pdf",
    "probable.pdf",
    "textbooks/Deisenroth_Math_For_ML.pdf",
    "web_syllabi/AI_ML_Archipelago_Corpus_Seed.md"
}
OUT_DIR = Path(__file__).resolve().parent / "training_data"
OUT_DIR.mkdir(exist_ok=True)

TARGET_TRAIN = 500
TARGET_TEST = 100
MAX_CONCEPTS = 5

# ─── Concept Extraction Patterns ─────────────────────────────────
# Key concept signatures in text -> concept definition
CONCEPT_PATTERNS = [
    # BERT / Transformer concepts
    (r"\bBERT\b", {
        "concept_name": "BERT",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Bidirectional Encoder Representations from Transformers - a bidirectional Transformer encoder pre-trained with masked language modeling and next sentence prediction.",
        "prerequisites": ["Transformer Encoder", "Masked Language Modeling", "Next Sentence Prediction"],
        "unlocks": ["Fine-Tuning Approach", "Feature-Based Approach"],
        "related_to": [{"concept": "GPT", "relation": "contrasts_with"}],
        "tags": ["bert", "pre-trained-model", "transformer"]
    }),
    (r"\bGPT\b", {
        "concept_name": "GPT",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Generative Pre-trained Transformer - a left-to-right language model using Transformer decoder architecture.",
        "prerequisites": ["Transformer Decoder", "Language Model Pre-training"],
        "unlocks": ["Fine-Tuning Approach"],
        "related_to": [{"concept": "BERT", "relation": "contrasts_with"}],
        "tags": ["gpt", "pre-trained-model", "transformer"]
    }),
    (r"masked language model(?:ing)?|MLM\b", {
        "concept_name": "Masked Language Modeling",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A pre-training objective where random tokens are masked and the model predicts them from bidirectional context.",
        "prerequisites": ["Language Model Pre-training", "Transformer Encoder"],
        "unlocks": ["Bidirectional Representation Learning"],
        "related_to": [{"concept": "Next Sentence Prediction", "relation": "contrasts_with"}],
        "tags": ["mlm", "pre-training", "bert"]
    }),
    (r"next sentence prediction|NSP\b", {
        "concept_name": "Next Sentence Prediction",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A pre-training task where the model predicts whether two sentences are consecutive in the original text.",
        "prerequisites": ["Language Model Pre-training"],
        "unlocks": [],
        "related_to": [{"concept": "Masked Language Modeling", "relation": "contrasts_with"}],
        "tags": ["nsp", "pre-training", "bert"]
    }),
    (r"fine.?tun(?:e|ing)", {
        "concept_name": "Fine-Tuning",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
        "prerequisites": ["Pre-Trained Model"],
        "unlocks": ["Task-Specific Model"],
        "related_to": [{"concept": "Feature-Based Approach", "relation": "contrasts_with"}],
        "tags": ["fine-tuning", "transfer-learning"]
    }),
    (r"feature.?based", {
        "concept_name": "Feature-Based Approach",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A transfer learning approach where fixed pre-trained representations are extracted and used as features for a downstream model.",
        "prerequisites": ["Pre-Trained Model", "Contextual Embeddings"],
        "unlocks": ["Downstream Classifier"],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["feature-extraction", "transfer-learning"]
    }),
    (r"\bELMo\b", {
        "concept_name": "ELMo",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Embeddings from Language Models - context-sensitive word representations from bidirectional LSTMs.",
        "prerequisites": ["BiLSTM", "Language Model Pre-training"],
        "unlocks": ["Contextual Word Embeddings"],
        "related_to": [{"concept": "BERT", "relation": "contrasts_with"}],
        "tags": ["elmo", "feature-based", "bilstm"]
    }),
    (r"\bBiLSTM\b|\bbidirectional LSTM\b", {
        "concept_name": "BiLSTM",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "Bidirectional LSTM that processes sequences in both forward and backward directions to capture contextual information.",
        "prerequisites": ["LSTM", "Recurrent Neural Network"],
        "unlocks": [],
        "related_to": [{"concept": "Transformer Encoder", "relation": "contrasts_with"}],
        "tags": ["bilstm", "recurrent"]
    }),
    (r"WordPiece", {
        "concept_name": "WordPiece Tokenization",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "A subword tokenization algorithm that splits words into frequent subword units.",
        "prerequisites": ["Tokenization"],
        "unlocks": ["BERT Input Representation"],
        "related_to": [],
        "tags": ["wordpiece", "tokenization", "subword"]
    }),
    (r"\[CLS\]", {
        "concept_name": "CLS Token",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "A special classification token ([CLS]) prepended to input sequences whose final hidden state serves as aggregate sequence representation.",
        "prerequisites": ["Transformer Encoder"],
        "unlocks": ["Sequence-Level Classification"],
        "related_to": [{"concept": "SEP Token", "relation": "related_to"}],
        "tags": ["cls-token", "classification", "bert"]
    }),
    (r"\[SEP\]", {
        "concept_name": "SEP Token",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "A special separator token ([SEP]) used to distinguish sentence pairs in a single input sequence.",
        "prerequisites": ["Transformer Encoder"],
        "unlocks": [],
        "related_to": [{"concept": "CLS Token", "relation": "related_to"}],
        "tags": ["sep-token", "bert", "sentence-pairs"]
    }),
    (r"segment embeddings?|sentence.?[AB] embeddings?", {
        "concept_name": "Segment Embeddings",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "Learned embeddings added to each token indicating whether it belongs to sentence A or sentence B.",
        "prerequisites": ["Position Embeddings"],
        "unlocks": [],
        "related_to": [{"concept": "Position Embeddings", "relation": "related_to"}],
        "tags": ["segment-embeddings", "sentence-pairs", "bert"]
    }),
    (r"position embeddings?", {
        "concept_name": "Position Embeddings",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "Learned or fixed embeddings that encode the position of each token in the sequence, enabling the Transformer to utilize sequence order.",
        "prerequisites": ["Transformer Encoder"],
        "unlocks": [],
        "related_to": [{"concept": "Segment Embeddings", "relation": "related_to"}],
        "tags": ["position-embeddings", "transformer", "bert"]
    }),
    (r"input representation|token.?segment.?position", {
        "concept_name": "BERT Input Representation",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "The construction of input representations by summing token, segment, and position embeddings for each token.",
        "prerequisites": ["Token Embeddings", "Segment Embeddings", "Position Embeddings"],
        "unlocks": [],
        "related_to": [],
        "tags": ["bert", "input-representation", "embeddings"]
    }),
    (r"transformer encoder", {
        "concept_name": "Transformer Encoder",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A bidirectional Transformer architecture using full self-attention over the entire input sequence.",
        "prerequisites": ["Attention Mechanism", "Transformer"],
        "unlocks": ["BERT", "Masked Language Modeling"],
        "related_to": [{"concept": "Transformer Decoder", "relation": "contrasts_with"}],
        "tags": ["transformer-encoder", "attention", "architecture"]
    }),
    (r"transformer decoder", {
        "concept_name": "Transformer Decoder",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A left-context-only Transformer architecture using masked self-attention, suitable for text generation.",
        "prerequisites": ["Attention Mechanism", "Transformer"],
        "unlocks": ["Left-to-Right Language Model"],
        "related_to": [{"concept": "Transformer Encoder", "relation": "contrasts_with"}],
        "tags": ["transformer-decoder", "generation", "architecture"]
    }),
    (r"attention mechanism|self.?attention", {
        "concept_name": "Attention Mechanism",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A mechanism allowing tokens to attend to other tokens in the sequence, computing weighted representations.",
        "prerequisites": [],
        "unlocks": ["Transformer", "Multi-Head Attention"],
        "related_to": [],
        "tags": ["attention", "transformer"]
    }),
    (r"multi.?head attention", {
        "concept_name": "Multi-Head Attention",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "Attention performed in parallel across multiple heads, allowing the model to attend to different representation subspaces.",
        "prerequisites": ["Attention Mechanism"],
        "unlocks": [],
        "related_to": [],
        "tags": ["multi-head-attention", "attention"]
    }),
    # LoRA concepts
    (r"\bLoRA\b|low.?rank adaptation", {
        "concept_name": "LoRA",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into Transformer layers.",
        "prerequisites": ["Pre-Trained Model", "Matrix Factorization"],
        "unlocks": ["Efficient Fine-Tuning", "Parameter-Efficient Adaptation"],
        "related_to": [{"concept": "Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["lora", "peft", "efficient-tuning"]
    }),
    (r"parameter.?efficient|PEFT", {
        "concept_name": "Parameter-Efficient Fine-Tuning",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "Methods that adapt pre-trained models with minimal trainable parameters and no inference latency.",
        "prerequisites": ["Pre-Trained Model", "Fine-Tuning"],
        "unlocks": ["Quick Task Switching"],
        "related_to": [{"concept": "Full Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["peft", "efficient-tuning"]
    }),
    (r"rank.?decomposition|low.?rank matrix", {
        "concept_name": "Low-Rank Decomposition",
        "concept_type": "technique",
        "difficulty": "advanced",
        "summary": "Decomposing a weight update matrix into two low-rank matrices to reduce trainable parameters.",
        "prerequisites": ["Matrix Factorization", "Linear Algebra"],
        "unlocks": [],
        "related_to": [],
        "tags": ["low-rank", "matrix-factorization"]
    }),
    # GraphRAG concepts
    (r"GraphRAG|graph.?rag", {
        "concept_name": "GraphRAG",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "A graph-based approach to question answering over private text corpora that builds an entity knowledge graph and pre-generates community summaries.",
        "prerequisites": ["RAG", "Knowledge Graph"],
        "unlocks": ["Global Question Answering"],
        "related_to": [{"concept": "RAG", "relation": "extends"}],
        "tags": ["graphrag", "graph-rag", "knowledge-graph"]
    }),
    (r"retrieval.?augmented generation|RAG\b", {
        "concept_name": "Retrieval-Augmented Generation",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A method that retrieves relevant documents from an external knowledge source and uses them to augment LLM generation.",
        "prerequisites": ["Information Retrieval", "Language Model"],
        "unlocks": ["GraphRAG"],
        "related_to": [{"concept": "Retrieve-and-Edit", "relation": "contrasts_with"}],
        "tags": ["rag", "retrieval", "generation"]
    }),
    (r"knowledge graph", {
        "concept_name": "Knowledge Graph",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A structured representation of entities and relationships extracted from text, enabling structured reasoning.",
        "prerequisites": ["Entity Extraction", "Relation Extraction"],
        "unlocks": ["GraphRAG"],
        "related_to": [],
        "tags": ["knowledge-graph", "entity-extraction"]
    }),
    (r"entity extraction", {
        "concept_name": "Entity Extraction",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "A process using language models to detect and capture meaningful entities from text to create knowledge graph elements.",
        "prerequisites": ["Language Model"],
        "unlocks": ["Knowledge Graph Construction"],
        "related_to": [{"concept": "Claim Extraction", "relation": "related_to"}],
        "tags": ["entity-extraction", "information-extraction"]
    }),
    (r"community detection|leiden|louvain", {
        "concept_name": "Community Detection",
        "concept_type": "method",
        "difficulty": "intermediate",
        "summary": "Algorithms that partition a graph into clusters of densely connected nodes (communities).",
        "prerequisites": ["Graph Theory", "Clustering"],
        "unlocks": ["Community Summaries"],
        "related_to": [],
        "tags": ["community-detection", "graph", "clustering"]
    }),
    # RAG concepts
    (r"retrieve.?and.?edit", {
        "concept_name": "Retrieve-and-Edit",
        "concept_type": "method",
        "difficulty": "advanced",
        "summary": "An approach that retrieves similar training examples and edits them to produce output for a new input.",
        "prerequisites": ["Retrieval", "Sequence-to-Sequence Model"],
        "unlocks": [],
        "related_to": [{"concept": "Retrieval-Augmented Generation", "relation": "contrasts_with"}],
        "tags": ["retrieve-and-edit", "retrieval"]
    }),
    # Math / ML foundations
    (r"\bvector\b", {
        "concept_name": "Vector",
        "concept_type": "definition",
        "difficulty": "foundational",
        "summary": "An element of a vector space that can be added to other vectors and multiplied by scalars.",
        "prerequisites": [],
        "unlocks": ["Linear Algebra"],
        "related_to": [],
        "tags": ["vector", "linear-algebra", "mathematics"]
    }),
    (r"matrix decomposition|SVD|eigen", {
        "concept_name": "Matrix Decomposition",
        "concept_type": "technique",
        "difficulty": "intermediate",
        "summary": "Factorizing a matrix into constituent matrices to reveal structural properties.",
        "prerequisites": ["Linear Algebra", "Matrix"],
        "unlocks": ["PCA", "Low-Rank Approximation"],
        "related_to": [],
        "tags": ["matrix-decomposition", "svd", "linear-algebra"]
    }),
    (r"gradient descent|optimization", {
        "concept_name": "Gradient Descent",
        "concept_type": "method",
        "difficulty": "foundational",
        "summary": "An iterative optimization algorithm that moves parameters in the direction of steepest descent of the loss function.",
        "prerequisites": ["Calculus", "Gradient"],
        "unlocks": ["Model Training"],
        "related_to": [],
        "tags": ["gradient-descent", "optimization", "training"]
    }),
]

# ─── Chunk Overrides (ground truth) ───────────────────────────────
CHUNK_OVERRIDES = {
    ("papers/Devlin2018_BERT.pdf", "chunk_001"): [{
        "concept_name": "BERT", "concept_type": "method", "difficulty": "intermediate",
        "summary": "Bidirectional Encoder Representations from Transformers - a bidirectional Transformer encoder pre-trained with masked language modeling and next sentence prediction.",
        "prerequisites": ["Transformer Encoder"], "unlocks": ["Fine-Tuning Approach", "Feature-Based Approach"],
        "related_to": [{"concept": "GPT", "relation": "contrasts_with"}, {"concept": "ELMo", "relation": "contrasts_with"}],
        "tags": ["bert", "pre-trained-model", "transformer"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_002"): [{
        "concept_name": "Feature-Based Strategy", "concept_type": "method", "difficulty": "intermediate",
        "summary": "An approach that uses pre-trained representations as fixed additional features within task-specific architectures, without fine-tuning the pre-trained parameters.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": [],
        "related_to": [{"concept": "Fine-Tuning Strategy", "relation": "contrasts_with"}],
        "tags": ["transfer-learning", "feature-extraction"]
    }, {
        "concept_name": "Fine-Tuning Strategy", "concept_type": "method", "difficulty": "intermediate",
        "summary": "An approach that adds minimal task-specific parameters and trains all model parameters end-to-end on a downstream task.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": [],
        "related_to": [{"concept": "Feature-Based Strategy", "relation": "contrasts_with"}],
        "tags": ["transfer-learning", "fine-tuning"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_003"): [{
        "concept_name": "Masked Language Modeling", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A pre-training objective where random tokens are masked and the model predicts them using bidirectional context.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": ["Bidirectional Representation Learning"],
        "related_to": [], "tags": ["masked-lm", "pre-training", "nlp"]
    }, {
        "concept_name": "Unidirectional Constraints", "concept_type": "theory", "difficulty": "intermediate",
        "summary": "A limitation in standard language models where token attention is restricted to left-to-right context during pre-training.",
        "prerequisites": [], "unlocks": ["Masked Language Modeling"],
        "related_to": [], "tags": ["language-modeling", "attention"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_004"): [{
        "concept_name": "Next Sentence Prediction", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A pre-training task where the model predicts whether two sentences are consecutive in the original text.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": [],
        "related_to": [{"concept": "Masked Language Modeling", "relation": "contrasts_with"}],
        "tags": ["pre-training", "nsp", "sentence-relationships"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_005"): [{
        "concept_name": "Pre-Trained Word Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Vector representations of words learned from large corpora and used to initialize NLP models, offering significant improvements over random initialization.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": ["Sentence Embeddings", "Paragraph Embeddings"],
        "related_to": [], "tags": ["word-embeddings", "pre-training", "nlp"]
    }, {
        "concept_name": "Sentence Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Fixed-length vector representations of entire sentences, trained using objectives like next-sentence ranking or auto-encoding.",
        "prerequisites": ["Pre-Trained Word Embeddings"], "unlocks": [],
        "related_to": [{"concept": "Paragraph Embeddings", "relation": "contrasts_with"}],
        "tags": ["sentence-embeddings", "pre-training"]
    }, {
        "concept_name": "Paragraph Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Vector representations of paragraphs or documents extending word embedding methods to coarser granularities (e.g., Paragraph Vector / doc2vec).",
        "prerequisites": ["Pre-Trained Word Embeddings"], "unlocks": [],
        "related_to": [{"concept": "Sentence Embeddings", "relation": "contrasts_with"}],
        "tags": ["paragraph-embeddings", "doc2vec", "pre-training"]
    }, {
        "concept_name": "Language Modeling Objective", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A training objective where the model predicts the next token or discriminates correct words from context, used to pre-train embeddings.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": ["Pre-Trained Word Embeddings"],
        "related_to": [{"concept": "Next-Sentence Ranking", "relation": "contrasts_with"}],
        "tags": ["language-modeling", "pre-training"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_006"): [{
        "concept_name": "ELMo", "concept_type": "method", "difficulty": "intermediate",
        "summary": "Embeddings from Language Models — a feature-based approach using bidirectional LSTM representations as context-sensitive features for downstream tasks.",
        "prerequisites": ["BiLSTM", "Language Model Pre-training"], "unlocks": ["Contextual Word Embeddings"],
        "related_to": [{"concept": "BERT", "relation": "contrasts_with"}],
        "tags": ["elmo", "feature-based", "bilstm", "contextual-embeddings"]
    }, {
        "concept_name": "Contextual Word Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Token representations that vary based on surrounding context, produced by models like ELMo using concatenated left-to-right and right-to-left LSTM states.",
        "prerequisites": ["BiLSTM", "Pre-Trained Word Embeddings"], "unlocks": [],
        "related_to": [{"concept": "Pre-Trained Word Embeddings", "relation": "contrasts_with"}],
        "tags": ["contextual-embeddings", "elmo"]
    }, {
        "concept_name": "Cloze Task", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A fill-in-the-blank task where a single word is predicted from both left and right context, used to learn contextual representations.",
        "prerequisites": ["Language Model Pre-training"], "unlocks": ["Masked Language Modeling"],
        "related_to": [], "tags": ["cloze-task", "pre-training"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_012"): [{
        "concept_name": "WordPiece Tokenization", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A subword tokenization algorithm that splits vocabulary into frequent subword units, balancing vocabulary size and sequence length.",
        "prerequisites": [], "unlocks": ["BERT Input Representation"],
        "related_to": [], "tags": ["wordpiece", "tokenization", "subword"]
    }, {
        "concept_name": "CLS Token", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A special classification token ([CLS]) prepended to every input sequence, whose final hidden state serves as the aggregate sequence representation for classification tasks.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "SEP Token", "relation": "related_to"}],
        "tags": ["cls-token", "classification", "bert"]
    }, {
        "concept_name": "SEP Token", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A special separator token ([SEP]) used to distinguish sentence pairs in a single input sequence.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "CLS Token", "relation": "related_to"}],
        "tags": ["sep-token", "bert", "sentence-pairs"]
    }, {
        "concept_name": "Segment Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Learned embeddings added to each token indicating whether it belongs to sentence A or sentence B, enabling the model to distinguish between sentence pairs.",
        "prerequisites": ["Position Embeddings"], "unlocks": [],
        "related_to": [], "tags": ["segment-embeddings", "sentence-pairs", "bert"]
    }, {
        "concept_name": "BERT Input Representation", "concept_type": "method", "difficulty": "intermediate",
        "summary": "The construction of input representations by summing token, segment, and position embeddings for each token in the sequence.",
        "prerequisites": ["Token Embeddings", "Segment Embeddings", "Position Embeddings"], "unlocks": [],
        "related_to": [], "tags": ["bert", "input-representation", "embeddings"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_013"): [{
        "concept_name": "Masked Language Modeling", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A pre-training objective where random tokens are masked and the model predicts them using bidirectional context, enabling deep bidirectional representations.",
        "prerequisites": ["Language Model Pre-training", "Transformer Encoder"], "unlocks": ["Bidirectional Representation Learning"],
        "related_to": [{"concept": "Left-to-Right Language Model", "relation": "contrasts_with"}],
        "tags": ["mlm", "masked-lm", "pre-training", "bert"]
    }, {
        "concept_name": "Bidirectional Representation Learning", "concept_type": "method", "difficulty": "advanced",
        "summary": "Learning representations that jointly condition on both left and right context, as opposed to unidirectional left-to-right or right-to-left models.",
        "prerequisites": ["Transformer Encoder", "Masked Language Modeling"], "unlocks": [],
        "related_to": [{"concept": "Left-to-Right Language Model", "relation": "contrasts_with"}],
        "tags": ["bidirectional", "representation-learning", "bert"]
    }, {
        "concept_name": "Transformer Encoder", "concept_type": "method", "difficulty": "intermediate",
        "summary": "The bidirectional Transformer architecture using full self-attention over the entire input sequence, as opposed to the masked self-attention in a Transformer decoder.",
        "prerequisites": ["Attention Mechanism", "Transformer"], "unlocks": ["BERT", "Masked Language Modeling"],
        "related_to": [{"concept": "Transformer Decoder", "relation": "contrasts_with"}],
        "tags": ["transformer-encoder", "attention", "architecture"]
    }, {
        "concept_name": "Transformer Decoder", "concept_type": "method", "difficulty": "intermediate",
        "summary": "The left-context-only Transformer architecture using masked self-attention, suitable for text generation.",
        "prerequisites": ["Attention Mechanism", "Transformer"], "unlocks": ["Left-to-Right Language Model"],
        "related_to": [{"concept": "Transformer Encoder", "relation": "contrasts_with"}],
        "tags": ["transformer-decoder", "generation", "architecture"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_014"): [{
        "concept_name": "Masked Language Modeling", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A pre-training procedure masking 15% of tokens and predicting them, using [MASK] 80%, random token 10%, and unchanged token 10% of the time.",
        "prerequisites": ["Language Model Pre-training", "Transformer Encoder"], "unlocks": ["Bidirectional Representation Learning"],
        "related_to": [{"concept": "Cloze Task", "relation": "variant_of"}],
        "tags": ["mlm", "masking", "pre-training", "bert"]
    }, {
        "concept_name": "Mask Token", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A special [MASK] token used to replace masked positions during pre-training, with a mitigation strategy to reduce pre-train/fine-tune mismatch.",
        "prerequisites": ["Tokenization", "Masked Language Modeling"], "unlocks": [],
        "related_to": [], "tags": ["mask-token", "bert", "pre-training"]
    }, {
        "concept_name": "Pre-Train Fine-Tune Mismatch", "concept_type": "theory", "difficulty": "advanced",
        "summary": "The discrepancy between pre-training (where [MASK] tokens appear) and fine-tuning (where they don't), mitigated by not always replacing with [MASK].",
        "prerequisites": ["Masked Language Modeling", "Fine-Tuning"], "unlocks": [],
        "related_to": [], "tags": ["domain-shift", "pre-training", "fine-tuning"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_028"): [{
        "concept_name": "Ablation Study", "concept_type": "method", "difficulty": "intermediate",
        "summary": "An experimental method that systematically removes components of a model to measure their individual contribution to performance.",
        "prerequisites": ["Controlled Experiment"], "unlocks": ["Component Analysis"],
        "related_to": [], "tags": ["ablation", "experimental-design"]
    }, {
        "concept_name": "Next Sentence Prediction", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A pre-training task where the model predicts whether two sentences are consecutive in the original text.",
        "prerequisites": ["Language Modeling"], "unlocks": [],
        "related_to": [{"concept": "Masked Language Modeling", "relation": "contrasts_with"}],
        "tags": ["pre-training", "nsp"]
    }, {
        "concept_name": "Masked Language Modeling", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A pre-training objective where random tokens are masked and the model predicts them from context.",
        "prerequisites": ["Language Modeling"], "unlocks": [],
        "related_to": [{"concept": "Next Sentence Prediction", "relation": "contrasts_with"}],
        "tags": ["pre-training", "mlm"]
    }, {
        "concept_name": "Left-to-Right Language Model", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A language modeling approach that predicts tokens sequentially from left to right, as used in GPT.",
        "prerequisites": ["Language Modeling"], "unlocks": [],
        "related_to": [{"concept": "Masked Language Modeling", "relation": "contrasts_with"}],
        "tags": ["language-modeling", "ltr"]
    }, {
        "concept_name": "BiLSTM", "concept_type": "method", "difficulty": "advanced",
        "summary": "A bidirectional LSTM that processes sequences in both forward and backward directions to capture contextual information.",
        "prerequisites": ["Recurrent Neural Network", "LSTM"], "unlocks": [],
        "related_to": [{"concept": "Transformer Encoder", "relation": "contrasts_with"}],
        "tags": ["bilstm", "recurrent"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_033"): [{
        "concept_name": "BERT", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A bidirectional Transformer encoder pre-trained with masked language modeling and next sentence prediction.",
        "prerequisites": ["Transformer Encoder", "Masked Language Modeling", "Next Sentence Prediction"], "unlocks": ["Fine-Tuning Approach", "Feature-Based Approach"],
        "related_to": [{"concept": "GPT", "relation": "contrasts_with"}],
        "tags": ["bert", "pre-trained-model", "transformer"]
    }, {
        "concept_name": "Fine-Tuning Approach", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
        "prerequisites": ["Pre-Trained Model"], "unlocks": ["Task-Specific Model"],
        "related_to": [{"concept": "Feature-Based Approach", "relation": "contrasts_with"}],
        "tags": ["fine-tuning", "transfer-learning"]
    }, {
        "concept_name": "Feature-Based Approach", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A transfer learning approach where fixed pre-trained representations are extracted and used as features for a downstream model.",
        "prerequisites": ["Pre-Trained Model", "Contextual Embeddings"], "unlocks": ["Downstream Classifier"],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["feature-extraction", "transfer-learning"]
    }, {
        "concept_name": "Named Entity Recognition", "concept_type": "task", "difficulty": "intermediate",
        "summary": "A sequence labeling task that identifies and classifies named entities in text into predefined categories.",
        "prerequisites": ["Token Representation", "Sequence Labeling"], "unlocks": [],
        "related_to": [], "tags": ["ner", "sequence-labeling", "information-extraction"]
    }, {
        "concept_name": "Contextual Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Token representations extracted from specific layers of a pre-trained model that capture context-dependent meaning.",
        "prerequisites": ["Transformer Encoder", "Pre-Trained Model"], "unlocks": ["Feature-Based Approach"],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["embeddings", "contextual"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_034"): [{
        "concept_name": "Fine-Tuning Approach", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
        "prerequisites": ["Pre-Trained Model"], "unlocks": ["Task-Specific Model"],
        "related_to": [{"concept": "Feature-Based Approach", "relation": "contrasts_with"}],
        "tags": ["fine-tuning", "transfer-learning"]
    }, {
        "concept_name": "Feature-Based Approach", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A transfer learning approach where fixed pre-trained representations are extracted and used as features for a downstream model.",
        "prerequisites": ["Pre-Trained Model", "Contextual Embeddings"], "unlocks": ["Downstream Classifier"],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["feature-extraction", "transfer-learning"]
    }, {
        "concept_name": "Named Entity Recognition", "concept_type": "task", "difficulty": "intermediate",
        "summary": "A sequence labeling task that identifies and classifies named entities in text into predefined categories.",
        "prerequisites": ["Token Representation", "Sequence Labeling"], "unlocks": [],
        "related_to": [], "tags": ["ner", "sequence-labeling", "information-extraction"]
    }, {
        "concept_name": "Contextual Embeddings", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "Token representations extracted from specific layers of a pre-trained model that capture context-dependent meaning.",
        "prerequisites": ["Transformer Encoder", "Pre-Trained Model"], "unlocks": ["Feature-Based Approach"],
        "related_to": [{"concept": "Fine-Tuning Approach", "relation": "contrasts_with"}],
        "tags": ["embeddings", "contextual"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_054"): [{
        "concept_name": "Sequence-Level Classification", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A classification task that operates on an entire sequence of tokens, utilizing representations like the special [CLS] token.",
        "prerequisites": ["Token Representation"], "unlocks": [],
        "related_to": [{"concept": "Token-Level Classification", "relation": "contrasts_with"}],
        "tags": ["classification", "sequence-level"]
    }, {
        "concept_name": "Token-Level Classification", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A classification task that assigns labels to individual tokens within a sequence using contextual representations.",
        "prerequisites": ["Token Representation"], "unlocks": [],
        "related_to": [{"concept": "Sequence-Level Classification", "relation": "contrasts_with"}],
        "tags": ["classification", "token-level"]
    }],
    ("papers/Devlin2018_BERT.pdf", "chunk_055"): [{
        "concept_name": "GLUE Benchmark", "concept_type": "dataset", "difficulty": "foundational",
        "summary": "A collection of diverse natural language understanding tasks used to evaluate model performance.",
        "prerequisites": [], "unlocks": ["Model Evaluation"],
        "related_to": [{"concept": "SuperGLUE", "relation": "variant_of"}],
        "tags": ["benchmark", "nlu", "evaluation"]
    }, {
        "concept_name": "MNLI", "concept_type": "dataset", "difficulty": "intermediate",
        "summary": "Multi-Genre Natural Language Inference dataset for entailment classification.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "GLUE Benchmark", "relation": "part_of"}],
        "tags": ["nli", "entailment"]
    }, {
        "concept_name": "QQP", "concept_type": "dataset", "difficulty": "intermediate",
        "summary": "Quora Question Pairs dataset for semantic equivalence classification.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "GLUE Benchmark", "relation": "part_of"}],
        "tags": ["paraphrase", "semantic-similarity"]
    }, {
        "concept_name": "QNLI", "concept_type": "dataset", "difficulty": "intermediate",
        "summary": "Question Natural Language Inference dataset converted from SQuAD for binary classification.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "GLUE Benchmark", "relation": "part_of"}],
        "tags": ["qa", "nli"]
    }, {
        "concept_name": "SQuAD", "concept_type": "dataset", "difficulty": "intermediate",
        "summary": "Stanford Question Answering Dataset for reading comprehension.",
        "prerequisites": [], "unlocks": [],
        "related_to": [{"concept": "QNLI", "relation": "variant_of"}],
        "tags": ["qa", "reading-comprehension"]
    }],
    ("papers/Edge2024_GraphRAG.pdf", "chunk_014"): [{
        "concept_name": "Entity Extraction", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A process using language models to detect and capture meaningful entities, relationships, or claims from text to create knowledge graph elements.",
        "prerequisites": [], "unlocks": ["Knowledge Graph Construction"],
        "related_to": [{"concept": "Claim Extraction", "relation": "related_to"}],
        "tags": ["entity-extraction", "information-extraction"]
    }, {
        "concept_name": "Claim Extraction", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A technique for identifying and capturing explicit assertions or structured statements within source text.",
        "prerequisites": [], "unlocks": ["Knowledge Graph Construction"],
        "related_to": [{"concept": "Entity Extraction", "relation": "related_to"}],
        "tags": ["claim-extraction", "information-extraction"]
    }],
    ("papers/Edge2024_GraphRAG.pdf", "chunk_018"): [{
        "concept_name": "Entity Extraction", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A process using language models to detect and capture meaningful entities, relationships, or claims from text to create knowledge graph elements.",
        "prerequisites": [], "unlocks": ["Knowledge Graph Construction"],
        "related_to": [{"concept": "Claim Extraction", "relation": "related_to"}],
        "tags": ["entity-extraction", "information-extraction"]
    }, {
        "concept_name": "Claim Extraction", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A technique for identifying and capturing explicit assertions or structured statements within source text.",
        "prerequisites": [], "unlocks": ["Knowledge Graph Construction"],
        "related_to": [{"concept": "Entity Extraction", "relation": "related_to"}],
        "tags": ["claim-extraction", "information-extraction"]
    }],
    ("papers/Edge2024_GraphRAG.pdf", "chunk_034"): [{
        "concept_name": "Agglomerative Clustering", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A hierarchical clustering algorithm that merges clusters iteratively based on a linkage criterion.",
        "prerequisites": ["Clustering", "Distance Metric"], "unlocks": ["Cluster Hierarchy"],
        "related_to": [], "tags": ["clustering", "hierarchical"]
    }, {
        "concept_name": "Complete Linkage", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A linkage criterion where clusters are merged only if the maximum distance between their points is below a threshold.",
        "prerequisites": ["Agglomerative Clustering"], "unlocks": [],
        "related_to": [], "tags": ["linkage", "clustering"]
    }, {
        "concept_name": "ROUGE-L Distance", "concept_type": "metric", "difficulty": "intermediate",
        "summary": "A distance metric based on 1 minus ROUGE-L score, measuring summary overlap for clustering.",
        "prerequisites": ["ROUGE Metric", "Longest Common Subsequence"], "unlocks": [],
        "related_to": [], "tags": ["rouge", "distance-metric"]
    }],
    ("papers/Hu2021_LoRA.pdf", "chunk_021"): [],  # Empty - table
    ("papers/Hu2021_LoRA.pdf", "chunk_028"): [{
        "concept_name": "LoRA", "concept_type": "method", "difficulty": "advanced",
        "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
        "prerequisites": ["Pre-Trained Model", "Matrix Factorization"], "unlocks": ["Efficient Fine-Tuning", "Parameter-Efficient Adaptation"],
        "related_to": [{"concept": "Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["lora", "peft", "efficient-tuning"]
    }, {
        "concept_name": "Parameter Budget", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "A constraint on the number of trainable parameters to control computational cost during adaptation.",
        "prerequisites": ["Model Parameters", "Compute Constraints"], "unlocks": [],
        "related_to": [{"concept": "Rank", "relation": "uses"}],
        "tags": ["parameter-efficiency", "budget"]
    }, {
        "concept_name": "Rank", "concept_type": "technique", "difficulty": "intermediate",
        "summary": "The rank of the low-rank decomposition matrices in LoRA, controlling expressiveness vs parameter count.",
        "prerequisites": ["Matrix Factorization", "Linear Algebra"], "unlocks": [],
        "related_to": [{"concept": "Parameter Budget", "relation": "uses"}],
        "tags": ["rank", "low-rank"]
    }, {
        "concept_name": "Self-Attention Module", "concept_type": "method", "difficulty": "intermediate",
        "summary": "The attention mechanism in transformers where queries, keys, and values are projected and attended.",
        "prerequisites": ["Attention Mechanism", "Transformer"], "unlocks": [],
        "related_to": [{"concept": "Weight Matrix", "relation": "part_of"}],
        "tags": ["attention", "transformer"]
    }],
    ("papers/Hu2021_LoRA.pdf", "chunk_033"): [{
        "concept_name": "LoRA", "concept_type": "method", "difficulty": "advanced",
        "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
        "prerequisites": ["Pre-Trained Model", "Matrix Factorization"], "unlocks": ["Efficient Fine-Tuning", "Quick Task Switching"],
        "related_to": [{"concept": "Full Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["lora", "peft", "efficient-tuning"]
    }, {
        "concept_name": "Efficient Adaptation", "concept_type": "method", "difficulty": "advanced",
        "summary": "Methods that adapt pre-trained models with minimal trainable parameters and no inference latency.",
        "prerequisites": ["Pre-Trained Model"], "unlocks": ["Parameter-Efficient Fine-Tuning"],
        "related_to": [{"concept": "Full Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["efficient-adaptation", "peft"]
    }, {
        "concept_name": "Orthogonal Improvement", "concept_type": "method", "difficulty": "advanced",
        "summary": "Combining multiple efficient adaptation methods that improve different aspects for cumulative gains.",
        "prerequisites": ["LoRA", "Adapter"], "unlocks": [],
        "related_to": [{"concept": "LoRA", "relation": "extends"}],
        "tags": ["orthogonal", "ensemble"]
    }, {
        "concept_name": "Rank Deficiency", "concept_type": "theory", "difficulty": "advanced",
        "summary": "The observation that weight updates during fine-tuning have low intrinsic rank, motivating low-rank adaptation.",
        "prerequisites": ["Linear Algebra", "Matrix Rank"], "unlocks": ["LoRA"],
        "related_to": [], "tags": ["rank", "theory", "fine-tuning-analysis"]
    }],
    ("papers/Hu2021_LoRA.pdf", "chunk_034"): [{
        "concept_name": "LoRA", "concept_type": "method", "difficulty": "advanced",
        "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
        "prerequisites": ["Pre-Trained Model", "Matrix Factorization"], "unlocks": ["Efficient Fine-Tuning", "Quick Task Switching"],
        "related_to": [{"concept": "Full Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["lora", "peft", "efficient-tuning"]
    }, {
        "concept_name": "Efficient Adaptation", "concept_type": "method", "difficulty": "advanced",
        "summary": "Methods that adapt pre-trained models with minimal trainable parameters and no inference latency.",
        "prerequisites": ["Pre-Trained Model"], "unlocks": ["Parameter-Efficient Fine-Tuning"],
        "related_to": [{"concept": "Full Fine-Tuning", "relation": "contrasts_with"}],
        "tags": ["efficient-adaptation", "peft"]
    }, {
        "concept_name": "Orthogonal Improvement", "concept_type": "method", "difficulty": "advanced",
        "summary": "Combining multiple efficient adaptation methods that improve different aspects for cumulative gains.",
        "prerequisites": ["LoRA", "Adapter"], "unlocks": [],
        "related_to": [{"concept": "LoRA", "relation": "extends"}],
        "tags": ["orthogonal", "ensemble"]
    }, {
        "concept_name": "Rank Deficiency", "concept_type": "theory", "difficulty": "advanced",
        "summary": "The observation that weight updates during fine-tuning have low intrinsic rank, motivating low-rank adaptation.",
        "prerequisites": ["Linear Algebra", "Matrix Rank"], "unlocks": ["LoRA"],
        "related_to": [], "tags": ["rank", "theory", "fine-tuning-analysis"]
    }],
    ("papers/Lewis2020_RAG.pdf", "chunk_034"): [{
        "concept_name": "Retrieve-and-Edit", "concept_type": "method", "difficulty": "advanced",
        "summary": "An approach that retrieves similar training examples and edits them to produce output for a new input.",
        "prerequisites": ["Retrieval", "Sequence-to-Sequence Model"], "unlocks": [],
        "related_to": [{"concept": "Retrieval-Augmented Generation", "relation": "contrasts_with"}],
        "tags": ["retrieve-and-edit", "retrieval"]
    }],
    ("papers/Vaswani2017_Attention_Is_All_You_Need.pdf", "chunk_001"): [{
        "concept_name": "Attention Mechanism", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A mechanism allowing tokens to attend to other tokens in the sequence, computing weighted representations based on relevance.",
        "prerequisites": [], "unlocks": ["Transformer", "Multi-Head Attention"],
        "related_to": [], "tags": ["attention", "transformer"]
    }],
    ("papers/Vaswani2017_Attention_Is_All_You_Need.pdf", "chunk_004"): [{
        "concept_name": "Attention Mechanism", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A mechanism allowing tokens to attend to other tokens in the sequence, computing weighted representations based on relevance.",
        "prerequisites": [], "unlocks": ["Transformer", "Multi-Head Attention"],
        "related_to": [], "tags": ["attention", "transformer"]
    }],
    ("papers/Vaswani2017_Attention_Is_All_You_Need.pdf", "chunk_013"): [{
        "concept_name": "Fully Connected Feed-Forward Network", "concept_type": "method", "difficulty": "intermediate",
        "summary": "A position-wise feed-forward network applied to each token representation independently, consisting of two linear transformations with a ReLU activation.",
        "prerequisites": ["Linear Transformation"], "unlocks": [],
        "related_to": [], "tags": ["feed-forward", "transformer", "mlp"]
    }],
}

# ─── Helpers ──────────────────────────────────────────────────────
def extract_concepts_from_text(text: str) -> list:
    """Extract concepts from text using pattern matching."""
    found = []
    text_lower = text.lower()
    
    for pattern, concept in CONCEPT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            # Check if already found
            if not any(c["concept_name"] == concept["concept_name"] for c in found):
                found.append(concept.copy())
    return found

def clean_and_validate(concepts: list, text: str) -> list:
    """Clean prerequisites, cap at 5, remove dangling refs."""
    if not concepts:
        return []
    
    # Collect all concept names in this chunk
    chunk_names = {c["concept_name"] for c in concepts}
    
    cleaned = []
    for c in concepts:
        # Prune prerequisites to only those in this chunk
        c["prerequisites"] = [p for p in c.get("prerequisites", []) if p in chunk_names]
        # Prune unlocks
        c["unlocks"] = [u for u in c.get("unlocks", []) if u in chunk_names]
        # Prune related_to
        c["related_to"] = [r for r in c.get("related_to", []) if r.get("concept") in chunk_names]
        
        # Ensure all required fields
        for k in ["concept_name", "concept_type", "difficulty", "summary", "prerequisites", "unlocks", "related_to", "tags"]:
            if k not in c:
                c[k] = [] if k in ("prerequisites", "unlocks", "related_to", "tags") else ""
        
        cleaned.append(c)
    
    # Cap at MAX_CONCEPTS
    return cleaned[:MAX_CONCEPTS]

# ─── Main Generation ─────────────────────────────────────────────
def main():
    print("Ingesting documents...")
    chunks = ingest_folder(str(PDF_DIR))
    chunks = [c for c in chunks if c["doc_id"] in CORE_DOCS and c.get("chunk_kind") == "prose" and c["text"].strip()]
    
    print(f"Loaded {len(chunks)} prose chunks from core papers.")
    
    # Build training records
    records = []
    for chunk in chunks:
        key = (chunk["doc_id"], chunk["chunk_id"])
        
        if key in CHUNK_OVERRIDES:
            concepts = CHUNK_OVERRIDES[key]
        else:
            concepts = extract_concepts_from_text(chunk["text"])
        
        concepts = clean_and_validate(concepts, chunk["text"])
        
        # Create record
        instruction_template = "You are an OKF extraction engine for the Archipelago knowledge graph.\nFrom the TEXT below, extract 1-5 teachable CONCEPTS as a JSON array.\n\nEach object MUST have exactly these keys:\nconcept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags\n\nRules:\n- concept_name: ≤ 5 words, a reusable noun phrase, Title Case.\n- Only concepts actually explained in the text. No authors, citations, section titles.\n- prerequisites = what a learner needs FIRST; unlocks = what this ENABLES next.\n- A concept must NEVER appear in its own prerequisites or unlocks.\n- Keep names stable across documents so the same concept merges into one node.\n- If the text has no real teachable concept, return [].\n- Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures.\n- Output ONLY the JSON array. No prose, no markdown fences.\n\nTEXT:\n{chunk_text}\n\nReturn ONLY the JSON array, no other text:"
        
        record = {
            "instruction": instruction_template.format(chunk_text=chunk["text"]),
            "input": "",
            "output": json.dumps(concepts, ensure_ascii=False, indent=2),
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk.get("page_number", 0),
            "section_title": chunk.get("section_title", "")
        }
        records.append(record)
    
    print(f"Generated {len(records)} total records.")
    
    # Split train/test by chunk (no leakage)
    by_doc_chunk = defaultdict(list)
    for r in records:
        by_doc_chunk[(r["doc_id"], r["chunk_id"])].append(r)
    
    keys = list(by_doc_chunk.keys())
    import random
    random.seed(42)
    random.shuffle(keys)
    
    n_test = max(1, int(len(keys) * 0.15))
    test_keys = set(keys[:n_test])
    
    train_records = []
    test_records = []
    for k, v in by_doc_chunk.items():
        if k in test_keys:
            test_records.extend(v)
        else:
            train_records.extend(v)
    
    # Ensure targets
    if len(train_records) < TARGET_TRAIN:
        print(f"Warning: Only {len(train_records)} train records (target {TARGET_TRAIN})")
    if len(test_records) < TARGET_TEST:
        print(f"Warning: Only {len(test_records)} test records (target {TARGET_TEST})")
    
    # Write JSONL
    for name, recs in [("okf_train_pairs.jsonl", train_records), ("okf_test_pairs.jsonl", test_records)]:
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(recs)} records to {name}")
    
    # Also write combined
    with open(OUT_DIR / "okf_training_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to okf_training_pairs.jsonl")

    # Stats
    all_concepts = []
    for r in records:
        out = json.loads(r["output"])
        if out != []:
            for c in out:
                all_concepts.append(c["concept_name"])
    
    print(f"\nTotal concept targets: {len(all_concepts)} ({len(set(all_concepts))} unique)")
    print(f"Train: {len(train_records)} records, Test: {len(test_records)} records")

if __name__ == "__main__":
    main()
