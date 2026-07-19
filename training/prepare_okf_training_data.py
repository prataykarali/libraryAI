#!/usr/bin/env python3
"""Build a 500-700 case fine-tuning dataset from the current OKF pipeline outputs.

Treats the existing extraction as a noisy starting point, then applies a strict
auto-cleaner so the training targets approach "utter perfect" OKF v1.6 JSON:
  - valid enums only
  - no self-loops
  - canonical names / no junk authors/chairs/underlined metrics
  - page-level provenance is attached as *metadata*, not emitted by the model
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

from pdf_ingestion import ingest_folder, _numeric_token_fraction
from okf_pipeline import VALID_TYPES, VALID_DIFFICULTIES, VALID_RELATIONS

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
RESULTS_FILE = BASE_DIR / "okf_results.json"
OUT_DIR = BASE_DIR / "training_data"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Lightweight canonical aliases for the current seed corpus. The runtime
# pipeline stays domain-agnostic; these aliases only shape this generated
# fine-tuning dataset.
# ---------------------------------------------------------------------------
_ACRONYM_ALIASES = {
    "bert": "BERT",
    "bertbase": "BERT",
    "bert base": "BERT",
    "lora": "LoRA",
    "low rank adaptation": "Low-Rank Adaptation",
    "rag": "RAG",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "graph rag": "GraphRAG",
    "graphrag": "GraphRAG",
    "next sentence prediction nsp": "Next Sentence Prediction",
    "masked language modeling lm": "Masked Language Modeling",
    "masked language modeling mlm": "Masked Language Modeling",
    "bert bert model": "BERT",
    "cls token": "CLS Token",
    "e cls token": "CLS Token",
    "su superglue": "SuperGLUE",
    "gpt": "GPT",
    "openai gpt": "GPT",
    "generative pre trained transformer": "GPT",
    "gpt generative pre trained transformer": "GPT",
    "elmo": "ELMo",
    "elmo model": "ELMo",
    "elmo embedding based model": "ELMo",
    "embedding based model": "ELMo",
    "masked lm": "Masked Language Modeling",
    "bidirectional model": "Bidirectional Language Model",
    "bidirectional language model": "Bidirectional Language Model",
}

_ALIAS_TO_SURFACES = defaultdict(set)
for _surface_key, _canonical in _ACRONYM_ALIASES.items():
    _ALIAS_TO_SURFACES[_canonical].add(_surface_key)


def coerce_concept_name(value) -> str:
    """Coerce a possibly list-valued legacy concept_name to a string.

    Mirrors okf_pipeline.normalize_okf_item: some legacy records store
    concept_name as a list, so keep the first non-empty string element.
    Normal string values pass through unchanged.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        names = [n.strip() for n in value if isinstance(n, str) and n.strip()]
        return names[0] if names else ""
    return ""


def normalize_training_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", " ", name.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    if key in _ACRONYM_ALIASES:
        return _ACRONYM_ALIASES[key]
    # Preserve existing casing but ensure title case for display consistency.
    if name == name.lower() or name == name.upper():
        return name.title()
    return name


# ---------------------------------------------------------------------------
# Heuristic non-concept blocker
# ---------------------------------------------------------------------------
_JUNK_RE = re.compile(
    r"(?i)\b(authors?|contributors?|chairs?|funding|acknowledg|best model without|"
    r"underlined|thank|grants?|projects?|universit|institute|mila quebec|"
    r"nserc|canada cifar|research chair|table\s+\d+|caption|phd program|"
    r"fellowship|scholarship|discovery grant|ai chairs?|computational resources provided|"
    r"ablation\s*(experiment|results?|studies|study|metrics?)|model\s*training\s*regime|training\s*regime|model\s*regime|"
    r"task-level\s*tasks?|task\s*level\s*tasks?|implementation\s*details|pipeline\s*design\s*parameters|"
    r"pipeline\s*design|design\s*parameters|feed-forward/filter\s*size|filter\s*size|parameter\s*settings|"
    r"experimental\s*setup|hardware\s*configuration|training\s*settings|model\s*parameters|architecture\s*parameters|"
    # Table row labels / ablation conditions that are NOT standalone concepts
    r"\bno\s+nsp\b|\bltr\s*&\s*no\s+nsp\b|\bltr\s+no\s+nsp\b|\+\s*bilstm\b|"
    r"bertlarge|bertbase|bert\s+base|bert\s+large|"
    r"\bdev\s+set\b|\btest\s+set\b|\bhyperparameters?\b|"
    r"\bconll\b|\bsquad\b|\bmnli\b|\bqnli\b|\bmrpc\b|\bsst\b|"
    r"weight\s+type|rank\s+r|parameter\s+budget|trainable\s+parameters)\b"
)
_NUMERIC_CONCEPT_RE = re.compile(r"^\d+[\d\s%\.x\-/]*$")
_FORMULA_OR_VALUE_RE = re.compile(
    r"(?i)([{}=∑∆Δ]|\.{2,}|"
    r"\bvs\.?\b|\bcomparison\b|\b\d+%|\b\d+\s*of\s+tokens\b|"
    r"\breplacement\s+token\s*\(\d|"
    r"\b(system|hyperparameter)\s+dev\b|"
    r"\b(dev|test)\s+(f1|acc|accuracy|score)\b|"
    r"^test\s+scores?$|"
    r"\bstate-of-the-art\b.*\b(bleu|f1|accuracy|score)\b|"
    r"\bbatch\b|\bwithout\s+gold\s+access\b)"
)

_META_RE = re.compile(
    r"\b(this (section|paper|chapter|table|figure|work|study|subsection)"
    r"|the (paper|authors?|section|following|table|figure|appendix)"
    r"|(this|the) (section|paper|chapter) (describes|presents|introduces|discusses|covers)"
    r"|section (describes|presents|introduces)"
    r"|is described in|are described in|described in (this|the) (section|paper|table))\b",
    re.I,
)

# Table-row-like content detection: if a chunk has many short lines with numbers, it's likely a table
_TABLE_ROW_RE = re.compile(r"(?m)^\s*\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d")
CORE_DOCS = {
    "papers/Devlin2018_BERT.pdf",
    "papers/Edge2024_GraphRAG.pdf",
    "papers/Hu2021_LoRA.pdf",
    "papers/Lewis2020_RAG.pdf",
    "papers/Vaswani2017_Attention_Is_All_You_Need.pdf",
    "probable.pdf"
}

def is_table_like(text: str) -> bool:
    """Detect if text is primarily a table/results block (not prose)."""
    if not text or len(text.strip()) < 50:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    # Count lines that look like table rows (multiple numbers separated by spaces)
    table_like = sum(1 for ln in lines if _TABLE_ROW_RE.search(ln))
    # Also check numeric token fraction for the whole text
    numeric_frac = _numeric_token_fraction(text)
    return table_like >= 2 or numeric_frac > 0.35


def is_concept_grounded(text: str, name: str) -> bool:
    if not text or not name:
        return False
    
    # 1. Clean the concept name (remove parentheticals)
    core = re.sub(r"\s*\([^)]*\)", "", name).strip()
    surfaces = {core.lower(), name.lower()}
    
    # 2. Add aliases from _ACRONYM_ALIASES
    for surface_key, canonical in _ACRONYM_ALIASES.items():
        if canonical.lower() == name.lower() or name.lower() == surface_key:
            surfaces.add(surface_key.lower())
            surfaces.add(canonical.lower())
            
    # 3. Add custom aliases
    if name in _ALIAS_TO_SURFACES:
        for s in _ALIAS_TO_SURFACES[name]:
            surfaces.add(s.lower())
            
    # 4. Check for direct substring matches
    text_lower = text.lower()
    for surface in surfaces:
        if not surface:
            continue
        if surface in text_lower:
            return True
            
    # 5. Check if the first two words (if they are long enough) appear close together
    words = [w for w in re.findall(r"[A-Za-z0-9]+", core) if len(w) > 2]
    if len(words) > 1:
        if words[0].lower() in text_lower and words[1].lower() in text_lower:
            pat = re.compile(re.escape(words[0]) + r".{0,100}?" + re.escape(words[1]), re.IGNORECASE | re.DOTALL)
            if pat.search(text):
                return True
    elif len(words) == 1:
        if words[0].lower() in text_lower:
            return True
            
    return False


# Misclassification type overrides - expanded to fix BERT/GraphRAG/LoRA type confusion
CONCEPT_TYPE_OVERRIDES = {
    "bert": "method",
    "lora": "method",
    "rag": "method",
    "graphrag": "method",
    "transformer": "method",
    "attention mechanism": "method",
    "multi-head attention": "technique",
    "self-attention": "technique",
    "elmo": "method",
    "gpt": "method",
    "bidirectional language model": "method",
    "masked language modeling": "technique",
    "next sentence prediction": "technique",
    "wordpiece": "technique",
    "cls token": "technique",
    "transformer encoder": "method",
    "fine-tuning": "technique",
    "feature-based approach": "method",
    "contextual embeddings": "technique",
    "named entity recognition": "task",
    "glue benchmark": "dataset",
    "conll-2003": "dataset",
    "squad": "dataset",
    "mnli": "dataset",
    "qnli": "dataset",
    "mrpc": "dataset",
    "sst-2": "dataset",
    "superglue": "dataset",
}

# Explicit chunk-level ground truth overrides (resolves noise / bad parsing)
CHUNK_OVERRIDES = {
    ("papers/Devlin2018_BERT.pdf", "chunk_054"): [
        {
            "concept_name": "Sequence-Level Classification",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A classification task that operates on an entire sequence of tokens, utilizing representations like the special [CLS] token.",
            "prerequisites": ["Token Representation"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Token-Level Classification",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["classification", "sequence-level"]
        },
        {
            "concept_name": "Token-Level Classification",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A classification task that assigns labels to individual tokens within a sequence using contextual representations.",
            "prerequisites": ["Token Representation"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Sequence-Level Classification",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["classification", "token-level"]
        }
    ],
    # chunk_001: BERT Abstract - introduces BERT concept
    ("papers/Devlin2018_BERT.pdf", "chunk_001"): [
        {
            "concept_name": "BERT",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "Bidirectional Encoder Representations from Transformers — a language representation model pre-trained on unlabeled text using masked language modeling and next sentence prediction.",
            "prerequisites": ["Transformer Encoder"],
            "unlocks": ["Fine-Tuning Approach", "Feature-Based Approach"],
            "related_to": [
                {
                    "concept": "GPT",
                    "relation": "contrasts_with"
                },
                {
                    "concept": "ELMo",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["bert", "pre-trained-model", "transformer"]
        }
    ],
    # chunk_002: Introduction - Feature-based vs Fine-tuning strategies
    ("papers/Devlin2018_BERT.pdf", "chunk_002"): [
        {
            "concept_name": "Feature-Based Strategy",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "An approach that uses pre-trained representations as fixed additional features within task-specific architectures, without fine-tuning the pre-trained parameters.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Fine-Tuning Strategy",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["transfer-learning", "feature-extraction"]
        },
        {
            "concept_name": "Fine-Tuning Strategy",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "An approach that adds minimal task-specific parameters and trains all model parameters end-to-end on a downstream task.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Feature-Based Strategy",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["transfer-learning", "fine-tuning"]
        }
    ],
    # chunk_003: Introduction - Masked Language Modeling
    ("papers/Devlin2018_BERT.pdf", "chunk_003"): [
        {
            "concept_name": "Masked Language Modeling",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A pre-training objective that randomly masks input tokens and trains the model to predict their original vocabulary IDs using bidirectional context.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": ["Bidirectional Representation Learning"],
            "related_to": [],
            "tags": ["masked-lm", "pre-training", "nlp"]
        },
        {
            "concept_name": "Unidirectional Constraints",
            "concept_type": "theory",
            "difficulty": "intermediate",
            "summary": "A limitation in standard language models where token attention is restricted to left-to-right context during pre-training.",
            "prerequisites": [],
            "unlocks": ["Masked Language Modeling"],
            "related_to": [],
            "tags": ["language-modeling", "attention"]
        }
    ],
    # chunk_004: Introduction - Next Sentence Prediction
    ("papers/Devlin2018_BERT.pdf", "chunk_004"): [
        {
            "concept_name": "Next Sentence Prediction",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A pre-training task where the model predicts whether two sentences are consecutive in the original text, capturing inter-sentence relationships.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Masked Language Modeling",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["pre-training", "nsp", "sentence-relationships"]
        }
    ],
    # chunk_005: Section 2.1 - Pre-trained word/sentence/paragraph embeddings
    ("papers/Devlin2018_BERT.pdf", "chunk_005"): [
        {
            "concept_name": "Pre-Trained Word Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Vector representations of words learned from large corpora and used to initialize NLP models, offering significant improvements over random initialization.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": ["Sentence Embeddings", "Paragraph Embeddings"],
            "related_to": [],
            "tags": ["word-embeddings", "pre-training", "nlp"]
        },
        {
            "concept_name": "Sentence Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Fixed-length vector representations of entire sentences, trained using objectives like next-sentence ranking or auto-encoding.",
            "prerequisites": ["Pre-Trained Word Embeddings"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Paragraph Embeddings",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["sentence-embeddings", "pre-training"]
        },
        {
            "concept_name": "Paragraph Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Vector representations of paragraphs or documents, such as Paragraph Vector (doc2vec), extending word embedding methods to coarser granularities.",
            "prerequisites": ["Pre-Trained Word Embeddings"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Sentence Embeddings",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["paragraph-embeddings", "doc2vec", "pre-training"]
        },
        {
            "concept_name": "Language Modeling Objective",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A training objective where the model predicts the next token or discriminates correct words from context, used to pre-train embeddings.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": ["Pre-Trained Word Embeddings"],
            "related_to": [
                {
                    "concept": "Next-Sentence Ranking",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["language-modeling", "pre-training"]
        }
    ],
    # chunk_006: Section 2.1 - ELMo and Contextual Embeddings
    ("papers/Devlin2018_BERT.pdf", "chunk_006"): [
        {
            "concept_name": "ELMo",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "Embeddings from Language Models — a feature-based approach using bidirectional LSTM representations as context-sensitive features for downstream tasks.",
            "prerequisites": ["BiLSTM", "Language Model Pre-training"],
            "unlocks": ["Contextual Word Embeddings"],
            "related_to": [
                {
                    "concept": "BERT",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["elmo", "feature-based", "bilstm", "contextual-embeddings"]
        },
        {
            "concept_name": "Contextual Word Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Token representations that vary based on surrounding context, produced by models like ELMo using concatenated left-to-right and right-to-left LSTM states.",
            "prerequisites": ["BiLSTM", "Pre-Trained Word Embeddings"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Pre-Trained Word Embeddings",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["contextual-embeddings", "elmo"]
        },
        {
            "concept_name": "Cloze Task",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A fill-in-the-blank task where a single word is predicted from both left and right context, used to learn contextual representations.",
            "prerequisites": ["Language Model Pre-training"],
            "unlocks": ["Masked Language Modeling"],
            "related_to": [],
            "tags": ["cloze-task", "pre-training"]
        }
    ],
    # chunk_012: WordPiece, [CLS], [SEP], Segment/Position Embeddings
    ("papers/Devlin2018_BERT.pdf", "chunk_012"): [
        {
            "concept_name": "WordPiece Tokenization",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A subword tokenization algorithm that splits words into frequent subword units, balancing vocabulary size and sequence length.",
            "prerequisites": ["Tokenization"],
            "unlocks": ["BERT Input Representation"],
            "related_to": [],
            "tags": ["wordpiece", "tokenization", "subword"]
        },
        {
            "concept_name": "CLS Token",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A special classification token ([CLS]) prepended to every input sequence, whose final hidden state serves as the aggregate sequence representation for classification tasks.",
            "prerequisites": ["Transformer Encoder"],
            "unlocks": ["Sequence-Level Classification"],
            "related_to": [
                {
                    "concept": "SEP Token",
                    "relation": "related_to"
                }
            ],
            "tags": ["cls-token", "classification", "bert"]
        },
        {
            "concept_name": "SEP Token",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A special separator token ([SEP]) used to distinguish sentence pairs in a single input sequence.",
            "prerequisites": ["Transformer Encoder"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "CLS Token",
                    "relation": "related_to"
                }
            ],
            "tags": ["sep-token", "bert", "sentence-pairs"]
        },
        {
            "concept_name": "Segment Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Learned embeddings added to each token indicating whether it belongs to sentence A or sentence B, enabling the model to distinguish between sentence pairs.",
            "prerequisites": ["Position Embeddings"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Position Embeddings",
                    "relation": "related_to"
                }
            ],
            "tags": ["segment-embeddings", "sentence-pairs", "bert"]
        },
        {
            "concept_name": "BERT Input Representation",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "The construction of input representations by summing token, segment, and position embeddings for each token in the sequence.",
            "prerequisites": ["Segment Embeddings", "Position Embeddings"],
            "unlocks": [],
            "related_to": [],
            "tags": ["bert", "input-representation", "embeddings"]
        }
    ],
    # chunk_013: BERT Pre-training Tasks - Masked LM and Bidirectional Models
    ("papers/Devlin2018_BERT.pdf", "chunk_013"): [
        {
            "concept_name": "Masked Language Modeling",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A pre-training objective where random tokens are masked and the model predicts them using bidirectional context, enabling deep bidirectional representations.",
            "prerequisites": ["Language Model Pre-training", "Transformer Encoder"],
            "unlocks": ["Bidirectional Representation Learning"],
            "related_to": [
                {
                    "concept": "Left-to-Right Language Model",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["mlm", "masked-lm", "pre-training", "bert"]
        },
        {
            "concept_name": "Bidirectional Representation Learning",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Learning representations that jointly condition on both left and right context, as opposed to unidirectional left-to-right or right-to-left models.",
            "prerequisites": ["Transformer Encoder", "Masked Language Modeling"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Left-to-Right Language Model",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["bidirectional", "representation-learning", "bert"]
        },
        {
            "concept_name": "Transformer Encoder",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "The bidirectional Transformer architecture that uses full self-attention over the entire sequence, as opposed to the masked self-attention in a Transformer decoder.",
            "prerequisites": ["Attention Mechanism", "Transformer"],
            "unlocks": ["BERT", "Masked Language Modeling"],
            "related_to": [
                {
                    "concept": "Transformer Decoder",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["transformer-encoder", "attention", "architecture"]
        },
        {
            "concept_name": "Transformer Decoder",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "The left-context-only Transformer architecture using masked self-attention, suitable for text generation.",
            "prerequisites": ["Attention Mechanism", "Transformer"],
            "unlocks": ["Left-to-Right Language Model"],
            "related_to": [
                {
                    "concept": "Transformer Encoder",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["transformer-decoder", "generation", "architecture"]
        }
    ],
    # chunk_014: Masked LM Details - 15% masking, [MASK] token, random/unchanged tokens
    ("papers/Devlin2018_BERT.pdf", "chunk_014"): [
        {
            "concept_name": "Masked Language Modeling",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A pre-training procedure masking 15% of tokens and predicting them, using [MASK] 80%, random token 10%, and unchanged token 10% of the time.",
            "prerequisites": ["Language Model Pre-training", "Transformer Encoder"],
            "unlocks": ["Bidirectional Representation Learning"],
            "related_to": [
                {
                    "concept": "Cloze Task",
                    "relation": "variant_of"
                }
            ],
            "tags": ["mlm", "masking", "pre-training", "bert"]
        },
        {
            "concept_name": "Mask Token",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A special [MASK] token used to replace masked positions during pre-training, with a mitigation strategy to reduce pre-train/fine-tune mismatch.",
            "prerequisites": ["Tokenization", "Masked Language Modeling"],
            "unlocks": [],
            "related_to": [],
            "tags": ["mask-token", "bert", "pre-training"]
        },
        {
            "concept_name": "Pre-Train Fine-Tune Mismatch",
            "concept_type": "theory",
            "difficulty": "advanced",
            "summary": "The discrepancy between pre-training (where [MASK] tokens appear) and fine-tuning (where they don't), mitigated by not always replacing with [MASK].",
            "prerequisites": ["Masked Language Modeling", "Fine-Tuning"],
            "unlocks": [],
            "related_to": [],
            "tags": ["domain-shift", "pre-training", "fine-tuning"]
        }
    ],
    # chunk_028: BERT Ablation Study (Table 5) - pre-training tasks ablation
    ("papers/Devlin2018_BERT.pdf", "chunk_028"): [
        {
            "concept_name": "Ablation Study",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "An experimental method that systematically removes components of a model to measure their individual contribution to performance.",
            "prerequisites": ["Controlled Experiment"],
            "unlocks": ["Component Analysis"],
            "related_to": [],
            "tags": ["ablation", "experimental-design"]
        },
        {
            "concept_name": "Next Sentence Prediction",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A pre-training task where the model predicts whether two sentences are consecutive in the original text.",
            "prerequisites": ["Language Modeling"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Masked Language Modeling",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["pre-training", "nsp"]
        },
        {
            "concept_name": "Masked Language Modeling",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A pre-training objective where random tokens are masked and the model predicts them from context.",
            "prerequisites": ["Language Modeling"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Next Sentence Prediction",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["pre-training", "mlm"]
        },
        {
            "concept_name": "Left-to-Right Language Model",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A language modeling approach that predicts tokens sequentially from left to right, as used in GPT.",
            "prerequisites": ["Language Modeling"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Masked Language Modeling",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["language-modeling", "ltr"]
        },
        {
            "concept_name": "BiLSTM",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "A bidirectional LSTM that processes sequences in both forward and backward directions to capture contextual information.",
            "prerequisites": ["Recurrent Neural Network", "LSTM"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Transformer Encoder",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["bilstm", "recurrent"]
        }
    ],
    # chunk_034: BERT NER Results (Table 7) - Fine-tuning vs Feature-based
    ("papers/Devlin2018_BERT.pdf", "chunk_034"): [
        {
            "concept_name": "Fine-Tuning Approach",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
            "prerequisites": ["Pre-Trained Model"],
            "unlocks": ["Task-Specific Model"],
            "related_to": [
                {
                    "concept": "Feature-Based Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["fine-tuning", "transfer-learning"]
        },
        {
            "concept_name": "Feature-Based Approach",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A transfer learning approach where fixed pre-trained representations are extracted and used as features for a downstream model.",
            "prerequisites": ["Pre-Trained Model", "Contextual Embeddings"],
            "unlocks": ["Downstream Classifier"],
            "related_to": [
                {
                    "concept": "Fine-Tuning Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["feature-extraction", "transfer-learning"]
        },
        {
            "concept_name": "Named Entity Recognition",
            "concept_type": "task",
            "difficulty": "intermediate",
            "summary": "A sequence labeling task that identifies and classifies named entities in text into predefined categories.",
            "prerequisites": ["Token Representation", "Sequence Labeling"],
            "unlocks": [],
            "related_to": [],
            "tags": ["ner", "sequence-labeling", "information-extraction"]
        },
        {
            "concept_name": "Contextual Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Token representations extracted from specific layers of a pre-trained model that capture context-dependent meaning.",
            "prerequisites": ["Transformer Encoder", "Pre-Trained Model"],
            "unlocks": ["Feature-Based Approach"],
            "related_to": [
                {
                    "concept": "Fine-Tuning Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["embeddings", "contextual"]
        }
    ],
    # chunk_033: BERT Feature-based Approach (Section 5.3)
    ("papers/Devlin2018_BERT.pdf", "chunk_033"): [
        {
            "concept_name": "BERT",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A bidirectional Transformer encoder pre-trained with masked language modeling and next sentence prediction.",
            "prerequisites": ["Transformer Encoder", "Masked Language Modeling", "Next Sentence Prediction"],
            "unlocks": ["Fine-Tuning Approach", "Feature-Based Approach"],
            "related_to": [
                {
                    "concept": "GPT",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["bert", "pre-trained-model", "transformer"]
        },
        {
            "concept_name": "Fine-Tuning Approach",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A transfer learning approach where all pre-trained model parameters are updated on a downstream task.",
            "prerequisites": ["Pre-Trained Model"],
            "unlocks": ["Task-Specific Model"],
            "related_to": [
                {
                    "concept": "Feature-Based Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["fine-tuning", "transfer-learning"]
        },
        {
            "concept_name": "Feature-Based Approach",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A transfer learning approach where fixed pre-trained representations are extracted and used as features for a downstream model.",
            "prerequisites": ["Pre-Trained Model", "Contextual Embeddings"],
            "unlocks": ["Downstream Classifier"],
            "related_to": [
                {
                    "concept": "Fine-Tuning Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["feature-extraction", "transfer-learning"]
        },
        {
            "concept_name": "Named Entity Recognition",
            "concept_type": "task",
            "difficulty": "intermediate",
            "summary": "A sequence labeling task that identifies and classifies named entities in text into predefined categories.",
            "prerequisites": ["Token Representation", "Sequence Labeling"],
            "unlocks": [],
            "related_to": [],
            "tags": ["ner", "sequence-labeling", "information-extraction"]
        },
        {
            "concept_name": "Contextual Embeddings",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "Token representations extracted from specific layers of a pre-trained model that capture context-dependent meaning.",
            "prerequisites": ["Transformer Encoder", "Pre-Trained Model"],
            "unlocks": ["Feature-Based Approach"],
            "related_to": [
                {
                    "concept": "Fine-Tuning Approach",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["embeddings", "contextual"]
        }
    ],
    # chunk_055: GLUE Benchmark Descriptions
    ("papers/Devlin2018_BERT.pdf", "chunk_055"): [
        {
            "concept_name": "GLUE Benchmark",
            "concept_type": "dataset",
            "difficulty": "foundational",
            "summary": "A collection of diverse natural language understanding tasks used to evaluate model performance.",
            "prerequisites": [],
            "unlocks": ["Model Evaluation"],
            "related_to": [
                {
                    "concept": "SuperGLUE",
                    "relation": "variant_of"
                }
            ],
            "tags": ["benchmark", "nlu", "evaluation"]
        },
        {
            "concept_name": "MNLI",
            "concept_type": "dataset",
            "difficulty": "intermediate",
            "summary": "Multi-Genre Natural Language Inference dataset for entailment classification.",
            "prerequisites": [],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "GLUE Benchmark",
                    "relation": "part_of"
                }
            ],
            "tags": ["nli", "entailment"]
        },
        {
            "concept_name": "QQP",
            "concept_type": "dataset",
            "difficulty": "intermediate",
            "summary": "Quora Question Pairs dataset for semantic equivalence classification.",
            "prerequisites": [],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "GLUE Benchmark",
                    "relation": "part_of"
                }
            ],
            "tags": ["paraphrase", "semantic-similarity"]
        },
        {
            "concept_name": "QNLI",
            "concept_type": "dataset",
            "difficulty": "intermediate",
            "summary": "Question Natural Language Inference dataset converted from SQuAD for binary classification.",
            "prerequisites": [],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "GLUE Benchmark",
                    "relation": "part_of"
                }
            ],
            "tags": ["qa", "nli"]
        },
        {
            "concept_name": "SQuAD",
            "concept_type": "dataset",
            "difficulty": "intermediate",
            "summary": "Stanford Question Answering Dataset for reading comprehension.",
            "prerequisites": [],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "QNLI",
                    "relation": "variant_of"
                }
            ],
            "tags": ["qa", "reading-comprehension"]
        }
    ],
    # LoRA paper - chunk_028: Parameter budget / weight types table
    ("papers/Hu2021_LoRA.pdf", "chunk_028"): [
        {
            "concept_name": "LoRA",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
            "prerequisites": ["Pre-Trained Model", "Matrix Factorization"],
            "unlocks": ["Efficient Fine-Tuning", "Parameter-Efficient Adaptation"],
            "related_to": [
                {
                    "concept": "Fine-Tuning",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["lora", "peft", "efficient-tuning"]
        },
        {
            "concept_name": "Parameter Budget",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A constraint on the number of trainable parameters to control computational cost during adaptation.",
            "prerequisites": ["Model Parameters", "Compute Constraints"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Rank",
                    "relation": "uses"
                }
            ],
            "tags": ["parameter-efficiency", "budget"]
        },
        {
            "concept_name": "Rank",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "The rank of the low-rank decomposition matrices in LoRA, controlling expressiveness vs parameter count.",
            "prerequisites": ["Matrix Factorization", "Linear Algebra"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Parameter Budget",
                    "relation": "uses"
                }
            ],
            "tags": ["rank", "low-rank"]
        },
        {
            "concept_name": "Self-Attention Module",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "The attention mechanism in transformers where queries, keys, and values are projected and attended.",
            "prerequisites": ["Attention Mechanism", "Transformer"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Weight Matrix",
                    "relation": "part_of"
                }
            ],
            "tags": ["attention", "transformer"]
        }
    ],
    # LoRA paper - chunk_034: Conclusion / Future Work
    ("papers/Hu2021_LoRA.pdf", "chunk_034"): [
        {
            "concept_name": "LoRA",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
            "prerequisites": ["Pre-Trained Model", "Matrix Factorization"],
            "unlocks": ["Efficient Fine-Tuning", "Quick Task Switching"],
            "related_to": [
                {
                    "concept": "Full Fine-Tuning",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["lora", "peft", "efficient-tuning"]
        },
        {
            "concept_name": "Efficient Adaptation",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Methods that adapt pre-trained models with minimal trainable parameters and no inference latency.",
            "prerequisites": ["Pre-Trained Model"],
            "unlocks": ["Parameter-Efficient Fine-Tuning"],
            "related_to": [
                {
                    "concept": "Full Fine-Tuning",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["efficient-adaptation", "peft"]
        },
        {
            "concept_name": "Orthogonal Improvement",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Combining multiple efficient adaptation methods that improve different aspects for cumulative gains.",
            "prerequisites": ["LoRA", "Adapter"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "LoRA",
                    "relation": "extends"
                }
            ],
            "tags": ["orthogonal", "ensemble"]
        },
        {
            "concept_name": "Rank Deficiency",
            "concept_type": "theory",
            "difficulty": "advanced",
            "summary": "The observation that weight updates during fine-tuning have low intrinsic rank, motivating low-rank adaptation.",
            "prerequisites": ["Linear Algebra", "Matrix Rank"],
            "unlocks": ["LoRA"],
            "related_to": [],
            "tags": ["rank", "theory", "fine-tuning-analysis"]
        }
    ],
    # LoRA paper - chunk_033: Conclusion / Future Work (contains Table 7 math artifacts)
    ("papers/Hu2021_LoRA.pdf", "chunk_033"): [
        {
            "concept_name": "LoRA",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Low-Rank Adaptation freezes pre-trained weights and injects trainable rank-decomposition matrices into transformer layers.",
            "prerequisites": ["Pre-Trained Model", "Matrix Factorization"],
            "unlocks": ["Efficient Fine-Tuning", "Quick Task Switching"],
            "related_to": [
                {
                    "concept": "Full Fine-Tuning",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["lora", "peft", "efficient-tuning"]
        },
        {
            "concept_name": "Efficient Adaptation",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Methods that adapt pre-trained models with minimal trainable parameters and no inference latency.",
            "prerequisites": ["Pre-Trained Model"],
            "unlocks": ["Parameter-Efficient Fine-Tuning"],
            "related_to": [
                {
                    "concept": "Full Fine-Tuning",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["efficient-adaptation", "peft"]
        },
        {
            "concept_name": "Orthogonal Improvement",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "Combining multiple efficient adaptation methods that improve different aspects for cumulative gains.",
            "prerequisites": ["LoRA", "Adapter"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "LoRA",
                    "relation": "extends"
                }
            ],
            "tags": ["orthogonal", "ensemble"]
        },
        {
            "concept_name": "Rank Deficiency",
            "concept_type": "theory",
            "difficulty": "advanced",
            "summary": "The observation that weight updates during fine-tuning have low intrinsic rank, motivating low-rank adaptation.",
            "prerequisites": ["Linear Algebra", "Matrix Rank"],
            "unlocks": ["LoRA"],
            "related_to": [],
            "tags": ["rank", "theory", "fine-tuning-analysis"]
        }
    ],
    # LoRA paper - chunk_055: Appendix E (LoRA + Prefix Tuning table)
    ("papers/Hu2021_LoRA.pdf", "chunk_055"): [],
    # LoRA paper - chunk_021: Evaluation on DeBERTa/GPT-2 (model comparison table)
    ("papers/Hu2021_LoRA.pdf", "chunk_021"): [],
    # GraphRAG paper - chunk_034: Clustering methodology (Diversity evaluation)
    ("papers/Edge2024_GraphRAG.pdf", "chunk_034"): [
        {
            "concept_name": "Agglomerative Clustering",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A hierarchical clustering algorithm that merges clusters iteratively based on a linkage criterion.",
            "prerequisites": ["Clustering", "Distance Metric"],
            "unlocks": ["Cluster Hierarchy"],
            "related_to": [],
            "tags": ["clustering", "hierarchical"]
        },
        {
            "concept_name": "Complete Linkage",
            "concept_type": "technique",
            "difficulty": "intermediate",
            "summary": "A linkage criterion where clusters are merged only if the maximum distance between their points is below a threshold.",
            "prerequisites": ["Agglomerative Clustering"],
            "unlocks": [],
            "related_to": [],
            "tags": ["linkage", "clustering"]
        },
        {
            "concept_name": "ROUGE-L Distance",
            "concept_type": "metric",
            "difficulty": "intermediate",
            "summary": "A distance metric based on 1 minus ROUGE-L score, measuring summary overlap for clustering.",
            "prerequisites": ["ROUGE Metric", "Longest Common Subsequence"],
            "unlocks": [],
            "related_to": [],
            "tags": ["rouge", "distance-metric"]
        }
    ],
    # GraphRAG paper - chunk_055: References
    ("papers/Edge2024_GraphRAG.pdf", "chunk_055"): [],
    # RAG paper - chunk_034: Retrieve-and-Edit (Related Work)
    ("papers/Lewis2020_RAG.pdf", "chunk_034"): [
        {
            "concept_name": "Retrieve-and-Edit",
            "concept_type": "method",
            "difficulty": "advanced",
            "summary": "An approach that retrieves similar training examples and edits them to produce output for a new input.",
            "prerequisites": ["Retrieval", "Sequence-to-Sequence Model"],
            "unlocks": [],
            "related_to": [
                {
                    "concept": "Retrieval-Augmented Generation",
                    "relation": "contrasts_with"
                }
            ],
            "tags": ["retrieve-and-edit", "retrieval"]
        }
    ],
    ("papers/Edge2024_GraphRAG.pdf", "chunk_018"): [
        {
            "concept_name": "Entity Extraction",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A process using language models to detect and capture meaningful entities, relationships, or claims from text to create knowledge graph elements.",
            "prerequisites": [],
            "unlocks": [
                "Knowledge Graph Construction"
            ],
            "related_to": [
                {
                    "concept": "Claim Extraction",
                    "relation": "related_to"
                }
            ],
            "tags": [
                "entity-extraction",
                "information-extraction"
            ]
        },
        {
            "concept_name": "Claim Extraction",
            "concept_type": "method",
            "difficulty": "intermediate",
            "summary": "A technique for identifying and capturing explicit assertions or structured statements within source text.",
            "prerequisites": [],
            "unlocks": [
                "Knowledge Graph Construction"
            ],
            "related_to": [
                {
                    "concept": "Entity Extraction",
                    "relation": "related_to"
                }
            ],
            "tags": [
                "claim-extraction",
                "information-extraction"
            ]
        }
    ]
}

def break_dependency_cycles(concepts: list) -> list:
    """Detect cycles in the directed graph of prerequisites/unlocks and break them."""
    if not concepts:
        return concepts
    
    # Map concept_name to its object (case-insensitive key)
    concept_map = {c["concept_name"].lower(): c for c in concepts}
    names = list(concept_map.keys())
    
    # Adjacency list: node -> set of successor nodes
    adj = {name: set() for name in names}
    
    # Populate adjacency list based on prerequisites and unlocks
    for c in concepts:
        u_name = c["concept_name"].lower()
        # Prerequisites: p is a prerequisite for u (p -> u)
        for p in c.get("prerequisites", []):
            p_name = p.lower()
            if p_name in adj:
                adj[p_name].add(u_name)
        # Unlocks: u unlocks unl (u -> unl)
        for unl in c.get("unlocks", []):
            unl_name = unl.lower()
            if unl_name in adj:
                adj[u_name].add(unl_name)
                
    # DFS status: 0=unvisited, 1=visiting, 2=visited
    visited = {name: 0 for name in names}
    edges_to_remove = set()
    
    def dfs(u):
        visited[u] = 1  # visiting
        for v in sorted(list(adj[u])):
            if visited[v] == 1:
                # Found a cycle back-edge u -> v!
                edges_to_remove.add((u, v))
            elif visited[v] == 0:
                dfs(v)
        visited[u] = 2  # visited
        
    for name in names:
        if visited[name] == 0:
            dfs(name)
            
    # Remove the edges that form cycles
    for u_name, v_name in edges_to_remove:
        v_obj = concept_map[v_name]
        u_obj = concept_map[u_name]
        
        # Remove u from v's prerequisites
        v_obj["prerequisites"] = [p for p in v_obj.get("prerequisites", []) if p.lower() != u_name]
        # Remove v from u's unlocks
        u_obj["unlocks"] = [unl for unl in u_obj.get("unlocks", []) if unl.lower() != v_name]
        
    return concepts

def clean_logical_dependencies(concepts: list, text_content: str) -> list:
    bilstm_grounded = ("bilstm" in text_content.lower() or "lstm" in text_content.lower())
    transformer_grounded = ("transformer" in text_content.lower() or "attention" in text_content.lower())
    bert_grounded = "bert" in text_content.lower()
    nsp_grounded = "next sentence prediction" in text_content.lower() or "nsp" in text_content.lower()
    mlm_grounded = "masked language model" in text_content.lower() or "masked lm" in text_content.lower()
    
    # Filter out table-row artifacts BEFORE processing dependencies
    filtered_concepts = []
    for c in concepts:
        name = c["concept_name"]
        # Skip table row labels and ablation conditions that aren't teachable concepts
        if name in {"LTR & No NSP", "No NSP", "LTR", "LTR No NSP", "+ BiLSTM", "BiLSTM Ablation"}:
            continue
        # Skip numeric-only or metric-like names
        if re.match(r"^\d+[\d\s%\.x\-/]*$", name):
            continue
        # Skip if concept_name is a table header pattern
        if re.match(r"^(Dev|Test|Acc|F1|Score|Set|Tasks?)$", name, re.IGNORECASE):
            continue
        filtered_concepts.append(c)
    
    concepts = filtered_concepts
    
    for c in concepts:
        name = c["concept_name"]
        
        # Remove self-loops or recursive dependencies
        c["prerequisites"] = [p for p in c.get("prerequisites", []) if p != name]
        c["unlocks"] = [u for u in c.get("unlocks", []) if u != name]
        
        # Prevent paper-specific model roots from being prerequisites for sub-components
        ROOT_MODEL_NAMES = {"BERT", "GraphRAG", "Graph RAG", "RAG", "LoRA", "ELMo", "GPT", "BERTBASE", "BERTLARGE", "Bertlarge", "Bertbase"}
        if name not in ROOT_MODEL_NAMES:
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ROOT_MODEL_NAMES]
            
        # 1. Bidirectional Language Model fixes
        if name == "Bidirectional Language Model":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p != "BiLSTM"]
            
        # 2. ELMo fixes
        if name == "ELMo":
            c["unlocks"] = [u for u in c.get("unlocks", []) if u != "BiLSTM"]
            if bilstm_grounded and "BiLSTM" not in c.get("prerequisites", []):
                c["prerequisites"].append("BiLSTM")
                
        # 3. BiLSTM fixes
        if name == "BiLSTM":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("ELMo", "BERT", "GPT", "Bidirectional Language Model", "BERTBASE", "BERTLARGE")]
            
        # 4. BERT fixes - BERT is a model, not a prerequisite for its components
        if name == "BERT":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("Bertlarge", "Bertbase", "BERTBASE", "BERTLARGE", "Masked Language Modeling", "Next Sentence Prediction", "Transformer Encoder", "WordPiece", "CLS Token")]
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("Bertlarge", "Bertbase", "BERTBASE", "BERTLARGE", "Masked Language Modeling", "Next Sentence Prediction")]
            # BERT unlocks fine-tuning and feature-based approaches
            if "Fine-Tuning" not in c.get("unlocks", []) and "fine-tuning" in text_content.lower():
                c["unlocks"].append("Fine-Tuning")
            if "Feature-Based Approach" not in c.get("unlocks", []) and "feature-based" in text_content.lower():
                c["unlocks"].append("Feature-Based Approach")
                
        # 5. BERTBASE / BERTLARGE fixes - these are model sizes, not separate concepts
        if name in ("BERTBASE", "BERTLARGE", "Bertbase", "Bertlarge"):
            # Merge into BERT - remove these concepts entirely
            c["concept_name"] = "BERT"
            c["prerequisites"] = []
            c["unlocks"] = []
            
        # 6. Embeddings / Contextual Embeddings fixes
        if name in ("Embeddings", "Contextual Embeddings", "Contextual Word Embeddings", "Word Embeddings"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "ELMo", "GPT")]
            # Embeddings are a building block, not dependent on specific models
            if "Transformer" not in c.get("prerequisites", []) and transformer_grounded:
                c["prerequisites"].append("Transformer")
                
        # 7. Fine-Tuning Approach fixes
        if name in ("Fine-Tuning", "Fine-Tuning Approach", "Fine Tuning"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE")]
            if "Pre-Trained Model" not in c.get("prerequisites", []):
                c["prerequisites"].append("Pre-Trained Model")
                
        # 8. Feature-Based Approach fixes
        if name in ("Feature-Based Approach", "Feature-Based Transfer Learning", "Feature Based Approach"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "Contextual Embeddings")]
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("BERT", "BERTBASE", "BERTLARGE")]
            if "Pre-Trained Model" not in c.get("prerequisites", []):
                c["prerequisites"].append("Pre-Trained Model")
            if "Contextual Embeddings" not in c.get("prerequisites", []):
                c["prerequisites"].append("Contextual Embeddings")
                
        # 9. Next Sentence Prediction fixes
        if name in ("Next Sentence Prediction", "NSP"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE")]
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("BERT", "BERTBASE", "BERTLARGE")]
            # NSP is a pre-training task, prerequisite is language modeling concept
            if "Language Modeling" not in c.get("prerequisites", []):
                c["prerequisites"].append("Language Modeling")
                
        # 10. Masked Language Modeling fixes
        if name in ("Masked Language Modeling", "MLM", "Masked LM"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "Transformer Encoder")]
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("BERT", "BERTBASE", "BERTLARGE", "Transformer Encoder")]
            if "Language Modeling" not in c.get("prerequisites", []):
                c["prerequisites"].append("Language Modeling")
            if transformer_grounded and "Transformer Encoder" not in c.get("prerequisites", []):
                c["prerequisites"].append("Transformer Encoder")
                
        # 11. Left-to-Right Language Model / LTR fixes
        if name in ("Left-to-Right Language Model", "LTR", "Left To Right Language Model"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "GPT", "BERTBASE", "BERTLARGE")]
            
        # 12. BiLSTM in ablation context fixes
        if name == "BiLSTM":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "LTR & No NSP", "No NSP")]
            if "Recurrent Neural Network" not in c.get("prerequisites", []) and "rnn" in text_content.lower():
                c["prerequisites"].append("Recurrent Neural Network")
                
        # 13. Transformer Encoder fixes
        if name == "Transformer Encoder":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "Masked Language Modeling")]
            if "Attention Mechanism" not in c.get("prerequisites", []) and "attention" in text_content.lower():
                c["prerequisites"].append("Attention Mechanism")
                
        # 14. LoRA fixes - LoRA is a general method, doesn't require BERT specifically
        if name == "LoRA":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "BERTBASE", "BERTLARGE", "DeBERTa", "GPT", "GPT-2", "GPT-3")]
            if "Pre-Trained Model" not in c.get("prerequisites", []):
                c["prerequisites"].append("Pre-Trained Model")
            if "Matrix Factorization" not in c.get("prerequisites", []) and "low-rank" in text_content.lower():
                c["prerequisites"].append("Matrix Factorization")
                
        # 15. Feature-Based Approach fixes - BERT enables it, not the other way around
        if name in ("Feature-Based Approach", "Feature-Based Transfer Learning", "Feature Based Approach"):
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("BERT", "BERTBASE", "BERTLARGE", "ELMo", "GPT")]
            if "Pre-Trained Model" not in c.get("prerequisites", []):
                c["prerequisites"].append("Pre-Trained Model")
            if "Contextual Embeddings" not in c.get("prerequisites", []):
                c["prerequisites"].append("Contextual Embeddings")
                
        # 16. Fine-Tuning Approach fixes
        if name in ("Fine-Tuning", "Fine-Tuning Approach", "Fine Tuning"):
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in ("BERT", "BERTBASE", "BERTLARGE")]
            if "Pre-Trained Model" not in c.get("prerequisites", []):
                c["prerequisites"].append("Pre-Trained Model")
                
        # 17. RAG fixes
        if name in ("RAG", "Retrieval-Augmented Generation", "Retrieval Augmented Generation"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "GPT", "Transformer")]
            if "Information Retrieval" not in c.get("prerequisites", []) and "retrieval" in text_content.lower():
                c["prerequisites"].append("Information Retrieval")
            if "Language Model" not in c.get("prerequisites", []) and "language model" in text_content.lower():
                c["prerequisites"].append("Language Model")
                
        # 18. GraphRAG fixes
        if name in ("GraphRAG", "Graph RAG"):
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("RAG", "Graph Database", "Knowledge Graph")]
            if "RAG" not in c.get("prerequisites", []) and "rag" in text_content.lower():
                c["prerequisites"].append("RAG")
            if "Knowledge Graph" not in c.get("prerequisites", []) and "knowledge graph" in text_content.lower():
                c["prerequisites"].append("Knowledge Graph")
                
        # 19. ELMo fixes
        if name == "ELMo":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "GPT", "Transformer")]
            if "BiLSTM" not in c.get("prerequisites", []) and bilstm_grounded:
                c["prerequisites"].append("BiLSTM")
            if "Language Model" not in c.get("prerequisites", []) and "language model" in text_content.lower():
                c["prerequisites"].append("Language Model")
                
        # 20. GPT fixes
        if name == "GPT":
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in ("BERT", "Transformer")]
            if "Language Model" not in c.get("prerequisites", []) and "language model" in text_content.lower():
                c["prerequisites"].append("Language Model")
            if "Transformer" not in c.get("prerequisites", []) and transformer_grounded:
                c["prerequisites"].append("Transformer")
                
        # 21. DeBERTa fixes - variant of BERT
        if name in ("DeBERTa", "Deberta", "DeBERTa XXL", "Deberta XXL"):
            # These are model variants, not standalone concepts to extract
            # Return None to drop them
            c["concept_name"] = ""
            c["_drop"] = True
                
        # General check: a building block should not list a high-level model as its prerequisite
        building_blocks = {
            "BiLSTM", "Transformer", "Attention Mechanism", "WordPiece", "CLS Token", 
            "Embeddings", "Transformer Encoder", "Contextual Embeddings", "Fine-Tuning",
            "Feature-Based Approach", "Next Sentence Prediction", "Masked Language Modeling",
            "Language Modeling", "Pre-Trained Model", "Word Embeddings", "Positional Embeddings",
            "Self-Attention", "Multi-Head Attention", "Feed-Forward Network", "Layer Normalization"
        }
        high_level_models = {"BERT", "ELMo", "GPT", "RAG", "GraphRAG", "LoRA", "BERTBASE", "BERTLARGE"}
        
        if name in building_blocks:
            c["prerequisites"] = [p for p in c.get("prerequisites", []) if p not in high_level_models]
            
        if name in high_level_models:
            c["unlocks"] = [u for u in c.get("unlocks", []) if u not in building_blocks]
            
        # Sort arrays for stability
        c["prerequisites"] = sorted(list(set(c["prerequisites"])))
        c["unlocks"] = sorted(list(set(c["unlocks"])))
        
    # Break cyclic dependencies
    concepts = break_dependency_cycles(concepts)
    
    # Final pass: remove any concepts that were marked for dropping (e.g., DeBERTa variants)
    concepts = [c for c in concepts if not c.get("_drop", False)]
    
    # Also remove any concepts that got merged (BERTBASE/BERTLARGE -> BERT)
    seen = {}
    deduped = []
    for c in concepts:
        key = c["concept_name"].lower()
        if key in seen:
            # Merge prerequisites and unlocks
            existing = seen[key]
            existing["prerequisites"] = sorted(list(set(existing.get("prerequisites", []) + c.get("prerequisites", []))))
            existing["unlocks"] = sorted(list(set(existing.get("unlocks", []) + c.get("unlocks", []))))
            # Keep the better summary
            if len(c.get("summary", "")) > len(existing.get("summary", "")):
                existing["summary"] = c["summary"]
        else:
            seen[key] = c
            deduped.append(c)
    
    return deduped

# Guided JSON schema for Ollama completion mode
OKF_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "concept_name": {"type": "string"},
            "concept_type": {"type": "string"},
            "difficulty": {"type": "string"},
            "summary": {"type": "string"},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "unlocks": {"type": "array", "items": {"type": "string"}},
            "related_to": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "relation": {"type": "string"}
                    },
                    "required": ["concept", "relation"]
                }
            },
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": [
            "concept_name", "concept_type", "difficulty", 
            "summary", "prerequisites", "unlocks", "related_to", "tags"
        ]
    }
}

SYSTEM_PROMPT = """You are an OKF extraction engine for the Archipelago knowledge graph.
From the TEXT provided, extract 1-5 teachable CONCEPTS as a JSON array.

Strict Rules:
- concept_name: ≤ 5 words, Title Case, reusable noun phrase.
- ONLY extract concepts that are explicitly explained in the input TEXT.
- DO NOT invent prerequisites or unlocks that are not mentioned in the text.
- NEVER list author names, paper section titles, or specific table headers as concepts or prerequisites.
- DO NOT output duplicate concepts in the array.
- Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures."""

def enforce_strict_schema(concepts_list: list) -> list:
    """Ensure every concept dict has exactly all 8 required keys in the correct order."""
    required_keys = [
        "concept_name",
        "concept_type",
        "difficulty",
        "summary",
        "prerequisites",
        "unlocks",
        "related_to",
        "tags"
    ]
    strict_list = []
    for c in concepts_list:
        if not isinstance(c, dict):
            continue
        strict_c = {}
        for key in required_keys:
            val = c.get(key)
            if val is None:
                if key in ("prerequisites", "unlocks", "related_to", "tags"):
                    strict_c[key] = []
                elif key == "concept_type":
                    strict_c[key] = "definition"
                elif key == "difficulty":
                    strict_c[key] = "intermediate"
                else:
                    strict_c[key] = ""
            else:
                strict_c[key] = val
        strict_list.append(strict_c)
    return strict_list


def clean_and_deduplicate_array(concepts: list, text_content: str) -> list:
    """Post-processing filter to drop ungrounded concepts and duplicates."""
    seen_names = set()
    cleaned_concepts = []

    for item in concepts:
        name = item.get("concept_name", "").strip()
        
        # Skip empty or duplicate concept names
        if not name or name.lower() in seen_names:
            continue
            
        # Clean prerequisites: remove self-references and unmentioned prerequisites
        prereqs = [
            p for p in item.get("prerequisites", []) 
            if p.lower() != name.lower() and p.lower() in text_content.lower()
        ]
        item["prerequisites"] = prereqs
        
        # Clean unlocks: remove self-references
        unlocks = [
            u for u in item.get("unlocks", []) 
            if u.lower() != name.lower()
        ]
        item["unlocks"] = unlocks
        
        # Override misclassified types
        name_lower = name.lower()
        if name_lower in CONCEPT_TYPE_OVERRIDES:
            item["concept_type"] = CONCEPT_TYPE_OVERRIDES[name_lower]

        seen_names.add(name_lower)
        cleaned_concepts.append(item)

    return clean_logical_dependencies(cleaned_concepts, text_content)

def is_junk(name: str) -> bool:
    name = (name or "").strip()
    if not name or len(name) < 3:
        return True
    if len(name.split()) > 5:
        return True
    if _NUMERIC_CONCEPT_RE.match(name):
        return True
    if _JUNK_RE.search(name):
        return True
    if _FORMULA_OR_VALUE_RE.search(name):
        return True
    return False



def _concept_key(name: str) -> str:
    key = re.sub(r"\([^)]*\)", "", name or "").lower()
    key = key.replace("-", " ")
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    words = [w for w in key.split() if w not in {"full", "basic", "standard", "general"}]
    normalized = []
    for word in words:
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return " ".join(normalized)


def same_training_concept(a: str, b: str) -> bool:
    """Return true for exact/near-exact aliases, not broader prerequisites."""
    a_key = _concept_key(a)
    b_key = _concept_key(b)
    return bool(a_key and b_key and a_key == b_key)


def clean_okf_record(rec: dict) -> dict | None:
    """Return a strict OKF v1.6 object or None if it should be discarded."""
    name = normalize_training_name(coerce_concept_name(rec.get("concept_name", "")).strip())
    if is_junk(name):
        return None
    ctype = str(rec.get("concept_type", "definition")).lower()
    if ctype not in VALID_TYPES:
        ctype = "definition"

    diff = str(rec.get("difficulty", "intermediate")).lower()
    if diff not in VALID_DIFFICULTIES:
        diff = "intermediate"

    summary = (rec.get("summary") or "").strip()
    if not summary or _META_RE.search(summary):
        # No summary or meta summary -> drop. The model must learn to write a 1-2 sentence definition.
        return None

    name_l = name.lower()
    prereqs = sorted(set(
        normalize_training_name(p) for p in rec.get("prerequisites", [])
        if isinstance(p, str)
        and p.strip()
        and p.lower() != name_l
        and not same_training_concept(p, name)
        and not is_junk(normalize_training_name(p))
    ))
    unlocks = sorted(set(
        normalize_training_name(u) for u in rec.get("unlocks", [])
        if isinstance(u, str)
        and u.strip()
        and u.lower() != name_l
        and not same_training_concept(u, name)
        and not is_junk(normalize_training_name(u))
    ))

    related = []
    seen_rel = set()
    for r in rec.get("related_to", []):
        if not isinstance(r, dict):
            continue
        c = normalize_training_name(str(r.get("concept", "")).strip())
        rel = str(r.get("relation", "uses")).lower()
        if not c or c.lower() == name_l or same_training_concept(c, name) or is_junk(c):
            continue
        if rel not in VALID_RELATIONS:
            rel = "uses"
        key = (c.lower(), rel)
        if key not in seen_rel:
            related.append({"concept": c, "relation": rel})
            seen_rel.add(key)

    tags = sorted(set(
        t.lower().replace(" ", "-") for t in rec.get("tags", []) if isinstance(t, str) and t.strip()
    ))

    return {
        "concept_name": name,
        "concept_type": ctype,
        "difficulty": diff,
        "summary": summary,
        "prerequisites": prereqs,
        "unlocks": unlocks,
        "related_to": related,
        "tags": tags,
    }


def concept_window(text: str, name: str, window: int = 500) -> str | None:
    """Return a short snippet surrounding a grounded mention of a concept."""
    if not text or not name:
        return None
    core = re.sub(r"\s*\([^)]*\)", "", name).strip()
    surfaces = {core, name}
    surfaces.update(_ALIAS_TO_SURFACES.get(name, set()))
    # Known acronym aliases may be the only literal form in a source passage.
    if name in {"BERT", "LoRA", "RAG", "GraphRAG"}:
        surfaces.add(name)
    if name == "Low-Rank Adaptation":
        surfaces.add("LoRA")
    if name == "Retrieval-Augmented Generation":
        surfaces.add("RAG")

    m = None
    for surface in sorted(surfaces, key=len, reverse=True):
        if not surface:
            continue
        surface_pat = re.sub(r"\s+", r"\\s+", re.escape(surface))
        pat = re.compile(surface_pat, re.IGNORECASE)
        m = pat.search(text)
        if m:
            break

    if not m:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", core) if len(w) > 2]
        if len(words) > 1:
            pat = re.compile(re.escape(words[0]) + r".*?" + re.escape(words[1]), re.IGNORECASE | re.DOTALL)
            m = pat.search(text)
    if not m:
        return None
    start = max(0, m.start() - window // 2)
    end = min(len(text), m.end() + window // 2)
    return text[start:end]


def make_alpaca_prompt(text: str) -> str:
    return (
        "You are an OKF extraction engine for the Archipelago knowledge graph.\n"
        "From the TEXT below, extract 1-5 teachable CONCEPTS as a JSON array.\n\n"
        "Each object MUST have exactly these keys:\n"
        "concept_name, concept_type, difficulty, summary, prerequisites, unlocks, related_to, tags\n\n"
        "Rules:\n"
        "- concept_name: ≤ 5 words, a reusable noun phrase, Title Case.\n"
        "- Only concepts actually explained in the text. No authors, citations, section titles.\n"
        "- prerequisites = what a learner needs FIRST; unlocks = what this ENABLES next.\n"
        "- A concept must NEVER appear in its own prerequisites or unlocks.\n"
        "- Keep names stable across documents so the same concept merges into one node.\n"
        "- If the text has no real teachable concept, return [].\n"
        "- Basic mathematical or statistical concepts (e.g., Linear Regression, Matrix Inverse) must usually be PREREQUISITES, not UNLOCKS for advanced architectures.\n"
        "- Output ONLY the JSON array. No prose, no markdown fences.\n\n"
        f"TEXT:\n{text}\n\n"
        "Return ONLY the JSON array, no other text:"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="Run structured completions via Ollama")
    parser.add_argument("--model", type=str, default="qwen2.5:0.5b", help="Ollama model to use")
    args = parser.parse_args()

    # Clean up old files in OUT_DIR
    print("Cleaning up old dataset files in training_data/...")
    for path in OUT_DIR.glob("*"):
        if path.is_file() and path.suffix in (".jsonl", ".json", ".bak"):
            print(f"  Deleting old file: {path.name}")
            try:
                path.unlink()
            except Exception as e:
                print(f"  Failed to delete {path.name}: {e}")

    print("Re-chunking source documents...")
    chunks = ingest_folder(str(PDF_DIR))
    # Restrict to core documents only
    chunks = [c for c in chunks if c["doc_id"] in CORE_DOCS]
    chunk_index = {(c["doc_id"], c["chunk_id"]): c for c in chunks}
    empty_chunks = [c for c in chunks if c.get("chunk_kind") != "prose" or not c["text"].strip()]

    # Filter out table-like prose chunks that passed the initial classifier
    print("Filtering table-like prose chunks...")
    filtered_chunks = []
    table_like_count = 0
    for c in chunks:
        if c.get("chunk_kind") == "prose" and c["text"].strip():
            if is_table_like(c["text"]):
                table_like_count += 1
                c["chunk_kind"] = "table"  # reclassify
            else:
                filtered_chunks.append(c)
        else:
            filtered_chunks.append(c)
    
    chunks = filtered_chunks
    # Rebuild empty_chunks from the filtered chunks to avoid duplicates
    empty_chunks = [c for c in chunks if c.get("chunk_kind") != "prose" or not c["text"].strip()]
    print(f"Loaded {len(chunks)} prose chunks ({table_like_count} reclassified as table) from core papers.")

    records = []
    chunk_level_count = 0
    discarded_reasons = Counter()
    discarded_examples = Counter()

    if args.ollama:
        import ollama
        print(f"Running structured completions via Ollama using model {args.model}...")
        prose_chunks = [c for c in chunks if c.get("chunk_kind") == "prose"]
        for i, chunk in enumerate(prose_chunks):
            progress = f"[{i+1}/{len(prose_chunks)}]"
            print(f"  {progress} {chunk['doc_id']} | {chunk['chunk_id']} (p.{chunk['page_number']})", end="")
            
            # Check for chunk override first
            override_key = (chunk["doc_id"], chunk["chunk_id"])
            if override_key in CHUNK_OVERRIDES:
                cleaned_output = CHUNK_OVERRIDES[override_key]
                print(" -> [override applied]", end="")
            else:
                try:
                    response = ollama.chat(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"TEXT:\n{chunk['text']}"}
                        ],
                        format=OKF_SCHEMA
                    )
                    extracted_raw = json.loads(response["message"]["content"])
                    
                    # Apply type coercions and cleanup on raw extracted items if needed
                    for r in extracted_raw:
                        r["concept_name"] = normalize_training_name(coerce_concept_name(r.get("concept_name", "")))
                    
                    # Apply our strict clean and deduplicate filter!
                    cleaned_output = clean_and_deduplicate_array(extracted_raw, chunk["text"])
                except Exception as exc:
                    print(f" -> FAILED: {exc}")
                    continue
                
            if cleaned_output:
                # Apply logical dependencies cleaner
                cleaned_output = clean_logical_dependencies(cleaned_output, chunk["text"])
                
                # Prune ungrounded prerequisite / unlock / related_to targets
                group_names = {c["concept_name"].lower() for c in cleaned_output}
                for c in cleaned_output:
                    own = c["concept_name"].lower()
                    
                    def keep_target(tgt):
                        t = (tgt or "").strip()
                        if not t:
                            return False
                        if is_concept_grounded(chunk["text"], t):
                            return True
                        if t.lower() in group_names:
                            return True
                        return False
                        
                    c["prerequisites"] = [p for p in c.get("prerequisites", []) if p.lower() != own and keep_target(p)]
                    c["unlocks"] = [u for u in c.get("unlocks", []) if u.lower() != own and keep_target(u)]
                    
                    clean_related = []
                    for rel in c.get("related_to", []):
                        tgt = rel.get("concept", "")
                        if tgt.lower() != own and keep_target(tgt):
                            clean_related.append(rel)
                    c["related_to"] = clean_related
                    
                records.append({
                    "instruction": make_alpaca_prompt(chunk["text"]),
                    "input": "",
                    "output": json.dumps(enforce_strict_schema(cleaned_output), ensure_ascii=False, indent=2),
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk.get("page_number", 0),
                    "section_title": chunk.get("section_title", "")
                })
                chunk_level_count += 1
                print(f" -> extracted {len(cleaned_output)} concepts")
            else:
                print(" -> 0 concepts (empty)")
    else:
        raw_results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        # Restrict raw results to core documents only
        raw_results = [r for r in raw_results if r.get("doc_id") in CORE_DOCS]
        print(f"Loaded {len(raw_results)} raw concept records for core papers.")

        # Apply only lightweight acronym aliases; keep distinct concepts distinct.
        for r in raw_results:
            r["concept_name"] = normalize_training_name(coerce_concept_name(r.get("concept_name", "")))

        by_chunk = defaultdict(list)
        for r in raw_results:
            by_chunk[(r["doc_id"], r["chunk_id"])].append(r)

        # Keep only the first record per chunk (raw_results has 5 duplicates from multiple pipeline runs)
        for key in by_chunk:
            by_chunk[key] = [by_chunk[key][0]]

        # 1) Multi-concept chunk-level examples (real task distribution)
        def output_for_concepts(concepts, text, include_metadata=True):
            cleaned = []
            for r in concepts:
                c = clean_okf_record(r)
                if c:
                    # Check if concept is grounded in chunk text
                    if not is_concept_grounded(text, c["concept_name"]):
                        discarded_reasons["ungrounded"] += 1
                        discarded_examples[c["concept_name"]] += 1
                        continue
                    cleaned.append(c)
                else:
                    discarded_reasons["failed_cleaner"] += 1
                    discarded_examples[coerce_concept_name(r.get("concept_name", "?")) or "?"] += 1

            cleaned = clean_and_deduplicate_array(cleaned, text)

            if not cleaned:
                return None

            # Prune ungrounded prerequisite / unlock / related_to targets
            group_names = {c["concept_name"].lower() for c in cleaned}
            for c in cleaned:
                own = c["concept_name"].lower()

                def keep_target(tgt):
                    t = (tgt or "").strip()
                    if not t:
                        return False
                    if is_concept_grounded(text, t):
                        return True
                    if t.lower() in group_names:
                        return True
                    return False

                c["prerequisites"] = [p for p in c["prerequisites"] if p.lower() != own and keep_target(p)]
                c["unlocks"] = [u for u in c["unlocks"] if u.lower() != own and keep_target(u)]

                clean_related = []
                for rel in c.get("related_to", []):
                    tgt = rel.get("concept", "")
                    if tgt.lower() != own and keep_target(tgt):
                        clean_related.append(rel)
                c["related_to"] = clean_related

            out = cleaned
            if include_metadata:
                # Training metadata is kept separate; model should NOT emit this.
                out = {
                    "output": cleaned,
                    "metadata": {
                        "doc_id": concepts[0].get("doc_id", ""),
                        "chunk_id": concepts[0].get("chunk_id", ""),
                        "page_number": concepts[0].get("page_number", 0),
                        "section_title": concepts[0].get("section_title", ""),
                    }
                }
            return out

        for key, concepts in by_chunk.items():
            chunk = chunk_index.get(key)
            if not chunk or not chunk["text"].strip():
                continue
            
            # Check for chunk override first - process even if non-prose
            override_key = (chunk["doc_id"], chunk["chunk_id"])
            has_override = override_key in CHUNK_OVERRIDES
            
            if not has_override and chunk.get("chunk_kind") != "prose":
                continue
            
            if has_override:
                out_concepts = CHUNK_OVERRIDES[override_key]
                # Apply logical dependencies cleaner on override
                out_concepts = clean_logical_dependencies(out_concepts, chunk["text"])
                out = {
                    "output": out_concepts,
                    "metadata": {
                        "doc_id": chunk["doc_id"],
                        "chunk_id": chunk["chunk_id"],
                        "page_number": chunk.get("page_number", 0),
                        "section_title": chunk.get("section_title", ""),
                    }
                }
            else:
                out = output_for_concepts(concepts, chunk["text"])
                
            if out is None:
                continue
            chunk_level_count += 1
            records.append({
                "instruction": make_alpaca_prompt(chunk["text"]),
                "input": "",
                "output": json.dumps(enforce_strict_schema(out["output"]), ensure_ascii=False, indent=2),
                **out["metadata"]
            })

    # 1b) Process CHUNK_OVERRIDES that weren't in raw_results (e.g., early BERT chunks)
    processed_chunk_ids = set((r["doc_id"], r["chunk_id"]) for r in records)
    for override_key, override_concepts in CHUNK_OVERRIDES.items():
        if override_key in processed_chunk_ids:
            continue
        chunk = chunk_index.get(override_key)
        if not chunk or not chunk["text"].strip():
            continue
        out_concepts = override_concepts
        out_concepts = clean_logical_dependencies(out_concepts, chunk["text"])
        out = {
            "output": out_concepts,
            "metadata": {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk.get("page_number", 0),
                "section_title": chunk.get("section_title", ""),
            }
        }
        chunk_level_count += 1
        records.append({
            "instruction": make_alpaca_prompt(chunk["text"]),
            "input": "",
            "output": json.dumps(enforce_strict_schema(out["output"]), ensure_ascii=False, indent=2),
            **out["metadata"]
        })

    # 2) Single-concept snippet examples (more granular supervision) - DISABLED
    single_count = 0

    # 3) Empty-response examples (math, references, tables, frontmatter)
    empty_added = 0
    override_chunk_ids = set(CHUNK_OVERRIDES.keys())
    for c in empty_chunks:
        if not c["text"].strip():
            continue
        # Skip chunks that have overrides (they were already processed)
        if (c["doc_id"], c["chunk_id"]) in override_chunk_ids:
            continue
        empty_added += 1
        records.append({
            "instruction": make_alpaca_prompt(c["text"][:1200]),
            "input": "",
            "output": "[]",
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "page_number": c.get("page_number", 0),
            "section_title": c.get("section_title", ""),
        })

    records.sort(key=lambda r: (r.get("doc_id", ""), r.get("chunk_id", ""), r["output"] == "[]", r["instruction"]))

    output_lengths = []
    concept_names = []
    empty_response_count = 0
    for rec in records:
        parsed = json.loads(rec["output"])
        if parsed == []:
            empty_response_count += 1
        if isinstance(parsed, list):
            output_lengths.append(len(parsed))
            for item in parsed:
                if isinstance(item, dict):
                    concept_names.append(item.get("concept_name", ""))

    report = {
        "source_results": str(RESULTS_FILE.relative_to(BASE_DIR)),
        "total_records": len(records),
        "chunk_level_multi_concept_records": chunk_level_count,
        "single_concept_snippet_records": single_count,
        "empty_response_records": empty_added,
        "non_empty_records": len(records) - empty_response_count,
        "total_concept_targets": len(concept_names),
        "unique_concept_targets": len(set(concept_names)),
        "discarded_raw_concept_records": sum(discarded_examples.values()),
        "top_discarded_examples": discarded_examples.most_common(25),
    }

    # Write JSONL
    jsonl_path = OUT_DIR / "okf_training_pairs_v3.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write human-reviewable JSON
    json_path = OUT_DIR / "okf_training_pairs_v3.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = OUT_DIR / "okf_dataset_report_v3.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(records)} training cases -> {jsonl_path}")
    print(f"  - chunk-level multi-concept: {chunk_level_count}")
    print(f"  - single-concept snippets:   {single_count}")
    print(f"  - empty-response examples:   {empty_added}")
    print(f"  - concept targets:           {len(concept_names)} ({len(set(concept_names))} unique)")
    print(f"Discarded raw concept records during cleaning: {sum(discarded_examples.values())}")
    for name, cnt in discarded_examples.most_common(20):
        print(f"  {cnt:3d}  {name}")
    print(f"See also: {json_path}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()

