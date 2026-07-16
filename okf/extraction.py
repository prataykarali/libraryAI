"""SLM extraction: local-model state, generation, and OKF v1.5 parsing.

This module OWNS the local PyTorch model state (LOCAL_MODEL / LOCAL_TOKENIZER
/ LOCAL_MODE). load_local_model mutates these module globals; other modules
must query the state through is_local_mode()/is_model_loaded() instead of
importing the raw globals (a `from ... import LOCAL_MODEL` would snapshot the
pre-load None forever).
"""

import json
import sys
import time

import ollama

from okf.cleanup import is_valid_concept_name
from okf.config import (
    BASE_DIR,
    EXTRACTION_PROMPT_V15,
    MAX_CHARS_TO_SLM,
    MAX_RETRIES,
    MODEL_NAME,
    VALID_DIFFICULTIES,
    VALID_RELATIONS,
    VALID_TYPES,
    _local_path,
    infer_source_category,
)

# Local PyTorch model configuration (aura-qwen fine-tuned)
LOCAL_MODEL = None
LOCAL_TOKENIZER = None
LOCAL_MODE = _local_path.exists()


def is_local_mode() -> bool:
    """Live LOCAL_MODE value (may flip after a failed load_local_model)."""
    return LOCAL_MODE


def is_model_loaded() -> bool:
    """True once load_local_model has successfully populated LOCAL_MODEL."""
    return LOCAL_MODEL is not None


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_json_payload(raw: str) -> str:
    """Extract the first JSON array/object from a chatty model response."""
    cleaned = _strip_json_fences(raw)
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if array_start != -1 and array_end > array_start:
        if object_start == -1 or array_start < object_start:
            return cleaned[array_start:array_end + 1]
    if object_start != -1 and object_end > object_start:
        return cleaned[object_start:object_end + 1]
    return cleaned


def _string_list(value) -> list:
    """Normalize scalar/nested list model output into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if isinstance(item, list):
            items.extend(str(x).strip() for x in item if isinstance(x, str) and x.strip())
        elif isinstance(item, str) and item.strip():
            items.append(item.strip())
    return list(dict.fromkeys(items))


def _normalize_related(value) -> list:
    if not isinstance(value, list):
        return []

    related = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            concept = item.strip()
            relation = "uses"
        elif isinstance(item, dict):
            concept = str(item.get("concept", "")).strip()
            relation = str(item.get("relation", "uses")).strip().lower()
        else:
            continue
        if not concept:
            continue
        if relation not in VALID_RELATIONS:
            relation = "uses"
        key = (concept.lower(), relation)
        if key not in seen:
            related.append({"concept": concept, "relation": relation})
            seen.add(key)
    return related


def normalize_okf_item(data: dict, doc_id: str, chunk_id: str,
                       page_number: int, section_title: str) -> dict | None:
    """Validate and normalize one OKF concept object with provenance."""
    if not isinstance(data, dict):
        return None

    concept_name = data.get("concept_name", "")
    if isinstance(concept_name, list):
        names = [n.strip() for n in concept_name if isinstance(n, str) and n.strip()]
        concept_name = names[0] if names else ""
    if not isinstance(concept_name, str) or not concept_name.strip():
        return None
    concept_name = concept_name.strip()
    if not is_valid_concept_name(concept_name):
        return None

    ctype = data.get("concept_type", "definition")
    if isinstance(ctype, list) and ctype:
        ctype = ctype[0]
    ctype = str(ctype).lower() if ctype else "definition"
    if ctype not in VALID_TYPES:
        ctype = "definition"

    difficulty = data.get("difficulty", "intermediate")
    if isinstance(difficulty, list) and difficulty:
        difficulty = difficulty[0]
    difficulty = str(difficulty).lower() if difficulty else "intermediate"
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "intermediate"

    summary = data.get("summary", "")
    if isinstance(summary, list):
        summary = " ".join(str(x).strip() for x in summary if str(x).strip())
    summary = str(summary).strip()

    name_lower = concept_name.lower()
    prerequisites = [p for p in _string_list(data.get("prerequisites")) if p.lower() != name_lower]
    unlocks = [u for u in _string_list(data.get("unlocks")) if u.lower() != name_lower]
    tags = [t.lower().replace(" ", "-") for t in _string_list(data.get("tags"))]

    return {
        "concept_name": concept_name,
        "concept_type": ctype,
        "difficulty": difficulty,
        "summary": summary,
        "prerequisites": prerequisites,
        "unlocks": unlocks,
        "related_to": _normalize_related(data.get("related_to")),
        "tags": list(dict.fromkeys(tags)),
        "doc_id": doc_id,
        "source_category": infer_source_category(doc_id),
        "chunk_id": chunk_id,
        "page_number": page_number,
        "section_title": section_title,
    }


def load_local_model():
    """Load the local model and tokenizer directly from disk."""
    global LOCAL_MODEL, LOCAL_TOKENIZER, LOCAL_MODE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = BASE_DIR.parent / "aura-qwen"
    if not local_path.exists():
        local_path = _local_path

    print(f"  Loading local model from {local_path}...")
    try:
        LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True, fix_mistral_regex=True
        )
        LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  ✓ Local model loaded successfully!")
        LOCAL_MODE = True
    except Exception as exc:
        print(f"  Error loading local model: {exc}")
        print("  ⚠️ Falling back to Ollama mode...")
        LOCAL_MODE = False


def _generate_local(prompt: str, max_new_tokens: int = 512) -> str:
    """One local-model generation with the same chat templating as extraction."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = LOCAL_TOKENIZER(templated_prompt, return_tensors="pt").to(LOCAL_MODEL.device)
    with torch.no_grad():
        outputs = LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )
    input_len = inputs.input_ids.shape[1]
    return LOCAL_TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def extract_okf_v15(text: str, doc_id: str = "", chunk_id: str = "",
                    page_number: int = 0, section_title: str = "") -> list:
    """Extract OKF v1.6 concepts from a text chunk using Ollama or local model.

    We never send a whole page/book: the chunk is already section/paragraph
    scale, and we truncate to MAX_CHARS_TO_SLM at a paragraph/sentence boundary
    so the SLM only reasons over a small, coherent span.
    """
    if len(text) > MAX_CHARS_TO_SLM:
        window = text[:MAX_CHARS_TO_SLM]
        # Prefer to cut at the last paragraph break, else the last sentence end.
        cut = window.rfind("\n\n")
        if cut < MAX_CHARS_TO_SLM // 2:
            cut = max(window.rfind(". "), window.rfind("\n"))
        text = window[:cut] if cut > MAX_CHARS_TO_SLM // 2 else window

    prompt = EXTRACTION_PROMPT_V15.format(text=text)

    for attempt in range(MAX_RETRIES + 1):
        try:
            if LOCAL_MODE:
                # Local PyTorch/Transformers inference
                messages = [{"role": "user", "content": prompt}]
                try:
                    templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception:
                    templated_prompt = LOCAL_TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                inputs = LOCAL_TOKENIZER(templated_prompt, return_tensors="pt").to(LOCAL_MODEL.device)

                import torch
                with torch.no_grad():
                    outputs = LOCAL_MODEL.generate(
                        **inputs,
                        max_new_tokens=768,
                        temperature=0.1,
                        do_sample=False,
                    )
                input_len = inputs.input_ids.shape[1]
                full_response = LOCAL_TOKENIZER.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            else:
                # Ollama inference
                response_stream = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    think=False,
                    # num_predict caps output so a rambling chunk fails in ~3s
                    # instead of generating until the context window fills (~150s).
                    options={"temperature": 0.1, "top_p": 0.9, "num_predict": 768},
                )
                full_response = ""
                for chunk in response_stream:
                    token = chunk["message"]["content"]
                    full_response += token

            cleaned = _extract_json_payload(full_response)
            data = json.loads(cleaned)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                if attempt < MAX_RETRIES:
                    continue
                return []

            normalized = [
                item for item in (
                    normalize_okf_item(x, doc_id, chunk_id, page_number, section_title)
                    for x in data[:5]
                )
                if item
            ]
            if normalized:
                return normalized
            if attempt < MAX_RETRIES:
                continue
            return []

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                continue
            return []
        except Exception as exc:
            print(f"    Error: {exc}")
            if attempt < MAX_RETRIES:
                continue
            return []

    return []


def extract_chunks_with_model(chunks: list) -> tuple[list, int]:
    """Run OKF extraction over prose chunks with per-chunk progress output.

    Returns (okf_results, successful_chunk_count). Shared by run_pipeline and
    add_document so the extraction loop stays identical in both modes.
    """
    okf_results = []
    successful_chunk_count = 0
    for i, chunk in enumerate(chunks):
        progress = f"[{i+1}/{len(chunks)}]"
        section = chunk.get("section_title", "?")[:40]
        print(f"  {progress} {chunk['doc_id']} | {section} (p.{chunk['page_number']})", end="")
        sys.stdout.flush()

        start_time = time.time()
        results = extract_okf_v15(
            text=chunk["text"],
            doc_id=chunk["doc_id"],
            chunk_id=chunk["chunk_id"],
            page_number=chunk["page_number"],
            section_title=chunk["section_title"]
        )
        elapsed = time.time() - start_time

        if results:
            # Attach the original source passage so the UI can show the
            # highlighted chunk and deep-link back to the source page.
            passage = chunk["text"][:1600]
            for r in results:
                r["source_passage"] = passage
                r.setdefault("section_title", chunk.get("section_title", ""))
            okf_results.extend(results)
            successful_chunk_count += 1
            names = ", ".join(r["concept_name"] for r in results[:3])
            if len(results) > 3:
                names += f", +{len(results) - 3} more"
            print(f" -> {len(results)} concepts: {names[:70]} ({elapsed:.1f}s)")
        else:
            print(f" -> FAILED ({elapsed:.1f}s)")

    return okf_results, successful_chunk_count
