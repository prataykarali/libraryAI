#!/usr/bin/env python3
"""
Section-aware PDF ingestion with provenance tagging.
Splits PDFs by structural boundaries (headings, ToC) instead of arbitrary char counts.
Also supports Markdown and plain text files.
"""

import re
import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def _detect_heading(span_text: str, span_size: float, span_font: str,
                    body_size: float) -> bool:
    """Heuristic: is this text span a heading?"""
    text = span_text.strip()
    if not text or len(text) < 2:
        return False
    # Larger font than body = heading
    if span_size > body_size + 1.5:
        return True
    # Bold font and reasonably short
    if "Bold" in span_font or "Medi" in span_font or "bold" in span_font:
        if len(text) < 120:
            return True
    # Numbered section pattern: "3.1 Something" or "Chapter 4"
    if re.match(r'^(\d+\.?\d*\.?\d*)\s+[A-Z]', text):
        return True
    if re.match(r'^(Chapter|Section|Appendix)\s+\w', text, re.IGNORECASE):
        return True
    return False


def _get_body_font_size(doc) -> float:
    """Find the most common font size in the document (= body text size)."""
    size_counts = {}
    for page_num in range(min(10, doc.page_count)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sz = round(span["size"], 1)
                        char_count = len(span["text"].strip())
                        if char_count > 0:
                            size_counts[sz] = size_counts.get(sz, 0) + char_count
    if not size_counts:
        return 10.0
    return max(size_counts, key=size_counts.get)


def chunk_pdf(pdf_path: str, max_chunk_chars: int = 3000,
              min_chunk_chars: int = 200) -> list:
    """
    Split a PDF into section-aware chunks with provenance metadata.

    Returns list of dicts:
        {doc_id, chunk_id, section_title, page_number, text}
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

    doc = fitz.open(pdf_path)
    doc_id = Path(pdf_path).name
    body_size = _get_body_font_size(doc)

    # Try ToC first for section boundaries
    toc = doc.get_toc()
    toc_sections = []
    if toc:
        for level, title, page_num in toc:
            if level <= 2:  # Only top-level and second-level headings
                toc_sections.append((title.strip(), page_num - 1))  # 0-indexed

    # Extract text with section awareness
    sections = []
    current_section = "Introduction"
    current_text = ""
    current_page = 1

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue
            block_text = ""
            is_heading = False

            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    text = span["text"]
                    line_text += text
                    if not is_heading and _detect_heading(
                        text, span["size"], span["font"], body_size
                    ):
                        is_heading = True
                block_text += line_text.strip() + " "

            block_text = block_text.strip()
            if not block_text:
                continue

            # New section detected
            if is_heading and len(block_text) < 150 and len(block_text) > 2:
                # Save previous section if it has content
                if current_text.strip() and len(current_text.strip()) >= min_chunk_chars:
                    sections.append({
                        "section_title": current_section,
                        "page_number": current_page,
                        "text": current_text.strip()
                    })
                elif current_text.strip() and sections:
                    # Merge tiny section into previous
                    sections[-1]["text"] += "\n\n" + current_text.strip()

                current_section = block_text.strip()
                current_text = ""
                current_page = page_num + 1
            else:
                current_text += block_text + "\n"

    # Don't forget the last section
    if current_text.strip() and len(current_text.strip()) >= min_chunk_chars:
        sections.append({
            "section_title": current_section,
            "page_number": current_page,
            "text": current_text.strip()
        })
    elif current_text.strip() and sections:
        sections[-1]["text"] += "\n\n" + current_text.strip()

    doc.close()

    # Now split oversized sections into sub-chunks
    chunks = []
    chunk_counter = 0

    for section in sections:
        text = section["text"]
        if len(text) <= max_chunk_chars:
            chunk_counter += 1
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "section_title": section["section_title"],
                "page_number": section["page_number"],
                "text": text
            })
        else:
            # Split by paragraphs, then merge up to max_chunk_chars
            paragraphs = text.split("\n")
            current_chunk_text = ""
            for para in paragraphs:
                if len(current_chunk_text) + len(para) > max_chunk_chars and current_chunk_text:
                    chunk_counter += 1
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"chunk_{chunk_counter:03d}",
                        "section_title": section["section_title"],
                        "page_number": section["page_number"],
                        "text": current_chunk_text.strip()
                    })
                    current_chunk_text = para + "\n"
                else:
                    current_chunk_text += para + "\n"
            if current_chunk_text.strip():
                chunk_counter += 1
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "section_title": section["section_title"],
                    "page_number": section["page_number"],
                    "text": current_chunk_text.strip()
                })

    return chunks


def chunk_markdown(md_path: str, max_chunk_chars: int = 3000,
                   min_chunk_chars: int = 200) -> list:
    """Split a Markdown file by headings into chunks."""
    doc_id = Path(md_path).name
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by headings (# or ##)
    sections = re.split(r'^(#{1,3}\s+.+)$', content, flags=re.MULTILINE)

    chunks = []
    chunk_counter = 0
    current_title = "Introduction"

    i = 0
    while i < len(sections):
        part = sections[i].strip()
        if re.match(r'^#{1,3}\s+', part):
            current_title = re.sub(r'^#{1,3}\s+', '', part).strip()
            i += 1
            continue
        if part and len(part) >= min_chunk_chars:
            # Sub-chunk if too large
            if len(part) > max_chunk_chars:
                for start in range(0, len(part), max_chunk_chars):
                    sub = part[start:start + max_chunk_chars].strip()
                    if sub and len(sub) >= min_chunk_chars:
                        chunk_counter += 1
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_id": f"chunk_{chunk_counter:03d}",
                            "section_title": current_title,
                            "page_number": 0,
                            "text": sub
                        })
            else:
                chunk_counter += 1
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "section_title": current_title,
                    "page_number": 0,
                    "text": part
                })
        i += 1

    return chunks


def chunk_text(txt_path: str, max_chunk_chars: int = 3000,
               min_chunk_chars: int = 200) -> list:
    """Split a plain text file by paragraphs into chunks."""
    doc_id = Path(txt_path).name
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = content.split("\n\n")
    chunks = []
    chunk_counter = 0
    current_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_text) + len(para) > max_chunk_chars and current_text:
            chunk_counter += 1
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "section_title": "Section",
                "page_number": 0,
                "text": current_text.strip()
            })
            current_text = para + "\n\n"
        else:
            current_text += para + "\n\n"

    if current_text.strip() and len(current_text.strip()) >= min_chunk_chars:
        chunk_counter += 1
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"chunk_{chunk_counter:03d}",
            "section_title": "Section",
            "page_number": 0,
            "text": current_text.strip()
        })

    return chunks


def ingest_document(path: str) -> list:
    """Universal document ingester. Routes to the right chunker by file extension."""
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return chunk_pdf(path)
    elif ext in (".md", ".markdown"):
        return chunk_markdown(path)
    elif ext in (".txt", ".text"):
        return chunk_text(path)
    else:
        # Try as plain text
        return chunk_text(path)


def ingest_folder(folder_path: str) -> list:
    """Ingest all supported documents from a folder."""
    supported_exts = {".pdf", ".md", ".markdown", ".txt", ".text"}
    all_chunks = []

    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in supported_exts:
            print(f"  Ingesting: {file_path.name}")
            chunks = ingest_document(str(file_path))
            all_chunks.extend(chunks)
            print(f"    -> {len(chunks)} chunks extracted")

    return all_chunks


if __name__ == "__main__":
    import json

    pdf_folder = Path(__file__).parent / "pdfs"
    print(f"Ingesting documents from: {pdf_folder}")
    print("=" * 60)

    chunks = ingest_folder(str(pdf_folder))

    print(f"\nTotal chunks: {len(chunks)}")
    print("\nSample chunks:")
    for chunk in chunks[:5]:
        print(f"\n  [{chunk['chunk_id']}] {chunk['section_title']} (p.{chunk['page_number']})")
        print(f"    {chunk['text'][:150]}...")

    # Save chunks for inspection
    with open("pdf_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\nChunks saved to pdf_chunks.json")
