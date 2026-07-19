from archipelago.ingestion._pdf_base import (  # noqa: F401
    fitz, docx, hashlib, json, re, os, Path, Counter,
    _CITATION_RE, _EMAIL_RE, _ARXIV_HEADER_RE, _MATH_CHARS,
)
from archipelago.ingestion.pdf_utils import (
    _nonspace, _numeric_token_fraction, _alpha_ratio,
    sanitize_section_title, classify_chunk_kind, annotate_chunks,
    _detect_heading, _get_body_font_size, _extract_page_labels,
    _compute_doc_hash, _extract_title_from_pdf, _extract_edition_from_pdf,
)
from archipelago.ingestion.pdf_utils import *  # noqa: F403

def chunk_pdf(pdf_path: str, max_chunk_chars: int = 1600,
              min_chunk_chars: int = 200, max_pages: int = None) -> list:
    """
    Split a PDF into section-aware chunks with provenance metadata.

    Returns list of dicts:
        {doc_id, chunk_id, section_title, page_number, text,
         text_offset_start, text_offset_end, block_x, block_y, block_w, block_h}

    max_pages: if set, only the first N pages are read (used to cap huge books
    like Math-for-ML to their first couple of chapters).
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

    doc = fitz.open(pdf_path)
    doc_id = Path(pdf_path).name
    body_size = _get_body_font_size(doc)
    page_cap = doc.page_count if max_pages is None else min(max_pages, doc.page_count)

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
    current_blocks = []

    for page_num in range(page_cap):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_text_accumulator = ""  # Track character offset within the page

        for block in blocks:
            if "lines" not in block:
                continue
            block_text = ""
            is_heading = False
            block_bbox = block.get("bbox", None)  # (x0, y0, x1, y1)

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

            # Record text offset within the page (character position)
            text_offset_start = len(page_text_accumulator)
            page_text_accumulator += block_text + "\n"
            text_offset_end = len(page_text_accumulator)

            # New section detected
            if is_heading and len(block_text) < 150 and len(block_text) > 2:
                # Save previous section if it has content
                total_chars = sum(len(b["text"]) for b in current_blocks)
                if total_chars >= min_chunk_chars:
                    sections.append({
                        "section_title": current_section,
                        "blocks": current_blocks
                    })
                elif total_chars > 0:
                    if sections:
                        # Merge tiny section into previous
                        sections[-1]["blocks"].extend(current_blocks)
                    else:
                        # No previous section to merge into, keep it as its own section
                        sections.append({
                            "section_title": current_section,
                            "blocks": current_blocks
                        })

                current_section = block_text.strip()
                current_blocks = []
            else:
                block_info = {
                    "text": block_text,
                    "page_number": page_num + 1,
                    "text_offset_start": text_offset_start,
                    "text_offset_end": text_offset_end,
                }
                if block_bbox:
                    block_info["block_x"] = block_bbox[0]
                    block_info["block_y"] = block_bbox[1]
                    block_info["block_w"] = block_bbox[2] - block_bbox[0]
                    block_info["block_h"] = block_bbox[3] - block_bbox[1]
                current_blocks.append(block_info)

    # Don't forget the last section
    total_chars = sum(len(b["text"]) for b in current_blocks)
    if total_chars >= min_chunk_chars:
        sections.append({
            "section_title": current_section,
            "blocks": current_blocks
        })
    elif total_chars > 0:
        if sections:
            sections[-1]["blocks"].extend(current_blocks)
        else:
            sections.append({
                "section_title": current_section,
                "blocks": current_blocks
            })

    doc.close()

    # Now split oversized sections into sub-chunks
    chunks = []
    chunk_counter = 0

    for section in sections:
        blocks = section["blocks"]
        current_chunk_text = ""
        current_page_number = None
        current_text_offset_start = None
        current_text_offset_end = None
        current_block_x = None
        current_block_y = None
        current_block_w = None
        current_block_h = None

        # Track char mass per page so page_number is the *dominant* page of the
        # chunk (not just the section start / first block) — fixes bibliography
        # and multi-page section provenance noise.
        page_char_mass = {}

        def _dominant_page(fallback):
            if not page_char_mass:
                return fallback
            return max(page_char_mass.items(), key=lambda kv: kv[1])[0]

        def _flush_chunk():
            nonlocal chunk_counter, current_chunk_text, current_page_number
            nonlocal current_page_start, current_page_end
            nonlocal current_text_offset_start, current_text_offset_end
            nonlocal current_block_x, current_block_y, current_block_w, current_block_h
            nonlocal page_char_mass
            text = current_chunk_text.strip()
            if not text:
                return
            chunk_counter += 1
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"chunk_{chunk_counter:03d}",
                "section_title": section["section_title"],
                "page_number": _dominant_page(current_page_number),
                "page_start": current_page_start,
                "page_end": current_page_end,
                "text": text,
                "text_offset_start": current_text_offset_start,
                "text_offset_end": current_text_offset_end,
                "block_x": current_block_x,
                "block_y": current_block_y,
                "block_w": current_block_w,
                "block_h": current_block_h,
            })
            current_chunk_text = ""
            current_page_number = None
            current_page_start = None
            current_page_end = None
            current_text_offset_start = None
            current_text_offset_end = None
            current_block_x = None
            current_block_y = None
            current_block_w = None
            current_block_h = None
            page_char_mass = {}

        for block in blocks:
            b_text = block["text"]
            b_page = block["page_number"]
            b_offset_start = block.get("text_offset_start")
            b_offset_end = block.get("text_offset_end")
            b_x = block.get("block_x")
            b_y = block.get("block_y")
            b_w = block.get("block_w")
            b_h = block.get("block_h")

            if not b_text.strip():
                continue

            if current_page_number is None:
                current_page_number = b_page
                current_page_start = b_page
                current_page_end = b_page
                current_text_offset_start = b_offset_start
                current_block_x = b_x
                current_block_y = b_y
                current_block_w = b_w
                current_block_h = b_h

            if len(current_chunk_text) + len(b_text) > max_chunk_chars and current_chunk_text:
                _flush_chunk()
                current_chunk_text = b_text + "\n"
                current_page_number = b_page
                current_page_start = b_page
                current_page_end = b_page
                current_text_offset_start = b_offset_start
                current_text_offset_end = b_offset_end
                current_block_x = b_x
                current_block_y = b_y
                current_block_w = b_w
                current_block_h = b_h
                page_char_mass = {b_page: len(b_text)}
            else:
                current_chunk_text += b_text + "\n"
                current_page_end = b_page
                current_text_offset_end = b_offset_end
                page_char_mass[b_page] = page_char_mass.get(b_page, 0) + len(b_text)

        _flush_chunk()

    return chunks
