from archipelago.ingestion._pdf_base import *  # noqa: F403
from archipelago.ingestion.pdf_utils import *  # noqa: F403

def chunk_markdown(md_path: str, max_chunk_chars: int = 1600,
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
                            "page_start": 0,
                            "page_end": 0,
                            "text": sub
                        })
            else:
                chunk_counter += 1
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "section_title": current_title,
                    "page_number": 0,
                    "page_start": 0,
                    "page_end": 0,
                    "text": part
                })
        i += 1

    return chunks

def chunk_text(txt_path: str, max_chunk_chars: int = 1600,
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
                "page_start": 0,
                "page_end": 0,
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
            "page_start": 0,
            "page_end": 0,
            "text": current_text.strip()
        })

    return chunks

def chunk_docx(docx_path: str, max_chunk_chars: int = 1600,
               min_chunk_chars: int = 200) -> list:
    """Split a Word .docx file by heading styles into chunks."""
    if docx is None:
        raise ImportError("python-docx is required for .docx files. "
                          "Install with: pip install python-docx")
    document = docx.Document(docx_path)
    doc_id = Path(docx_path).name

    sections = []
    current_title = "Introduction"
    current_text = ""
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading") or style == "title":
            if current_text.strip():
                sections.append((current_title, current_text.strip()))
            current_title = text
            current_text = ""
        else:
            current_text += text + "\n"
    if current_text.strip():
        sections.append((current_title, current_text.strip()))

    chunks = []
    counter = 0
    for title, text in sections:
        for start in range(0, len(text), max_chunk_chars):
            sub = text[start:start + max_chunk_chars].strip()
            if sub and len(sub) >= min_chunk_chars:
                counter += 1
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"chunk_{counter:03d}",
                    "section_title": title,
                    "page_number": 0,
                    "page_start": 0,
                    "page_end": 0,
                    "text": sub,
                })
    return chunks
