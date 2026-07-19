from archipelago.ingestion._pdf_base import (  # noqa: F401
    fitz, docx, hashlib, json, re, os, Path, Counter,
    _CITATION_RE, _EMAIL_RE, _ARXIV_HEADER_RE, _MATH_CHARS,
)

def _nonspace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())

def _numeric_token_fraction(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if re.fullmatch(r'[-+]?\d[\d.,%/±]*', t))
    return numeric / len(tokens)

def _alpha_ratio(text: str) -> float:
    ns = _nonspace(text)
    if not ns:
        return 0.0
    return sum(1 for ch in ns if ch.isalpha()) / len(ns)

def sanitize_section_title(title: str) -> str:
    """Drop running headers / table-row noise that the heading heuristic mislabels."""
    t = (title or "").strip()
    if not t:
        return ""
    if _ARXIV_HEADER_RE.search(t):
        return ""
    # A "title" that is mostly numbers is a misdetected table row.
    if _numeric_token_fraction(t) > 0.3 or _alpha_ratio(t) < 0.4:
        return ""
    return t

def classify_chunk_kind(text: str, section_title: str = "") -> str:
    """Label a chunk as prose | reference | table | frontmatter | math."""
    body = (text or "").strip()
    if not body:
        return "empty"

    title_l = (section_title or "").lower()
    if title_l.startswith(("reference", "bibliograph", "works cited")):
        return "reference"

    lines = [ln for ln in body.splitlines() if ln.strip()]
    if lines:
        # Check if it looks like a real bibliography listing (lines start with citation keys e.g. [1] or 1. )
        starts_with_cite = sum(1 for ln in lines if re.match(r'^\s*(\[\d+\]|\d+\.\s)', ln))
        if starts_with_cite / len(lines) > 0.6:
            return "reference"
        # Or if the density of citations is extremely high (e.g. > 85% of lines have citations)
        cite_lines = sum(1 for ln in lines if _CITATION_RE.search(ln))
        if cite_lines / len(lines) > 0.85:
            return "reference"

    # Front-matter: author lists / affiliations (multiple emails, few sentences)
    if len(_EMAIL_RE.findall(body)) >= 2 and body.count(".") < 12:
        return "frontmatter"

    # Table / results block: number-dominated
    if _numeric_token_fraction(body) > 0.28:
        return "table"

    # Bare math: symbol-heavy, letter-poor
    math_chars = sum(1 for ch in body if ch in _MATH_CHARS)
    ns = _nonspace(body)
    if ns and (math_chars / len(ns) > 0.12 or _alpha_ratio(body) < 0.55):
        return "math"

    return "prose"

def annotate_chunks(chunks: list) -> list:
    """Sanitize section titles and attach a chunk_kind to every chunk in place."""
    for c in chunks:
        c["section_title"] = sanitize_section_title(c.get("section_title", ""))
        c["chunk_kind"] = classify_chunk_kind(c.get("text", ""), c["section_title"])
    return chunks

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

def _extract_page_labels(doc) -> dict:
    """
    Extract PDF page labels (printed page numbers) and build a mapping
    from printed label to physical PDF page number (1-indexed).
    
    Returns a dict like {"1": 1, "2": 2, "iii": 3, "112": 112, ...}
    or empty dict if no page labels exist.
    """
    label_map = {}
    try:
        labels = doc.get_page_labels()
        if labels:
            for i, label in enumerate(labels):
                if label and label.strip():
                    label_map[label.strip()] = i + 1  # 1-indexed PDF page
    except Exception:
        pass
    return label_map

def _compute_doc_hash(file_path: str) -> str:
    """Compute SHA-256 hash of the file contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def _extract_title_from_pdf(doc) -> str:
    """Extract document title from PDF metadata or first heading."""
    # Try PDF metadata first
    metadata = doc.metadata
    if metadata and metadata.get("title"):
        title = metadata["title"].strip()
        if title and len(title) > 1:
            return title
    
    # Fall back to first substantial heading
    for page_num in range(min(3, doc.page_count)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) > 10 and len(text) < 200:
                        # Likely a title if it's the first large text
                        if span["size"] > 14:
                            return text
    return ""

def _extract_edition_from_pdf(doc) -> str:
    """Try to extract edition info from PDF metadata or early pages."""
    metadata = doc.metadata
    if metadata:
        # Check subject, keywords, or other metadata fields
        for key in ["subject", "keywords", "creator", "producer"]:
            val = metadata.get(key, "")
            if val and re.search(r'\b(\d+(?:st|nd|rd|th)\s+edition|edition\s+\d+)\b', val, re.IGNORECASE):
                return val
    return ""
