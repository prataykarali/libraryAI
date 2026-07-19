from archipelago.ingestion._pdf_base import *  # noqa: F403
from archipelago.ingestion.pdf_chunk import chunk_pdf
from archipelago.ingestion.pdf_formats import chunk_markdown, chunk_text, chunk_docx
from archipelago.ingestion.pdf_utils import *  # noqa: F403
# import * skips leading-underscore names — bind the helpers we call directly.
from archipelago.ingestion.pdf_utils import (
    _compute_doc_hash,
    _extract_title_from_pdf,
    _extract_edition_from_pdf,
    _extract_page_labels,
    annotate_chunks,
)

def ingest_document(path: str, max_pages: int = None) -> list:
    """Universal document ingester. Routes to the right chunker by file extension.

    Every returned chunk carries a sanitized `section_title` and a `chunk_kind`.
    Also computes document-level metadata: doc_hash, page_count, title, edition, page_label_map.
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    doc_hash = None
    page_count = None
    title = None
    edition = None
    page_label_map = None

    if ext == ".pdf":
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
        
        # Open once to extract document metadata
        doc = fitz.open(path)
        doc_hash = _compute_doc_hash(path)
        page_count = doc.page_count
        title = _extract_title_from_pdf(doc)
        edition = _extract_edition_from_pdf(doc)
        page_label_map = _extract_page_labels(doc)
        doc.close()
        
        chunks = chunk_pdf(path, max_pages=max_pages)
    elif ext in (".md", ".markdown"):
        doc_hash = _compute_doc_hash(path)
        chunks = chunk_markdown(path)
    elif ext in (".txt", ".text"):
        doc_hash = _compute_doc_hash(path)
        chunks = chunk_text(path)
    elif ext == ".docx":
        doc_hash = _compute_doc_hash(path)
        chunks = chunk_docx(path)
    elif ext == ".doc":
        # Legacy binary .doc is NOT readable by python-docx (which only handles
        # the .docx OOXML format). Skip with a clear message rather than routing
        # it to chunk_docx (fails) or chunk_text (reads binary garbage).
        raise ValueError("legacy .doc not supported, convert to .docx")
    else:
        doc_hash = _compute_doc_hash(path)
        chunks = chunk_text(path)  # best-effort plain text

    # Annotate all chunks with chunk_kind
    chunks = annotate_chunks(chunks)
    
    # Attach document-level metadata to each chunk for downstream storage
    for chunk in chunks:
        chunk["doc_hash"] = doc_hash
        chunk["page_count"] = page_count
        chunk["doc_title"] = title
        chunk["edition"] = edition
        chunk["page_label_map"] = page_label_map

    return chunks

def ingest_folder(folder_path: str, max_pages: int = None) -> list:
    """Ingest all supported documents from a folder and its subfolders.

    max_pages: optional UNIFORM page cap applied to every PDF (not per-file).
    None reads whole documents so any source ingests fully.
    """
    supported_exts = {".pdf", ".md", ".markdown", ".txt", ".text", ".docx"}
    all_chunks = []

    folder = Path(folder_path)
    for file_path in sorted(folder.rglob("*")):
        if file_path.suffix.lower() in supported_exts:
            display_name = file_path.relative_to(folder)
            note = f" (first {max_pages} pages)" if max_pages else ""
            print(f"  Ingesting: {display_name}{note}")
            try:
                chunks = ingest_document(str(file_path), max_pages=max_pages)
            except Exception as exc:
                print(f"    -> skipped: {exc}")
                continue
            for chunk in chunks:
                chunk["doc_id"] = str(display_name).replace(os.sep, "/")
            all_chunks.extend(chunks)
            kinds = Counter(c.get("chunk_kind", "?") for c in chunks)
            prose = kinds.get("prose", 0)
            print(f"    -> {len(chunks)} chunks ({prose} prose, "
                  f"{len(chunks) - prose} non-prose)")

    return all_chunks
