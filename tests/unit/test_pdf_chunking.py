import os
import fitz
import pytest
from pdf_ingestion import (
    chunk_pdf,
    classify_chunk_kind,
    sanitize_section_title,
    chunk_markdown,
    chunk_text
)

@pytest.mark.unit
def test_chunk_pdf_preserves_short_sections(tmp_path):
    pdf_path = os.path.join(tmp_path, "test_doc.pdf")
    doc = fitz.open()
    
    # Page 1: Short Introduction
    page1 = doc.new_page()
    page1.insert_text((50, 50), "This is a short intro. It has very few chars.", fontsize=10)
    
    # Page 2: Chapter 1 Heading and some content
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Chapter 1", fontsize=16)  # Heading
    page2.insert_text((50, 100), "This is chapter 1 body content. It spans a little bit.", fontsize=10)
    
    # Page 3: Chapter 2 Heading and content
    page3 = doc.new_page()
    page3.insert_text((50, 50), "Chapter 2", fontsize=16)  # Heading
    page3.insert_text((50, 100), "This is chapter 2 content.", fontsize=10)
    
    doc.save(pdf_path)
    doc.close()
    
    # Let's run chunk_pdf with min_chunk_chars = 100 (which is larger than the intro length of ~45 chars)
    chunks = chunk_pdf(pdf_path, min_chunk_chars=100)
    
    # The short intro should not be discarded.
    intro_chunks = [c for c in chunks if "intro" in c["text"].lower()]
    assert len(intro_chunks) > 0, "Introduction chunk was discarded!"

@pytest.mark.unit
def test_sub_chunk_page_assignments(tmp_path):
    pdf_path = os.path.join(tmp_path, "test_pages.pdf")
    doc = fitz.open()
    
    # Page 1: Chapter 1 Heading and page 1 content
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Chapter 1", fontsize=16)
    page1.insert_text((50, 100), "This is page 1 content of Chapter 1.", fontsize=10)
    
    # Page 2: More Chapter 1 content
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is page 2 content of Chapter 1.", fontsize=10)
    
    doc.save(pdf_path)
    doc.close()
    
    # We want max_chunk_chars = 40, so it will split into multiple sub-chunks
    # We set min_chunk_chars = 5 so it allows small chunks
    chunks = chunk_pdf(pdf_path, max_chunk_chars=40, min_chunk_chars=5)
    
    p1_chunks = [c for c in chunks if "page 1" in c["text"]]
    p2_chunks = [c for c in chunks if "page 2" in c["text"]]
    
    assert len(p1_chunks) > 0
    assert len(p2_chunks) > 0
    
    for c in p1_chunks:
        assert c["page_number"] == 1
        
    for c in p2_chunks:
        assert c["page_number"] == 2

@pytest.mark.unit
def test_chunk_pdf_no_headings_short(tmp_path):
    pdf_path = os.path.join(tmp_path, "test_short_no_headings.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a very short PDF with no headings.", fontsize=10)
    doc.save(pdf_path)
    doc.close()
    
    # Running chunk_pdf with min_chunk_chars = 100 (which is larger than the text length of ~42 chars)
    chunks = chunk_pdf(pdf_path, min_chunk_chars=100)
    
    # Short PDF without headings should not be discarded
    assert len(chunks) > 0, "Short PDF without headings was completely discarded!"

@pytest.mark.unit
def test_chunk_kind_classification():
    # Test empty text
    assert classify_chunk_kind("") == "empty"
    
    # Test reference classification by section title
    assert classify_chunk_kind("Some text content", "References") == "reference"
    assert classify_chunk_kind("Some text content", "Bibliography") == "reference"
    assert classify_chunk_kind("Some text content", "Works Cited") == "reference"
    
    # Test reference classification by citation lists
    bib_text = "1. Smith et al. (2020)\n2. Doe et al. (2021)\n3. Johnson et al. (2022)"
    assert classify_chunk_kind(bib_text, "Introduction") == "reference"
    
    bib_text_bracket = "[1] Smith et al.\n[2] Doe et al.\n[3] Johnson et al."
    assert classify_chunk_kind(bib_text_bracket, "Introduction") == "reference"
    
    # Test frontmatter classification (emails, low sentence count)
    frontmatter_text = "Author A (a@example.com), Author B (b@example.com). Affiliation details."
    assert classify_chunk_kind(frontmatter_text) == "frontmatter"
    
    # Test table classification (number-dominated)
    table_text = "1.5 2.3 4.5 10.1\n99.2 18.3 4.7 0.05\n3.1 5.9 8.2 12.1"
    assert classify_chunk_kind(table_text) == "table"
    
    # Test math classification (math characters, low alpha ratio)
    math_text = r"f(x) = \int_a^b g(x) dx + \sum_{i=1}^n x_i"
    assert classify_chunk_kind(math_text) == "math"
    
    # Test prose classification (normal text)
    prose_text = "This is a regular paragraph containing standard English sentences. It has standard characters and normal structure."
    assert classify_chunk_kind(prose_text) == "prose"

@pytest.mark.unit
def test_section_title_sanitization():
    # Empty title
    assert sanitize_section_title("") == ""
    
    # arXiv header should be sanitized out
    assert sanitize_section_title("arXiv:2101.12345 [cs.CL]") == ""
    assert sanitize_section_title("Arxiv: 1234.5678") == ""
    
    # Number dominated title (mislabeled table row)
    assert sanitize_section_title("12.3 45.6 78.9 0.12") == ""
    
    # Normal section titles
    assert sanitize_section_title("Introduction") == "Introduction"
    assert sanitize_section_title("Section 1 Introduction to Deep Learning") == "Section 1 Introduction to Deep Learning"
    assert sanitize_section_title("Chapter 3: Methodology") == "Chapter 3: Methodology"

@pytest.mark.unit
def test_markdown_chunking(tmp_path):
    md_file = tmp_path / "test.md"
    content = """# Section 1
This is a paragraph in section 1. It is longer than the minimum character count of 200 characters. Let's make sure it has enough characters to be parsed. This is a paragraph in section 1. It is longer than the minimum character count of 200 characters. Let's make sure it has enough characters to be parsed.

## Section 2
This is section 2 text. It also needs to be long enough. This is section 2 text. It also needs to be long enough. This is section 2 text. It also needs to be long enough. This is section 2 text. It also needs to be long enough.
"""
    md_file.write_text(content, encoding="utf-8")
    
    chunks = chunk_markdown(str(md_file), max_chunk_chars=1000, min_chunk_chars=50)
    assert len(chunks) == 2
    assert chunks[0]["section_title"] == "Section 1"
    assert "section 1" in chunks[0]["text"]
    assert chunks[1]["section_title"] == "Section 2"
    assert "section 2" in chunks[1]["text"]

@pytest.mark.unit
def test_text_chunking(tmp_path):
    txt_file = tmp_path / "test.txt"
    content = """Paragraph 1 is here. It is a paragraph. It is long enough to be included. Paragraph 1 is here. It is a paragraph. It is long enough to be included. Paragraph 1 is here. It is a paragraph. It is long enough to be included.

Paragraph 2 is here. It is also long enough. Paragraph 2 is here. It is also long enough. Paragraph 2 is here. It is also long enough. Paragraph 2 is here. It is also long enough."""
    txt_file.write_text(content, encoding="utf-8")
    
    # Text chunking with a small max size to force multiple chunks
    chunks = chunk_text(str(txt_file), max_chunk_chars=200, min_chunk_chars=50)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["section_title"] == "Section"
        assert len(chunk["text"]) >= 50
