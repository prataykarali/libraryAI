import os
import fitz
import pytest
from pdf_ingestion import chunk_pdf

@pytest.mark.unit
def test_multipage_chunk_preserves_correct_spans(tmp_path):
    pdf_path = os.path.join(tmp_path, "multipage.pdf")
    doc = fitz.open()
    
    # Page 1 (index 0, physical 1)
    p1 = doc.new_page()
    p1.insert_text((50, 50), "This is page one content.")
    
    # Page 2 (index 1, physical 2)
    p2 = doc.new_page()
    p2.insert_text((50, 50), "This is page two content.")
    
    # Page 3 (index 2, physical 3)
    p3 = doc.new_page()
    p3.insert_text((50, 50), "This is page three content that spans multiple pages.")
    
    # Page 4 (index 3, physical 4)
    p4 = doc.new_page()
    p4.insert_text((50, 50), "This is page four content continuing the same section.")
    
    # Page 5 (index 4, physical 5)
    p5 = doc.new_page()
    p5.insert_text((50, 50), "This is page five content completing the multipage chunk.")
    
    doc.save(pdf_path)
    doc.close()
    
    # We want a very large chunk size so it merges pages 3, 4, 5
    # The first 2 pages should form a chunk because they might be separate,
    # or the sectioning will merge them. Let's see: we have no headings, so they are all in "Introduction" section.
    # With a large max_chunk_chars, all pages 1 to 5 will merge into one single chunk!
    # Let's verify that the chunk's page_start is 1 and page_end is 5.
    chunks = chunk_pdf(pdf_path, max_chunk_chars=5000, min_chunk_chars=10)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page_start"] == 1
    assert chunk["page_end"] == 5
    
    # Now let's test a case where we have a heading on page 3.
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Introduction text.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "More intro text.")
    
    # Page 3 heading
    p3 = doc.new_page()
    p3.insert_text((50, 50), "Chapter 1", fontsize=16)  # Bold/Heading
    p3.insert_text((50, 100), "This starts Chapter 1 on page three.")
    
    p4 = doc.new_page()
    p4.insert_text((50, 50), "Continuing Chapter 1 on page four.")
    
    p5 = doc.new_page()
    p5.insert_text((50, 50), "Finishing Chapter 1 on page five.")
    
    pdf_path_headings = os.path.join(tmp_path, "headings_multipage.pdf")
    doc.save(pdf_path_headings)
    doc.close()
    
    # The first two pages will be in Introduction.
    # Page 3, 4, 5 will start a new section "Chapter 1" and merge into one chunk.
    chunks = chunk_pdf(pdf_path_headings, max_chunk_chars=5000, min_chunk_chars=10)
    
    # Find the chunk for Chapter 1
    ch1_chunks = [c for c in chunks if "Chapter 1" in c["section_title"]]
    assert len(ch1_chunks) == 1
    ch1_chunk = ch1_chunks[0]
    assert ch1_chunk["page_start"] == 3
    assert ch1_chunk["page_end"] == 5

@pytest.mark.unit
def test_text_offsets_within_page(tmp_path):
    """Offsets must index real page text (substring match), not merely satisfy start < end."""
    pdf_path = os.path.join(tmp_path, "offsets.pdf")
    doc = fitz.open()
    page = doc.new_page()

    phrases = [
        "Paragraph one content here.",
        "Paragraph two text is located here.",
        "Paragraph three holds other details.",
    ]
    page.insert_text((50, 50), phrases[0])
    page.insert_text((50, 100), phrases[1])
    page.insert_text((50, 150), phrases[2])

    doc.save(pdf_path)
    doc.close()

    chunks = chunk_pdf(pdf_path, max_chunk_chars=2000, min_chunk_chars=10)
    assert len(chunks) == 1
    chunk = chunks[0]

    # Rebuild page text exactly as pdf_ingestion.chunk_pdf does (block_text + "\\n").
    doc = fitz.open(pdf_path)
    blocks = doc[0].get_text("dict")["blocks"]
    page_text = ""
    for block in blocks:
        if "lines" not in block:
            continue
        block_text = ""
        for line in block["lines"]:
            line_text = ""
            for span in line["spans"]:
                line_text += span["text"]
            block_text += line_text.strip() + " "
        block_text = block_text.strip()
        if block_text:
            page_text += block_text + "\n"
    doc.close()

    start = chunk["text_offset_start"]
    end = chunk["text_offset_end"]

    assert isinstance(start, int) and isinstance(end, int)
    assert 0 <= start < end <= len(page_text)

    # Core contract: offsets slice into the real page text at the chunk's content.
    sliced = page_text[start:end]
    assert sliced.strip() == chunk["text"].strip()
    for phrase in phrases:
        assert phrase in sliced, f"phrase not found at offsets [{start}:{end}]: {phrase!r}"
        # Each phrase must also appear at a concrete index within the sliced region.
        phrase_at = page_text.find(phrase)
        assert phrase_at >= 0
        assert start <= phrase_at < end
        assert page_text[phrase_at : phrase_at + len(phrase)] == phrase

@pytest.mark.unit
def test_section_title_attached_to_chunks(tmp_path):
    pdf_path = os.path.join(tmp_path, "sections.pdf")
    doc = fitz.open()
    
    # Page 1: Introduction
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Introduction Heading", fontsize=16)
    p1.insert_text((50, 100), "This is introduction text content.")
    
    # Page 2: Chapter 1
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Chapter 1: Deep Learning", fontsize=16)
    p2.insert_text((50, 100), "This is deep learning content.")
    
    doc.save(pdf_path)
    doc.close()
    
    # Set min_chunk_chars small to ensure they don't get merged out
    chunks = chunk_pdf(pdf_path, max_chunk_chars=2000, min_chunk_chars=5)
    
    intro_chunks = [c for c in chunks if "introduction" in c["text"].lower()]
    dl_chunks = [c for c in chunks if "deep learning" in c["text"].lower()]
    
    assert len(intro_chunks) > 0
    assert len(dl_chunks) > 0
    
    for c in intro_chunks:
        assert "Introduction" in c["section_title"]
        
    for c in dl_chunks:
        assert "Chapter 1" in c["section_title"]
