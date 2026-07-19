"""Unit tests for library route detection and library render helpers."""
from archipelago.inference.routing import _detect_library_intent
from archipelago.inference.synthesis import (
    prettify_doc_title, render_library_books,
    render_library_chapters, render_library_chapter_lookup,
)


def _intent(query):
    r = _detect_library_intent(query)
    return r["intent"] if r else None


class TestLibraryIntentDetection:
    def test_which_chapters_contain_concept(self):
        assert _intent("which chapters of Math for ML contain PCA") == "library_chapter_lookup"

    def test_which_chapter_discusses_concept(self):
        assert _intent("which chapter of the book discusses backpropagation") == "library_chapter_lookup"

    def test_chapters_of_book_that_cover_concept(self):
        assert _intent("chapters of Math for ML that cover PCA") == "library_chapter_lookup"

    def test_where_in_book_is_concept_discussed(self):
        assert _intent("where in Math for ML is PCA discussed") == "library_chapter_lookup"

    def test_plain_chapters_of_book(self):
        assert _intent("what are the chapters of Math for ML") == "library_chapters"

    def test_show_chapters(self):
        assert _intent("show chapters of the attention paper") == "library_chapters"

    def test_table_of_contents(self):
        assert _intent("table of contents of the BERT paper") == "library_chapters"

    def test_book_recommendation(self):
        assert _intent("suggest top 5 books for linear algebra") == "library_books"


class TestPrettifyDocTitle:
    def test_known_doc_with_path(self):
        assert prettify_doc_title("textbooks/Deisenroth_Math_For_ML.pdf") == \
            "Mathematics for Machine Learning (Deisenroth et al.)"

    def test_known_doc_bare(self):
        assert prettify_doc_title("Hu2021_LoRA.pdf") == \
            "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)"

    def test_known_markdown_doc(self):
        assert prettify_doc_title("AI_ML_Archipelago_Corpus_Seed.md") == \
            "AI/ML Corpus Seed Syllabus"

    def test_unknown_doc_generic_cleanup(self):
        assert prettify_doc_title("papers/Some_Unknown_Doc.pdf") == "Some Unknown Doc"

    def test_already_human_title_untouched(self):
        assert prettify_doc_title("Mathematics for Machine Learning") == \
            "Mathematics for Machine Learning"

    def test_empty(self):
        assert prettify_doc_title("") == ""


class TestRenderFunctions:
    def test_books_render_uses_pretty_title_keeps_doc_id(self):
        books = [{
            "id": "textbooks/Deisenroth_Math_For_ML.pdf",
            "title": "textbooks/Deisenroth_Math_For_ML.pdf",
            "mentions": 12,
            "matched": ["PCA"],
        }]
        out = render_library_books("PCA", books)
        assert "Mathematics for Machine Learning (Deisenroth et al.)" in out
        assert "`textbooks/Deisenroth_Math_For_ML.pdf`" in out

    def test_chapter_lookup_render_pretty_title(self):
        out = render_library_chapter_lookup(
            "textbooks/Deisenroth_Math_For_ML.pdf", "PCA",
            [{"section_title": "10 Dimensionality Reduction with PCA", "page_number": 317}],
        )
        assert "Mathematics for Machine Learning (Deisenroth et al.)" in out
        assert "Page 317" in out

    def test_chapters_render_small_list_unfiltered(self):
        chapters = [{"section_title": f"{i} Chapter", "page_number": i} for i in range(1, 11)]
        out = render_library_chapters("Devlin2018_BERT.pdf", chapters)
        assert "BERT (Devlin et al., 2018)" in out
        assert len([l for l in out.splitlines() if l.strip().startswith(tuple("0123456789"))]) == 10

    def test_chapters_render_large_list_filters_to_top_level(self):
        chapters = [{"section_title": "Foreword", "page_number": 1}]
        for i in range(1, 13):
            chapters.append({"section_title": f"{i} Chapter Title", "page_number": i * 25})
            for j in range(1, 10):
                chapters.append({"section_title": f"{i}.{j} Subsection", "page_number": i * 25 + j})
        assert len(chapters) > 40
        out = render_library_chapters("Deisenroth_Math_For_ML.pdf", chapters)
        assert "showing 13 top-level chapters of 121 sections" in out
        assert "1.1 Subsection" not in out
        assert "Foreword" in out
        assert "12 Chapter Title" in out

    def test_chapters_render_large_list_no_top_level_truncates(self):
        chapters = [{"section_title": f"Weird heading {i}", "page_number": i} for i in range(1, 61)]
        out = render_library_chapters("SomeBook.pdf", chapters)
        assert "showing first 40 of 60 sections" in out
