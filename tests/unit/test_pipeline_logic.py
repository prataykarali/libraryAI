import unittest
import sys
import pytest
from pathlib import Path

# Add parent directory to path so we can import from libraryAI
sys.path.append(str(Path(__file__).resolve().parent))

from okf_pipeline import (
    break_global_cycles,
    prune_unresolved_references,
    merge_duplicate_results,
    canonicalize_name,
    build_canonical_map,
    apply_canonicalization
)

@pytest.mark.unit
class TestPipelineLogic(unittest.TestCase):

    def test_break_global_cycles_simple(self):
        """Test cycle breaking with a simple direct cycle: A -> B -> A."""
        # A has difficulty "foundational" (rank 1), unlocks B.
        # B has difficulty "intermediate" (rank 2), unlocks A.
        # Cycle is A -> B -> A.
        # Edges:
        # A -> B (unlock from A): score = diff(B) - diff(A) = 2 - 1 = 1
        # B -> A (unlock from B): score = diff(A) - diff(B) = 1 - 2 = -1 (weaker)
        # So B -> A should be broken (B's unlock of A is removed).
        okf_results = [
            {
                "concept_name": "A",
                "difficulty": "foundational",
                "prerequisites": [],
                "unlocks": ["B"]
            },
            {
                "concept_name": "B",
                "difficulty": "intermediate",
                "prerequisites": [],
                "unlocks": ["A"]
            }
        ]
        
        removed = break_global_cycles(okf_results)
        self.assertEqual(removed, 1)
        self.assertNotIn("A", [x.title() for x in okf_results[1]["unlocks"]])
        self.assertIn("B", okf_results[0]["unlocks"])

    def test_break_global_cycles_with_placeholders(self):
        """Test cycle breaking when the cycle passes through an unresolved placeholder node."""
        # Cycle: A -> P -> B -> A. P is placeholder (unresolved).
        # A unlocks P (A -> P).
        # B has prerequisite P (P -> B).
        # A has prerequisite B (B -> A).
        # Scores:
        # A -> P: diff(P) - diff(A) = 2 - 1 = 1
        # P -> B: diff(B) - diff(P) = 2 - 2 = 0
        # B -> A: diff(A) - diff(B) = 1 - 2 = -1 (weakest, should be broken)
        # So B -> A should be broken (remove B from A's prerequisites).
        okf_results = [
            {
                "concept_name": "A",
                "difficulty": "foundational",
                "prerequisites": ["B"],
                "unlocks": ["P"]
            },
            {
                "concept_name": "B",
                "difficulty": "intermediate",
                "prerequisites": ["P"],
                "unlocks": []
            }
        ]
        
        removed = break_global_cycles(okf_results)
        self.assertEqual(removed, 1)
        self.assertEqual(okf_results[0]["prerequisites"], [])
        self.assertEqual(okf_results[0]["unlocks"], ["P"])
        self.assertEqual(okf_results[1]["prerequisites"], ["P"])

    def test_prune_unresolved_references_does_not_prune(self):
        """Test that prune_unresolved_references does not prune unresolved references."""
        # Unresolved references should not be dropped, keeping placeholder nodes intact.
        okf_results = [
            {
                "concept_name": "A",
                "prerequisites": ["UnresolvedPrereq"],
                "unlocks": ["UnresolvedUnlock"],
                "related_to": [{"concept": "UnresolvedRelated", "relation": "uses"}]
            }
        ]
        stats = prune_unresolved_references(okf_results)
        self.assertEqual(stats, {"prerequisites": 0, "unlocks": 0, "related": 0, "self_references": 0})
        self.assertIn("UnresolvedPrereq", okf_results[0]["prerequisites"])
        self.assertIn("UnresolvedUnlock", okf_results[0]["unlocks"])
        self.assertEqual(okf_results[0]["related_to"][0]["concept"], "UnresolvedRelated")

    def test_merge_duplicate_results(self):
        """Test merge_duplicate_results merges entries and accumulates all provenance sources."""
        okf_results = [
            {
                "concept_name": "A",
                "summary": "Short desc",
                "prerequisites": ["B"],
                "unlocks": ["C"],
                "doc_id": "doc1.pdf",
                "chunk_id": "chunk1",
                "page_number": 1,
                "section_title": "Sec 1",
                "source_passage": "passage 1"
            },
            {
                "concept_name": "A",
                "summary": "Longer and richer description of A",
                "prerequisites": ["D"],
                "unlocks": [],
                "doc_id": "doc2.pdf",
                "chunk_id": "chunk2",
                "page_number": 2,
                "section_title": "Sec 2",
                "source_passage": "passage 2"
            }
        ]
        
        merged, removed = merge_duplicate_results(okf_results)
        self.assertEqual(len(merged), 1)
        self.assertEqual(removed, 1)
        
        merged_item = merged[0]
        # Should pick the longer summary
        self.assertEqual(merged_item["summary"], "Longer and richer description of A")
        # Should union prerequisites and unlocks
        self.assertCountEqual(merged_item["prerequisites"], ["B", "D"])
        self.assertCountEqual(merged_item["unlocks"], ["C"])
        
        # Should union sources correctly
        self.assertEqual(len(merged_item["sources"]), 2)
        doc_ids = [s["doc_id"] for s in merged_item["sources"]]
        self.assertCountEqual(doc_ids, ["doc1.pdf", "doc2.pdf"])

    def test_canonicalize_name(self):
        """Test concept name canonicalization."""
        self.assertEqual(canonicalize_name("some_concept_name"), "Some Concept Name")
        self.assertEqual(canonicalize_name("concept (abbrev)"), "Concept")
        self.assertEqual(canonicalize_name("concept name."), "Concept Name")

if __name__ == "__main__":
    unittest.main()
