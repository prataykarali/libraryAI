import unittest
import os
import shutil
import sys
from pathlib import Path

# Add libraryAI to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from okf_pipeline import ingest_to_kuzu, export_graph

class TestKuzuProvenance(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_provenance.db"
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)

    def test_ingestion_and_provenance(self):
        # We will create mock results where concept B has concept A as prerequisite.
        # B is processed first, so A is first created as a placeholder.
        # Then A is processed as a main concept with rich details.
        mock_results = [
            {
                "concept_name": "Concept B",
                "concept_type": "technique",
                "difficulty": "intermediate",
                "summary": "This is concept B summary.",
                "prerequisites": ["Concept A"],
                "unlocks": [],
                "related_to": [],
                "tags": ["b-tag"],
                "doc_id": "doc_b.pdf",
                "chunk_id": "chunk_b1",
                "page_number": 12,
                "section_title": "Section B",
                "source_passage": "Detailed passage for B."
            },
            {
                "concept_name": "Concept A",
                "concept_type": "method",
                "difficulty": "foundational",
                "summary": "This is concept A summary.",
                "prerequisites": [],
                "unlocks": ["Concept B"],
                "related_to": [],
                "tags": ["a-tag"],
                "doc_id": "doc_a.pdf",
                "chunk_id": "chunk_a1",
                "page_number": 5,
                "section_title": "Section A",
                "source_passage": "Detailed passage for A."
            }
        ]

        conn, db, export = ingest_to_kuzu(mock_results, db_path=self.db_path)

        # Check export
        concepts = export["concepts"]
        
        # Verify Concept A is fully populated (not left as placeholder defaults!)
        self.assertIn("concept_a", concepts)
        self.assertEqual(concepts["concept_a"]["name"], "Concept A")
        self.assertEqual(concepts["concept_a"]["concept_type"], "method")
        self.assertEqual(concepts["concept_a"]["difficulty"], "foundational")
        self.assertEqual(concepts["concept_a"]["summary"], "This is concept A summary.")

        # Verify sources list is reconstructed correctly
        self.assertEqual(len(concepts["concept_a"]["sources"]), 1)
        self.assertEqual(concepts["concept_a"]["sources"][0]["doc_id"], "doc_a.pdf")
        self.assertEqual(concepts["concept_a"]["sources"][0]["chunk_id"], "chunk_a1")
        self.assertEqual(concepts["concept_a"]["sources"][0]["page_number"], 5)
        self.assertEqual(concepts["concept_a"]["sources"][0]["section_title"], "Section A")
        self.assertEqual(concepts["concept_a"]["sources"][0]["text_passage"], "Detailed passage for A.")

if __name__ == "__main__":
    unittest.main()
