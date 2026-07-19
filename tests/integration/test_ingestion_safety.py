import os
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import kuzu

from ingestion_jobs import JobStatus, JobStore
from ingestion_worker import IngestionWorker, GraphLock

# Mark as integration tests
pytestmark = pytest.mark.integration

@pytest.fixture
def tmp_store(tmp_path):
    return JobStore(jobs_dir=tmp_path / "jobs")

@pytest.fixture
def dummy_pdf_creator():
    def _create(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 dummy content")
    return _create

def test_failed_kuzu_write_aborts_ingestion(tmp_path, tmp_store, dummy_pdf_creator):
    """
    Simulate a failure during the Kuzu database write or file swap phase.
    Verify that the live graph remains completely untouched and the status flips to FAILED.
    """
    live_db = tmp_path / "okf_graph.db"
    live_db.mkdir(parents=True, exist_ok=True)
    (live_db / "live_marker").write_bytes(b"ORIGINAL_DB_STATE")
    
    job = tmp_store.create_job("failed_write.pdf")
    upload = tmp_store.job_dir(job.job_id) / "upload.pdf"
    dummy_pdf_creator(upload)
    
    temp_db_path = tmp_store.job_dir(job.job_id) / "temp_graph.db"
    
    mock_graph_export = {
        "stats": {"total_concepts": 5, "total_edges": 8},
        "concepts": {},
        "edges": [],
        "visualization": {"nodes": [], "edges": []},
    }
    
    def fake_pipeline(source_path, temp_db_path, on_progress, check_cancelled=None):
        import kuzu
        db = kuzu.Database(temp_db_path)
        conn = kuzu.Connection(db)
        conn.execute("CREATE NODE TABLE Concept (id STRING PRIMARY KEY, name STRING, concept_type STRING, difficulty STRING, summary STRING)")
        conn.execute("CREATE NODE TABLE Document (id STRING PRIMARY KEY)")
        conn.execute("CREATE NODE TABLE Chunk (id STRING PRIMARY KEY, chunk_id STRING, page_number INT64, section_title STRING, text_passage STRING)")
        conn.execute("CREATE REL TABLE HAS_CHUNK (FROM Document TO Chunk)")
        conn.execute("CREATE REL TABLE MENTIONS (FROM Chunk TO Concept)")
        conn.execute("CREATE REL TABLE REQUIRES (FROM Concept TO Concept, relation_type STRING, source STRING)")
        conn.execute("CREATE REL TABLE UNLOCKS (FROM Concept TO Concept, relation_type STRING, source STRING)")
        conn.execute("CREATE REL TABLE RELATED (FROM Concept TO Concept, relation_type STRING, source STRING)")
        return [], db, mock_graph_export

    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)
    
    with patch("ingestion_worker.run_pipeline_staged", side_effect=fake_pipeline), \
         patch("shutil.copytree", side_effect=IOError("Simulated disk write failure")), \
         patch("shutil.copy2", side_effect=IOError("Simulated disk write failure")), \
         patch("ingestion_worker.BASE_DIR", tmp_path):
        w._process_job(job.job_id)
        
    # Verify live graph directory still contains the original marker (untouched)
    assert (live_db / "live_marker").exists()
    assert (live_db / "live_marker").read_bytes() == b"ORIGINAL_DB_STATE"
    
    # Verify status is recorded as FAILED
    final = tmp_store.get_job(job.job_id)
    assert final.status == JobStatus.FAILED
    assert "Simulated disk write failure" in final.error

def test_readd_document_idempotent(tmp_path, tmp_store, dummy_pdf_creator):
    """
    Verifies that ingesting the exact same document twice produces an identical graph.
    """
    live_db = tmp_path / "okf_graph.db"
    
    mock_graph_export = {
        "stats": {"total_concepts": 2, "total_edges": 1},
        "concepts": {
            "c1": {"name": "Concept One", "sources": [{"doc_id": "same.pdf", "page_number": 1}]},
            "c2": {"name": "Concept Two", "sources": [{"doc_id": "same.pdf", "page_number": 2}]}
        },
        "edges": [],
        "visualization": {
            "nodes": [{"id": "c1", "label": "Concept One"}, {"id": "c2", "label": "Concept Two"}],
            "edges": []
        },
    }
    
    def fake_pipeline(source_path, temp_db_path, on_progress, check_cancelled=None):
        import kuzu
        db = kuzu.Database(temp_db_path)
        conn = kuzu.Connection(db)
        conn.execute("CREATE NODE TABLE Concept (id STRING PRIMARY KEY, name STRING, concept_type STRING, difficulty STRING, summary STRING)")
        conn.execute("CREATE NODE TABLE Document (id STRING PRIMARY KEY)")
        conn.execute("CREATE NODE TABLE Chunk (id STRING PRIMARY KEY, chunk_id STRING, page_number INT64, section_title STRING, text_passage STRING)")
        conn.execute("CREATE REL TABLE HAS_CHUNK (FROM Document TO Chunk)")
        conn.execute("CREATE REL TABLE MENTIONS (FROM Chunk TO Concept)")
        conn.execute("CREATE REL TABLE REQUIRES (FROM Concept TO Concept, relation_type STRING, source STRING)")
        conn.execute("CREATE REL TABLE UNLOCKS (FROM Concept TO Concept, relation_type STRING, source STRING)")
        conn.execute("CREATE REL TABLE RELATED (FROM Concept TO Concept, relation_type STRING, source STRING)")
        return [], db, mock_graph_export

    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)
    
    # First Ingestion
    job1 = tmp_store.create_job("same.pdf")
    upload1 = tmp_store.job_dir(job1.job_id) / "upload.pdf"
    dummy_pdf_creator(upload1)
    
    with patch("ingestion_worker.run_pipeline_staged", side_effect=fake_pipeline), \
         patch("ingestion_worker.BASE_DIR", tmp_path):
        w._process_job(job1.job_id)
        
    assert (tmp_path / "okf_graph.json").exists()
    graph_v1 = (tmp_path / "okf_graph.json").read_text()
    
    # Second Ingestion
    job2 = tmp_store.create_job("same.pdf")
    upload2 = tmp_store.job_dir(job2.job_id) / "upload.pdf"
    dummy_pdf_creator(upload2)
    
    with patch("ingestion_worker.run_pipeline_staged", side_effect=fake_pipeline), \
         patch("ingestion_worker.BASE_DIR", tmp_path):
        w._process_job(job2.job_id)
        
    graph_v2 = (tmp_path / "okf_graph.json").read_text()
    
    # Assert that the graph structure exported is identical
    assert graph_v1 == graph_v2

def test_live_upload_cannot_corrupt_active_graph(tmp_path, tmp_store, dummy_pdf_creator):
    """
    Test reader-writer safety during ingestion:
    Concurrent reads must all succeed without corruption or throwing errors.
    """
    live_db = tmp_path / "okf_graph.db"
    live_db.mkdir(parents=True, exist_ok=True)
    
    job = tmp_store.create_job("concurrent.pdf")
    upload = tmp_store.job_dir(job.job_id) / "upload.pdf"
    dummy_pdf_creator(upload)
    
    mock_graph_export = {
        "stats": {"total_concepts": 3, "total_edges": 2},
        "concepts": {},
        "edges": [],
        "visualization": {"nodes": [], "edges": []},
    }
    
    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)
    
    read_results = []
    errors = []
    stop_event = threading.Event()
    
    # Reader thread
    def reader_loop():
        while not stop_event.is_set():
            try:
                with lock.read_lock():
                    # Simulate reading files or DB
                    read_results.append(True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
                
    readers = [threading.Thread(target=reader_loop) for _ in range(5)]
    for r in readers:
        r.start()
        
    # Process job which triggers write lock during final swap
    # We mock the staged pipeline and kuzu database
    def fake_pipeline(source_path, temp_db_path, on_progress, check_cancelled=None):
        time.sleep(0.02)  # Simulate some ingestion time
        return [], None, mock_graph_export
        
    with patch("ingestion_worker.run_pipeline_staged", side_effect=fake_pipeline), \
         patch("ingestion_worker.BASE_DIR", tmp_path), \
         patch("kuzu.Database"), \
         patch("shutil.copytree"):
        w._process_job(job.job_id)
        
    stop_event.set()
    for r in readers:
        r.join()
        
    # Verify no read errors occurred
    assert len(errors) == 0
    assert len(read_results) > 0
