"""
tests/integration/test_ingestion_worker.py — Integration tests for IngestionWorker,
GraphLock, and end-to-end status updates.
"""

import json
import os
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion_jobs import JobStatus, JobStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """JobStore pointing to a temporary directory."""
    return JobStore(jobs_dir=tmp_path / "jobs")


@pytest.fixture
def mock_staged_pipeline():
    """Mock run_pipeline_staged to simulate successful stages."""
    mock_db = MagicMock()
    mock_db.execute.return_value.has_next.return_value = False
    mock_graph_export = {
        "stats": {"total_concepts": 10, "total_edges": 20},
        "concepts": {},
        "edges": [],
        "visualization": {"nodes": [], "edges": []},
    }
    
    def fake_pipeline(source_path, temp_db_path, on_progress, check_cancelled=None):
        for stage in ["PARSING", "EXTRACTION", "CANONICALIZATION", "GRAPH_BUILD", "GRAPH_VALIDATION"]:
            on_progress(stage, 100, {"message": f"{stage} done"})
        return [], mock_db, mock_graph_export

    with patch("ingestion_worker.run_pipeline_staged", side_effect=fake_pipeline):
        yield mock_graph_export


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_pdf(path: Path) -> None:
    """Create a minimal dummy file representing a quarantined PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 dummy contents")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_successful_ingestion(tmp_path, tmp_store, mock_staged_pipeline):
    """
    End-to-end test of a successful ingestion.
    Verifies that files are swapped under write lock, JSON files are written,
    and job finishes with COMPLETE status and version incremented.
    """
    from ingestion_worker import IngestionWorker, GraphLock

    live_db = tmp_path / "okf_graph.db"
    live_db.mkdir(parents=True, exist_ok=True)
    (live_db / "old_db_file").write_bytes(b"OLD")

    job = tmp_store.create_job("success.pdf")
    upload = tmp_store.job_dir(job.job_id) / "upload.pdf"
    _make_dummy_pdf(upload)

    temp_db_path = tmp_store.job_dir(job.job_id) / "temp_graph.db"
    temp_db_path.mkdir(parents=True, exist_ok=True)
    (temp_db_path / "new_db_file").write_bytes(b"NEW")

    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)

    # Patch BASE_DIR and other internal calls to use our tmp directory path
    with patch("ingestion_worker.BASE_DIR", tmp_path), \
         patch("kuzu.Database") as mock_kuzu_db:
        mock_kuzu_db.return_value.execute.return_value.has_next.return_value = False
        w._process_job(job.job_id)

    # Verify atomic DB swap took place
    assert not (live_db / "old_db_file").exists()
    assert (live_db / "new_db_file").exists()

    # Verify live JSON files are written to tmp_path (patched BASE_DIR)
    assert (tmp_path / "okf_results.json").exists()
    assert (tmp_path / "okf_graph.json").exists()
    assert (tmp_path / "graph_audit.json").exists()

    # Verify uploaded PDF is copied to live pdfs directory
    # (worker categorizes uploads: PDFs without "textbook" in the name → papers/)
    assert (tmp_path / "pdfs" / "papers" / "success.pdf").exists()

    # Check job record status
    final = tmp_store.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETE
    assert final.graph_version == 1
    assert final.result["nodes"] == 10
    assert final.result["edges"] == 20

    # Quarantine DB directory should be cleaned up
    assert not temp_db_path.exists()


def test_failed_ingestion_no_graph_change(tmp_path, tmp_store):
    """
    If any staged pipeline step raises an exception, the job status must
    flip to FAILED, the error is recorded, and the live graph is untouched.
    """
    from ingestion_worker import IngestionWorker, GraphLock

    # Create dummy live DB path
    live_db = tmp_path / "okf_graph.db"
    live_db.mkdir(parents=True, exist_ok=True)
    (live_db / "live_db_file").write_bytes(b"LIVE_STAYS")

    job = tmp_store.create_job("corrupted.pdf")
    upload = tmp_store.job_dir(job.job_id) / "upload.pdf"
    _make_dummy_pdf(upload)

    def error_pipeline(*args, **kwargs):
        raise ValueError("Malformed PDF structure")

    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)

    with patch("ingestion_worker.run_pipeline_staged", side_effect=error_pipeline):
        w._process_job(job.job_id)

    # Live DB files must remain untouched
    assert (live_db / "live_db_file").read_bytes() == b"LIVE_STAYS"

    # Job status must be FAILED with the error saved
    final = tmp_store.get_job(job.job_id)
    assert final.status == JobStatus.FAILED
    assert "Malformed PDF structure" in final.error


def test_cancelled_ingestion(tmp_path, tmp_store):
    """
    If a job's cancelled flag is checked and found True during stages,
    the job must abort immediately, status flips to CANCELLED, and live graph
    remains untouched.
    """
    from ingestion_worker import IngestionWorker, GraphLock
    from okf.pipeline import PipelineAborted

    live_db = tmp_path / "okf_graph.db"
    live_db.mkdir(parents=True, exist_ok=True)
    (live_db / "live_db_file").write_bytes(b"LIVE_STAYS")

    job = tmp_store.create_job("cancelled.pdf")
    upload = tmp_store.job_dir(job.job_id) / "upload.pdf"
    _make_dummy_pdf(upload)
    
    # Request cancellation
    tmp_store.cancel_job(job.job_id)

    def cancel_pipeline(*args, **kwargs):
        raise PipelineAborted("Job cancelled")

    lock = GraphLock()
    w = IngestionWorker(tmp_store, live_db_path=str(live_db), graph_lock=lock)

    with patch("ingestion_worker.run_pipeline_staged", side_effect=cancel_pipeline):
        w._process_job(job.job_id)

    # Live DB is untouched
    assert (live_db / "live_db_file").read_bytes() == b"LIVE_STAYS"

    # Status is CANCELLED
    final = tmp_store.get_job(job.job_id)
    assert final.status == JobStatus.CANCELLED


def test_readd_document_idempotent(tmp_path, tmp_store):
    """
    Re-adding the same PDF creates separate job entries, but the JobStore is
    idempotent and lists all jobs correctly without internal dictionary key collision.
    """
    j1 = tmp_store.create_job("same.pdf")
    j2 = tmp_store.create_job("same.pdf")

    assert j1.job_id != j2.job_id
    jobs = tmp_store.list_jobs()
    assert len(jobs) == 2


def test_concurrent_read_during_ingestion():
    """
    Verify reader-writer lock semantics: multiple readers can hold the lock
    simultaneously, but writers get exclusive access and block new readers.
    """
    from ingestion_worker import GraphLock

    lock = GraphLock()
    events = []

    def reader(idx):
        with lock.read_lock():
            events.append(f"read_start_{idx}")
            time.sleep(0.05)
            events.append(f"read_end_{idx}")

    def writer():
        time.sleep(0.01)  # start shortly after first readers
        with lock.write_lock():
            events.append("write_start")
            time.sleep(0.05)
            events.append("write_end")

    threads = [
        threading.Thread(target=reader, args=(1,)),
        threading.Thread(target=reader, args=(2,)),
        threading.Thread(target=writer),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Assert that all read starts occurred before the write start, or similar order.
    # Specifically, writer cannot acquire lock while readers hold it.
    write_start_idx = events.index("write_start")
    
    # Read end events of reader 1 and 2 must happen BEFORE write_start
    assert events.index("read_end_1") < write_start_idx
    assert events.index("read_end_2") < write_start_idx
