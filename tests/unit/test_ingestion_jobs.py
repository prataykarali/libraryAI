"""
tests/unit/test_ingestion_jobs.py — Unit tests for the JobStore / JobRecord
ingestion job management layer.
"""

import pytest
import threading
import time
from pathlib import Path

from ingestion_jobs import JobStatus, JobRecord, JobStore, TERMINAL_STATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh JobStore backed by a temporary directory."""
    return JobStore(jobs_dir=tmp_path / "jobs")


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------

def test_job_status_values():
    expected = {
        "QUEUED", "PARSING", "EXTRACTION", "CANONICALIZATION",
        "GRAPH_BUILD", "GRAPH_VALIDATION", "COMPLETE", "FAILED", "CANCELLED",
    }
    assert {s.value for s in JobStatus} == expected


def test_terminal_states():
    assert JobStatus.COMPLETE  in TERMINAL_STATES
    assert JobStatus.FAILED    in TERMINAL_STATES
    assert JobStatus.CANCELLED in TERMINAL_STATES
    assert JobStatus.QUEUED    not in TERMINAL_STATES


# ---------------------------------------------------------------------------
# JobRecord serialisation round-trip
# ---------------------------------------------------------------------------

def test_job_record_round_trip():
    r = JobRecord(
        job_id="abc-123",
        status=JobStatus.PARSING,
        source_filename="test.pdf",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:01:00+00:00",
        progress={"PARSING": {"pct": 50}},
        error=None,
        result={"nodes": 10},
        graph_version=7,
        cancelled=False,
    )
    d = r.to_dict()
    r2 = JobRecord.from_dict(d)
    assert r2.job_id          == r.job_id
    assert r2.status          == r.status
    assert r2.source_filename == r.source_filename
    assert r2.progress        == r.progress
    assert r2.result          == r.result
    assert r2.graph_version   == r.graph_version
    assert r2.cancelled       == r.cancelled


def test_job_record_summary():
    r = JobRecord(
        job_id="x", status=JobStatus.COMPLETE, source_filename="f.pdf",
        created_at="t1", updated_at="t2", graph_version=3,
    )
    s = r.summary()
    assert s["job_id"]   == "x"
    assert s["status"]   == "COMPLETE"
    assert "progress" not in s
    assert "error"    not in s


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------

def test_create_job_returns_queued(store):
    job = store.create_job("paper.pdf")
    assert job.status          == JobStatus.QUEUED
    assert job.source_filename == "paper.pdf"
    assert job.job_id
    assert job.created_at
    assert job.cancelled is False


def test_create_job_persists(store):
    job = store.create_job("a.pdf")
    fetched = store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id
    assert fetched.status == JobStatus.QUEUED


def test_create_job_creates_quarantine_dir(store, tmp_path):
    job = store.create_job("b.pdf")
    qdir = store.job_dir(job.job_id)
    assert qdir.exists()


def test_create_job_writes_job_json(store, tmp_path):
    job = store.create_job("c.pdf")
    job_json = store.job_dir(job.job_id) / "job.json"
    assert job_json.exists()


def test_create_multiple_jobs_distinct_ids(store):
    ids = {store.create_job("x.pdf").job_id for _ in range(5)}
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

def test_update_status_transitions(store):
    job = store.create_job("doc.pdf")
    for status in [
        JobStatus.PARSING, JobStatus.EXTRACTION,
        JobStatus.CANONICALIZATION, JobStatus.GRAPH_BUILD,
        JobStatus.GRAPH_VALIDATION, JobStatus.COMPLETE,
    ]:
        updated = store.update_status(job.job_id, status)
        assert updated.status == status
    fetched = store.get_job(job.job_id)
    assert fetched.status == JobStatus.COMPLETE


def test_update_status_merges_progress(store):
    job = store.create_job("d.pdf")
    store.update_status(job.job_id, JobStatus.PARSING,
                        progress={"PARSING": {"pct": 10}})
    store.update_status(job.job_id, JobStatus.EXTRACTION,
                        progress={"EXTRACTION": {"pct": 50}})
    fetched = store.get_job(job.job_id)
    assert "PARSING"    in fetched.progress
    assert "EXTRACTION" in fetched.progress


def test_update_status_sets_error(store):
    job = store.create_job("e.pdf")
    store.update_status(job.job_id, JobStatus.FAILED, error="Bad PDF")
    fetched = store.get_job(job.job_id)
    assert fetched.error == "Bad PDF"
    assert fetched.status == JobStatus.FAILED


def test_update_status_sets_result(store):
    job = store.create_job("f.pdf")
    store.update_status(job.job_id, JobStatus.COMPLETE,
                        result={"nodes": 42, "edges": 100},
                        graph_version=5)
    fetched = store.get_job(job.job_id)
    assert fetched.result["nodes"] == 42
    assert fetched.graph_version   == 5


def test_update_status_unknown_id_returns_none(store):
    result = store.update_status("nonexistent-id", JobStatus.FAILED)
    assert result is None


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

def test_get_job_unknown_returns_none(store):
    assert store.get_job("does-not-exist") is None


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

def test_list_jobs_empty(store):
    assert store.list_jobs() == []


def test_list_jobs_multiple(store):
    j1 = store.create_job("a.pdf")
    j2 = store.create_job("b.pdf")
    j3 = store.create_job("c.pdf")
    summaries = store.list_jobs()
    assert len(summaries) == 3
    ids = {s["job_id"] for s in summaries}
    assert ids == {j1.job_id, j2.job_id, j3.job_id}


def test_list_jobs_summary_keys(store):
    store.create_job("x.pdf")
    s = store.list_jobs()[0]
    required_keys = {"job_id", "status", "source_filename", "created_at", "updated_at"}
    assert required_keys.issubset(s.keys())


def test_list_jobs_newest_first(store):
    j1 = store.create_job("1.pdf")
    time.sleep(0.01)  # ensure different timestamps
    j2 = store.create_job("2.pdf")
    summaries = store.list_jobs()
    # newest first
    assert summaries[0]["job_id"] == j2.job_id
    assert summaries[1]["job_id"] == j1.job_id


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------

def test_cancel_queued_job_becomes_cancelled(store):
    job = store.create_job("q.pdf")
    cancelled = store.cancel_job(job.job_id)
    assert cancelled.status    == JobStatus.CANCELLED
    assert cancelled.cancelled is True


def test_cancel_active_job_sets_flag(store):
    job = store.create_job("r.pdf")
    store.update_status(job.job_id, JobStatus.EXTRACTION)
    cancelled = store.cancel_job(job.job_id)
    # Active job keeps its current status but gets the flag
    assert cancelled.cancelled is True
    assert store.is_cancelled(job.job_id) is True


def test_cancel_terminal_job_no_change(store):
    job = store.create_job("s.pdf")
    store.update_status(job.job_id, JobStatus.COMPLETE)
    result = store.cancel_job(job.job_id)
    # Status must remain COMPLETE
    assert result.status == JobStatus.COMPLETE


def test_cancel_unknown_job_returns_none(store):
    assert store.cancel_job("ghost-id") is None


# ---------------------------------------------------------------------------
# is_cancelled
# ---------------------------------------------------------------------------

def test_is_cancelled_false_for_new_job(store):
    job = store.create_job("t.pdf")
    assert store.is_cancelled(job.job_id) is False


def test_is_cancelled_true_after_cancel(store):
    job = store.create_job("u.pdf")
    store.cancel_job(job.job_id)
    assert store.is_cancelled(job.job_id) is True


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------

def test_concurrent_create_jobs(store):
    results = []
    errors  = []

    def create():
        try:
            j = store.create_job("concurrent.pdf")
            results.append(j.job_id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=create) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(set(results)) == 20   # all unique IDs
    assert len(store.list_jobs()) == 20
