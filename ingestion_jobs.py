"""
ingestion_jobs.py — Lightweight, thread-safe, JSON-backed job store for the
safe live ingestion pipeline.

Each job is quarantined in jobs/<job_id>/ and tracked in jobs/jobs.json.
The store never holds open file handles — every write loads the current file,
applies the change, and saves atomically via rename-swap.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED            = "QUEUED"
    PARSING           = "PARSING"
    EXTRACTION        = "EXTRACTION"
    CANONICALIZATION  = "CANONICALIZATION"
    GRAPH_BUILD       = "GRAPH_BUILD"
    GRAPH_VALIDATION  = "GRAPH_VALIDATION"
    COMPLETE          = "COMPLETE"
    FAILED            = "FAILED"
    CANCELLED         = "CANCELLED"


# Ordered active stages — useful for progress display.
ACTIVE_STAGES: List[JobStatus] = [
    JobStatus.PARSING,
    JobStatus.EXTRACTION,
    JobStatus.CANONICALIZATION,
    JobStatus.GRAPH_BUILD,
    JobStatus.GRAPH_VALIDATION,
]

# Terminal states — the job will never progress beyond these.
TERMINAL_STATES = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}


# ---------------------------------------------------------------------------
# JobRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class JobRecord:
    job_id:          str
    status:          JobStatus
    source_filename: str
    created_at:      str            # ISO-8601 UTC
    updated_at:      str            # ISO-8601 UTC
    progress:        Dict[str, Any] = field(default_factory=dict)
    error:           Optional[str]  = None
    result:          Optional[Dict[str, Any]] = field(default_factory=dict)
    graph_version:   Optional[int]  = None
    cancelled:       bool           = False   # soft-cancel flag polled by worker

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "job_id":          self.job_id,
            "status":          self.status.value if isinstance(self.status, JobStatus) else self.status,
            "source_filename": self.source_filename,
            "created_at":      self.created_at,
            "updated_at":      self.updated_at,
            "progress":        self.progress,
            "error":           self.error,
            "result":          self.result if self.result is not None else {},
            "graph_version":   self.graph_version,
            "cancelled":       self.cancelled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobRecord":
        return cls(
            job_id=d["job_id"],
            status=JobStatus(d["status"]),
            source_filename=d["source_filename"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            progress=d.get("progress") or {},
            error=d.get("error"),
            result=d.get("result") or {},
            graph_version=d.get("graph_version"),
            cancelled=bool(d.get("cancelled", False)),
        )

    def summary(self) -> dict:
        """Lightweight summary for list_jobs()."""
        return {
            "job_id":          self.job_id,
            "status":          self.status.value,
            "source_filename": self.source_filename,
            "created_at":      self.created_at,
            "updated_at":      self.updated_at,
            "graph_version":   self.graph_version,
        }


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------

class JobStore:
    """
    Thread-safe, JSON-backed job store.

    Layout on disk::

        jobs/
            jobs.json           ← index of all job records
            <job_id>/           ← per-job quarantine directory
                job.json        ← mirror of the record for this job
                upload.pdf      ← quarantined upload (written by worker)
                temp_graph.db/  ← temporary KùzuDB (written by worker)
                temp_nodes.json
                temp_edges.json
    """

    _JOBS_FILE = "jobs.json"

    def __init__(self, jobs_dir: str | Path = "jobs"):
        self._dir = Path(jobs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Ensure the index file exists
        with self._lock:
            if not self._jobs_path().exists():
                self._save_unlocked({})

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _jobs_path(self) -> Path:
        return self._dir / self._JOBS_FILE

    def _load_unlocked(self) -> Dict[str, dict]:
        """Load raw jobs dict from disk. Caller MUST hold self._lock."""
        p = self._jobs_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_unlocked(self, jobs: Dict[str, dict]) -> None:
        """Atomically save jobs dict via rename. Caller MUST hold self._lock."""
        p = self._jobs_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def _mirror_job_json(self, record: JobRecord) -> None:
        """Write per-job job.json into its quarantine directory (lock held)."""
        job_dir = self._dir / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        p = job_dir / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def create_job(self, filename: str) -> JobRecord:
        """Create a new job record with status=QUEUED.

        Creates the per-job quarantine directory and mirrors job.json.
        Returns the new JobRecord.
        """
        now = self._now()
        record = JobRecord(
            job_id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            source_filename=filename,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            jobs = self._load_unlocked()
            jobs[record.job_id] = record.to_dict()
            self._save_unlocked(jobs)
            self._mirror_job_json(record)
        return record

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        graph_version: Optional[int] = None,
    ) -> Optional[JobRecord]:
        """Update the status (and optional fields) of an existing job.

        Silently returns None if job_id is unknown.
        Progress dict is *merged* into the existing per-stage dict, not replaced.
        Returns the updated JobRecord.
        """
        with self._lock:
            jobs = self._load_unlocked()
            if job_id not in jobs:
                return None
            raw = jobs[job_id]
            raw["status"] = status.value
            raw["updated_at"] = self._now()
            if progress is not None:
                raw.setdefault("progress", {}).update(progress)
            if error is not None:
                raw["error"] = error
            if result is not None:
                raw["result"] = result
            if graph_version is not None:
                raw["graph_version"] = graph_version
            jobs[job_id] = raw
            self._save_unlocked(jobs)
            record = JobRecord.from_dict(raw)
            self._mirror_job_json(record)
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Return the JobRecord for job_id, or None if not found."""
        with self._lock:
            jobs = self._load_unlocked()
            raw = jobs.get(job_id)
            if raw is None:
                return None
            return JobRecord.from_dict(raw)

    def list_jobs(self) -> List[dict]:
        """Return a list of lightweight job summaries, newest first."""
        with self._lock:
            jobs = self._load_unlocked()
        records = [JobRecord.from_dict(v) for v in jobs.values()]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [r.summary() for r in records]

    def cancel_job(self, job_id: str) -> Optional[JobRecord]:
        """Request cancellation of a job.

        - Terminal-state jobs are returned unchanged.
        - QUEUED jobs are immediately flipped to CANCELLED.
        - In-progress jobs get the ``cancelled`` flag set so the worker's
          ``check_cancelled()`` callback can detect and abort them.
        """
        with self._lock:
            jobs = self._load_unlocked()
            raw = jobs.get(job_id)
            if raw is None:
                return None
            record = JobRecord.from_dict(raw)
            if record.status in TERMINAL_STATES:
                return record
            # Immediately cancel queued jobs; signal active ones via flag.
            if record.status == JobStatus.QUEUED:
                raw["status"] = JobStatus.CANCELLED.value
            raw["cancelled"] = True
            raw["updated_at"] = self._now()
            jobs[job_id] = raw
            self._save_unlocked(jobs)
            updated = JobRecord.from_dict(raw)
            self._mirror_job_json(updated)
        return updated

    def job_dir(self, job_id: str) -> Path:
        """Return the quarantine directory Path for a given job."""
        return self._dir / job_id

    def is_cancelled(self, job_id: str) -> bool:
        """Fast poll: return True if the job's cancelled flag is set."""
        with self._lock:
            jobs = self._load_unlocked()
            raw = jobs.get(job_id, {})
            return bool(raw.get("cancelled", False))
