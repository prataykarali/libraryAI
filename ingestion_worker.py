"""
ingestion_worker.py — Background worker thread and GraphLock reader-writer lock
for safe live ingestion.
"""

import json
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from okf.config import BASE_DIR
from okf.exports import export_vis_json
from okf.pipeline import run_pipeline_staged, PipelineAborted
from ingestion_jobs import JobStatus, JobStore


# ---------------------------------------------------------------------------
# GraphLock (Reader-Writer Lock)
# ---------------------------------------------------------------------------

class GraphLock:
    """
    A readers-writer lock to allow concurrent reads of the graph database while
    ensuring exclusive access for writes (database swaps and embedding updates).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    @contextmanager
    def read_lock(self):
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()


# Module-level singleton lock
graph_lock = GraphLock()


# ---------------------------------------------------------------------------
# IngestionWorker
# ---------------------------------------------------------------------------

class IngestionWorker(threading.Thread):
    """
    A single-threaded background worker that processes ingestion jobs sequentially.
    """

    def __init__(self, job_store: JobStore, live_db_path: str = None, graph_lock: GraphLock = None):
        super().__init__(name="IngestionWorker", daemon=True)
        self.job_store = job_store
        self.live_db_path = live_db_path or str(BASE_DIR / "okf_graph.db")
        self.graph_lock = graph_lock or globals().get("graph_lock")
        self.queue = []
        self._stop_event = threading.Event()
        self._cond = threading.Condition()

    def enqueue(self, job_id: str):
        with self._cond:
            self.queue.append(job_id)
            self._cond.notify()

    def stop(self):
        self._stop_event.set()
        with self._cond:
            self._cond.notify()

    def run(self):
        while not self._stop_event.is_set():
            job_id = None
            with self._cond:
                while not self.queue and not self._stop_event.is_set():
                    self._cond.wait()
                if self._stop_event.is_set():
                    break
                if self.queue:
                    job_id = self.queue.pop(0)

            if job_id:
                self._process_job(job_id)

    def _process_job(self, job_id: str):
        job = self.job_store.get_job(job_id)
        if not job:
            return

        if self.job_store.is_cancelled(job_id):
            self.job_store.update_status(job_id, JobStatus.CANCELLED)
            return

        job_dir = self.job_store.job_dir(job_id)
        # Resolve quarantined upload (PDF, MD, or TXT)
        upload_pdf = None
        for candidate in (
            job_dir / "upload.pdf",
            job_dir / "upload.md",
            job_dir / "upload.txt",
            job_dir / "upload.markdown",
        ):
            if candidate.is_file():
                upload_pdf = candidate
                break
        if upload_pdf is None:
            # Fallback: any upload.* file
            for p in sorted(job_dir.glob("upload.*")):
                if p.is_file() and p.suffix.lower() in (".pdf", ".md", ".txt", ".markdown"):
                    upload_pdf = p
                    break
        if upload_pdf is None:
            self.job_store.update_status(
                job_id, JobStatus.FAILED, error="No upload.pdf/.md/.txt found in job dir"
            )
            return
        temp_db_path = str(job_dir / "temp_graph.db")

        # 1. Update status to PARSING
        self.job_store.update_status(job_id, JobStatus.PARSING)

        def on_progress(stage, pct, detail):
            self.job_store.update_status(
                job_id,
                JobStatus(stage),
                progress={stage: {"pct": pct, "detail": detail}}
            )

        def check_cancelled():
            return self.job_store.is_cancelled(job_id)

        try:
            # Run the staged pipeline in the quarantine temporary database
            okf_results, temp_db, graph_export = run_pipeline_staged(
                source_path=str(upload_pdf),
                temp_db_path=temp_db_path,
                on_progress=on_progress,
                check_cancelled=check_cancelled
            )

            # Export temp nodes/edges visualization JSONs inside quarantine
            temp_nodes_path = job_dir / "temp_nodes.json"
            temp_edges_path = job_dir / "temp_edges.json"
            export_vis_json(temp_db, str(temp_nodes_path), str(temp_edges_path))

            # Explicitly delete temp_db to release Kuzu locks before copying
            del temp_db

            on_progress("GRAPH_VALIDATION", 100, {"message": "Beginning atomic database swap"})

            # 2. COMPLETE (atomic swap under lock)
            with self.graph_lock.write_lock():
                backup_path = self.live_db_path + ".bak"
                has_backup = False
                
                # Move existing live database to backup
                if os.path.exists(self.live_db_path):
                    if os.path.exists(backup_path):
                        if os.path.isdir(backup_path):
                            shutil.rmtree(backup_path, ignore_errors=True)
                        else:
                            os.remove(backup_path)
                    os.rename(self.live_db_path, backup_path)
                    has_backup = True

                try:
                    # Move temp_graph.db to live okf_graph.db path
                    if os.path.isdir(temp_db_path):
                        shutil.copytree(temp_db_path, self.live_db_path)
                    else:
                        shutil.copy2(temp_db_path, self.live_db_path)
                except Exception as swap_err:
                    # Rollback if copy failed
                    if os.path.exists(self.live_db_path):
                        if os.path.isdir(self.live_db_path):
                            shutil.rmtree(self.live_db_path, ignore_errors=True)
                        else:
                            os.remove(self.live_db_path)
                    if has_backup:
                        os.rename(backup_path, self.live_db_path)
                    raise swap_err

                # Clean up backup on success
                if has_backup and os.path.exists(backup_path):
                    if os.path.isdir(backup_path):
                        shutil.rmtree(backup_path, ignore_errors=True)
                    else:
                        os.remove(backup_path)

                # Reload live database handle in inference_server if running
                try:
                    import inference_server
                    # reload_db() reopens read-only so the server never holds
                    # the exclusive write lock (a raw kuzu.Database here would
                    # silently reintroduce it).
                    inference_server.reload_db()
                except ImportError:
                    pass
                except Exception as reload_err:
                    # The swap itself succeeded; a failed hot-reload (e.g. an
                    # external server process holds the Kuzu lock) must not
                    # fail the job. The other process reloads on its own.
                    print(f"Warning: inference_server.reload_db failed after swap: {reload_err}")

                # Reset default GraphDB handle to force recreation on next read
                try:
                    import okf.graph_db as _gdb
                    _gdb._DEFAULT_GRAPH_DB = None
                except ImportError:
                    pass

                # Write live JSON exports to BASE_DIR
                with open(BASE_DIR / "okf_results.json", "w", encoding="utf-8") as f:
                    json.dump(okf_results, f, indent=2, ensure_ascii=False)

                # Write every downstream artifact (graph_audit.json, root +
                # graph_ui okf_graph.json, _graph_nodes/_graph_edges,
                # accuracy.json) via the shared writer so the live swap can
                # never drift from finalize_and_build's outputs.
                from okf.exports import write_all_artifacts
                import kuzu
                new_db = kuzu.Database(self.live_db_path)
                write_all_artifacts(graph_export, okf_results, new_db,
                                    base_dir=BASE_DIR)

                # Rebuild embeddings in memory on inference_server if possible
                try:
                    import inference_server
                    inference_server.build_concept_embeddings()
                except (ImportError, AttributeError):
                    pass

            # 3. FINALIZE (after releasing write lock)
            # Copy uploaded source into pdfs/ with a sensible category folder
            pdf_dest_dir = BASE_DIR / "pdfs"
            pdf_dest_dir.mkdir(parents=True, exist_ok=True)
            src_name = job.source_filename or upload_pdf.name
            low = src_name.lower()
            if low.endswith((".md", ".markdown", ".txt")):
                sub = pdf_dest_dir / "web_syllabi"
            elif "textbook" in low:
                sub = pdf_dest_dir / "textbooks"
            else:
                sub = pdf_dest_dir / "papers"
            sub.mkdir(parents=True, exist_ok=True)
            dest_pdf_path = sub / Path(src_name).name
            shutil.copy2(str(upload_pdf), str(dest_pdf_path))

            # Compute graph version by checking previous completed jobs
            max_v = 0
            jobs_list = self.job_store.list_jobs()
            for j_sum in jobs_list:
                j_rec = self.job_store.get_job(j_sum["job_id"])
                if j_rec and j_rec.status == JobStatus.COMPLETE and j_rec.graph_version:
                    max_v = max(max_v, j_rec.graph_version)
            next_version = max_v + 1

            # Find concepts that are in the new document
            new_node_ids = []
            merged_concepts = []
            source_pages = set()
            
            doc_id = job.source_filename
            for cid, cinfo in graph_export.get("concepts", {}).items():
                is_from_doc = False
                for src in cinfo.get("sources", []):
                    if src.get("doc_id") == doc_id:
                        is_from_doc = True
                        if src.get("page_number"):
                            source_pages.add(int(src["page_number"]))
                if is_from_doc:
                    new_node_ids.append(cid)
                    merged_concepts.append(cinfo.get("name"))

            # Update job status to COMPLETE
            res_meta = {
                "nodes": graph_export.get("stats", {}).get("total_concepts", 0),
                "edges": graph_export.get("stats", {}).get("total_edges", 0),
                "new_node_ids": new_node_ids,
                "merged_concepts": merged_concepts,
                "source_pages": sorted(list(source_pages)),
            }
            self.job_store.update_status(
                job_id,
                JobStatus.COMPLETE,
                result=res_meta,
                graph_version=next_version
            )

        except PipelineAborted:
            self.job_store.update_status(job_id, JobStatus.CANCELLED)
        except Exception as e:
            self.job_store.update_status(job_id, JobStatus.FAILED, error=str(e))
        finally:
            # Clean up temp_graph.db from quarantine directory, but keep upload.pdf and job.json
            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path, ignore_errors=True)


# Module-level singletons
job_store = JobStore(BASE_DIR / "jobs")
worker = IngestionWorker(job_store)


def get_worker() -> IngestionWorker:
    """Get the active IngestionWorker thread, starting it if not alive."""
    global worker
    if not worker.is_alive():
        if getattr(worker, "_started", None) and worker._started.is_set():
            worker = IngestionWorker(job_store)
        worker.start()
    return worker
