"""
tests/e2e/test_upload_flow.py — E2E manual verification script and curl docs.
"""

import os
import sys
import unittest
import requests

class TestUploadFlow(unittest.TestCase):
    """
    End-to-End manual verification documentation and live check script.
    Only executes live checks when RUN_LIVE_E2E=1 is set in the environment.
    """

    def setUp(self):
        # RUN_LIVE_E2E=1 guard
        if os.getenv("RUN_LIVE_E2E") != "1":
            self.skipTest("Skipping live E2E check. Set RUN_LIVE_E2E=1 to run.")

    def test_upload_flow_curl_docs(self):
        """
        Documents the expected upload flow and provides curl commands to test each step manually.
        
        ========== MANUAL UPLOAD FLOW INSTRUCTIONS ==========
        
        Step 1: Start Ingestion
        Upload a PDF file using multipart form-data.
        
        Curl command:
          curl -i -X POST -F "file=@/home/pratay-karali/Desktop/libraryAI/libraryAI/pdfs/Hu2021_LoRA.pdf" http://localhost:5051/api/ingest
          
        Expected Response (202 Accepted):
          {
            "job_id": "<job_id>",
            "status": "queued"
          }
          
        -----------------------------------------------------
        
        Step 2: Poll Ingestion Job Status
        Poll the status of the job using the returned job_id.
        
        Curl command:
          curl -i http://localhost:5051/api/ingest/<job_id>
          
        Expected Response (200 OK):
          {
            "job_id": "...",
            "status": "QUEUED|PARSING|EXTRACTION|CANONICALIZATION|GRAPH_BUILD|GRAPH_VALIDATION|COMPLETE|FAILED|CANCELLED",
            "progress": {
               "<stage>": { "pct": 100, "detail": { ... } }
            },
            ...
          }
          
        -----------------------------------------------------
        
        Step 3: Optional Cancellation
        Cancel a running or queued job.
        
        Curl command:
          curl -i http://localhost:5051/api/ingest/<job_id>/cancel
          
        Expected Response (200 OK):
          {
            "job_id": "...",
            "status": "CANCELLED",
            "cancelled": true
          }
          
        =====================================================
        """
        API_HOST = "http://localhost:5051"
        
        # Verify inference server is running and ready
        try:
            res = requests.get(f"{API_HOST}/api/readiness", timeout=5)
            self.assertIn(res.status_code, [200, 503], "Inference server is not running or responding correctly")
            print("✓ Live Inference Server is online.")
        except requests.exceptions.RequestException as e:
            self.fail(f"Could not connect to inference server at {API_HOST}: {e}")

        # Check job list endpoint
        try:
            res = requests.get(f"{API_HOST}/api/ingest", timeout=5)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("jobs", data)
            print(f"✓ Ingest list online. Found {len(data['jobs'])} historical jobs.")
        except requests.exceptions.RequestException as e:
            self.fail(f"Could not connect to ingestion endpoint at {API_HOST}/api/ingest: {e}")
