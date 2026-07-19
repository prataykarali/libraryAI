#!/usr/bin/env python3
import time
import subprocess
import requests
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_tests():
    print("Starting Archipelago Graph UI Server (port 5050)...")
    ui_process = subprocess.Popen(
        [".venv/bin/python", "graph_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(BASE_DIR),
        text=True
    )
    
    print("Starting Archipelago Inference Server (port 5051)...")
    inf_process = subprocess.Popen(
        [".venv/bin/python", "inference_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(BASE_DIR),
        text=True
    )
    
    # Wait for the Flask servers to bind to ports and load models
    print("Waiting 12 seconds for servers to initialize and load models...")
    time.sleep(12)
    
    success = True
    
    try:
        # Test 1: Verify UI Server (Graph metadata)
        print("\nTesting Graph UI Server (port 5050)...")
        graph_url = "http://localhost:5050/api/graph"
        graph_res = requests.get(graph_url)
        if graph_res.status_code == 200:
            print("✓ Graph metadata endpoint successful!")
            nodes_count = len(graph_res.json().get("nodes", []))
            print(f"  Total concepts in visualization payload: {nodes_count}")
        else:
            print(f"✗ Graph metadata endpoint failed (Status: {graph_res.status_code})")
            success = False

        # Test 2: Verify PDF Serving on Inference Server
        print("\nTesting PDF Serving Endpoint (port 5051)...")
        pdf_url = "http://localhost:5051/pdfs/papers/Devlin2018_BERT.pdf"
        pdf_res = requests.get(pdf_url, stream=True)
        if pdf_res.status_code == 200:
            print("✓ PDF serve successful!")
        else:
            print(f"✗ PDF serve failed (Status: {pdf_res.status_code})")
            success = False

        # Test 3: Verify RAG Synthesis Mode
        print("\nTesting /api/chat RAG Synthesis (port 5051)...")
        chat_url = "http://localhost:5051/api/chat"
        payload_rag = {
            "query": "What is LoRA?",
            "mode": "rag_synthesis",
            "history": []
        }
        res_rag = requests.post(chat_url, json=payload_rag)
        if res_rag.status_code == 200:
            data = res_rag.json()
            print("✓ RAG Synthesis Response received!")
            print(f"  Matched Anchor: {data.get('anchor_concept', {}).get('label', 'None')}")
            print(f"  Prereqs found: {len(data.get('prerequisites', []))}")
            print(f"  Citations found: {len(data.get('citations', []))}")
            print(f"  Response preview:\n{data.get('response', '')[:300]}...\n")
        else:
            print(f"✗ RAG Synthesis failed (Status: {res_rag.status_code})")
            print(res_rag.text)
            success = False

        # Test 4: Verify Conversational Agent Mode
        print("\nTesting /api/chat Conversational Agent (port 5051)...")
        payload_agent = {
            "query": "Hello! Can you help me query the database?",
            "mode": "conversational_agent",
            "history": []
        }
        res_agent = requests.post(chat_url, json=payload_agent)
        if res_agent.status_code == 200:
            data = res_agent.json()
            print("✓ Conversational Agent Response received!")
            print(f"  Response preview:\n{data.get('response', '')[:300]}...\n")
        else:
            print(f"✗ Conversational Agent failed (Status: {res_agent.status_code})")
            print(res_agent.text)
            success = False
            
    except Exception as e:
        print(f"Error during tests: {e}")
        success = False
    finally:
        print("Shutting down Archipelago Servers...")
        ui_process.terminate()
        inf_process.terminate()
        
        ui_stdout, ui_stderr = ui_process.communicate()
        inf_stdout, inf_stderr = inf_process.communicate()
        
        print("\n--- UI SERVER STDOUT ---")
        print(ui_stdout)
        print("\n--- UI SERVER STDERR ---")
        print(ui_stderr)
        
        print("\n--- INF SERVER STDOUT ---")
        print(inf_stdout)
        print("\n--- INF SERVER STDERR ---")
        print(inf_stderr)
        print("---------------------\n")
        print("Servers stopped.")
        
    if success:
        print("\n✓ ALL ENDPOINT TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
