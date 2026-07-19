# Local Inference & Chat Architecture

This document describes the completely local architecture for the Archipelago Library AI system. It runs entirely on your local machine using the 7 pre-downloaded source papers/textbooks, an embedded graph database (KuzuDB), and a local LLM runner (Ollama) with no external cloud dependencies.

---

## 1. System Architecture Diagram

```mermaid
graph TD
    %% Frontend Component
    subgraph Client Browser (localhost:5050)
        UI[D3.js Graph Visualizer & Chat Sidebar]
        UI -->|1. User Message / Query| API[Flask Chat Endpoint: /api/chat]
        UI -->|Open PDF Page| PDFViewer[Browser PDF Viewer with #page=N]
    end

    %% Backend Server
    subgraph Local Flask Server (graph_server.py)
        API -->|2. Prompt + Tools| LLM[Local LLM Client: Ollama]
        LLM -->|3. Decides to Call Tool| Tool[Python KuzuDB Executor]
        Tool -->|4. Runs Cypher Query| DB[(Embedded KuzuDB: okf_graph.db)]
        DB -->|5. Returns Nodes & Relations| Tool
        Tool -->|6. Raw Context Data| LLM
        LLM -->|7. Friendly Response| API
        
        API_PDF[Flask Serve PDFs: /pdfs/*] -->|Read File| LocalPDFs[Local pdfs/ directory]
        API_PDF -->|8. Serve PDF Stream| PDFViewer
    end

    %% Local Model Runner
    subgraph Local LLM Runner (Ollama / Local GPU)
        LLM <-->|Instruct / Chat Generation| OllamaServer[Ollama: qwen2.5:7b-instruct / llama3]
    end
```

---

## 2. In-Depth Component Walkthrough

### Component A: The Local PDF Server
* **Where files live**: All 7 papers and the textbook are stored in the local `/home/pratay-karali/Desktop/libraryAI/libraryAI/pdfs/` folder.
* **How redirection works**: When the UI wants to show a reference, it redirects to `/pdfs/papers/Devlin2018_BERT.pdf#page=4`. Flask handles this using `send_from_directory`. The browser opens its built-in PDF reader and automatically jumps to the correct page.

### Component B: The Embedded Graph Database (KuzuDB)
* **No Server Configuration**: KuzuDB is embedded. Flask imports it directly using `import kuzu` and opens the database folder `okf_graph.db` directly from the local disk.
* **Schema Mapping**:
  * `(Document)` nodes representing the 7 local files.
  * `(Chunk)` nodes representing the text segments.
  * `(Concept)` nodes representing the 526 concept definitions.

### Component C: The Local Chat & Tool-Use Agent (Ollama)
* **Connection**: Flask connects to your local Ollama server (running on `http://localhost:11434`) using standard chat APIs.
* **Model Choices**: You can run models like `qwen2.5:7b-instruct` or `llama3:8b` locally. These models are large enough to handle conversational chat and function-calling (tool use) to query the database.
* **Tool-Use Mechanism**:
  1. The student asks: *"Show me where LoRA is explained."*
  2. The Flask server passes this to the local Ollama model alongside a list of available python functions.
  3. The model outputs a tool call: `query_database("MATCH (d:Document)-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->(c:Concept {name: 'Low-Rank Adaptation'}) RETURN d.id, chk.page_number, chk.section_title")`.
  4. Flask runs this Cypher query against the local `okf_graph.db`, collects the results, and passes them back to the model.
  5. The model writes the final answer: *"LoRA is explained in papers/Hu2021_LoRA.pdf on Page 4 (Section 4.1)."*
