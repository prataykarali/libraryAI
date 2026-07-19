# ⚠️ SUPERSEDED (2026-07-16)

> This plan is stale (it predates the local inference stack and still assumes
> Hugging Face Serverless / Spaces deployment). The current single source of
> truth for status and priorities is **`FULL_AUDIT_REPORT.md`** (repo root).
> Content below is kept for historical reference only.

---

# Implementation Plan - Phase 2: Inference & Recommendation System

This plan details the architecture and implementation of the conversational chat, dynamic PDF uploading, and multi-attribute recommendation engine for the university-specific library AI.

---

## User Review Required

> [!IMPORTANT]
> **External API Dependency**: To keep the Hugging Face Space running completely for free, the chat assistant uses the **Hugging Face Serverless Inference API** (running a model like `Qwen/Qwen2.5-72B-Instruct`). This requires a free Hugging Face User Access Token (read permissions).
>
> **Dynamic DB Persistence**: When a librarian uploads a PDF live, the database is updated directly on the Space's local disk. To ensure this data is not lost when the Space restarts, the Space must be configured with a persistent storage directory.

---

## Proposed Changes

We will modify the backend Flask app, implement database catalog syncing, and build the frontend chat/upload components.

---

### Backend Service (Flask API)

#### [MODIFY] [graph_server.py](file:///home/pratay-karali/Desktop/libraryAI/libraryAI/graph_server.py)
* **Library Catalog Integration**: Add endpoints to update book metadata (authors, call numbers, shelf locations, and availability) from a local catalog or CSV.
* **POST `/api/chat` Endpoint**:
  * Set up the Hugging Face Serverless API client.
  * Define database tools (functions) that the agent can call:
    * `query_database(cypher_query)`
    * `get_book_location_and_availability(book_title)`
    * `recommend_books_for_concept(concept_name)`
  * Run the agentic loop: LLM decides to call a database tool, Flask executes the Cypher query on `okf_graph.db` using `kuzu`, and the LLM translates the query results into a conversational answer.
* **POST `/api/upload` Endpoint**:
  * Save uploaded PDF files to the `pdfs/` directory.
  * Extract text blocks and generate chunks using the updated page-aware and citation-ratio robust `pdf_ingestion.py`.
  * Run the small local extraction model (`qwen2.5:0.8b`) to extract concepts, prerequisites, and unlocks.
  * Merge the new concepts into KuzuDB and regenerate `okf_graph.json` so the visual UI updates dynamically.

---

### Frontend Dashboard UI

#### [MODIFY] [index.html](file:///home/pratay-karali/Desktop/libraryAI/libraryAI/graph_ui/index.html)
* **Chat Panel**: Add an interactive chat interface panel next to the visual graph, letting students talk directly to the library assistant.
* **File Upload Component**: Add an upload modal or button allowing librarians to upload new PDFs, displaying a processing spinner with real-time estimation (e.g. 30-45 seconds for a short paper).
* **Metadata Displays**: Update the graph node card view to show Call Number, Shelf Location, and Availability status for each book. Highlight newly uploaded documents in a distinct color (e.g., glowing gold) to visualize their integration.

---

## Verification Plan

We will verify both local development and cloud Hugging Face Space operations.

### Automated Verification
* Run unit tests on the upload and chunking logic:
  `python3 -m pytest libraryAI/test_pdf_ingestion.py`
* Run a mock request script testing `/api/chat` and `/api/upload` endpoints to confirm response formats and correct Cypher execution.

### Manual Verification
1. **Mock Upload Test**: Upload a short 5-page PDF via the UI, verify the progress bar updates, check that the new book node is shown on the graph, and verify it correctly links to existing concept nodes.
2. **Chat Relevance Check**: Ask the chat assistant: *"Which rack is the BERT paper located on?"* or *"What prerequisites do I need before studying LoRA?"*. Verify the model triggers the database tools and responds with the correct page citations and shelf locations.
3. **Availability Toggle Check**: Toggle a book's availability to `FALSE` in the catalog, ask the chat model for a recommendation, and verify it prioritizes alternative available books.
