import pytest
import kuzu
import shutil
import os
import re
from unittest.mock import MagicMock, patch

@pytest.fixture
def tmp_kuzu_db(tmp_path):
    """Creates and returns a temporary Kuzu DB connection, sets up schema, and cleans up."""
    db_path = str(tmp_path / "test_kuzu.db")
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    # Set up schema (Document, Chunk, Concept, and relationships: HAS_CHUNK, MENTIONS, REQUIRES, UNLOCKS, RELATED)
    conn.execute("CREATE NODE TABLE Document (id STRING PRIMARY KEY)")
    conn.execute("""
        CREATE NODE TABLE Chunk (
            id STRING PRIMARY KEY,
            chunk_id STRING,
            page_number INT64,
            section_title STRING,
            text_passage STRING
        )
    """)
    conn.execute("""
        CREATE NODE TABLE Concept (
            id STRING PRIMARY KEY,
            name STRING,
            concept_type STRING,
            difficulty STRING,
            summary STRING
        )
    """)
    conn.execute("CREATE REL TABLE HAS_CHUNK (FROM Document TO Chunk)")
    conn.execute("CREATE REL TABLE MENTIONS (FROM Chunk TO Concept)")
    conn.execute("CREATE REL TABLE REQUIRES (FROM Concept TO Concept, relation_type STRING, source STRING)")
    conn.execute("CREATE REL TABLE UNLOCKS (FROM Concept TO Concept, relation_type STRING, source STRING)")
    conn.execute("CREATE REL TABLE RELATED (FROM Concept TO Concept, relation_type STRING, source STRING)")
    
    # Expose db and db_path as custom attributes on the connection for test utility
    conn.db = db
    conn.db_path = db_path
    
    yield conn
    
    # Explicitly delete connection and database objects to release locks
    del conn
    del db
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture
def sample_chunks():
    """Realistic test chunks with proper provenance."""
    return [
        {
            "doc_id": "math_for_ml.pdf",
            "chunk_id": "chunk_001",
            "section_title": "Introduction to Linear Algebra",
            "page_number": 5,
            "text": "Linear Algebra is the branch of mathematics concerning linear equations and linear functions. It is key for machine learning.",
            "chunk_kind": "prose"
        },
        {
            "doc_id": "math_for_ml.pdf",
            "chunk_id": "chunk_002",
            "section_title": "Introduction to Probability",
            "page_number": 12,
            "text": "Probability theory provides the mathematical language for modeling uncertainty and randomness in data.",
            "chunk_kind": "prose"
        },
        {
            "doc_id": "math_for_ml.pdf",
            "chunk_id": "chunk_003",
            "section_title": "Introduction to Machine Learning",
            "page_number": 20,
            "text": "Machine Learning is a subset of AI that allows systems to learn patterns directly from data. It builds on math foundations.",
            "chunk_kind": "prose"
        },
        {
            "doc_id": "math_for_ml.pdf",
            "chunk_id": "chunk_004",
            "section_title": "Introduction to Deep Learning",
            "page_number": 45,
            "text": "Deep Learning utilizes artificial neural networks with multiple layers to model complex relationships in high-dimensional data.",
            "chunk_kind": "prose"
        }
    ]


@pytest.fixture
def sample_concepts():
    """Gold-standard concepts."""
    return [
        {
            "concept_name": "Basic Algebra",
            "concept_type": "definition",
            "difficulty": "foundational",
            "summary": "Basic Algebra teaches you how to solve equations using variables and fundamental operations.",
            "prerequisites": [],
            "unlocks": ["Linear Algebra"],
            "related_to": [],
            "sources": [
                {
                    "doc_id": "math_for_ml.pdf",
                    "chunk_id": "chunk_001",
                    "page_number": 5,
                    "section_title": "Introduction to Linear Algebra",
                    "text_passage": "Basic Algebra teaches you how to solve equations using variables..."
                }
            ]
        },
        {
            "concept_name": "Linear Algebra",
            "concept_type": "definition",
            "difficulty": "intermediate",
            "summary": "Linear Algebra extends basic algebra to work with vectors and matrices.",
            "prerequisites": ["Basic Algebra"],
            "unlocks": ["Machine Learning"],
            "related_to": [],
            "sources": [
                {
                    "doc_id": "math_for_ml.pdf",
                    "chunk_id": "chunk_002",
                    "page_number": 12,
                    "section_title": "Introduction to Probability",
                    "text_passage": "Linear Algebra extends basic algebra to work with vectors..."
                }
            ]
        },
        {
            "concept_name": "Probability Theory",
            "concept_type": "definition",
            "difficulty": "intermediate",
            "summary": "Probability Theory is the mathematics of randomness and uncertainty.",
            "prerequisites": [],
            "unlocks": ["Machine Learning"],
            "related_to": [],
            "sources": [
                {
                    "doc_id": "math_for_ml.pdf",
                    "chunk_id": "chunk_003",
                    "page_number": 20,
                    "section_title": "Introduction to Machine Learning",
                    "text_passage": "Probability Theory is the mathematics of randomness..."
                }
            ]
        },
        {
            "concept_name": "Machine Learning",
            "concept_type": "definition",
            "difficulty": "advanced",
            "summary": "Machine Learning is a subset of AI that enables systems to learn from data.",
            "prerequisites": ["Linear Algebra", "Probability Theory"],
            "unlocks": ["Deep Learning"],
            "related_to": [],
            "sources": [
                {
                    "doc_id": "math_for_ml.pdf",
                    "chunk_id": "chunk_004",
                    "page_number": 45,
                    "section_title": "Introduction to Deep Learning",
                    "text_passage": "Machine Learning is a subset of AI..."
                }
            ]
        },
        {
            "concept_name": "Deep Learning",
            "concept_type": "definition",
            "difficulty": "advanced",
            "summary": "Deep Learning uses neural networks with multiple layers to process complex patterns.",
            "prerequisites": ["Machine Learning"],
            "unlocks": [],
            "related_to": [],
            "sources": [
                {
                    "doc_id": "math_for_ml.pdf",
                    "chunk_id": "chunk_004",
                    "page_number": 45,
                    "section_title": "Introduction to Deep Learning",
                    "text_passage": "Deep Learning uses neural networks with multiple layers..."
                }
            ]
        }
    ]


@pytest.fixture
def sample_graph(tmp_kuzu_db, sample_concepts):
    """Populates tmp_kuzu_db with a small graph: 5 concepts, 4 edges, plus documents/chunks and mentions."""
    conn = tmp_kuzu_db
    
    def escape(s):
        return str(s).replace("\\", "\\\\").replace("'", "\\'")
        
    created_docs = set()
    created_chunks = set()
    
    # 1. Insert Concept nodes and their source chunk/document lineage
    for concept in sample_concepts:
        name = concept["concept_name"]
        cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
        cid = re.sub(r'_+', '_', cid).strip('_') or 'concept'
        
        concept_type = concept.get("concept_type", "definition")
        difficulty = concept.get("difficulty", "intermediate")
        summary = concept.get("summary", "")
        
        conn.execute(f"""
            CREATE (c:Concept {{
                id: '{cid}',
                name: '{escape(name)}',
                concept_type: '{concept_type}',
                difficulty: '{difficulty}',
                summary: '{escape(summary)}'
            }})
        """)
        
        for src in concept.get("sources", []):
            doc_id = src.get("doc_id")
            chunk_id = src.get("chunk_id")
            if not doc_id or not chunk_id:
                continue
                
            safe_doc_id = escape(doc_id)
            if safe_doc_id not in created_docs:
                conn.execute(f"MERGE (d:Document {{id: '{safe_doc_id}'}})")
                created_docs.add(safe_doc_id)
                
            chunk_db_id = f"{safe_doc_id}_{escape(chunk_id)}"
            if chunk_db_id not in created_chunks:
                page_number = int(src.get("page_number", 0))
                section_title = escape(src.get("section_title", ""))
                text_passage = escape(src.get("text_passage", ""))
                
                conn.execute(f"""
                    CREATE (ch:Chunk {{
                        id: '{chunk_db_id}',
                        chunk_id: '{escape(chunk_id)}',
                        page_number: {page_number},
                        section_title: '{section_title}',
                        text_passage: '{text_passage}'
                    }})
                """)
                conn.execute(f"""
                    MATCH (d:Document {{id: '{safe_doc_id}'}}), (ch:Chunk {{id: '{chunk_db_id}'}})
                    CREATE (d)-[:HAS_CHUNK]->(ch)
                """)
                created_chunks.add(chunk_db_id)
                
            # Link Chunk -> Concept
            conn.execute(f"""
                MATCH (ch:Chunk {{id: '{chunk_db_id}'}}), (c:Concept {{id: '{cid}'}})
                CREATE (ch)-[:MENTIONS]->(c)
            """)
            
    # 2. Insert relationship edges (exactly 4 REQUIRES edges between the 5 concepts)
    edges = [
        ("linear_algebra", "basic_algebra", "REQUIRES", "requires", "math_for_ml.pdf:chunk_002"),
        ("machine_learning", "linear_algebra", "REQUIRES", "requires", "math_for_ml.pdf:chunk_004"),
        ("machine_learning", "probability_theory", "REQUIRES", "requires", "math_for_ml.pdf:chunk_004"),
        ("deep_learning", "machine_learning", "REQUIRES", "requires", "math_for_ml.pdf:chunk_004")
    ]
    for from_id, to_id, rel_table, rel_type, source in edges:
        conn.execute(f"""
            MATCH (a:Concept {{id: '{from_id}'}}), (b:Concept {{id: '{to_id}'}})
            CREATE (a)-[r:{rel_table} {{relation_type: '{rel_type}', source: '{escape(source)}'}}]->(b)
        """)
        
    return conn


@pytest.fixture
def mock_ollama():
    """Patches/mocks ollama.Client calls to avoid connecting to a local Ollama server during tests."""
    mock_client_instance = MagicMock()
    
    mock_list_response = {'models': [{'name': 'qwen3.5:0.8b'}]}
    mock_client_instance.list.return_value = mock_list_response
    
    mock_chat_response = {
        'message': {
            'role': 'assistant',
            'content': 'Mocked Ollama response content.',
            'tool_calls': []
        }
    }
    mock_client_instance.chat.return_value = mock_chat_response
    
    with patch('ollama.Client', return_value=mock_client_instance) as mock_class:
        yield mock_client_instance


@pytest.fixture
def flask_test_client(tmp_kuzu_db, sample_concepts):
    """Initializes concepts data and returns flask client, overriding the db with the temporary one."""
    import inference_server
    
    # Save original values to restore later
    old_db = inference_server.db
    old_concepts = inference_server.CONCEPTS_DATA
    
    # Point server db to the temporary test db
    inference_server.db = tmp_kuzu_db.db
    
    # Build CONCEPTS_DATA from sample_concepts
    concepts_dict = {}
    for concept in sample_concepts:
        name = concept["concept_name"]
        cid = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
        cid = re.sub(r'_+', '_', cid).strip('_') or 'concept'
        concepts_dict[cid] = {
            "id": cid,
            "label": name,
            "name": name,
            "concept_type": concept.get("concept_type", "definition"),
            "difficulty": concept.get("difficulty", "intermediate"),
            "summary": concept.get("summary", "")
        }
    inference_server.CONCEPTS_DATA = concepts_dict
    
    with inference_server.app.test_client() as client:
        yield client
        
    # Restore original values
    inference_server.db = old_db
    inference_server.CONCEPTS_DATA = old_concepts
