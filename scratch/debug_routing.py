import sys
import shutil
import re
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kuzu
import inference_server

tmp_dir = Path(__file__).resolve().parent / "test_kuzu_db_temp"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)

db_path = str(tmp_dir / "test_kuzu.db")
db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

# Set up schema
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

# Gold-standard concepts
sample_concepts = [
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

def escape(s):
    return str(s).replace("\\", "\\\\").replace("'", "\\'")
    
created_docs = set()
created_chunks = set()

# Insert Concept nodes
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
        
# Insert relationship edges
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

# Rebind inference_server state
inference_server.db = db
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
import archipelago.inference.routing as routing
routing._run_dual_pass_guard = lambda q: False

# Run query routing
q = "hi i wanna start learning AIML"
res = inference_server.resolve_query_routing(q)
print("Routing result for 'What is Linear Algebra?':")
import pprint
pprint.pprint(res)

# Clean up
del conn
del db
shutil.rmtree(tmp_dir)
