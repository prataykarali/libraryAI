"""library_queries.py — Cypher queries and fuzzy matching for book/chapter library metadata."""
from __future__ import annotations
import re
import kuzu
from thefuzz import fuzz
from ingestion_worker import graph_lock
from archipelago.inference import state as st
from archipelago.inference.ranking import rank_concepts
from okf.config import infer_source_category

# Pedagogy weights: prefer textbooks for "books", papers for "papers"
_BOOK_CAT_WEIGHT = {
    "textbook": 3.0,
    "web_syllabus": 1.4,
    "markdown": 1.2,
    "paper": 0.35,
    "pdf": 1.0,
    "text": 0.9,
    "unknown": 0.8,
}
_PAPER_CAT_WEIGHT = {
    "paper": 3.0,
    "textbook": 1.2,
    "web_syllabus": 0.9,
    "markdown": 0.8,
    "pdf": 1.5,
    "text": 0.7,
    "unknown": 0.8,
}

def clean_topic_query(query: str) -> str:
    q = re.sub(
        r"\b(suggest|recommend|top|best|\d+|books?|papers?|readings?|textbooks?|"
        r"for|the|topic|about|on|of|me|show|find|list|a|an|some|regarding)\b",
        " ",
        query,
        flags=re.I,
    )
    q = re.sub(r"\s+", " ", q).strip(" :\"'?.")
    return q

def clean_book_query(query: str) -> str:
    q = re.sub(r"\b(what|are|the|chapters|sections|of|in|book|paper|show|me|table|contents)\b", "", query, flags=re.I)
    return q.strip(" :\"'?.")

def parse_chapter_lookup_query(query: str) -> tuple[str, str]:
    query_lower = query.lower()
    splitters = ["discusses", "discussed", "discuss", "mentioning", "mentions", "mentioned", "mention", "containing", "contains", "contained", "contain", "covering", "covers", "covered", "cover", "is in", "about", "has"]
    for splitter in splitters:
        if splitter in query_lower:
            idx = query_lower.find(splitter)
            book_part = query[:idx]
            concept_part = query[idx + len(splitter):]
            book = re.sub(r"\b(which|chapter|section|of|book|paper|where|in)\b", "", book_part, flags=re.I).strip(" :\"'?.")
            concept = concept_part.strip(" :\"'?.")
            return book, concept
    return query, query

def resolve_matching_doc(book_query: str) -> tuple[str, str, float] | None:
    """Find the closest matching document ID and title in KuzuDB using fuzzy matching."""
    cleaned = clean_book_query(book_query)
    if not cleaned:
        return None
        
    docs = []
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            res = conn.execute("MATCH (d:Document) RETURN d.id, d.title")
            while res.has_next():
                row = res.get_next()
                docs.append((row[0], row[1] or ""))
    except Exception as e:
        print(f"Failed to fetch documents: {e}")
        return None

    if not docs:
        return None

    best_match = None
    best_score = -1.0
    for doc_id, title in docs:
        # Score against filename (id) and title
        s1 = fuzz.token_set_ratio(cleaned.lower(), doc_id.lower())
        s2 = fuzz.token_set_ratio(cleaned.lower(), title.lower()) if title else 0.0
        score = max(s1, s2)
        if score > best_score:
            best_score = score
            best_match = (doc_id, title or doc_id)
            
    if best_match and best_score >= 45:
        return best_match[0], best_match[1], best_score
    return None

def _prefer_papers_query(topic_query: str) -> bool:
    ql = (topic_query or "").lower()
    wants_papers = bool(re.search(r"\bpapers?\b|\barticles?\b|\bbibliography\b", ql))
    wants_books = bool(re.search(r"\bbooks?\b|\btextbooks?\b|\breadings?\b", ql))
    # Explicit papers win; pure "books" prefers textbooks
    return wants_papers and not wants_books


def _doc_pedagogy_score(rec: dict, prefer_papers: bool) -> float:
    """Combine mention count with source-category weight (textbook vs paper).

    For book-style queries, also boost known starter textbooks by title/path.
    """
    mentions = float(rec.get("mentions") or 0)
    cat = rec.get("source_category") or infer_source_category(rec.get("id") or "")
    weights = _PAPER_CAT_WEIGHT if prefer_papers else _BOOK_CAT_WEIGHT
    w = float(weights.get(cat, 1.0))
    doc_id = (rec.get("id") or "").lower()
    title = (rec.get("title") or "").lower()
    # Starter pedagogy boosts (book queries only)
    if not prefer_papers:
        if "textbooks/" in doc_id or "deisenroth" in doc_id or "math" in title:
            w *= 1.35
        if "syllab" in doc_id or "seed" in doc_id:
            w *= 1.15
        # Pure research papers stay downranked for "books on…"
        if doc_id.startswith("papers/") or cat == "paper":
            w *= 0.85
    # log-ish dampening so one huge paper mention dump doesn't dominate
    return (1.0 + mentions) * w


def get_books_for_topic(topic_query: str, limit: int = 5) -> list[dict]:
    """Suggest books/papers related to a topic.

    Ranking = concept mention counts × source-category pedagogy weight
    (textbooks preferred for book queries; papers preferred for paper queries).
    """
    cleaned_topic = clean_topic_query(topic_query)
    if not cleaned_topic:
        return []
        
    ranked = rank_concepts(cleaned_topic, top_k=5)
    if not ranked:
        return []
        
    concept_ids = [r["id"] for r in ranked if float(r.get("cos", 0)) > 0.15]
    if not concept_ids:
        # Fallback to top ranked if similarity is low
        concept_ids = [ranked[0]["id"]]
        
    concept_id_to_name = {r["id"]: r["label"] for r in ranked}
    prefer_papers = _prefer_papers_query(topic_query)

    doc_mentions = {}
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            # Find document chunks mentioning these concepts
            for cid in concept_ids:
                safe_cid = cid.replace("'", "\\'")
                res = conn.execute(f"""
                    MATCH (d:Document)-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->(co:Concept {{id: '{safe_cid}'}})
                    RETURN d.id, d.title, count(chk)
                """)
                while res.has_next():
                    row = res.get_next()
                    doc_id, title, count = row[0], row[1] or row[0], int(row[2])
                    rec = doc_mentions.setdefault(
                        doc_id,
                        {
                            "id": doc_id,
                            "title": title,
                            "mentions": 0,
                            "matched": set(),
                            "source_category": infer_source_category(doc_id),
                        },
                    )
                    rec["mentions"] += count
                    rec["matched"].add(concept_id_to_name.get(cid, cid))
    except Exception as e:
        print(f"get_books_for_topic query failed: {e}")
        return []

    # Format and sort by pedagogy-weighted score
    results = []
    for rec in doc_mentions.values():
        rec["matched"] = sorted(list(rec["matched"]))
        rec["source_category"] = rec.get("source_category") or infer_source_category(rec["id"])
        rec["score"] = _doc_pedagogy_score(rec, prefer_papers)
        results.append(rec)
        
    results.sort(key=lambda x: (x.get("score") or 0, x.get("mentions") or 0), reverse=True)
    return results[:limit]

def get_chapters_of_book(book_query: str) -> tuple[str, list[dict]] | None:
    """Get the structured list of chapters/sections in a book, ordered by page number."""
    matched = resolve_matching_doc(book_query)
    if not matched:
        return None
        
    doc_id, title, _ = matched
    chapters_map = {}
    
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            safe_doc = doc_id.replace("'", "\\'")
            res = conn.execute(f"""
                MATCH (d:Document {{id: '{safe_doc}'}})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.section_title, c.page_number
            """)
            while res.has_next():
                row = res.get_next()
                sect, page = row[0], row[1]
                if not sect or not sect.strip():
                    continue
                sect = sect.strip()
                page = int(page) if page is not None else 0
                if sect not in chapters_map or page < chapters_map[sect]:
                    chapters_map[sect] = page
    except Exception as e:
        print(f"get_chapters_of_book failed: {e}")
        return None

    # Sort chapters by page number
    sorted_chapters = [{"section_title": k, "page_number": v} for k, v in chapters_map.items()]
    sorted_chapters.sort(key=lambda x: x["page_number"])
    return title, sorted_chapters

def get_chapters_containing_concept(query: str) -> tuple[str, str, list[dict]] | None:
    """Find chapters of book X that mention concept Y."""
    book_part, concept_part = parse_chapter_lookup_query(query)
    
    doc_match = resolve_matching_doc(book_part)
    if not doc_match:
        return None
    doc_id, title, _ = doc_match

    # Find the matching concept
    ranked = rank_concepts(concept_part, top_k=2)
    if not ranked:
        return None
    concept_id = ranked[0]["id"]
    concept_name = ranked[0]["label"]

    chapters_map = {}
    try:
        with graph_lock.read_lock():
            conn = kuzu.Connection(st.db)
            safe_doc = doc_id.replace("'", "\\'")
            safe_cid = concept_id.replace("'", "\\'")
            res = conn.execute(f"""
                MATCH (d:Document {{id: '{safe_doc}'}})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(co:Concept {{id: '{safe_cid}'}})
                RETURN c.section_title, c.page_number
            """)
            while res.has_next():
                row = res.get_next()
                sect, page = row[0], row[1]
                if not sect or not sect.strip():
                    continue
                sect = sect.strip()
                page = int(page) if page is not None else 0
                if sect not in chapters_map or page < chapters_map[sect]:
                    chapters_map[sect] = page
    except Exception as e:
        print(f"get_chapters_containing_concept failed: {e}")
        return None

    sorted_chapters = [{"section_title": k, "page_number": v} for k, v in chapters_map.items()]
    sorted_chapters.sort(key=lambda x: x["page_number"])
    return title, concept_name, sorted_chapters
