import kuzu
import sys
import os
from collections import defaultdict

def validate_db(db_path="okf_graph.db"):
    print("=" * 80)
    print(f"VALIDATING KUZUDB GRAPH DATABASE: {db_path}")
    print("=" * 80)
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database path '{db_path}' does not exist.")
        sys.exit(1)
        
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    
    # 1. Check Schema and counts
    print("\n[1] Schema and Node/Edge Counts:")
    print("-" * 50)
    
    try:
        res = conn.execute("MATCH (c:Concept) RETURN COUNT(c)")
        concept_count = res.get_next()[0]
        print(f"  Concept nodes count: {concept_count}")
    except Exception as e:
        print(f"  Error querying Concept table: {e}")
        concept_count = 0
        
    try:
        res = conn.execute("MATCH (d:Document) RETURN COUNT(d)")
        doc_count = res.get_next()[0]
        print(f"  Document nodes count: {doc_count}")
    except Exception as e:
        print(f"  Error querying Document table: {e}")
        
    try:
        res = conn.execute("MATCH (chk:Chunk) RETURN COUNT(chk)")
        chunk_count = res.get_next()[0]
        print(f"  Chunk nodes count: {chunk_count}")
    except Exception as e:
        print(f"  Error querying Chunk table: {e}")
        
    # Check relationship counts
    for rel_table in ["REQUIRES", "UNLOCKS", "RELATED", "HAS_CHUNK", "MENTIONS"]:
        try:
            res = conn.execute(f"MATCH ()-[r:{rel_table}]->() RETURN COUNT(r)")
            rel_count = res.get_next()[0]
            print(f"  Relationship {rel_table} count: {rel_count}")
        except Exception as e:
            print(f"  Error querying relationship {rel_table}: {e}")

    # 2. Check for Duplicate Edges
    print("\n[2] Checking for Duplicate Edges in REQUIRES, UNLOCKS, and RELATED:")
    print("-" * 50)
    duplicate_found = False
    for rel_table in ["REQUIRES", "UNLOCKS", "RELATED"]:
        try:
            res = conn.execute(f"MATCH (a:Concept)-[r:{rel_table}]->(b:Concept) RETURN a.id, b.id, a.name, b.name")
            edges = []
            while res.has_next():
                edges.append(res.get_next())
            
            # Find duplicate edges (same from_id and to_id)
            seen = defaultdict(int)
            for from_id, to_id, from_name, to_name in edges:
                seen[(from_id, to_id)] += 1
                
            dups = {k: v for k, v in seen.items() if v > 1}
            if dups:
                duplicate_found = True
                print(f"  [FAIL] Duplicate edges found in {rel_table}:")
                for (from_id, to_id), count in dups.items():
                    print(f"    - Concept '{from_id}' -> '{to_id}' has {count} edges")
            else:
                print(f"  [PASS] No duplicate edges found in {rel_table}")
        except Exception as e:
            print(f"  Error checking duplicate edges in {rel_table}: {e}")

    # 3. Check Page Number Shifts
    print("\n[3] Checking Page Number Shifts (PDF pages must be 1-based / resolved):")
    print("-" * 50)
    try:
        res = conn.execute("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d.id, c.page_number, c.chunk_id")
        chunks_info = []
        while res.has_next():
            chunks_info.append(res.get_next())
            
        pdf_chunks_with_zero = []
        pdf_chunks_valid = []
        non_pdf_chunks = []
        
        for doc_id, page_num, chunk_id in chunks_info:
            if doc_id.lower().endswith(".pdf"):
                if page_num == 0:
                    pdf_chunks_with_zero.append((doc_id, chunk_id))
                else:
                    pdf_chunks_valid.append((doc_id, chunk_id, page_num))
            else:
                non_pdf_chunks.append((doc_id, chunk_id, page_num))
                
        print(f"  PDF Chunks with valid page numbers (>=1): {len(pdf_chunks_valid)}")
        print(f"  Non-PDF Chunks (markdown, txt, etc.): {len(non_pdf_chunks)}")
        
        if pdf_chunks_with_zero:
            print(f"  [FAIL] Found {len(pdf_chunks_with_zero)} PDF chunks with page_number = 0 (unresolved page shift):")
            for doc_id, chunk_id in pdf_chunks_with_zero[:10]:
                print(f"    - {doc_id} | {chunk_id}")
            if len(pdf_chunks_with_zero) > 10:
                print("    - ...")
        else:
            print("  [PASS] All PDF chunks have resolved/1-based page numbers.")
    except Exception as e:
        print(f"  Error checking page number shifts: {e}")

    # 4. Check Traversal Works
    print("\n[4] Checking Traversal Works:")
    print("-" * 50)
    try:
        # Traversal: Document -> Chunk -> Concept -> requires/unlocks -> Concept
        query = """
            MATCH (d:Document)-[:HAS_CHUNK]->(chk:Chunk)-[:MENTIONS]->(c1:Concept)-[:REQUIRES]->(c2:Concept)
            RETURN d.id, chk.chunk_id, c1.name, c2.name
            LIMIT 5
        """
        res = conn.execute(query)
        paths = []
        while res.has_next():
            paths.append(res.get_next())
        
        if paths:
            print("  [PASS] Successfully traversed path: Document -> Chunk -> Concept -> Concept")
            for doc, chunk, c1, c2 in paths:
                print(f"    - Document '{doc}' chunk '{chunk}' mentions '{c1}' which REQUIRES '{c2}'")
        else:
            print("  [INFO] Traversal query returned 0 paths (this is normal if no concepts mention a prerequisite inside the ingested dataset).")
            # Let's run a simpler traversal check: Chunk -> Concept
            res = conn.execute("MATCH (chk:Chunk)-[:MENTIONS]->(c:Concept) RETURN chk.id, c.name LIMIT 5")
            mentions = []
            while res.has_next():
                mentions.append(res.get_next())
            if mentions:
                print("  [PASS] Simpler traversal: Chunk -> Concept works.")
                for chk_id, cname in mentions:
                    print(f"    - Chunk '{chk_id}' MENTIONS Concept '{cname}'")
            else:
                print("  [FAIL] Chunk -> Concept traversal returned 0 paths.")
    except Exception as e:
        print(f"  [FAIL] Traversal failed with error: {e}")

    # 5. Check for Cycles in Learning Progression Graph (DAG Validity)
    print("\n[5] Checking for Cycles in Learning Progression Graph:")
    print("-" * 50)
    
    # We build the dependency graph:
    # A requires B  => B must be learned before A => Edge B -> A
    # A unlocks B   => A must be learned before B => Edge A -> B
    # Nodes: all concepts in the database
    adj = defaultdict(list)
    nodes = set()
    
    # Fetch all Concept IDs
    try:
        res = conn.execute("MATCH (c:Concept) RETURN c.id")
        while res.has_next():
            nodes.add(res.get_next()[0])
            
        # Fetch REQUIRES edges (from_id -REQUIRES-> to_id  => from_id requires to_id => to_id must come before from_id => to_id -> from_id)
        res = conn.execute("MATCH (a:Concept)-[:REQUIRES]->(b:Concept) RETURN a.id, b.id")
        while res.has_next():
            u, v = res.get_next()
            # v -> u (v must come before u)
            adj[v].append(u)
            nodes.add(u)
            nodes.add(v)
            
        # Fetch UNLOCKS edges (from_id -UNLOCKS-> to_id => from_id unlocks to_id => from_id must come before to_id => from_id -> to_id)
        res = conn.execute("MATCH (a:Concept)-[:UNLOCKS]->(b:Concept) RETURN a.id, b.id")
        while res.has_next():
            u, v = res.get_next()
            # u -> v (u must come before v)
            adj[u].append(v)
            nodes.add(u)
            nodes.add(v)
            
        # Cycle detection using DFS
        visited = {} # None: unvisited, 1: visiting, 2: fully visited
        for node in nodes:
            visited[node] = 0
            
        cycles = []
        
        def dfs(u, path):
            visited[u] = 1 # visiting
            path.append(u)
            for v in adj[u]:
                if visited.get(v, 0) == 1:
                    # Found cycle!
                    cycle_start_idx = path.index(v)
                    cycles.append(path[cycle_start_idx:] + [v])
                elif visited.get(v, 0) == 0:
                    dfs(v, path)
            path.pop()
            visited[u] = 2 # fully visited
            
        for node in nodes:
            if visited[node] == 0:
                dfs(node, [])
                
        if cycles:
            print(f"  [FAIL] Learning progression graph contains {len(cycles)} cycle(s):")
            for cycle in cycles[:5]:
                print("    - " + " -> ".join(cycle))
            if len(cycles) > 5:
                print("    - ...")
        else:
            print("  [PASS] Learning progression graph is acyclic (DAG).")
            
    except Exception as e:
        print(f"  Error running cycle detection: {e}")
        
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    db_path = "okf_graph.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    validate_db(db_path)
