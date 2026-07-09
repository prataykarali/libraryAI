#!/usr/bin/env python3
"""
KuzuDB Graph Database for OKF Concepts and Relationships
Creates a knowledge graph with concepts as nodes and prerequisites/unlocks as edges
"""

import kuzu
import os
import json
import re
from okf_extraction import extract_batch
from mock_data import MOCK_TEXT_CHUNKS

class OKFGraphDB:
    def __init__(self, db_path="okf_graph.db"):
        """Initialize KuzuDB database"""
        self.db_path = db_path
        
        # Remove existing database if it exists
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                import shutil
                shutil.rmtree(db_path, ignore_errors=True)
        
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
    
    def create_schema(self):
        """Create tables for concepts and relationships"""
        print("Creating schema...")
        
        # Create Concept node table
        try:
            self.conn.execute("""
                CREATE NODE TABLE Concept (
                    id STRING PRIMARY KEY,
                    name STRING,
                    summary STRING
                )
            """)
            print("[OK] Created Concept table")
        except Exception as e:
            print(f"Concept table: {e}")
        
        # Create PREREQUISITE relationship (points from A to B means B is prereq of A)
        try:
            self.conn.execute("""
                CREATE REL TABLE PREREQUISITE (
                    FROM Concept TO Concept,
                    relation_type STRING
                )
            """)
            print("[OK] Created PREREQUISITE relationship")
        except Exception as e:
            print(f"PREREQUISITE relationship: {e}")
        
        # Create UNLOCKS relationship (A unlocks B)
        try:
            self.conn.execute("""
                CREATE REL TABLE UNLOCKS (
                    FROM Concept TO Concept,
                    relation_type STRING
                )
            """)
            print("[OK] Created UNLOCKS relationship")
        except Exception as e:
            print(f"UNLOCKS relationship: {e}")
    
    def add_concepts(self, okf_data):
        """Add concept nodes to the database"""
        print("\nAdding concepts to graph...")
        
        for i, concept in enumerate(okf_data, 1):
            name = concept.get('concept_name', '')
            summary = concept.get('summary', '')
            concept_id = ''.join(ch if ch.isalnum() else '_' for ch in name.lower())
            concept_id = re.sub(r'_+', '_', concept_id).strip('_') or 'concept'
            
            try:
                self.conn.execute(
                    f"""
                    CREATE (c:Concept {{
                        id: '{concept_id}',
                        name: '{name.replace("'", "''")}',
                        summary: '{summary.replace("'", "''")}' 
                    }})
                    """
                )
                print(f"  {i}. Added: {name}")
            except Exception as e:
                print(f"  Error adding {name}: {e}")
    
    def add_relationships(self, okf_data):
        """Add prerequisite and unlock relationships"""
        print("\nAdding relationships...")
        
        # Create a mapping of concept names to IDs
        concept_map = {
            c.get('concept_name', '').lower(): ''.join(ch if ch.isalnum() else '_' for ch in c.get('concept_name', '').lower())
            for c in okf_data
        }
        concept_map = {k: re.sub(r'_+', '_', v).strip('_') or 'concept' for k, v in concept_map.items()}
        
        rel_count = 0
        
        for concept in okf_data:
            concept_name = concept.get('concept_name', '')
            concept_id = ''.join(ch if ch.isalnum() else '_' for ch in concept_name.lower())
            concept_id = re.sub(r'_+', '_', concept_id).strip('_') or 'concept'

            prerequisites = concept.get('prerequisites', [])
            if isinstance(prerequisites, str):
                prerequisites = [prerequisites]
            for prereq in prerequisites:
                if not isinstance(prereq, str):
                    continue
                prereq_lower = prereq.lower()
                matched_concept = None
                for existing_name, existing_id in concept_map.items():
                    if (prereq_lower in existing_name or existing_name in prereq_lower or
                        prereq_lower.split()[0] in existing_name):
                        matched_concept = existing_id
                        break
                if matched_concept and matched_concept != concept_id:
                    try:
                        self.conn.execute(f"""
                            MATCH (from:Concept {{id: '{concept_id}'}}),
                                  (to:Concept {{id: '{matched_concept}'}})
                            CREATE (from)-[:PREREQUISITE {{relation_type: 'requires'}}]->(to)
                        """)
                        rel_count += 1
                    except Exception:
                        pass

            unlocks = concept.get('unlocks', [])
            if isinstance(unlocks, str):
                unlocks = [unlocks]
            for unlock in unlocks:
                if not isinstance(unlock, str):
                    continue
                unlock_lower = unlock.lower()
                matched_concept = None
                for existing_name, existing_id in concept_map.items():
                    if (unlock_lower in existing_name or existing_name in unlock_lower or
                        unlock_lower.split()[0] in existing_name):
                        matched_concept = existing_id
                        break
                if matched_concept and matched_concept != concept_id:
                    try:
                        self.conn.execute(f"""
                            MATCH (from:Concept {{id: '{concept_id}'}}),
                                  (to:Concept {{id: '{matched_concept}'}})
                            CREATE (from)-[:UNLOCKS {{relation_type: 'enables'}}]->(to)
                        """)
                        rel_count += 1
                    except Exception:
                        pass
        
        print(f"[OK] Added {rel_count} relationships")
    
    def query_learning_path(self):
        """Query the graph to find learning paths"""
        print("\n" + "="*70)
        print("LEARNING PATHS IN GRAPH")
        print("="*70)
        
        try:
            result = self.conn.execute("""
                MATCH (c:Concept)
                RETURN c.name AS concept_name, 
                       c.summary AS summary
                ORDER BY c.name
            """)
            
            concepts = []
            while result.has_next():
                concepts.append(result.get_next())
            print(f"\nTotal Concepts: {len(concepts)}")
            for concept in concepts:
                print(f"  - {concept[0]}")
        except Exception as e:
            print(f"Error querying concepts: {e}")
    
    def query_prerequisites(self, concept_name):
        """Find prerequisites for a concept"""
        concept_id = concept_name.lower().replace(" ", "_")
        
        try:
            result = self.conn.execute(f"""
                MATCH (from:Concept)-[:PREREQUISITE]->(to:Concept)
                WHERE from.id = '{concept_id}'
                RETURN from.name AS concept, to.name AS prerequisite
            """)
            
            prereqs = []
            while result.has_next():
                prereqs.append(result.get_next())
            if prereqs:
                print(f"\n{concept_name} requires:")
                for prereq in prereqs:
                    print(f"  <- {prereq[1]}")
            else:
                print(f"\n{concept_name} has no prerequisites")
        except Exception as e:
            print(f"Error querying prerequisites: {e}")
    
    def query_unlocks(self, concept_name):
        """Find what a concept unlocks"""
        concept_id = concept_name.lower().replace(" ", "_")
        
        try:
            result = self.conn.execute(f"""
                MATCH (from:Concept)-[:UNLOCKS]->(to:Concept)
                WHERE from.id = '{concept_id}'
                RETURN from.name AS concept, to.name AS unlocks
            """)
            
            unlocks = []
            while result.has_next():
                unlocks.append(result.get_next())
            if unlocks:
                print(f"\n{concept_name} unlocks:")
                for unlock in unlocks:
                    print(f"  -> {unlock[1]}")
            else:
                print(f"\n{concept_name} doesn't unlock any concepts in the graph")
        except Exception as e:
            print(f"Error querying unlocks: {e}")
    
    def get_graph_stats(self):
        """Get statistics about the graph"""
        print("\n" + "="*70)
        print("GRAPH STATISTICS")
        print("="*70)
        
        try:
            # Count concepts
            result = self.conn.execute("MATCH (c:Concept) RETURN COUNT(c) AS count")
            concept_count = result.get_next()[0]
            print(f"Total Concepts: {concept_count}")
            
            # Count relationships
            result = self.conn.execute("""
                MATCH ()-[r]->() 
                RETURN COUNT(r) AS count
            """)
            rel_count = result.get_next()[0]
            print(f"Total Relationships: {rel_count}")
            
            # Most connected concepts
            result = self.conn.execute("""
                MATCH (c:Concept)-[r]->()
                RETURN c.name AS concept, COUNT(r) AS outgoing
                ORDER BY outgoing DESC
                LIMIT 5
            """)
            
            print(f"\nMost Connected Concepts (by outgoing edges):")
            while result.has_next():
                row = result.get_next()
                print(f"  {row[0]}: {row[1]} connections")
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def export_graph_json(self, output_file="okf_graph.json"):
        """Export graph structure to JSON"""
        try:
            # Get all concepts
            result = self.conn.execute("""
                MATCH (c:Concept)
                RETURN c.id AS id, c.name AS name, c.summary AS summary
            """)
            
            concepts = {}
            while result.has_next():
                row = result.get_next()
                concepts[row[0]] = {"name": row[1], "summary": row[2]}
            
            # Get all relationships
            result = self.conn.execute("""
                MATCH (from:Concept)-[r]->(to:Concept)
                RETURN from.id AS from_id, to.id AS to_id, r.relation_type AS rel_type
            """)
            
            relationships = []
            while result.has_next():
                row = result.get_next()
                relationships.append({
                    "from": row[0],
                    "to": row[1],
                    "type": row[2]
                })
            
            export_data = {
                "concepts": concepts,
                "relationships": relationships,
                "stats": {
                    "concept_count": len(concepts),
                    "relationship_count": len(relationships)
                }
            }
            
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            
            print(f"\n[OK] Graph exported to {output_file}")
        except Exception as e:
            print(f"Error exporting graph: {e}")
    
    def close(self):
        """Close database connection"""
        pass  # KuzuDB handles cleanup automatically

def main():
    print("="*70)
    print("OKF GRAPH DATABASE - KuzuDB")
    print("="*70)
    
    # Extract OKF data
    print("\nStep 1: Extracting OKF data from mock chunks...")
    text_chunks = [chunk["text"] for chunk in MOCK_TEXT_CHUNKS]
    okf_data = extract_batch(text_chunks)
    
    if not okf_data:
        print("Failed to extract OKF data")
        return
    
    # Create graph database
    print("\nStep 2: Creating KuzuDB graph...")
    graph = OKFGraphDB()
    
    # Create schema
    graph.create_schema()
    
    # Add concepts
    graph.add_concepts(okf_data)
    
    # Add relationships
    graph.add_relationships(okf_data)
    
    # Query graph
    graph.query_learning_path()
    graph.get_graph_stats()
    
    # Show specific relationships
    print("\n" + "="*70)
    print("CONCEPT RELATIONSHIPS")
    print("="*70)
    
    key_concepts = ["Basic Algebra", "Machine Learning", "Deep Learning"]
    for concept in key_concepts:
        try:
            graph.query_prerequisites(concept)
            graph.query_unlocks(concept)
        except:
            pass
    
    # Export graph
    graph.export_graph_json("mock_graph.json")
    
    # Close connection
    graph.close()
    
    print("\n" + "="*70)
    print("[OK] Graph database created and populated successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
