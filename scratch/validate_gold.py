import json

attention_concepts = [
    "Sequence-to-Sequence Models",
    "Recurrent Neural Networks",
    "Convolutional Neural Networks",
    "Attention Mechanism",
    "Self-Attention",
    "Scaled Dot-Product Attention",
    "Query, Key, and Value Vectors",
    "Softmax Function",
    "Scale Factor",
    "Masked Self-Attention",
    "Multi-Head Attention",
    "Attention Heads",
    "Encoder-Decoder Attention",
    "Transformer Encoder",
    "Transformer Decoder",
    "Feed-Forward Networks",
    "Residual Connections",
    "Layer Normalization",
    "Positional Encoding",
    "Sinusoidal Embeddings",
    "Linear Transformation",
    "Teacher Forcing",
    "Beam Search",
    "Label Smoothing",
    "Adam Optimizer"
]

attention_edges = [
    {"source": "Sequence-to-Sequence Models", "target": "Attention Mechanism", "type": "requires"},
    {"source": "Recurrent Neural Networks", "target": "Self-Attention", "type": "requires"},
    {"source": "Convolutional Neural Networks", "target": "Self-Attention", "type": "requires"},
    {"source": "Attention Mechanism", "target": "Self-Attention", "type": "requires"},
    {"source": "Self-Attention", "target": "Scaled Dot-Product Attention", "type": "requires"},
    {"source": "Query, Key, and Value Vectors", "target": "Scaled Dot-Product Attention", "type": "requires"},
    {"source": "Softmax Function", "target": "Scaled Dot-Product Attention", "type": "requires"},
    {"source": "Scale Factor", "target": "Scaled Dot-Product Attention", "type": "requires"},
    {"source": "Scaled Dot-Product Attention", "target": "Masked Self-Attention", "type": "requires"},
    {"source": "Scaled Dot-Product Attention", "target": "Multi-Head Attention", "type": "requires"},
    {"source": "Attention Heads", "target": "Multi-Head Attention", "type": "requires"},
    {"source": "Scaled Dot-Product Attention", "target": "Encoder-Decoder Attention", "type": "requires"},
    {"source": "Linear Transformation", "target": "Multi-Head Attention", "type": "requires"},
    {"source": "Multi-Head Attention", "target": "Transformer Encoder", "type": "requires"},
    {"source": "Feed-Forward Networks", "target": "Transformer Encoder", "type": "requires"},
    {"source": "Layer Normalization", "target": "Transformer Encoder", "type": "requires"},
    {"source": "Residual Connections", "target": "Transformer Encoder", "type": "requires"},
    {"source": "Transformer Encoder", "target": "Transformer Decoder", "type": "requires"},
    {"source": "Masked Self-Attention", "target": "Transformer Decoder", "type": "requires"},
    {"source": "Encoder-Decoder Attention", "target": "Transformer Decoder", "type": "requires"},
    {"source": "Positional Encoding", "target": "Sinusoidal Embeddings", "type": "requires"},
    {"source": "Positional Encoding", "target": "Transformer Encoder", "type": "requires"},
    {"source": "Positional Encoding", "target": "Transformer Decoder", "type": "requires"},
    {"source": "Teacher Forcing", "target": "Transformer Decoder", "type": "requires"},
    {"source": "Transformer Decoder", "target": "Beam Search", "type": "requires"},
    {"source": "Label Smoothing", "target": "Adam Optimizer", "type": "requires"}
]

lora_concepts = [
    "Fine-Tuning",
    "Parameter-Efficient Fine-Tuning",
    "Adapter Layers",
    "Prefix Tuning",
    "Low-Rank Adaptation",
    "Weight Matrix Reparameterization",
    "Intrinsic Rank",
    "Rank Decomposition Matrices",
    "Low-Rank Matrix",
    "Query and Value Projection Matrices",
    "Frozen Pre-trained Weights",
    "Scaling Factor Alpha",
    "Inference Latency",
    "Weight Merging",
    "VRAM Optimization",
    "Task Switching Cost",
    "Adam Optimizer",
    "Gradient Update",
    "Singular Value Decomposition",
    "Subspace Similarity"
]

lora_edges = [
    {"source": "Fine-Tuning", "target": "Parameter-Efficient Fine-Tuning", "type": "requires"},
    {"source": "Parameter-Efficient Fine-Tuning", "target": "Adapter Layers", "type": "requires"},
    {"source": "Parameter-Efficient Fine-Tuning", "target": "Prefix Tuning", "type": "requires"},
    {"source": "Parameter-Efficient Fine-Tuning", "target": "Low-Rank Adaptation", "type": "requires"},
    {"source": "Low-Rank Matrix", "target": "Rank Decomposition Matrices", "type": "requires"},
    {"source": "Rank Decomposition Matrices", "target": "Weight Matrix Reparameterization", "type": "requires"},
    {"source": "Intrinsic Rank", "target": "Low-Rank Adaptation", "type": "requires"},
    {"source": "Low-Rank Adaptation", "target": "Weight Matrix Reparameterization", "type": "requires"},
    {"source": "Frozen Pre-trained Weights", "target": "Weight Matrix Reparameterization", "type": "requires"},
    {"source": "Query and Value Projection Matrices", "target": "Weight Matrix Reparameterization", "type": "requires"},
    {"source": "Scaling Factor Alpha", "target": "Weight Matrix Reparameterization", "type": "requires"},
    {"source": "Adapter Layers", "target": "Inference Latency", "type": "requires"},
    {"source": "Prefix Tuning", "target": "Inference Latency", "type": "requires"},
    {"source": "Weight Matrix Reparameterization", "target": "Weight Merging", "type": "requires"},
    {"source": "Inference Latency", "target": "Weight Merging", "type": "requires"},
    {"source": "Weight Merging", "target": "Task Switching Cost", "type": "requires"},
    {"source": "VRAM Optimization", "target": "Low-Rank Adaptation", "type": "requires"},
    {"source": "Adam Optimizer", "target": "VRAM Optimization", "type": "requires"},
    {"source": "Gradient Update", "target": "Rank Decomposition Matrices", "type": "requires"},
    {"source": "Singular Value Decomposition", "target": "Subspace Similarity", "type": "requires"},
    {"source": "Rank Decomposition Matrices", "target": "Singular Value Decomposition", "type": "requires"}
]

def check_validity(name, concepts, edges):
    print(f"Checking {name}...")
    # Check concept count
    print(f"  Number of concepts: {len(concepts)}")
    
    # Check uniqueness of concepts
    if len(concepts) != len(set(concepts)):
        print("  WARNING: Duplicate concepts found!")
        
    concept_set = set(concepts)
    
    # Check edge endpoints
    for edge in edges:
        s = edge["source"]
        t = edge["target"]
        if s not in concept_set:
            print(f"  ERROR: Source concept '{s}' not in concepts list!")
        if t not in concept_set:
            print(f"  ERROR: Target concept '{t}' not in concepts list!")
            
    # Check for cycles using DFS
    adj = {c: [] for c in concepts}
    for edge in edges:
        s = edge["source"]
        t = edge["target"]
        if s in adj:
            adj[s].append(t)
            
    visited = {}
    
    def dfs(node):
        visited[node] = 1 # visiting
        for neighbor in adj.get(node, []):
            if visited.get(neighbor, 0) == 1:
                print(f"  ERROR: Cycle detected containing {node} -> {neighbor}")
                return False
            elif visited.get(neighbor, 0) == 0:
                if not dfs(neighbor):
                    return False
        visited[node] = 2 # visited
        return True

    is_dag = True
    for c in concepts:
        if visited.get(c, 0) == 0:
            if not dfs(c):
                is_dag = False
                break
                
    if is_dag:
        print("  DAG Check passed: No cycles found.")
    else:
        print("  DAG Check failed: Cycle(s) found.")

check_validity("Attention Is All You Need", attention_concepts, attention_edges)
check_validity("LoRA", lora_concepts, lora_edges)
