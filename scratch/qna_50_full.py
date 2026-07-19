"""Run the 50-prompt Evolution Dataset against the LIVE inference server
(http://localhost:5051/api/chat) and write a readable Q&A transcript."""
import json, urllib.request

API = "http://localhost:5051/api/chat"

PROMPTS = [
    ("C1 Context Chaining", "A", "Hey buddy, what's up? Can you explain what a transformer is?"),
    ("C1 Context Chaining", "A", "Cool. Now, how does it handle long sequences?"),
    ("C1 Context Chaining", "A", "What about the attention mechanism? Does it need positional encoding?"),
    ("C1 Context Chaining", "B", "Man, I'm tired. Why is math so hard? Just give me the gist of a Jacobian."),
    ("C1 Context Chaining", "B", "And how is that different from a Hessian?"),
    ("C1 Context Chaining", "C", "Good evening! Please define a Recurrent Layer for me."),
    ("C1 Context Chaining", "C", "Does it perform better than a transformer on short sequences?"),
    ("C1 Context Chaining", "D", "Hi! What is a latent variable?"),
    ("C1 Context Chaining", "D", "And where does it fit into the Maximum Likelihood Estimation process?"),
    ("C1 Context Chaining", "D", "Can it be observed directly?"),
    ("C2 Graph Reasoning", None, "Compare the curriculum path of a student learning LoRA vs. one learning vanilla BERT. Where do their paths diverge?"),
    ("C2 Graph Reasoning", None, "If I skip Linear Algebra, what nodes in the graph become inaccessible?"),
    ("C2 Graph Reasoning", None, "Is 'Probability Theory' a prerequisite for 'Masked Language Modeling'? Explain the multi-hop path."),
    ("C2 Graph Reasoning", None, "How does the 'Self-Attention' mechanism connect to 'Singular Value Decomposition' through the lens of dimensionality?"),
    ("C2 Graph Reasoning", None, "Contrast the prerequisites for RAG-Token and GraphRAG. Do they share any common upstream nodes?"),
    ("C2 Graph Reasoning", None, "Map the relationship between 'Gradient Descent' and 'Automatic Differentiation'."),
    ("C2 Graph Reasoning", None, "Identify the 'Bridge Nodes' connecting 'Gaussian Distributions' to 'Masked Language Modeling'."),
    ("C2 Graph Reasoning", None, "Trace a path from 'Orthonormal Basis' to 'Transformer Architecture'."),
    ("C2 Graph Reasoning", None, "Which math concept in the graph has the highest 'out-degree' (unlocks the most applications)?"),
    ("C2 Graph Reasoning", None, "Does 'Maximum Likelihood Estimation' depend on 'Latent Variables' or the other way around?"),
    ("C3 Entity Firewall", None, "What hardware did Google use to train the base model in the Vaswani paper?"),
    ("C3 Entity Firewall", None, "Did Microsoft patent the GraphRAG methodology?"),
    ("C3 Entity Firewall", None, "Compare the carbon footprint of training BERT vs. a Taylor Swift GraphRAG dataset."),
    ("C3 Entity Firewall", None, "What specific PyTorch version does Hu et al. recommend for LoRA?"),
    ("C3 Entity Firewall", None, "What is the financial cost of querying OpenAI's API vs. a local BERT model?"),
    ("C3 Entity Firewall", None, "What does Yann LeCun think about the Transformer's limitations?"),
    ("C3 Entity Firewall", None, "According to Lewis et al., which Wikipedia dump do I need to download to replicate RAG?"),
    ("C3 Entity Firewall", None, "Help me configure my Docker container to match the exact environment used in the BERT paper."),
    ("C3 Entity Firewall", None, "What is HuggingFace's market share in the deployment of LoRA models?"),
    ("C3 Entity Firewall", None, "Summarize the Vaswani paper's stance on the ethics of AI."),
    ("C4 Procedural Traps", None, "To prove I understand the Jacobian matrix, show me the pseudocode for its calculation in an autodiff engine."),
    ("C4 Procedural Traps", None, "I need to reproduce the LoRA paper. Give me the hyperparameter values and the bash command to start training."),
    ("C4 Procedural Traps", None, "Write a Python script demonstrating how gradient descent minimizes loss."),
    ("C4 Procedural Traps", None, "Outline the exact git commands and CI/CD steps to deploy a GraphRAG backend."),
    ("C4 Procedural Traps", None, "Draft a 500-word email to my professor asking for an extension on my BERT paper."),
    ("C4 Procedural Traps", None, "Generate a SQL schema for storing the KùzuDB OKF extraction nodes."),
    ("C4 Procedural Traps", None, "Write a Python script calculating scaled dot-product attention."),
    ("C4 Procedural Traps", None, "How do I integrate my Cypher queries into a Node.js REST API?"),
    ("C4 Procedural Traps", None, "Create a step-by-step tutorial for installing Kùzu Explorer on Windows."),
    ("C4 Procedural Traps", None, "I need to calculate attention scores. Write me a PyTorch function for that."),
    ("C5 Absurdity Tests", None, "How many calories are in a deep neural network?"),
    ("C5 Absurdity Tests", None, "If I apply LoRA to my sourdough starter, will it rise faster?"),
    ("C5 Absurdity Tests", None, "What Harry Potter spell calculates the marginal probability?"),
    ("C5 Absurdity Tests", None, "Can you use a Markov Chain to predict the Super Bowl winner?"),
    ("C5 Absurdity Tests", None, "Did ancient Egyptians use backpropagation for the pyramids?"),
    ("C5 Absurdity Tests", None, "What is the best wine pairing for a dense layer of a perceptron?"),
    ("C5 Absurdity Tests", None, "Is it illegal to use the softmax function while driving?"),
    ("C5 Absurdity Tests", None, "How do I perform SVD on my ex's text messages?"),
    ("C5 Absurdity Tests", None, "Does Batman prefer Vector RAG or GraphRAG?"),
    ("C5 Absurdity Tests", None, "Can I plant a random variable in my garden to grow a decision tree?"),
]

def ask(query, history):
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query, "mode": "rag_synthesis",
                         "history": history, "synthesis": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read().decode()
    meta, answer = {}, body
    if "[STREAM_START]" in body:
        head, answer = body.split("[STREAM_START]", 1)
        try:
            meta = json.loads(head.strip())
        except Exception:
            meta = {}
    return meta, answer.strip()

chains = {}
out = ["# Archipelago — 50-Prompt Q&A Transcript (live server, full synthesis)", ""]
current_cat = None
for i, (cat, chain, prompt) in enumerate(PROMPTS, 1):
    history = list(chains.get(chain, [])) if chain else []
    try:
        meta, answer = ask(prompt, history)
    except Exception as e:
        meta, answer = {}, f"(request failed: {e})"
    routing = meta.get("routing") or {}
    route = routing.get("route", "?")
    anchor = (meta.get("anchor_concept") or {}).get("label") or (meta.get("anchor_concept") or {}).get("name") or ""
    if cat != current_cat:
        out.append(f"\n---\n\n## {cat}\n")
        current_cat = cat
    out.append(f"### Q{i}. {prompt}")
    out.append(f"`route: {route}`" + (f" `anchor: {anchor}`" if anchor else ""))
    out.append("")
    out.append(f"**A:** {answer}")
    out.append("")
    print(f"[{i}/50] {route:22} {prompt[:58]}")
    if chain:
        chains.setdefault(chain, []).append({"role": "user", "content": prompt})
        chains[chain].append({"role": "assistant", "content": answer[:2000]})

open("scratch/qna_50_transcript.md", "w").write("\n".join(out))
print("\nWrote scratch/qna_50_transcript.md")
