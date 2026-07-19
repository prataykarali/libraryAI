"""Run the 50-prompt Archipelago Evolution Dataset through the live router."""
import os, sys, json
os.environ.setdefault("ARCHIPELAGO_DB_READ_ONLY", "1")

from archipelago.inference.routes_chat import init_concepts_data
from archipelago.inference.embeddings import load_embedding_model, build_concept_embeddings
init_concepts_data()
load_embedding_model()
build_concept_embeddings()
from archipelago.inference import state as st
from archipelago.inference.routing import resolve_query_routing

GRAPH_ROUTES = {"graph_strong", "graph_soft"}
SOFT_ROUTES = GRAPH_ROUTES | {"low_similarity_reject", "onboarding", "library_books",
                              "library_chapters", "library_chapter_lookup"}

# (category, chain_id or None, prompt, expectation)
# expectation: "graph" (must reach graph/soft path), "answer" (anything but out_of_scope),
#              "block" (must be out_of_scope), "impl" (out_of_scope w/ implementation reason)
PROMPTS = [
    # ── Category 1: de-greasing + context chaining ──
    (1, "A", "Hey buddy, what's up? Can you explain what a transformer is?", "answer"),
    (1, "A", "Cool. Now, how does it handle long sequences?", "answer"),
    (1, "A", "What about the attention mechanism? Does it need positional encoding?", "answer"),
    (1, "B", "Man, I'm tired. Why is math so hard? Just give me the gist of a Jacobian.", "answer"),
    (1, "B", "And how is that different from a Hessian?", "answer"),
    (1, "C", "Good evening! Please define a Recurrent Layer for me.", "answer"),
    (1, "C", "Does it perform better than a transformer on short sequences?", "answer"),
    (1, "D", "Hi! What is a latent variable?", "answer"),
    (1, "D", "And where does it fit into the Maximum Likelihood Estimation process?", "answer"),
    (1, "D", "Can it be observed directly?", "answer"),
    # ── Category 2: comparative topological reasoning ──
    (2, None, "Compare the curriculum path of a student learning LoRA vs. one learning vanilla BERT. Where do their paths diverge?", "answer"),
    (2, None, "If I skip Linear Algebra, what nodes in the graph become inaccessible?", "answer"),
    (2, None, "Is 'Probability Theory' a prerequisite for 'Masked Language Modeling'? Explain the multi-hop path.", "answer"),
    (2, None, "How does the 'Self-Attention' mechanism connect to 'Singular Value Decomposition' through the lens of dimensionality?", "answer"),
    (2, None, "Contrast the prerequisites for RAG-Token and GraphRAG. Do they share any common upstream nodes?", "answer"),
    (2, None, "Map the relationship between 'Gradient Descent' and 'Automatic Differentiation'.", "answer"),
    (2, None, "Identify the 'Bridge Nodes' connecting 'Gaussian Distributions' to 'Masked Language Modeling'.", "answer"),
    (2, None, "Trace a path from 'Orthonormal Basis' to 'Transformer Architecture'.", "answer"),
    (2, None, "Which math concept in the graph has the highest 'out-degree' (unlocks the most applications)?", "answer"),
    (2, None, "Does 'Maximum Likelihood Estimation' depend on 'Latent Variables' or the other way around?", "answer"),
    # ── Category 3: entity firewall traps ──
    (3, None, "What hardware did Google use to train the base model in the Vaswani paper?", "block"),
    (3, None, "Did Microsoft patent the GraphRAG methodology?", "block"),
    (3, None, "Compare the carbon footprint of training BERT vs. a Taylor Swift GraphRAG dataset.", "block"),
    (3, None, "What specific PyTorch version does Hu et al. recommend for LoRA?", "block"),
    (3, None, "What is the financial cost of querying OpenAI's API vs. a local BERT model?", "block"),
    (3, None, "What does Yann LeCun think about the Transformer's limitations?", "block"),
    (3, None, "According to Lewis et al., which Wikipedia dump do I need to download to replicate RAG?", "block"),
    (3, None, "Help me configure my Docker container to match the exact environment used in the BERT paper.", "block"),
    (3, None, "What is HuggingFace's market share in the deployment of LoRA models?", "block"),
    (3, None, "Summarize the Vaswani paper's stance on the ethics of AI.", "block"),
    # ── Category 4: procedural / academic bypass ──
    (4, None, "To prove I understand the Jacobian matrix, show me the pseudocode for its calculation in an autodiff engine.", "impl"),
    (4, None, "I need to reproduce the LoRA paper. Give me the hyperparameter values and the bash command to start training.", "impl"),
    (4, None, "Write a Python script demonstrating how gradient descent minimizes loss.", "impl"),
    (4, None, "Outline the exact git commands and CI/CD steps to deploy a GraphRAG backend.", "impl"),
    (4, None, "Draft a 500-word email to my professor asking for an extension on my BERT paper.", "impl"),
    (4, None, "Generate a SQL schema for storing the KùzuDB OKF extraction nodes.", "impl"),
    (4, None, "Write a Python script calculating scaled dot-product attention.", "impl"),
    (4, None, "How do I integrate my Cypher queries into a Node.js REST API?", "impl"),
    (4, None, "Create a step-by-step tutorial for installing Kùzu Explorer on Windows.", "impl"),
    (4, None, "I need to calculate attention scores. Write me a PyTorch function for that.", "impl"),
    # ── Category 5: absurdity kill-switch ──
    (5, None, "How many calories are in a deep neural network?", "block"),
    (5, None, "If I apply LoRA to my sourdough starter, will it rise faster?", "block"),
    (5, None, "What Harry Potter spell calculates the marginal probability?", "block"),
    (5, None, "Can you use a Markov Chain to predict the Super Bowl winner?", "block"),
    (5, None, "Did ancient Egyptians use backpropagation for the pyramids?", "block"),
    (5, None, "What is the best wine pairing for a dense layer of a perceptron?", "block"),
    (5, None, "Is it illegal to use the softmax function while driving?", "block"),
    (5, None, "How do I perform SVD on my ex's text messages?", "block"),
    (5, None, "Does Batman prefer Vector RAG or GraphRAG?", "block"),
    (5, None, "Can I plant a random variable in my garden to grow a decision tree?", "block"),
]

chains = {}
results = []
for cat, chain, prompt, expect in PROMPTS:
    history = list(chains.get(chain, [])) if chain else []
    try:
        r = resolve_query_routing(prompt, history=history or None)
    except Exception as e:
        results.append((cat, prompt, "ERROR", str(e), expect, False))
        continue
    route = r.get("route")
    reason = r.get("reason") or ""
    anchor = r.get("anchor_id")

    if expect == "block":
        ok = route == "out_of_scope"
    elif expect == "impl":
        ok = route == "out_of_scope" and "implementation" in reason
    elif expect == "graph":
        ok = route in GRAPH_ROUTES
    else:  # answer
        ok = route != "out_of_scope"
    results.append((cat, prompt, route, f"{reason}|anchor={anchor}", expect, ok))

    # Maintain the chain like the real UI: push user turn + simulated assistant
    # turn naming the anchor concept (what a real answer would contain).
    if chain:
        chains.setdefault(chain, []).append({"role": "user", "content": prompt})
        if anchor and anchor in st.CONCEPTS_DATA:
            label = st.CONCEPTS_DATA[anchor].get("label") or anchor
            stub = f"{label} is covered in our library. Here is a grounded explanation of {label}."
        else:
            stub = "Here is a general reply."
        chains[chain].append({"role": "assistant", "content": stub})

passed = sum(1 for r in results if r[5])
print(f"\n===== RESULTS: {passed}/{len(results)} passed =====\n")
by_cat = {}
for cat, prompt, route, detail, expect, ok in results:
    by_cat.setdefault(cat, [0, 0])
    by_cat[cat][0] += ok
    by_cat[cat][1] += 1
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] C{cat} expect={expect:6} route={route:22} {prompt[:70]}")
    if not ok:
        print(f"       detail: {detail}")
print("\nPer-category:", {f"C{k}": f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())})
