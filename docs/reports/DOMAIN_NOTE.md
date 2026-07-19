# Domain Note

This OKF / Archipelago pipeline is **domain-agnostic**.

Given any document (PDF, Markdown, text), it:

1. Chunks the document into coherent passages.
2. Extracts reusable concept nodes.
3. Maps prerequisite (`requires`), downstream (`unlocks`), and lateral (`related_to`) edges between concepts.
4. Visualises the resulting knowledge graph and links every concept back to its source page and passage.

The current repository is seeded with **AI/ML source documents** only as a test corpus.  
The schema, prompts, canonicaliser, graph UI, and training instructions contain **no AI/ML-specific assumptions**.  Feed it biology, history, law, or any other subject and it will extract concepts for that domain.

To specialise the graph for a new domain, add your documents under `pdfs/` and re-run:

```bash
python okf_pipeline.py
```

For domain-specific alias merging (e.g. collapsing variants of one term), add entries to the `ALIAS_MAP` in `okf_pipeline.py` or create a `domain_aliases.json` file.
