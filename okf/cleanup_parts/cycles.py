from okf.cleanup_parts._base import *  # noqa: F403

def break_global_cycles(okf_results: list) -> int:
    """Break all cycles in the learning progression graph (prerequisites and unlocks).

    Enforces a strict Directed Acyclic Graph (DAG) for learning paths.
    """
    import sys
    # Increase recursion limit just in case
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

    index = {r.get("concept_name", "").lower(): r for r in okf_results}
    total_removed = 0

    for _ in range(1000):  # limit iterations to prevent infinite loop
        # Build learning edges list (src -> dst means src must be learned before dst)
        edges = []
        for r in okf_results:
            u = r.get("concept_name", "")
            u_lower = u.lower()
            if not u_lower:
                continue

            # Prerequisites: p -> u (p is learned before u)
            for p in r.get("prerequisites", []):
                p_lower = p.lower()
                if p_lower and p_lower != u_lower:
                    edges.append((p_lower, u_lower, "prereq", p, u))

            # Unlocks: u -> val (u unlocks val, so u is learned before val)
            for val in r.get("unlocks", []):
                val_lower = val.lower()
                if val_lower and val_lower != u_lower:
                    edges.append((u_lower, val_lower, "unlock", u, val))

        # Build adjacency list
        adj = {}
        edge_lookup = {}
        for src, dst, etype, orig_src, orig_dst in edges:
            adj.setdefault(src, set()).add(dst)
            edge_lookup[(src, dst)] = (etype, orig_src, orig_dst)

        # DFS cycle detection
        state = {}  # node -> 0: unvisited, 1: visiting, 2: visited
        parent = {}
        cycle_found = None

        def dfs(node):
            nonlocal cycle_found
            if cycle_found:
                return
            state[node] = 1
            for neighbor in adj.get(node, []):
                if state.get(neighbor, 0) == 1:
                    # Cycle detected! Reconstruct cycle path
                    cycle = []
                    curr = node
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent.get(curr)
                    cycle.append(neighbor)
                    cycle.reverse()
                    cycle.append(neighbor)
                    cycle_found = cycle
                    return
                elif state.get(neighbor, 0) == 0:
                    parent[neighbor] = node
                    dfs(neighbor)
                    if cycle_found:
                        return
            state[node] = 2

        all_nodes = set(index.keys()) | set(adj.keys())
        for node in all_nodes:
            if state.get(node, 0) == 0:
                dfs(node)
                if cycle_found:
                    break

        if not cycle_found:
            break

        # We have a cycle path like [A, B, C, A]
        cycle_pairs = []
        for i in range(len(cycle_found) - 1):
            cycle_pairs.append((cycle_found[i], cycle_found[i+1]))

        # Find weakest edge in the cycle:
        # Score = diff(dst) - diff(src). Lower score = weaker/more likely incorrect.
        weakest_edge = None
        min_score = None

        for src, dst in cycle_pairs:
            src_res = index.get(src)
            dst_res = index.get(dst)

            src_diff = _DIFFICULTY_RANK.get(src_res.get("difficulty", "intermediate") if src_res else "intermediate", 2)
            dst_diff = _DIFFICULTY_RANK.get(dst_res.get("difficulty", "intermediate") if dst_res else "intermediate", 2)

            score = dst_diff - src_diff

            # Tie-breaker: score, -len(src), -len(dst)
            edge_score = (score, -len(src), -len(dst))

            if min_score is None or edge_score < min_score:
                min_score = edge_score
                weakest_edge = (src, dst)

        if weakest_edge:
            src, dst = weakest_edge
            etype, orig_src, orig_dst = edge_lookup[(src, dst)]

            if etype == "prereq":
                dst_res = index.get(dst)
                if dst_res:
                    dst_res["prerequisites"] = [x for x in dst_res.get("prerequisites", []) if x.lower() != src]
            else:
                src_res = index.get(src)
                if src_res:
                    src_res["unlocks"] = [x for x in src_res.get("unlocks", []) if x.lower() != dst]

            total_removed += 1

    return total_removed
