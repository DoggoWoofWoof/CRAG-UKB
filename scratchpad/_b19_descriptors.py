"""B1.9 STEP 1 — LABEL-FREE INDEX-TIME CORPUS DESCRIPTORS.

Every quantity here is computed from the corpus substrate ALONE:
    documents (master_nodes, type != question)  +  node embeddings (gte_qwen/nodes.npy)
    +  the SAME structural adjacency _b12_build.py uses (kb.txt for metaqa, master_nodes.neighbors otherwise)

For EVERY descriptor:  uses_corpus_only = YES · uses_target_queries = NO · uses_labels = NO · uses_dataset_id = NO
No query file is opened. No relevance label, supporting fact, hop label or split is read. No new encoder pass.
All descriptors are cacheable at ingestion and available BEFORE the first unseen query.

HYPOTHESIS UNDER TEST (stated before running):
    structural reserve utility is HIGH when structural edges lead to nodes semantic retrieval does NOT already
    reach (structural novelty), and LOW when structural neighbours are already semantically near (redundancy).
  supports  -> StructSemOverlap LOW for MetaQA (R=16), HIGH for SQuAD (R=0), 2Wiki intermediate
  refutes   -> overlap fails to separate them, or the ordering inverts
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from src.pipeline.standardizer import load_nodes

DSES = ["musique_clean", "squad_clean", "metaqa", "2wiki_clean", "hotpotqa_clean", "webqsp"]
OUT = "results/GENERALIZATION/_g2_b19_descriptors.json"

DEG_CAP = 300          # frozen _b12 contract: a node with deg>DEG_CAP is not expanded through
S_SAMPLE = 2000        # deterministic label-free node sample
NB_CAP = 32            # max structural neighbours examined per sampled node
K_TOP = 200            # running semantic top-K depth
CHUNK = 20000
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


def build_adj(ds, docs, id2row):
    """IDENTICAL construction to scratchpad/_b12_build.py::build_adj -- the graph the reserve actually runs on."""
    n = len(docs); adj = [set() for _ in range(n)]
    if ds == "metaqa":
        norm = lambda name: "metaqa_ent_" + name.strip().lower()
        ne = 0
        for line in open("data/original/metaqa/kb.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("|")
            if len(p) != 3:
                continue
            hi = id2row.get(norm(p[0])); ti = id2row.get(norm(p[2]))
            if hi is None or ti is None or hi == ti:
                continue
            adj[hi].add(ti); adj[ti].add(hi); ne += 1
        src = f"kb.txt structural (undirected), edges={ne}"
    else:
        ne = 0
        for i, nd in enumerate(docs):
            for x in nd.neighbors:
                j = id2row.get(x)
                if j is not None and j != i:
                    adj[i].add(j); adj[j].add(i); ne += 1
        src = f"master_nodes.neighbors (undirected), dir_edges={ne}"
    adj = [np.fromiter(s, np.int32) for s in adj]
    deg = np.array([len(a) for a in adj], np.int32)
    return adj, deg, src


def giant_component_frac(adj, n):
    seen = np.zeros(n, bool); best = 0
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; cnt = 0
        while stack:
            u = stack.pop(); cnt += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v)
        best = max(best, cnt)
    return best / n


def running_topk(Xs, path, n, d, k, chunk, skip_rows):
    """Exact top-k semantic neighbours of each sampled row over the WHOLE corpus, streamed (no S x N matrix)."""
    X = np.load(path, mmap_mode="r")
    S = Xs.shape[0]
    bv = np.full((S, k), -2.0, np.float32); bi = np.full((S, k), -1, np.int64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        C = np.array(X[a:b], dtype=np.float32)      # copy: a matching-dtype mmap slice is a read-only view
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        sims = Xs @ C.T                                   # (S, b-a)
        # mask self
        for si, r in enumerate(skip_rows):
            if a <= r < b:
                sims[si, r - a] = -2.0
        m = min(k, b - a)
        part = np.argpartition(-sims, m - 1, axis=1)[:, :m]
        pv = np.take_along_axis(sims, part, 1)
        cv = np.concatenate([bv, pv], 1); ci = np.concatenate([bi, part + a], 1)
        sel = np.argpartition(-cv, k - 1, axis=1)[:, :k]
        bv = np.take_along_axis(cv, sel, 1); bi = np.take_along_axis(ci, sel, 1)
        del sims, C
    o = np.argsort(-bv, axis=1)
    return np.take_along_axis(bv, o, 1), np.take_along_axis(bi, o, 1)


def descriptors(ds):
    log(f"--- {ds}")
    docs = [nd for nd in load_nodes(f"data/processed/master_nodes_{ds}.json") if nd.metadata.get("type") != "question"]
    id2row = {nd.node_id: i for i, nd in enumerate(docs)}
    n = len(docs)
    adj, deg, src = build_adj(ds, docs, id2row)
    log(f"    graph: {src}  n={n}")
    del docs

    rng = np.random.default_rng(0)
    z = {"_provenance": {"graph_src": src, "uses_corpus_only": "YES", "uses_target_queries": "NO",
                         "uses_labels": "NO", "uses_dataset_id": "NO", "encoder_passes": 0},
         "n_nodes": int(n)}

    # ---------------- GRAPH TOPOLOGY (exact, whole corpus)
    ue = int(deg.sum() // 2)
    dpos = deg[deg > 0]
    z.update({
        "n_edges_undirected": ue,
        "mean_degree": round(float(deg.mean()), 4),
        "median_degree": float(np.median(deg)),
        "deg_p90": float(np.percentile(deg, 90)), "deg_p95": float(np.percentile(deg, 95)),
        "deg_p99": float(np.percentile(deg, 99)),
        "degree_cv": round(float(deg.std() / (deg.mean() + 1e-9)), 4),
        "edge_density": float(2.0 * ue / max(n * (n - 1), 1)),
        "isolated_frac": round(float((deg == 0).mean()), 4),
        "hub_frac_over_DEGCAP": round(float((deg > DEG_CAP).mean()), 5),
        "mean_degree_nonisolated": round(float(dpos.mean()) if len(dpos) else 0.0, 4),
        "giant_component_frac": round(float(giant_component_frac(adj, n)), 4)})

    # ---------------- deterministic label-free sample (nodes WITH structure, for the alignment family)
    cand = np.where(deg > 0)[0]
    if len(cand) == 0:
        z["_status"] = "NO_STRUCTURE"; return z
    samp = np.sort(rng.choice(cand, size=min(S_SAMPLE, len(cand)), replace=False))

    # ---------------- LOCAL EXPANSION BEHAVIOUR (sampled)
    h1, h2, jac = [], [], []
    for u in samp:
        nb = adj[u]
        h1.append(len(nb))
        nb_use = nb if len(nb) <= NB_CAP else nb[:NB_CAP]
        s2 = set()
        for v in nb_use:
            if deg[v] <= DEG_CAP:                          # frozen contract: hubs are not expanded through
                s2.update(adj[v].tolist())
        s2 -= set(nb.tolist()); s2.discard(int(u))
        h2.append(len(s2))
        for v in nb_use[:8]:
            a, b = set(adj[u].tolist()), set(adj[v].tolist())
            un = len(a | b)
            if un:
                jac.append(len(a & b) / un)
    z.update({"hop1_unique_mean": round(float(np.mean(h1)), 3),
              "hop2_unique_mean": round(float(np.mean(h2)), 3),
              "hop2_over_hop1": round(float(np.mean(h2) / max(np.mean(h1), 1e-9)), 4),
              "neighbor_jaccard_redundancy": round(float(np.mean(jac)) if jac else 0.0, 5),
              "branch_p50": float(np.percentile(h1, 50)), "branch_p90": float(np.percentile(h1, 90))})

    # ---------------- SEMANTIC-STRUCTURAL ALIGNMENT  (the decisive family)
    path = f"data/ukb_storage/{ds}/gte_qwen/nodes.npy"
    X = np.load(path, mmap_mode="r"); d = X.shape[1]
    assert X.shape[0] == n, f"row-space mismatch {X.shape[0]} vs {n}"
    Xs = np.asarray(X[samp], np.float32)
    Xs /= (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    log(f"    streaming exact top-{K_TOP} for {len(samp)} sampled nodes over {n} ...")
    tv, ti = running_topk(Xs, path, n, d, K_TOP, CHUNK, samp)

    ov10, ov50, ov200, edge_cos, rnd_cos, top10_cos, top1_cos = [], [], [], [], [], [], []
    rnd_rows = rng.choice(n, size=min(512, n), replace=False)
    Xr = np.asarray(X[rnd_rows], np.float32); Xr /= (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
    for si, u in enumerate(samp):
        nb = adj[u]
        nb_use = nb if len(nb) <= NB_CAP else nb[:NB_CAP]
        nbs = set(int(x) for x in nb_use)
        t10, t50, t200 = set(ti[si, :10].tolist()), set(ti[si, :50].tolist()), set(ti[si, :K_TOP].tolist())
        m = len(nbs)
        ov10.append(len(nbs & t10) / m); ov50.append(len(nbs & t50) / m); ov200.append(len(nbs & t200) / m)
        Xn_ = np.asarray(X[np.array(sorted(nbs), np.int64)], np.float32)
        Xn_ /= (np.linalg.norm(Xn_, axis=1, keepdims=True) + 1e-9)
        edge_cos.append(float((Xs[si] @ Xn_.T).mean()))
        rnd_cos.append(float((Xs[si] @ Xr.T).mean()))
        top10_cos.append(float(tv[si, :10].mean())); top1_cos.append(float(tv[si, 0]))

    ec, rc, tc = float(np.mean(edge_cos)), float(np.mean(rnd_cos)), float(np.mean(top10_cos))
    z.update({
        "StructSemOverlap@10": round(float(np.mean(ov10)), 5),
        "StructSemOverlap@50": round(float(np.mean(ov50)), 5),
        "StructSemOverlap@200": round(float(np.mean(ov200)), 5),
        "struct_edges_outside_sem_top200": round(1.0 - float(np.mean(ov200)), 5),
        "edge_cos_mean": round(ec, 5), "random_cos_mean": round(rc, 5),
        "sem_top10_cos_mean": round(tc, 5), "sem_top1_cos_mean": round(float(np.mean(top1_cos)), 5),
        # where structural edges sit on the corpus' own semantic scale: 0 = random, 1 = as close as sem top-10
        "edge_semantic_lift": round(float((ec - rc) / max(tc - rc, 1e-9)), 5),
        "n_sampled": int(len(samp))})
    z["STRUCTURAL_NOVELTY_INDEX_v0"] = round(1.0 - float(np.mean(ov50)), 5)
    log(f"    ov@10 {z['StructSemOverlap@10']:.4f} ov@50 {z['StructSemOverlap@50']:.4f} "
        f"ov@200 {z['StructSemOverlap@200']:.4f} lift {z['edge_semantic_lift']:.4f} "
        f"meandeg {z['mean_degree']:.2f}")
    return z


def main():
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for ds in DSES:
        if ds in res and "_status" not in res[ds]:
            log(f"--- {ds} cached, skip"); continue
        try:
            res[ds] = descriptors(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            res[ds] = {"_status": f"ERROR {type(e).__name__}: {e}"}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    log("wrote " + OUT)

    keys = ["n_nodes", "mean_degree", "edge_density", "neighbor_jaccard_redundancy", "hop2_over_hop1",
            "StructSemOverlap@10", "StructSemOverlap@50", "StructSemOverlap@200", "edge_semantic_lift"]
    print("\n=== LABEL-FREE CORPUS DESCRIPTORS ===")
    print(f"{'descriptor':34s}" + "".join(f"{d[:11]:>13s}" for d in DSES))
    for k in keys:
        row = f"{k:34s}"
        for d in DSES:
            v = res.get(d, {}).get(k)
            row += f"{'--':>13s}" if v is None else (f"{v:13.5f}" if isinstance(v, float) else f"{v:13d}")
        print(row)


if __name__ == "__main__":
    main()
