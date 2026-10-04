#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 SMOKE S1b (MetaQA): the parameter-free Track-A geometry experiment (answers Q1).
Seed-anchored bounded expansion; measure ALL-gold co-scope coverage vs candidate budget, per hop,
per provenance. Compare:
  A5   EXPAND_ALL     : BFS from seeds, follow ALL edges, cap total by BFS/degree order.
  A2   EXPAND_DIR     : per-hop keep top-M edges by s_dir=cos(r_q, delta(e->v)) (directional pruning).
  A4   EXPAND_DIR_SEQ : as A2 but residual updated each hop (remove explained direction).
Budgets B in {50,100,200,500}. Fixed H=3 hop budget, beam M. Params fixed/predeclared (no target-test tuning).
Dev/val only, <=250 q/hop. Parameter-free (no learned weights). No gold/hops used to build scope."""
import json, sys, time, collections
import numpy as np
sys.argv = [sys.argv[0]]
import g2_l1_geo as G

H_BUDGET = 3          # max hops (fixed; NOT using true hop label)
BEAM_M = 32           # per-hop directional beam (predeclared)
BUDGETS = [50, 100, 200, 500]
NQ = 250
TAU = 8.0             # softmax temp for A3 voting (predeclared)

t0 = time.time()
A = G.load_artifacts()
j = A["j"]; si = j["split_indices"]; hops = j["hops"]; golds = j["golds"]; ids = j["ids"]
d2i = A["doc_id_to_idx"]
nodes = np.load(G.BASE + "nodes.npy")              # (40151,1536) load full for speed
qall = np.load(G.BASE + "queries_all.npy", mmap_mode="r")
Xn = nodes / (np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-9)
print(f"[load] {time.time()-t0:.1f}s")

qtext = {n.node_id: n.content for n in __import__("src.pipeline.standardizer", fromlist=["load_nodes"])
         .load_nodes(f"data/processed/master_nodes_{G.DS}.json") if n.metadata.get("type") == "question"}

# provenance-pure adjacencies (row space)
adj_struct, sinfo = G.build_structural_adj(d2i)
adj_knn = G.recompute_knn_adj(nodes, k=3)
def merged_adj(a, b):
    m = {k: list(v) for k, v in a.items()}
    for k, v in b.items(): m.setdefault(k, []).extend(v)
    return m
ADJ = {"structural": adj_struct, "knn": adj_knn, "structural+knn": merged_adj(adj_struct, adj_knn)}
print(f"[adj] struct={sinfo['n_edges']} knn={sum(len(v) for v in adj_knn.values())} {time.time()-t0:.1f}s")

# sample
rng = np.random.default_rng(0)
sample = {h: [] for h in (1, 2, 3)}
for i in si["val"]:
    h = hops[i]
    if h in sample and len(sample[h]) < NQ: sample[h].append(i)

def residual(qvec, seed_rows):
    """A1: remove the seed-entity subspace from the query embedding."""
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not seed_rows: return q
    E = Xn[seed_rows]                       # (m,d) unit
    # orthonormal basis of span(E) via SVD (handles collinear/multi)
    U, s, _ = np.linalg.svd(E.T, full_matrices=False)
    U = U[:, s > 1e-6]
    if U.shape[1] == 0: return q
    r = q - U @ (U.T @ q)
    n = np.linalg.norm(r)
    return r / n if n > 1e-6 else q          # fallback: nearly explained -> keep q

def expand(seed_rows, r_q, adj, mode, B):
    """Return set of candidate rows (<=B) reachable from seeds under the mode's pruning."""
    scope = set(seed_rows)
    frontier = list(seed_rows)
    r_active = r_q.copy()
    scored = []                              # (s_dir, row) accumulated for global budget cap
    for hop in range(H_BUDGET):
        cand = []                            # (s_dir, row, delta)
        seen_edge = set()
        for e in frontier:
            xe = Xn[e]
            for (v, rel, dirn) in adj.get(e, []):
                if v in scope or (e, v) in seen_edge: continue
                seen_edge.add((e, v))
                delta = Xn[v] - xe
                nd = np.linalg.norm(delta)
                if nd < 1e-9: continue
                delta = delta / nd
                sdir = float(r_active @ delta)
                cand.append((sdir, v, delta))
        if not cand: break
        if mode == "EXPAND_ALL":
            keep = cand                      # follow everything (budget cap applied globally later)
        else:                                # EXPAND_DIR / EXPAND_DIR_SEQ: top-M by s_dir
            cand.sort(key=lambda x: -x[0]); keep = cand[:BEAM_M]
        new_front = []
        for sdir, v, delta in keep:
            if v not in scope:
                scope.add(v); new_front.append(v); scored.append((sdir, v))
        frontier = new_front
        if mode == "EXPAND_DIR_SEQ" and keep:
            # A4: remove the mean chosen direction from the residual
            dmean = np.mean([d for _, _, d in keep[:BEAM_M]], axis=0)
            dn = np.linalg.norm(dmean)
            if dn > 1e-6:
                dmean /= dn
                r_active = r_active - (r_active @ dmean) * dmean
                rn = np.linalg.norm(r_active)
                if rn > 1e-6: r_active /= rn
        if len(scope) >= B * 4: break        # avoid runaway frontier before capping
    # apply global budget: seeds always kept; fill remaining by s_dir order
    out = set(seed_rows)
    for sdir, v in sorted(scored, key=lambda x: -x[0]):
        if len(out) >= B: break
        out.add(v)
    return out

report = {"harness": "G2_SMOKE_S1b", "dataset": G.DS, "split": "val", "H_BUDGET": H_BUDGET,
          "BEAM_M": BEAM_M, "BUDGETS": BUDGETS, "NQ": NQ, "note": "seed-anchored ALL-coverage; parameter-free",
          "RESULTS": {}}
MODES = ["EXPAND_ALL", "EXPAND_DIR", "EXPAND_DIR_SEQ"]
for prov in ["structural", "structural+knn"]:
    adj = ADJ[prov]
    for mode in MODES:
        for B in BUDGETS:
            per_hop = {}
            for h in (1, 2, 3):
                allc = 0; anyc = 0; nfe = 0; sizes = []
                for i in sample[h]:
                    seeds = G.resolve_seeds(qtext[ids[i]], d2i)
                    gidx = [d2i[g] for g in golds[i] if g in d2i]
                    if not gidx: continue
                    nfe += 1
                    r_q = residual(np.asarray(qall[i]), seeds)
                    scope = expand(seeds, r_q, adj, mode, B)
                    scope -= set(seeds)                # seeds are the query entities, not answers
                    ret = sum(1 for g in gidx if g in scope)
                    anyc += int(ret >= 1); allc += int(ret == len(gidx)); sizes.append(len(scope))
                per_hop[h] = {"n": nfe, "ANY": round(anyc/max(nfe,1),4), "ALL": round(allc/max(nfe,1),4),
                              "cand_mean": round(float(np.mean(sizes)),1) if sizes else 0}
            report["RESULTS"][f"{prov}|{mode}|B{B}"] = per_hop
            print(f"{prov:16s} {mode:15s} B={B:4d} | "
                  + " ".join(f"h{h}:ALL={per_hop[h]['ALL']:.3f}(any{per_hop[h]['ANY']:.2f},c{per_hop[h]['cand_mean']:.0f})" for h in (1,2,3)))

json.dump(report, open("results/GENERALIZATION/_g2_smoke_s1b.json", "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s  wrote _g2_smoke_s1b.json")
