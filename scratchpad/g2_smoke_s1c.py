#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 SMOKE S1c (MetaQA): beam-width sweep + P50-UNION (the real Q1 metric).
S1b showed EXPAND_DIR saturates at ~61 cands (beam M binds, not budget). Here: structural-only,
EXPAND_DIR, sweep M in {32,64,128,256}, H=3, generous B, and measure the money metric:
  P50-only ALL  vs  (P50 UNION directional-expansion) ALL  vs  expansion-only ALL,
per hop, with the ADDED candidate count. Answers Q1: does bounded directional geometry co-scope the
2/3-hop golds P50 misses? Parameter-free. Dev/val only. No gold/hops used to build scope."""
import json, sys, time
import numpy as np
sys.argv = [sys.argv[0]]
import g2_l1_geo as G

H_BUDGET = 3; B_CAP = 2000; NQ = 250
M_SWEEP = [32, 64, 128, 256]

t0 = time.time()
A = G.load_artifacts()
j = A["j"]; si = j["split_indices"]; hops = j["hops"]; golds = j["golds"]; ids = j["ids"]
d2i = A["doc_id_to_idx"]; hard = A["hard"]
nodes = np.load(G.BASE + "nodes.npy")
qall = np.load(G.BASE + "queries_all.npy", mmap_mode="r")
Xn = nodes / (np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-9)
adj_struct, sinfo = G.build_structural_adj(d2i)
qtext = {n.node_id: n.content for n in __import__("src.pipeline.standardizer", fromlist=["load_nodes"])
         .load_nodes(f"data/processed/master_nodes_{G.DS}.json") if n.metadata.get("type") == "question"}
print(f"[load] {time.time()-t0:.1f}s struct={sinfo['n_edges']}")

sample = {h: [] for h in (1, 2, 3)}
for i in si["val"]:
    h = hops[i]
    if h in sample and len(sample[h]) < NQ: sample[h].append(i)

def residual(qvec, seed_rows):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not seed_rows: return q
    E = Xn[seed_rows]
    U, s, _ = np.linalg.svd(E.T, full_matrices=False); U = U[:, s > 1e-6]
    if U.shape[1] == 0: return q
    r = q - U @ (U.T @ q); n = np.linalg.norm(r)
    return r / n if n > 1e-6 else q

def expand_dir(seed_rows, r_q, adj, M, B=B_CAP):
    scope = set(seed_rows); frontier = list(seed_rows); scored = []
    for hop in range(H_BUDGET):
        cand = []; seen = set()
        for e in frontier:
            xe = Xn[e]
            for (v, rel, dirn) in adj.get(e, []):
                if v in scope or (e, v) in seen: continue
                seen.add((e, v))
                delta = Xn[v] - xe; nd = np.linalg.norm(delta)
                if nd < 1e-9: continue
                cand.append((float(r_q @ (delta / nd)), v))
        if not cand: break
        cand.sort(key=lambda x: -x[0]); keep = cand[:M]
        nf = []
        for sdir, v in keep:
            if v not in scope: scope.add(v); nf.append(v); scored.append((sdir, v))
        frontier = nf
        if len(scope) >= B * 2: break
    out = set(seed_rows)
    for sdir, v in sorted(scored, key=lambda x: -x[0]):
        if len(out) >= B: break
        out.add(v)
    return out - set(seed_rows)

# precompute P50 selected partitions per hop-sample (frozen router)
P50_sel = {h: G.p50_scope_partitions(sample[h], A) for h in (1, 2, 3)}

report = {"harness": "G2_SMOKE_S1c", "dataset": G.DS, "H_BUDGET": H_BUDGET, "NQ": NQ,
          "provenance": "structural_only", "M_SWEEP": M_SWEEP, "RESULTS": {}}
# baseline P50-only ALL per hop (sanity, matches S1)
for M in M_SWEEP:
    for h in (1, 2, 3):
        p50_all = uni_all = exp_all = uni_any = 0; nfe = 0; added = []
        for qi, i in enumerate(sample[h]):
            gidx = [d2i[g] for g in golds[i] if g in d2i]
            if not gidx: continue
            nfe += 1
            sel = P50_sel[h][qi]
            seeds = G.resolve_seeds(qtext[ids[i]], d2i)
            r_q = residual(np.asarray(qall[i]), seeds)
            exp = expand_dir(seeds, r_q, adj_struct, M)
            added.append(len(exp))
            in_p50 = [int(hard[g]) in sel for g in gidx]
            in_exp = [g in exp for g in gidx]
            in_uni = [a or b for a, b in zip(in_p50, in_exp)]
            p50_all += int(all(in_p50)); exp_all += int(all(in_exp))
            uni_all += int(all(in_uni)); uni_any += int(any(in_uni))
        report["RESULTS"][f"M{M}|h{h}"] = {
            "n": nfe, "P50_ALL": round(p50_all/max(nfe,1),4), "UNION_ALL": round(uni_all/max(nfe,1),4),
            "EXP_ALL": round(exp_all/max(nfe,1),4), "UNION_ANY": round(uni_any/max(nfe,1),4),
            "added_cand_mean": round(float(np.mean(added)),1) if added else 0}
    print(f"M={M:4d} | " + "  ".join(
        f"h{h}: P50={report['RESULTS'][f'M{M}|h{h}']['P50_ALL']:.3f}->UNION={report['RESULTS'][f'M{M}|h{h}']['UNION_ALL']:.3f} "
        f"(+{report['RESULTS'][f'M{M}|h{h}']['added_cand_mean']:.0f})" for h in (1,2,3)))

json.dump(report, open("results/GENERALIZATION/_g2_smoke_s1c.json", "w"), indent=1)
print(f"[done] {time.time()-t0:.1f}s wrote _g2_smoke_s1c.json")
