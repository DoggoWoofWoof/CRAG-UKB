#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 SMOKE S1 (MetaQA): validate the L1 metric harness + leakage guards BEFORE any geometry/scale.
Gate 1: reproduce frozen P50 per-hop ANY/ALL coverage vs G1 (99.85/96.31/94.68 ANY ; 99.59/71.38/27.01 ALL).
Gate 2: provenance-pure structural adjacency loads in row space (kb.txt) with high entity coverage.
Gate 3: leakage asserts — seeds come ONLY from brackets; golds/hops never touch scope; no test rows used.
Dev/val split only. Sample <=300 q/hop."""
import json, sys, time, collections
import numpy as np
sys.argv = [sys.argv[0]]  # guard against arg parsing in imports
import g2_l1_geo as G

t0 = time.time()
A = G.load_artifacts()
j = A["j"]; si = j["split_indices"]; hops = j["hops"]; golds = j["golds"]; ids = j["ids"]
doc_id_to_idx = A["doc_id_to_idx"]
print(f"[load] N_docs={A['N']} npart={A['npart']} in {time.time()-t0:.1f}s")

# ---- leakage guard: use VAL only, never test ----
val = si["val"]; test_set = set(si["test"])
assert not (set(val) & test_set), "LEAKAGE: val intersects test"

rng = np.random.default_rng(0)
sample = {h: [] for h in (1, 2, 3)}
for i in val:
    h = hops[i]
    if h in sample and len(sample[h]) < 300:
        sample[h].append(i)
for h in sample:
    assert all(hops[i] == h for i in sample[h])
print("[sample] per-hop:", {h: len(sample[h]) for h in sample})

# ---- Gate 2: structural adjacency in row space ----
t1 = time.time()
adj_struct, sinfo = G.build_structural_adj(doc_id_to_idx)
print(f"[struct] {sinfo} in {time.time()-t1:.1f}s")
assert sinfo["n_edges"] > 120000, "structural edges too few"
assert sinfo["n_nodes_with_edges"] > 30000, "too few entities have edges"

# ---- Gate 3 + seed resolution (inference-safe) ----
seed_hit = collections.Counter(); seed_tot = collections.Counter()
for h in (1, 2, 3):
    for i in sample[h]:
        seeds = G.resolve_seeds(ids_text := next(n.content for n in A["doc_nodes"][:0]), doc_id_to_idx) if False else None
        # question text isn't in doc_nodes; pull from master question content via a light map built once
        break
# build qid->text once (questions only)
from src.pipeline.standardizer import load_nodes
qtext = {n.node_id: n.content for n in load_nodes(f"data/processed/master_nodes_{G.DS}.json")
         if n.metadata.get("type") == "question"}
for h in (1, 2, 3):
    for i in sample[h]:
        seeds = G.resolve_seeds(qtext[ids[i]], doc_id_to_idx)
        seed_tot[h] += 1; seed_hit[h] += int(len(seeds) > 0)
        # LEAKAGE ASSERT: no seed may be a gold answer row for this query
        gset = set(doc_id_to_idx[g] for g in golds[i] if g in doc_id_to_idx)
        assert not (set(seeds) & gset), f"LEAKAGE: seed is a gold for q {ids[i]}"
print("[seed] resolved frac per hop:", {h: round(seed_hit[h]/max(seed_tot[h],1), 4) for h in (1,2,3)})

# ---- Gate 1: reproduce frozen P50 per-hop coverage ----
report = {"harness": "G2_SMOKE_S1", "dataset": G.DS, "split": "val", "sample_per_hop": {h: len(sample[h]) for h in sample},
          "K": G.K_LOCK, "P_MAIN": G.P_MAIN, "K0": G.K0, "PER_HOP": {}, "STRUCT_INFO": sinfo,
          "SEED_RESOLVED_FRAC": {h: seed_hit[h]/max(seed_tot[h],1) for h in (1,2,3)}}
for h in (1, 2, 3):
    rows = sample[h]
    sels = G.p50_scope_partitions(rows, A)          # frozen router, top-P partitions
    hard = A["hard"]
    any_c = all_c = 0; scope_sizes = []
    part_sizes = np.bincount(hard, minlength=A["npart"])
    n_feasible = 0
    for qi, i in enumerate(rows):
        sel = sels[qi]
        gidx = [doc_id_to_idx[g] for g in golds[i] if g in doc_id_to_idx]
        if not gidx:
            continue
        n_feasible += 1
        retained = sum(1 for g in gidx if int(hard[g]) in sel)
        any_c += int(retained >= 1)
        all_c += int(retained == len(gidx))
        scope_sizes.append(int(part_sizes[list(sel)].sum()) if sel else 0)
    report["PER_HOP"][h] = {"n": n_feasible, "ANY": round(any_c/max(n_feasible,1), 4),
                             "ALL": round(all_c/max(n_feasible,1), 4),
                             "scope_mean": int(np.mean(scope_sizes)) if scope_sizes else 0,
                             "scope_max": int(np.max(scope_sizes)) if scope_sizes else 0}
    print(f"[hop {h}] n={n_feasible} ANY={report['PER_HOP'][h]['ANY']} ALL={report['PER_HOP'][h]['ALL']} "
          f"scope~{report['PER_HOP'][h]['scope_mean']}")

# ---- validation vs G1 (tolerance for 300-sample) ----
G1_ANY = {1: 0.9985, 2: 0.9631, 3: 0.9468}; G1_ALL = {1: 0.9959, 2: 0.7138, 3: 0.2701}
ok = True
for h in (1, 2, 3):
    da = abs(report["PER_HOP"][h]["ANY"] - G1_ANY[h]); dl = abs(report["PER_HOP"][h]["ALL"] - G1_ALL[h])
    within = da < 0.06 and dl < 0.08
    ok = ok and within
    print(f"[validate hop {h}] dANY={da:.3f} dALL={dl:.3f} -> {'OK' if within else 'OUT OF TOL'}")
report["HARNESS_VALIDATED"] = bool(ok)
json.dump(report, open("results/GENERALIZATION/_g2_smoke_s1.json", "w"), indent=1)
print("HARNESS_VALIDATED =", ok, "| wrote _g2_smoke_s1.json |", f"{time.time()-t0:.1f}s total")
