"""Tasks 5-11: A-vs-C graph diagnostic (induced-subgraph connectivity + gold reachability).
GRAPH_SCOPE_RULE = INDUCED_SUBGRAPH_OVER_L1_CANDIDATE_IDS.
Conditions: COND_1 A_scope+A_graph, COND_2 A_scope+C_graph, COND_3 C_scope+C_graph, COND_4 C_scope+A_graph.
Both UNCONDITIONAL and CONDITIONAL_ON_GOLD_PRESENT reported. Graph-family ablation STRUCT/B/A/C on fixed A_scope.
CPU-only. Routing bit-identical to ac_scope_analysis (fusion RRF K0=60). No retrieval recompute.
"""
import os, sys, json, pickle, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components
from l1_eval_phase1 import load_partition_topology
from ac_scope_analysis import _fusion_ranking, _test_idx

K0 = 60
SEED_BUDGET = 20   # dense topN + splade topN (each), intersect scope, as traversal anchors
MAX_HOP = 3

# ---------------- edge families -> symmetric boolean CSR over N doc nodes ----------------
def _sym_csr(u, v, N):
    u = np.asarray(u, np.int64); v = np.asarray(v, np.int64)
    m = u != v
    u, v = u[m], v[m]
    ru = np.concatenate([u, v]); rv = np.concatenate([v, u])
    data = np.ones(len(ru), np.uint8)
    A = coo_matrix((data, (ru, rv)), shape=(N, N)).tocsr()
    A.data[:] = 1
    A.sum_duplicates()
    A.data[:] = 1
    return A

def load_graph_families(dataset):
    """Returns dict family-> symmetric boolean CSR. STRUCT via CoreEngine neighbors; A/NER/C/B via files."""
    from src.core.engine import CoreEngine
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    N = len(eng.nodes)
    id2idx = eng.node_id_to_idx
    import torch
    # STRUCT
    su, sv = [], []
    for i, nd in enumerate(eng.nodes):
        for nb in nd.neighbors:
            j = id2idx.get(nb)
            if j is not None:
                su.append(i); sv.append(j)
    STRUCT = _sym_csr(su, sv, N)
    gA = torch.load(f"data/ukb_storage/{dataset}/gte_qwen/graph.pt", map_location='cpu', weights_only=False)
    A = _sym_csr(gA.edge_index[0].numpy(), gA.edge_index[1].numpy(), N)
    ner = pickle.load(open(f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER = _sym_csr(ner.row, ner.col, N)
    gC = torch.load(f"scratchpad/ablation_qwen/{dataset}/variant_C/graph.pt", map_location='cpu', weights_only=False)
    C = _sym_csr(gC.edge_index[0].numpy(), gC.edge_index[1].numpy(), N)
    gB = torch.load(f"scratchpad/ablation_qwen/{dataset}/variant_B/graph.pt", map_location='cpu', weights_only=False)
    B = _sym_csr(gB.edge_index[0].numpy(), gB.edge_index[1].numpy(), N)
    return {"STRUCT": STRUCT, "A": A, "NER": NER, "B": B, "C": C}, N

# ---------------- per-query induced-subgraph BFS reachability ----------------
def bfs_reach(G, cand_mask, seed_idx, gold_idx):
    """Multi-source depth-limited BFS on induced subgraph (both endpoints in cand_mask).
    Returns per-gold first-reach hop (0=is a seed, 1..MAX_HOP, -1 unreached within MAX_HOP)."""
    N = G.shape[0]
    reached = np.zeros(N, bool)
    frontier = np.zeros(N, bool)
    frontier[seed_idx] = True
    frontier &= cand_mask
    reached |= frontier
    gold_hop = {int(g): (0 if reached[g] else -1) for g in gold_idx}
    fv = frontier.astype(np.uint8)
    for hop in range(1, MAX_HOP + 1):
        nxt = (G.dot(fv) > 0) & cand_mask & (~reached)
        if not nxt.any():
            break
        reached |= nxt
        for g in gold_idx:
            if gold_hop[g] == -1 and reached[g]:
                gold_hop[g] = hop
        fv = nxt.astype(np.uint8)
    return gold_hop, reached

def induced_components(G, cand):
    sub = G[cand][:, cand]
    ncomp, labels = connected_components(sub, directed=False)
    if len(cand) == 0:
        return 0, 0.0
    _, counts = np.unique(labels, return_counts=True)
    return int(ncomp), float(counts.max() / len(cand))

# ---------------- main diagnostic ----------------
def run(dataset, P=50, limit=None, do_connectivity=True, sample_cap=None):
    t0 = time.time()
    base = f"data/ukb_storage/{dataset}/gte_qwen"
    j = json.load(open(f"{base}/query_ids_all.json"))
    golds = j["golds"]; hops = j.get("hops", [None] * len(j["ids"]))
    test_idx = _test_idx(dataset, j)
    sampled = False
    if limit:
        test_idx = test_idx[:limit]
    elif sample_cap and len(test_idx) > sample_cap:
        stride = len(test_idx) // sample_cap
        test_idx = test_idx[::stride][:sample_cap]
        sampled = True
    dense = np.load(f"{base}/dense_top200_all.npy", mmap_mode='r')
    splade = np.load(f"{base}/splade_top200_all.npy", mmap_mode='r')
    dense_t = np.array([dense[i] for i in test_idx])
    splade_t = np.array([splade[i] for i in test_idx])

    hA, memA, npA, doc_nodes, _ = load_partition_topology(dataset, "A")
    hC, memC, npC, _, _ = load_partition_topology(dataset, "C")
    id2idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
    tg = []
    for i in test_idx:
        tg.append([id2idx[g] for g in golds[i] if g in id2idx])

    rankA = _fusion_ranking(dense_t, splade_t, memA, npA, 100)
    rankC = _fusion_ranking(dense_t, splade_t, memC, npC, 100)
    # per-partition doc lists
    docs_A = [np.where(hA == p)[0] for p in range(npA)]
    docs_C = [np.where(hC == p)[0] for p in range(npC)]
    N = len(hA)

    fams, Ng = load_graph_families(dataset)
    assert Ng == N, f"graph N {Ng} vs partition N {N}"
    t_setup = time.time() - t0

    # conditions: (scope_topo, graph_family)
    conds = {"COND_1": ("A", "A"), "COND_2": ("A", "C"), "COND_3": ("C", "C"), "COND_4": ("C", "A")}
    abl_fams = ["STRUCT", "B", "A", "C"]  # graph-family ablation on fixed A_scope

    # accumulators
    def new_acc():
        return {"n_gold_present": 0, "n_gold_total": 0,
                "reach_cond": np.zeros(MAX_HOP + 1),   # golds present & reached by hop<=h (cumulative), index0=hop0
                "reach_uncond": np.zeros(MAX_HOP + 1),
                "allgold_cond_at3": 0, "allgold_uncond_at3": 0, "nq_goldpresent": 0, "nq": 0,
                "expanded_sum": 0, "gold_reached3_sum": 0}
    cond_acc = {c: new_acc() for c in conds}
    abl_acc = {f: new_acc() for f in abl_fams}
    conn_acc = {c: {"ncomp": [], "largest": []} for c in conds}

    tq0 = time.time()
    for qi in range(len(test_idx)):
        selA = rankA[qi][:P]; selC = rankC[qi][:P]
        candA = np.concatenate([docs_A[p] for p in selA]) if len(selA) else np.array([], np.int64)
        candC = np.concatenate([docs_C[p] for p in selC]) if len(selC) else np.array([], np.int64)
        maskA = np.zeros(N, bool); maskA[candA] = True
        maskC = np.zeros(N, bool); maskC[candC] = True
        # anchor seeds = (dense topN U splade topN) intersect scope
        dseed = dense_t[qi][:SEED_BUDGET]; sseed = splade_t[qi][:SEED_BUDGET]
        seeds_all = np.unique(np.concatenate([dseed, sseed]))
        seeds_all = seeds_all[(seeds_all >= 0) & (seeds_all < N)]
        gi = np.array(tg[qi], np.int64)
        n_gold_total = len(gi)

        def eval_cond(mask, cand, G, acc, want_conn=False, conn_store=None):
            gold_present = gi[mask[gi]] if n_gold_total else gi
            seeds = seeds_all[mask[seeds_all]]
            acc["nq"] += 1
            acc["n_gold_total"] += n_gold_total
            acc["n_gold_present"] += len(gold_present)
            if n_gold_total == 0:
                return
            gold_hop, reached = bfs_reach(G, mask, seeds, gi)
            acc["expanded_sum"] += int(reached.sum())
            # cumulative reach by hop, conditional (present golds) and unconditional (all golds)
            hops_arr = np.array([gold_hop[int(g)] for g in gi])
            present = mask[gi]
            for h in range(MAX_HOP + 1):
                rc = (hops_arr >= 0) & (hops_arr <= h)
                acc["reach_uncond"][h] += int(rc.sum())
                acc["reach_cond"][h] += int((rc & present).sum())
            reached3 = (hops_arr >= 0) & (hops_arr <= 3)
            acc["gold_reached3_sum"] += int(reached3.sum())
            if len(gold_present) > 0:
                acc["nq_goldpresent"] += 1
                if reached3[present].all():
                    acc["allgold_cond_at3"] += 1
            if reached3.all():
                acc["allgold_uncond_at3"] += 1
            if want_conn and conn_store is not None:
                nc, lg = induced_components(G, cand)
                conn_store["ncomp"].append(nc); conn_store["largest"].append(lg)

        for c, (stopo, gfam) in conds.items():
            mask = maskA if stopo == "A" else maskC
            cand = candA if stopo == "A" else candC
            eval_cond(mask, cand, fams[gfam], cond_acc[c],
                      want_conn=do_connectivity, conn_store=conn_acc[c])
        for f in abl_fams:
            eval_cond(maskA, candA, fams[f], abl_acc[f])
    tq = time.time() - tq0

    def summ(acc):
        ngt = max(acc["n_gold_total"], 1); ngp = max(acc["n_gold_present"], 1)
        return {
            "nq": acc["nq"], "n_gold_total": acc["n_gold_total"], "n_gold_present": acc["n_gold_present"],
            "gold_present_rate": round(acc["n_gold_present"] / ngt, 4),
            "reach_uncond_at1": round(acc["reach_uncond"][1] / ngt, 4),
            "reach_uncond_at2": round(acc["reach_uncond"][2] / ngt, 4),
            "reach_uncond_at3": round(acc["reach_uncond"][3] / ngt, 4),
            "reach_cond_at1": round(acc["reach_cond"][1] / ngp, 4),
            "reach_cond_at2": round(acc["reach_cond"][2] / ngp, 4),
            "reach_cond_at3": round(acc["reach_cond"][3] / ngp, 4),
            "allgold_cond_at3_pct": round(100 * acc["allgold_cond_at3"] / max(acc["nq_goldpresent"], 1), 2),
            "allgold_uncond_at3_pct": round(100 * acc["allgold_uncond_at3"] / max(acc["nq"], 1), 2),
            "mean_expanded": round(acc["expanded_sum"] / max(acc["nq"], 1), 1),
            "gold_reach_per_1k_expanded": round(1000 * acc["gold_reached3_sum"] / max(acc["expanded_sum"], 1), 3),
        }

    out = {"dataset": dataset, "P": P, "n_test": len(test_idx), "sampled": sampled, "seed_budget": SEED_BUDGET,
           "max_hop": MAX_HOP, "setup_sec": round(t_setup, 1), "query_sec": round(tq, 1),
           "sec_per_100q": round(100 * tq / max(len(test_idx), 1), 2),
           "conditions": {c: summ(cond_acc[c]) for c in conds},
           "graph_family_ablation_Ascope": {f: summ(abl_acc[f]) for f in abl_fams}}
    if do_connectivity:
        out["connectivity"] = {c: {"mean_ncomp": round(float(np.mean(conn_acc[c]["ncomp"])), 1),
                                   "mean_largest_comp_frac": round(float(np.mean(conn_acc[c]["largest"])), 4)}
                               for c in conds}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"])
    ap.add_argument("--P", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-cap", type=int, default=None)
    ap.add_argument("--no-conn", action="store_true")
    ap.add_argument("--out", default="results/L1/ac_graph_diagnostic.json")
    a = ap.parse_args()
    allres = {}
    for ds in a.datasets:
        print(f"=== {ds} (P{a.P}, limit={a.limit}, cap={a.sample_cap}) ===", flush=True)
        r = run(ds, P=a.P, limit=a.limit, do_connectivity=not a.no_conn, sample_cap=a.sample_cap)
        allres[ds] = r
        os.makedirs("results/L1", exist_ok=True)
        json.dump(allres, open(a.out, "w"), indent=2)
        c = r["conditions"]
        print(f"  setup {r['setup_sec']}s query {r['query_sec']}s ({r['sec_per_100q']}s/100q)", flush=True)
        print(f"  COND1 A+A cond@3 {c['COND_1']['reach_cond_at3']} | COND3 C+C cond@3 {c['COND_3']['reach_cond_at3']} "
              f"| gold_present A {c['COND_1']['gold_present_rate']} C {c['COND_3']['gold_present_rate']}", flush=True)
    print(f"WROTE {a.out}", flush=True)
