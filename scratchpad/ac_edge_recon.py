"""Task 8/3-correction: edge-family reconstruction + parity, overlap-aware.
Canonicalizes every edge to (min,max), encodes as int64 key = min*N + max.
Families: STRUCT (node.neighbors), A_EDGES (gte_qwen/graph.pt), NER (ner pkl), KNN = A-STRUCT.
Verifies C_GRAPH == A ∪ NER. Reports family overlaps. Also runs the doc-order alignment gate."""
import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch

def _load_doc_nodes(dataset):
    from src.pipeline.standardizer import load_nodes
    mpath = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mpath): mpath = "data/processed/master_nodes.json"
    alln = load_nodes(mpath)
    srcs = set(n.metadata.get("source", "") for n in alln)
    if len(srcs) > 1: alln = [n for n in alln if n.metadata.get("source") == dataset]
    return [n for n in alln if n.metadata.get("type") != "question"]

def _undir_keys(u, v, N):
    u = np.asarray(u, dtype=np.int64); v = np.asarray(v, dtype=np.int64)
    m = u != v                                   # drop self-loops
    u, v = u[m], v[m]
    lo = np.minimum(u, v); hi = np.maximum(u, v)
    return np.unique(lo * np.int64(N) + hi)

def recon(dataset):
    from src.core.engine import CoreEngine
    eng = CoreEngine(source=dataset, index_subdir="gte_qwen")
    N = len(eng.nodes)
    # ---- alignment gate: eng.nodes order == load_nodes doc order (graph/NER live in eng-index space) ----
    doc_nodes = _load_doc_nodes(dataset)
    align = (len(doc_nodes) == N) and all(doc_nodes[i].node_id == eng.nodes[i].node_id for i in range(min(N, len(doc_nodes))))
    id2idx = eng.node_id_to_idx
    # STRUCT from eng.nodes neighbors
    su, sv = [], []
    for i, nd in enumerate(eng.nodes):
        for nb in nd.neighbors:
            j = id2idx.get(nb)
            if j is not None: su.append(i); sv.append(j)
    STRUCT = _undir_keys(su, sv, N)
    # A_EDGES
    gA = torch.load(f"data/ukb_storage/{dataset}/gte_qwen/graph.pt", map_location='cpu', weights_only=False)
    A = _undir_keys(gA.edge_index[0].numpy(), gA.edge_index[1].numpy(), N)
    # NER
    ner = pickle.load(open(f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER = _undir_keys(ner.row, ner.col, N)
    # KNN = A - STRUCT
    KNN = np.setdiff1d(A, STRUCT, assume_unique=True)
    # C parity
    gC = torch.load(f"scratchpad/ablation_qwen/{dataset}/variant_C/graph.pt", map_location='cpu', weights_only=False)
    C = _undir_keys(gC.edge_index[0].numpy(), gC.edge_index[1].numpy(), N)
    C_expected = np.union1d(A, NER)
    c_parity = (len(C) == len(C_expected)) and bool(np.array_equal(C, C_expected))
    # B parity (struct + NER)
    gB = torch.load(f"scratchpad/ablation_qwen/{dataset}/variant_B/graph.pt", map_location='cpu', weights_only=False)
    B = _undir_keys(gB.edge_index[0].numpy(), gB.edge_index[1].numpy(), N)
    B_expected = np.union1d(STRUCT, NER)
    b_parity = (len(B) == len(B_expected)) and bool(np.array_equal(B, B_expected))
    # overlaps
    sn = len(np.intersect1d(STRUCT, NER, assume_unique=True))
    sk = len(np.intersect1d(STRUCT, KNN, assume_unique=True))
    nk = len(np.intersect1d(NER, KNN, assume_unique=True))
    return {
        "dataset": dataset, "N_nodes": int(N),
        "ALIGNMENT_GATE": "PASS" if align else "FAIL",
        "STRUCT": int(len(STRUCT)), "A_EDGES": int(len(A)), "NER": int(len(NER)),
        "KNN": int(len(KNN)), "C_EDGES": int(len(C)), "C_EXPECTED": int(len(C_expected)),
        "B_EDGES": int(len(B)), "B_EXPECTED": int(len(B_expected)),
        "C_PARITY": "PASS" if c_parity else "FAIL", "B_PARITY": "PASS" if b_parity else "FAIL",
        "overlap_STRUCT_NER": sn, "overlap_STRUCT_KNN": sk, "overlap_NER_KNN": nk,
        "EDGE_FAMILY_RECONSTRUCTION": "PASS" if (align and c_parity and b_parity) else "FAIL",
    }

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["2wiki_clean","musique_clean","hotpotqa_clean","squad_clean"])
    ap.add_argument("--out", default="results/L1/ac_edge_recon.json")
    a = ap.parse_args()
    res = {}
    for ds in a.datasets:
        r = recon(ds); res[ds] = r
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"{ds:16} align={r['ALIGNMENT_GATE']} STRUCT={r['STRUCT']} KNN={r['KNN']} NER={r['NER']} "
              f"A={r['A_EDGES']} C={r['C_EDGES']}(exp {r['C_EXPECTED']}) Cpar={r['C_PARITY']} Bpar={r['B_PARITY']} "
              f"| ovl S∩N={r['overlap_STRUCT_NER']} S∩K={r['overlap_STRUCT_KNN']} N∩K={r['overlap_NER_KNN']} "
              f"=> {r['EDGE_FAMILY_RECONSTRUCTION']}", flush=True)
    print(f"WROTE {a.out}")
