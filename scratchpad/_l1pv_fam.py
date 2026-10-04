"""CORE-EXIT PROVENANCE AUDIT -- STEP 1: edge family labels for the FROZEN traversal adjacency.

No new graph work.  Every family already exists on disk in the SAME node-index space that
`_ta_prepartition.load_topology` uses:

  STRUCT  the traversal adjacency itself -- master_nodes_{ds}.json `neighbors`
          (metaqa: official kb.txt triples; 2wiki: Wikipedia hyperlinks; musique/squad/hotpot:
           derived title-mention; webqsp: KB-derived -- see data/canonical/*/graph_manifest.json)
  NER     data/ukb_storage/{ds}/ner_edges_w_df25.pkl        "shares-entity", df-weighted
  A       data/ukb_storage/{ds}/gte_qwen/graph.pt           STRUCT u KNN (semantic Qwen-kNN)
  KNN     A \\ STRUCT                                        the semantic family, by the existing
          reconstruction convention in scratchpad/ac_edge_recon.py

`variant_C` (the frozen topology) is A u NER, and the traversal is STRUCT, so every traversal edge
carries STRUCT and may ALSO carry NER; KNN is disjoint from the traversal by construction of the
label.  That containment is asserted per corpus, not assumed.

Edges are canonicalised undirected as key = min(u,v)*N + max(u,v), matching ac_edge_recon.

  python scratchpad/_l1pv_fam.py [ds ...]
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

CACHE = "scratchpad/_l1pv"
KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PROVENANCE"
BIT = {"STRUCT": 1, "NER": 2, "KNN": 4}
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
# corpus -> the canonical dir whose graph_manifest.json documents the STRUCT provenance
CANON = {"metaqa": "metaqa", "webqsp": "webqsp", "2wiki_clean": "2wiki",
         "musique_clean": "musique", "hotpotqa_clean": "hotpotqa", "squad_clean": "squad"}
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def ukeys(u, v, N):
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    m = u != v
    u, v = u[m], v[m]
    return np.unique(np.minimum(u, v) * np.int64(N) + np.maximum(u, v))


def build(ds, log=log):
    """returns (N, trav_keys sorted, fam uint8 per trav key, stats dict).  Cached."""
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/fam_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        return int(z["N"][0]), z["trav"], z["fam"], json.loads(str(z["stats"]))
    import torch
    import _ta_prepartition as TA
    hard, mem, npart, (ap, ai), deg, id2row = TA.load_topology(ds, lambda *a: None)
    N = len(hard)
    src = np.repeat(np.arange(N, dtype=np.int64), np.diff(ap))
    TRAV = ukeys(src, ai, N)
    log(f"  {ds}: N={N} traversal undirected edges={len(TRAV)}")
    g = torch.load(f"data/ukb_storage/{ds}/gte_qwen/graph.pt", map_location="cpu",
                   weights_only=False)
    A = ukeys(g.edge_index[0].numpy(), g.edge_index[1].numpy(), N)
    del g
    ner = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER = ukeys(ner.row, ner.col, N)
    nnz = int(ner.nnz)
    del ner
    KNN = np.setdiff1d(A, TRAV, assume_unique=True)          # A \ STRUCT, existing convention
    fam = np.full(len(TRAV), BIT["STRUCT"], np.uint8)
    hit_ner = np.isin(TRAV, NER, assume_unique=True)
    fam[hit_ner] |= BIT["NER"]
    hit_knn = np.isin(TRAV, KNN, assume_unique=True)
    fam[hit_knn] |= BIT["KNN"]
    cm = {}
    cp = f"data/canonical/{CANON[ds]}/graph_manifest.json"
    if os.path.exists(cp):
        m = json.load(open(cp))
        cm = {k: m.get(k) for k in ("edge_family", "edge_subtype", "provenance", "n_edges",
                                    "n_relations", "directed")}
    stats = {"ds": ds, "N": N, "npart": npart,
             "traversal_undirected_edges": int(len(TRAV)),
             "A_undirected_edges": int(len(A)),
             "NER_undirected_edges": int(len(NER)), "NER_nnz": nnz,
             "KNN_undirected_edges": int(len(KNN)),
             "traversal_is_subset_of_A": bool(len(np.intersect1d(TRAV, A, assume_unique=True))
                                              == len(TRAV)),
             "traversal_not_in_A": int(len(TRAV)
                                       - len(np.intersect1d(TRAV, A, assume_unique=True))),
             "traversal_edges_also_NER": int(hit_ner.sum()),
             "traversal_edges_also_KNN": int(hit_knn.sum()),
             "frac_traversal_also_NER": round(float(hit_ner.mean()), 6),
             "canonical_struct_manifest": cm}
    np.savez_compressed(fp, N=np.array([N]), trav=TRAV, fam=fam, stats=json.dumps(stats))
    return N, TRAV, fam, stats


class Fam:
    """O(log E) family lookup for an arbitrary (u, v) edge array of the frozen traversal."""

    def __init__(self, ds, log=log):
        self.N, self.trav, self.fam, self.stats = build(ds, log)

    def of(self, u, v):
        u = np.asarray(u, np.int64)
        v = np.asarray(v, np.int64)
        k = np.minimum(u, v) * np.int64(self.N) + np.maximum(u, v)
        i = np.searchsorted(self.trav, k)
        i = np.clip(i, 0, len(self.trav) - 1)
        ok = self.trav[i] == k
        out = np.zeros(len(k), np.uint8)
        out[ok] = self.fam[i[ok]]
        return out, ok


def label(fam):
    """exclusive bucket name for a family bitmask (STRUCT is always set on traversal edges)."""
    n, k = bool(fam & BIT["NER"]), bool(fam & BIT["KNN"])
    if n and k:
        return "STRUCT_NER_KNN"
    if n:
        return "STRUCT_AND_NER"
    if k:
        return "STRUCT_AND_KNN"
    return "STRUCT_ONLY"


if __name__ == "__main__":
    todo = sys.argv[1:] or DS
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    out = {}
    for d in todo:
        N, TRAV, fam, st = build(d)
        if not st["traversal_is_subset_of_A"]:
            log(f"  NOTE {d}: {st['traversal_not_in_A']} traversal edges "
                f"({st['traversal_not_in_A']/st['traversal_undirected_edges']:.4%}) absent from "
                f"gte_qwen/graph.pt -- recorded, not fatal; KNN label is A \ TRAV so those edges "
                f"are STRUCT (+NER if present), which is what the audit uses.")
        u, c = np.unique(fam, return_counts=True)
        st["bucket_counts"] = {label(int(x)): int(y) for x, y in zip(u, c)}
        out[d] = st
        log(d, json.dumps({k: st[k] for k in ("traversal_undirected_edges", "traversal_edges_also_NER",
                                              "traversal_edges_also_KNN", "bucket_counts")}))
    fp = f"{KTD}/diag/step1_families.json"
    old = json.load(open(fp)) if os.path.exists(fp) else {}
    old.update(out)
    json.dump(old, open(fp, "w"), indent=1)
    print(f"wrote {fp}")
