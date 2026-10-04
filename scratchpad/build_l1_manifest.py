"""Build results/L1/L1_LOCKED_MANIFEST.json — the frozen L1 record (MASTER_TOPOLOGY=C, C partitioning,
Dense+SPLADE fusion, K100/P50). READ-ONLY over caches: hashes + shapes + stats. No recompute of L1."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

DATASETS = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa", "webqsp"]
TEXT4 = {"2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"}

# eval protocol + gold semantics (known/frozen; not recomputed here)
PROTO = {
    "2wiki_clean":   ("DERIVED_TEST", "supporting passages (ALL primary); 2-hop"),
    "musique_clean": ("DERIVED_TEST", "supporting passages (ALL primary); 2-4 hop"),
    "hotpotqa_clean":("DERIVED_TEST", "2 supporting passages (ALL primary)"),
    "squad_clean":   ("DERIVED_TEST", "single gold passage (ANY primary)"),
    "metaqa":        ("NATIVE_TEST",  "KB answer-entity nodes (ANY primary); per-hop 1/2/3"),
    "webqsp":        ("ALL_EVALUABLE","KB answer-entity nodes (ANY primary)"),
}
PRIMARY = {"2wiki_clean":"ALL","musique_clean":"ALL","hotpotqa_clean":"ALL",
           "squad_clean":"ANY","metaqa":"ANY","webqsp":"ANY"}

def sha256(path, cap_mb=None):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256(); n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
            if cap_mb and n >= cap_mb * (1 << 20):
                h.update(b"__CAPPED__"); break
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(path), "capped": bool(cap_mb and n >= cap_mb*(1<<20))}

def npy_meta(path):
    if not os.path.exists(path):
        return None
    a = np.load(path, mmap_mode='r')
    return {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": os.path.getsize(path),
            "sha256": sha256(path)["sha256"]}

def phase1_c_fusion(dataset, primary):
    """Pull the C dense+splade K100/P50 cell (canonical operating point) + A for reference."""
    fn = {"2wiki_clean":"phase1_2wiki","musique_clean":"phase1_musique","hotpotqa_clean":"phase1_hotpot",
          "squad_clean":"phase1_squad","metaqa":"phase1_metaqa","webqsp":"phase1_webqsp_corrected"}[dataset]
    p = f"results/L1/{fn}.json"
    cells = json.load(open(p))
    def cell(topo):
        for c in cells:
            if c["K"]==100 and c["router"]=="dense+splade" and c["topology"]==topo and c["P"]==50:
                return c
        return None
    cC, cA = cell("C"), cell("A")
    def pick(c):
        if not c: return None
        return {"ANY_GOLD_COV":c["ANY_GOLD_COV"],"ALL_GOLD_COV":c["ALL_GOLD_COV"],
                "GOLD_FRAC":c["GOLD_FRAC"],"mean_scope":c["mean_scope"],"reduction":c["reduction"],
                "PRIMARY":primary,"PRIMARY_VALUE":c[f"{primary}_GOLD_COV"]}
    return {"C_dense+splade_K100_P50": pick(cC), "A_dense+splade_K100_P50": pick(cA),
            "source_json": p, "source_sha256": sha256(p)["sha256"]}

def main():
    recon = json.load(open("results/L1/ac_edge_recon.json"))
    out = {
        "STATUS": {"L1_FROZEN": True, "L1_DOCUMENTED": True, "L1_RECOMPUTE_REQUIRED": False},
        "ARCHITECTURE": {
            "MASTER_TOPOLOGY": "C",
            "C_EDGE_FAMILIES": ["STRUCT", "NER", "QWEN_KNN"],
            "L1_PARTITION_TOPOLOGY": "C",
            "ROUTER": "Dense+SPLADE",
            "DENSE_MODEL": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "DENSE_EXACT_CEILING": "FAISS IndexFlatIP (top200)",
            "SPARSE": "SPLADE exact (naver/splade-cocondenser-ensembledistil, top200)",
            "FUSION": "partition-level RRF, K0=60",
            "K": 100, "P_MAIN": 50,
            "P20_ROLE": "aggressive-pruning / efficiency ablation only",
            "P100_ROLE": "high-recall ceiling only",
            "SELECTION_RATIONALE": ("C selected for ARCHITECTURAL CONSISTENCY + FUTURE L2/L3 SIGNAL "
                "AVAILABILITY, NOT universal L1 dominance. A and C have ~equal aggregate L1 quality "
                "(~+/-1pp at K100/fusion/P50); C wins some datasets, A wins others; scopes ~identical "
                "(~5k) but candidate sets differ (Jaccard 0.09-0.38, ~0% identical) and mutually rescue "
                "different queries. C is the superset graph (STRUCT+NER+QWEN_KNN) giving one unified "
                "representation from which downstream stages selectively consume edge families; L3 need "
                "not blindly traverse all C edges."),
        },
        "INTEGRITY": {"CACHE_INTEGRITY": "PASS", "EDGE_FAMILY_RECONSTRUCTION": "PASS",
                      "C_PARITY": "PASS (C == A U NER, verified 4 text substrates)",
                      "B_PARITY": "PASS", "ALIGNMENT_GATE": "PASS (eng.nodes order == load_nodes doc order)",
                      "note": "STRUCT ∩ KNN = 0 (A = STRUCT ⊔ KNN); big npy/graph hashes are full sha256."},
        "P50_DECISION": ("P20 too aggressive as L1->L2 boundary; K100/fusion P20->P50 recovered ~5-10pp "
            "primary gold cov for ~2.5x scope; P_MAIN=50 (~5k cand/query) is the frozen L1->L2 interface. "
            "Do not retune P during L2."),
        "ANY_VS_ALL": ("ANY_GOLD_COV = >=1 gold survives L1; ALL_GOLD_COV = all supporting golds survive. "
            "High ANY / lower ALL => partial evidence. Distinguish (A) within-scope reasoning failure from "
            "(B) L1 pruning failure. Under strict induced-P50, a gold pruned by L1 is UNRECOVERABLE; L3 must "
            "not silently reintroduce the full corpus. Bounded frontier expansion = explicit future L3 experiment."),
        "datasets": {},
        "L1_RESULT_ARTIFACTS": {},
        "AC_DIAGNOSTIC_ARTIFACTS": {},
        "SCRIPT_HASHES": {},
        "CAVEATS": [
            "C is ARCHITECTURALLY selected; C is NOT a universal empirical L1 winner (A wins hotpot on scope; "
            "C wins 2wiki/musique/squad marginally).",
            "MetaQA/WebQSP B/C NER variants remain LEGACY_EXPLORATORY: their entity-node substrate makes "
            "current NER provenance noncanonical; canonical L1 for metaqa/webqsp stays topology-A "
            "(metaqa winner = SPLADE-alone; webqsp = A-fusion). C partition_maps exist for them but are "
            "not the canonical operating topology.",
            "Hotpot/SQuAD A/C graph diagnostic used strided sampling (2500 of 9786/13033 queries); 2wiki/"
            "musique were full-N. The 648-cell L1 sweep itself is full-population (no sampling).",
        ],
    }

    for ds in DATASETS:
        cpm = f"scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json"
        cgraph = f"scratchpad/ablation_qwen/{ds}/variant_C/graph.pt"
        apm = f"data/ukb_storage/{ds}/gte_qwen/partition_map.json"
        qids = f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json"
        dense = f"data/ukb_storage/{ds}/gte_qwen/dense_top200_all.npy"
        splade = f"data/ukb_storage/{ds}/gte_qwen/splade_top200_all.npy"
        ner = f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl"

        pm = json.load(open(cpm))
        n_docs = len(pm)
        npart_c = max(int(v) for v in pm.values()) + 1
        qj = json.load(open(qids))
        n_all = len(qj["ids"])
        si = qj.get("split_indices", {})
        if ds == "webqsp":
            n_eval = len(si.get("all", qj["ids"]))
        else:
            n_eval = len(si["test"]) if si.get("test") else len(si.get("all", qj["ids"]))
        c_status = "CANONICAL" if ds in TEXT4 else "LEGACY_EXPLORATORY"

        rec = recon.get(ds)
        edge_fams = None
        if rec:
            edge_fams = {"STRUCT": rec["STRUCT"], "NER": rec["NER"], "QWEN_KNN": rec["KNN"],
                         "A_undirected": rec["A_EDGES"], "B_undirected": rec["B_EDGES"],
                         "C_undirected": rec["C_EDGES"], "C_PARITY": rec["C_PARITY"],
                         "ALIGNMENT_GATE": rec["ALIGNMENT_GATE"]}

        out["datasets"][ds] = {
            "C_STATUS": c_status,
            "N_docs": n_docs, "N_all_queries": n_all, "N_eval": n_eval,
            "eval_population": PROTO[ds][0], "gold_semantics": PROTO[ds][1],
            "PRIMARY_METRIC": PRIMARY[ds],
            "npart_C": npart_c,
            "edge_families_undirected": edge_fams,
            "paths": {
                "C_partition_map": cpm, "C_graph": cgraph, "A_partition_map": apm,
                "query_ids_all": qids, "dense_cache": dense, "splade_cache": splade,
                "ner_edges_pkl": ner,
            },
            "hashes": {
                "C_partition_map": sha256(cpm),
                "C_graph": sha256(cgraph),
                "A_partition_map": sha256(apm),
                "query_ids_all": sha256(qids),
                "ner_edges_pkl": sha256(ner),
                "dense_cache": npy_meta(dense),
                "splade_cache": npy_meta(splade),
            },
            "L1_operating_point": phase1_c_fusion(ds, PRIMARY[ds]),
        }
        print(f"{ds:16} docs={n_docs} q_all={n_all} eval={n_eval} npartC={npart_c} C={c_status}", flush=True)

    # result artifacts
    for tag, p in [("L1_FULL_TABLE","results/L1/L1_FULL_TABLE.md"),
                   ("L1_STOP_OUTPUTS","results/L1/L1_STOP_OUTPUTS.md"),
                   ("phase1_2wiki","results/L1/phase1_2wiki.json"),
                   ("phase1_musique","results/L1/phase1_musique.json"),
                   ("phase1_hotpot","results/L1/phase1_hotpot.json"),
                   ("phase1_squad","results/L1/phase1_squad.json"),
                   ("phase1_metaqa","results/L1/phase1_metaqa.json"),
                   ("phase1_webqsp_corrected","results/L1/phase1_webqsp_corrected.json")]:
        out["L1_RESULT_ARTIFACTS"][tag] = {"path": p, **(sha256(p) or {})}
    for tag, p in [("ac_scope_analysis","results/L1/ac_scope_analysis.json"),
                   ("ac_scope_hotpot","results/L1/ac_scope_hotpot.json"),
                   ("ac_scope_legacy","results/L1/ac_scope_legacy.json"),
                   ("ac_graph_diagnostic","results/L1/ac_graph_diagnostic.json"),
                   ("ac_graph_diagnostic_md","results/L1/ac_graph_diagnostic.md"),
                   ("ac_edge_recon","results/L1/ac_edge_recon.json")]:
        out["AC_DIAGNOSTIC_ARTIFACTS"][tag] = {"path": p, **(sha256(p) or {})}
    for tag, p in [("phase1_l1_allq","scratchpad/phase1_l1_allq.py"),
                   ("l1_eval_phase1","scratchpad/l1_eval_phase1.py"),
                   ("ac_scope_analysis_py","scratchpad/ac_scope_analysis.py"),
                   ("ac_edge_recon_py","scratchpad/ac_edge_recon.py"),
                   ("ac_graph_diagnostic_py","scratchpad/ac_graph_diagnostic.py"),
                   ("build_l1_manifest_py","scratchpad/build_l1_manifest.py")]:
        out["SCRIPT_HASHES"][tag] = {"path": p, **(sha256(p) or {})}

    os.makedirs("results/L1", exist_ok=True)
    json.dump(out, open("results/L1/L1_LOCKED_MANIFEST.json", "w"), indent=2)
    print("WROTE results/L1/L1_LOCKED_MANIFEST.json", flush=True)

if __name__ == "__main__":
    main()
