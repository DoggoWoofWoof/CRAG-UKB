"""W0 — FULL/OPEN-WORLD CORPUS SUBSTRATE manifest builder.
Discovery-only: re-reads existing local manifests/stats and emits
results/WORLD/W0_CORPUS_MANIFEST.{json,md}. No compute, no eval.
"""
import os, json, glob

ROOT = os.getcwd()
CAN = "data/canonical"
ABL = "scratchpad/ablation_qwen"
OUT = "results/WORLD"
os.makedirs(OUT, exist_ok=True)

def jload(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def enc_status(ds, enc, kind):
    """kind in {docs, queries}. Returns (present, complete, n_items, rows)."""
    m = jload(f"{CAN}/{ds}/encodings/{enc}/{kind}/manifest.json")
    if m is None:
        return dict(present=False, complete=False, n_items=None, rows=None)
    return dict(present=True, complete=bool(m.get("complete")),
                n_items=m.get("n_items"), rows=m.get("rows_covered"),
                encoder=m.get("encoder"), dim=m.get("dim"), dtype=m.get("dtype"))

def faiss_present(ds):
    hits = []
    for pat in ("*.faiss", "*.index", "index.faiss", "flat*.bin"):
        hits += glob.glob(f"{CAN}/{ds}/encodings/dense/**/{pat}", recursive=True)
    # exclude the shard index.json (that's a shard->row map, not a FAISS index)
    return [h for h in hits if not h.endswith("index.json")]

def clean_stats(clean_ds):
    s = jload(f"{ABL}/{clean_ds}/variant_C/stats.json")
    pm = os.path.exists(f"{ABL}/{clean_ds}/variant_C/partition_map.json")
    gp = os.path.exists(f"{ABL}/{clean_ds}/variant_C/graph.pt")
    if s is None:
        return dict(n_nodes=None, n_parts=None, partition_map=pm, graph_pt=gp)
    return dict(n_nodes=s.get("n_nodes"), n_parts=s.get("n_parts"),
                partition_map=pm, graph_pt=gp,
                synthetic_qwen_edges=s.get("synthetic_qwen_edges"))

def graph_families(ds):
    fam = {}
    gm = jload(f"{CAN}/{ds}/graph_manifest.json")
    fam["structural"] = dict(
        present=os.path.exists(f"{CAN}/{ds}/graph_structural.tsv"),
        n_edges=(gm or {}).get("n_edges"), subtype=(gm or {}).get("edge_subtype"),
        family=(gm or {}).get("edge_family"))
    nm = jload(f"{CAN}/{ds}/ner_manifest.json")
    fam["ner"] = dict(present=os.path.exists(f"{CAN}/{ds}/graph_ner.tsv"),
                      n_edges=(nm or {}).get("n_edges"))
    # qwen semantic-kNN edges for the WORLD corpus: look for a knn edge tsv
    knn = glob.glob(f"{CAN}/{ds}/graph_qwen*.tsv") + glob.glob(f"{CAN}/{ds}/graph_knn*.tsv")
    fam["qwen_knn"] = dict(present=len(knn) > 0, files=knn)
    return fam

# clean_family -> (clean_ablation_dir, [world canonical ds ...])
FAMILIES = {
    "2wiki":    dict(clean="2wiki_clean", world="2wiki",     universe="2wiki_universe", kind="TEXT"),
    "hotpotqa": dict(clean="hotpotqa_clean", world="hotpotqa", universe=None, kind="TEXT"),
    "musique":  dict(clean="musique_clean", world="musique",  universe=None, kind="TEXT"),
    "squad":    dict(clean="squad_clean", world="squad",      universe=None, kind="TEXT"),
    "webqsp":   dict(clean="webqsp", world="webqsp",          universe=None, kind="KB"),
    "metaqa":   dict(clean="metaqa", world="metaqa",          universe=None, kind="KB"),
}

RAW_LOCAL = {  # du -sh data/original/<x>
    "2wiki": "2.8G", "hotpotqa": "1.9G", "metaqa": "135M",
    "musique": "287M", "squad": "45M", "webqsp": "523M",
}

out = {"generated_by": "scratchpad/_w0_build_manifest.py",
       "note": "W0 discovery-only inventory; clean=controlled substrate, world=open retrieval corpus, universe=full graph universe",
       "datasets": {}}

for fam, cfg in FAMILIES.items():
    world = cfg["world"]
    dm = jload(f"{CAN}/{world}/document_manifest.json") or {}
    qm = jload(f"{CAN}/{world}/query_manifest.json")
    cst = clean_stats(cfg["clean"])
    gf = graph_families(world)
    dd = enc_status(world, "dense", "docs")
    sd = enc_status(world, "splade", "docs")
    dq = enc_status(world, "dense", "queries")
    faiss = faiss_present(world)

    # gold->world mapping from query_manifest (train/dev). denom = answerable where present.
    gold = {}
    if world == "metaqa":
        # metaqa queries live in per-hop dirs
        for h in ("1hop", "2hop", "3hop"):
            hm = jload(f"{CAN}/metaqa_{h}/query_manifest.json")
            if hm:
                for sp, v in hm.get("splits", {}).items():
                    gold[sp] = dict(n=v.get("n"), gold_mapped=v.get("gold_mapped"))
        dq = enc_status("metaqa_1hop", "dense", "queries")  # per-hop enc (representative)
    elif qm and "splits" in qm:
        for sp, v in qm["splits"].items():
            denom = v.get("answerable", v.get("n"))
            gm_ = v.get("gold_mapped", v.get("all_sf_titles_mapped", v.get("all_gold_mapped")))
            gold[sp] = dict(n=denom, gold_mapped=gm_)

    uni = None
    if cfg["universe"]:
        u = cfg["universe"]
        udm = jload(f"{CAN}/{u}/document_manifest.json") or {}
        uni = dict(
            path=f"{CAN}/{u}", n_docs=udm.get("n_docs"),
            dense=enc_status(u, "dense", "docs"), splade=enc_status(u, "splade", "docs"),
            graph=graph_families(u),
            crosswalk_to_view=os.path.exists(f"{CAN}/{u}/crosswalk_to_retrieval_view.json"))

    # ---- per-dataset gates ----
    world_raw_ready = fam in RAW_LOCAL  # verified local du
    world_id_map_ready = bool(dm.get("n_docs"))  # canonical id scheme in documents.jsonl
    clean_world_map_ready = (cfg["universe"] is not None and uni and uni["crosswalk_to_view"])
    _tr = {sp: v for sp, v in gold.items() if ("train" in sp or "dev" in sp)}
    gold_world_map_ready = bool(_tr) and all(
        (v["gold_mapped"] or 0) >= (v["n"] or 0) for v in _tr.values())
    world_qwen_ready = dd["complete"] and (dq["complete"] if dq["present"] else True)
    world_dense_index_ready = len(faiss) > 0
    world_splade_ready = sd["complete"]
    world_graph_c_ready = gf["structural"]["present"] and gf["qwen_knn"]["present"]  # C needs qwen-knn combined
    world_partitions_ready = os.path.exists(f"{ABL}/{world}_world/variant_C/partition_map.json") or \
                             os.path.exists(f"{CAN}/{world}/partitions_C/partition_map.json")
    world_retrieval_ready = all([world_qwen_ready, world_dense_index_ready, world_splade_ready,
                                 world_graph_c_ready, world_partitions_ready, gold_world_map_ready])

    out["datasets"][fam] = dict(
        kind=cfg["kind"],
        clean=dict(ablation_dir=f"{ABL}/{cfg['clean']}/variant_C",
                   n_nodes=cst["n_nodes"], n_parts=cst["n_parts"],
                   partition_map=cst["partition_map"], graph_pt=cst["graph_pt"],
                   C_uses_qwen_knn=cst.get("synthetic_qwen_edges")),
        world=dict(
            corpus_path=f"{CAN}/{world}/documents.jsonl",
            n_docs=dm.get("n_docs"), id_scheme=dm.get("id_scheme", dm.get("doc_unit")),
            provenance=dm.get("provenance"),
            raw_source_local=RAW_LOCAL.get(fam),
            docs_dense=dd, docs_splade=sd, queries_dense=dq,
            faiss_index=faiss, graph=gf, gold_map=gold),
        universe=uni,
        gates=dict(
            WORLD_RAW_READY=world_raw_ready,
            WORLD_ID_MAP_READY=world_id_map_ready,
            CLEAN_WORLD_MAP_READY=clean_world_map_ready,
            GOLD_WORLD_MAP_READY=gold_world_map_ready,
            WORLD_QWEN_READY=world_qwen_ready,
            WORLD_DENSE_INDEX_READY=world_dense_index_ready,
            WORLD_SPLADE_READY=world_splade_ready,
            WORLD_GRAPH_C_READY=world_graph_c_ready,
            WORLD_PARTITIONS_READY=world_partitions_ready,
            WORLD_RETRIEVAL_READY=world_retrieval_ready,
        ),
    )

# overall
any_retrieval = any(d["gates"]["WORLD_RETRIEVAL_READY"] for d in out["datasets"].values())
out["WORLD_RAW_INPUTS_LOCAL"] = "YES"  # all data/original/* verified present
out["SAFE_TO_START_FULL_WORLD_EXPERIMENT"] = "YES" if any_retrieval else "NO"
out["STOP_BEFORE_STARTING"] = True

out["clean_to_world_mapping_coverage"] = {
    "2wiki": {"materialized": True, "coverage": 1.0,
              "file": "data/canonical/2wiki_universe/crosswalk_to_retrieval_view.json"},
    "hotpotqa": {"materialized": False, "derivable": "identity on canonical curid"},
    "musique": {"materialized": False, "derivable": "identity on canonical text-hash id"},
    "squad": {"materialized": False, "derivable": "identity on canonical text-hash id"},
    "webqsp": {"materialized": False, "derivable": "identity on canonical entity id"},
    "metaqa": {"materialized": False, "derivable": "identity on canonical entity id"},
}
out["task7_dense_index_cost"] = {
    "note": "no FAISS index exists yet; exact IndexFlatIP stores fp32 (N*1536*4B); fp16 sharded GPU topk is the established parity-1.0 path",
    "exact_flatip_fp32_GB": {
        "2wiki_universe": 36.8, "hotpotqa": 32.2, "webqsp": 8.1,
        "2wiki": 2.45, "musique": 0.72, "metaqa": 0.27, "squad": 0.12},
    "recommended": "Option A: exact sharded fp16 GPU topk for 5M corpora + IndexFlatIP for <=1.32M; exact semantics preserved; STOP for sign-off before build",
    "STOP_for_index_approach_signoff": True,
}
out["WORLD_DENSE_METHOD"] = "EXHAUSTIVE_SHARDED_FP32_COMPUTE"
out["WORLD_DENSE_APPROXIMATE"] = "NO"
out["WORLD_DENSE_PARITY_REQUIRED"] = "YES"
out["WORLD_DENSE_METHOD_DESC"] = (
    "Exhaustive sharded fp16 inner-product retrieval, validated against fp32 IndexFlatIP exact retrieval. "
    "Shard-by-shard GPU fp16 inner products; local top-K per shard; exact merge to global top-K; "
    "no IVF/HNSW/ANN/corpus pruning. Global results return canonical world_node_ids. "
    "IndexFlatIP permitted only as an execution optimization for small corpora (same exhaustive definition). "
    "No monolithic fp32 FAISS index persisted for multi-million corpora; canonical assets = fp16 shards + world ID maps + checksums + retrieval manifest.")
out["WORLD_DENSE_PARITY_GATE"] = {
    "sample": "deterministic query sample per corpus",
    "report": ["top1_parity", "top10_overlap", "top50_overlap", "top100_overlap", "top200_overlap", "rank_agreement"],
    "required": "TOPK_PARITY == 1.0 for canonical downstream K; investigate fp16 ties before accepting",
}
out["remaining_work_to_retrieval_ready"] = [
    "dense index (after approach sign-off)",
    "qwen-kNN edge family from existing world dense shards (no re-encode)",
    "world C graph.pt = structural (+) NER (+) qwen-kNN per frozen C",
    "world C partitions (~100-node balanced) + stats",
    "clean<->world crosswalks for hotpot/musique/squad/webqsp/metaqa",
]
with open(f"{OUT}/W0_CORPUS_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("WROTE", f"{OUT}/W0_CORPUS_MANIFEST.json")
print(json.dumps({k: v["gates"] for k, v in out["datasets"].items()}, indent=2))
print("SAFE_TO_START_FULL_WORLD_EXPERIMENT =", out["SAFE_TO_START_FULL_WORLD_EXPERIMENT"])
