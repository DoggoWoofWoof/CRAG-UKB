"""FINAL LOCAL ARTIFACT AUDIT for G1. Verifies every artifact produced or needed during G1 is present locally,
non-empty, shape-consistent, and (for G1-uniquely-produced artifacts) hashed. Writes
results/GENERALIZATION/G1_LOCAL_ARTIFACT_AUDIT.json with EVERYTHING_REQUIRED_LOCAL / REMOTE_ONLY_ARTIFACTS /
MISSING_OR_CORRUPT_LOCAL / READY_FOR_OFFLINE_REBUILD.

Nothing is deleted. Remote copies are NOT touched. A required artifact that is missing locally but is known to
have been produced on a Modal workspace is reported under REMOTE_ONLY_ARTIFACTS.
"""
import os, sys, json, hashlib, time
import numpy as np
sys.path.insert(0, "scratchpad")
sys.path.insert(0, os.getcwd())   # so `import src...` resolves when run as `python scratchpad/_g1_local_audit.py`

T0 = time.time()
CHECKS = []            # every individual check {id, path, category, required, exists, size, shape, ok, issue}
MISSING = []           # required & (missing | zero | corrupt)
REMOTE_ONLY = []       # required & missing locally & known produced on a Modal workspace
SHA = {}               # sha256 of G1-uniquely-produced artifacts
NOTES = {}

# artifacts we KNOW were produced on a Modal workspace (so a local miss => remote-only, not just absent)
REMOTE_PRODUCED = {
    "results/L2/_heads/universal_offset_src_gteqwen.pt": "deepali",
    "results/L2/_heads/universal_mixture_src_gteqwen.pt": "deepali",
    "results/L2/_heads/_universal_heads_src_manifest.json": "deepali",
    "results/GENERALIZATION/_g1_build_squad_clean.json": "spanishorgay",
    "results/GENERALIZATION/_g1_eval_squad_clean.json": "spanishorgay",
}
for sp in ("train", "val"):
    for f in ("cand_ids.npy", "query_offsets.npy", "labels.npy", "dense_score.npy", "splade_scope_score.npy",
              "offset_score.npy", "mixture_score.npy", "relation_qwen_score.npy", "relation_mask.npy", "expert_meta.npz"):
        REMOTE_PRODUCED[f"data/l2_corpus/squad_clean/{sp}/{f}"] = "spanishorgay"

CORE = ["cand_ids.npy", "query_offsets.npy", "labels.npy", "dense_score.npy", "splade_scope_score.npy",
        "offset_score.npy", "mixture_score.npy", "relation_qwen_score.npy", "relation_mask.npy"]


def sha256(path, cap_gb=None):
    if cap_gb and os.path.getsize(path) > cap_gb * 1e9:
        return f"SKIPPED_>{cap_gb}GB"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rec(cid, path, category, required=True, want_shape=None, want_1d_len=None, hashit=False, hash_cap=None,
        extra_ok=None):
    """Record one artifact check. want_shape: (dim0_or_None, dim1) tuple to assert. want_1d_len: exact len."""
    c = {"id": cid, "path": path, "category": category, "required": required,
         "exists": os.path.exists(path), "size": None, "shape": None, "ok": False, "issue": None}
    if not c["exists"]:
        c["issue"] = "MISSING_LOCAL"
        if required:
            (REMOTE_ONLY if path in REMOTE_PRODUCED else MISSING).append(path)
        CHECKS.append(c); return c
    c["size"] = os.path.getsize(path)
    if c["size"] == 0:
        c["issue"] = "ZERO_SIZE"
        if required: MISSING.append(path)
        CHECKS.append(c); return c
    try:
        if path.endswith(".npy"):
            a = np.load(path, mmap_mode="r"); c["shape"] = list(a.shape)
            if want_shape is not None:
                d0, d1 = want_shape
                if (d1 is not None and (a.ndim != 2 or a.shape[1] != d1)) or (d0 is not None and a.shape[0] != d0):
                    c["issue"] = f"SHAPE_MISMATCH got={a.shape} want={want_shape}"
            if want_1d_len is not None and (a.ndim != 1 or a.shape[0] != want_1d_len):
                c["issue"] = c["issue"] or f"LEN_MISMATCH got={a.shape} want_len={want_1d_len}"
        elif path.endswith(".npz"):
            z = np.load(path, allow_pickle=True); c["shape"] = {k: list(np.asarray(z[k]).shape) for k in z.files}
        if extra_ok is not None:
            ok, msg = extra_ok(path)
            if not ok: c["issue"] = c["issue"] or msg
    except Exception as e:
        c["issue"] = f"LOAD_ERROR {type(e).__name__}: {e}"
    c["ok"] = c["issue"] is None
    if not c["ok"] and required:
        MISSING.append(path)
    if hashit and c["ok"]:
        SHA[path] = sha256(path, cap_gb=hash_cap)
    CHECKS.append(c); return c


def audit_substrate(ds, splits, category, produced_remote=False):
    """Verify one l2_corpus substrate (source or squad). Cross-checks offsets/labels/scores lengths vs cand_ids."""
    for sp in splits:
        d = f"data/l2_corpus/{ds}/{sp}"
        cid = rec(f"{ds}/{sp}/cand_ids", f"{d}/cand_ids.npy", category, hashit=produced_remote, hash_cap=8)
        M = None
        if cid["ok"]:
            M = int(np.load(f"{d}/cand_ids.npy", mmap_mode="r").shape[0])
        qo = rec(f"{ds}/{sp}/query_offsets", f"{d}/query_offsets.npy", category)
        nq = None
        if qo["ok"]:
            off = np.load(f"{d}/query_offsets.npy")
            nq = len(off) - 1
            if off[0] != 0 or (M is not None and off[-1] != M) or not np.all(np.diff(off) >= 0):
                qo["ok"] = False; qo["issue"] = f"OFFSETS_INCONSISTENT [0]={off[0]} [-1]={off[-1]} M={M}"; MISSING.append(qo["path"])
        for f in ["labels.npy", "dense_score.npy", "splade_scope_score.npy", "offset_score.npy",
                  "mixture_score.npy", "relation_qwen_score.npy", "relation_mask.npy"]:
            rec(f"{ds}/{sp}/{f[:-4]}", f"{d}/{f}", category, want_1d_len=M, hashit=produced_remote, hash_cap=8)
        rec(f"{ds}/{sp}/expert_meta", f"{d}/expert_meta.npz", category, hashit=produced_remote, hash_cap=8)
        NOTES.setdefault(f"{ds}_substrate", {})[sp] = {"n_candidates": M, "n_queries": nq}


def audit_target_inputs(ds, category, master_path, master_source_filter=None):
    """Verify build INPUTS for a target (substrate not yet built). Shape-cross-checks nodes/queries/topN/partition."""
    g = f"data/ukb_storage/{ds}/gte_qwen"
    qj = f"{g}/query_ids_all.json"
    N = Q = None
    # partition_map (topo C)
    pm_path = f"scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json"
    pmc = rec(f"{ds}/partition_map_C", pm_path, category)
    if pmc["ok"]:
        pm = json.load(open(pm_path)); N = len(pm); NOTES.setdefault(ds, {})["partition_map_N"] = N
    # nodes
    rec(f"{ds}/nodes", f"{g}/nodes.npy", category, want_shape=(N, 1536))
    # query ids + split_indices
    qi = rec(f"{ds}/query_ids_all", qj, category)
    if qi["ok"]:
        j = json.load(open(qj)); si = j.get("split_indices", {})
        Q = len(j.get("ids", []))
        ok = bool(si) and ("all" in si or all(k in si for k in ("train", "val")))
        qi["ok"] = qi["ok"] and ok
        qi["issue"] = None if ok else "split_indices_MISSING_OR_INCOMPLETE"
        if not ok: MISSING.append(qj)
        NOTES.setdefault(ds, {})["split_indices"] = {k: len(v) for k, v in si.items()}
        NOTES[ds]["ids_len"] = Q
    rec(f"{ds}/queries_all", f"{g}/queries_all.npy", category, want_shape=(Q, 1536))
    rec(f"{ds}/dense_top200", f"{g}/dense_top200_all.npy", category, want_shape=(Q, None))
    rec(f"{ds}/splade_top200", f"{g}/splade_top200_all.npy", category, want_shape=(Q, None))
    # splade doc matrix (rows == N)
    def _splade_ok(p):
        try:
            import pickle
            d = pickle.load(open(p, "rb")); m = d["matrix"]
            NOTES.setdefault(ds, {})["splade_doc_matrix_shape"] = list(m.shape)
            return (m.shape[0] == N if N else True), f"SPLADE_ROWS {m.shape[0]} != N {N}"
        except Exception as e:
            return False, f"SPLADE_LOAD_ERROR {e}"
    rec(f"{ds}/splade_doc_embs", f"{g.replace('/gte_qwen','')}/splade_doc_embs.pkl", category, extra_ok=_splade_ok)
    # graph + NER
    rec(f"{ds}/graph_pt", f"data/ukb_storage/{ds}/graph.pt", category, required=False)
    rec(f"{ds}/ner_edges", f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", category, required=False)
    # master_nodes (fallback-aware for metaqa)
    mrec = rec(f"{ds}/master_nodes", master_path, category)
    if mrec["ok"] and N is not None:
        from src.pipeline.standardizer import load_nodes
        nodes = load_nodes(master_path)
        docs = [n for n in nodes if n.metadata.get("type") != "question"]
        if master_source_filter is not None:
            docs = [n for n in docs if n.metadata.get("source") == master_source_filter]
        doc_ids = [n.node_id for n in docs]
        pm = json.load(open(pm_path)) if pmc["ok"] else {}
        set_eq = set(doc_ids) == set(pm.keys())
        info = {"master_path": master_path, "source_filter": master_source_filter,
                "doc_nodes": len(doc_ids), "partition_map_N": N, "count_match": len(doc_ids) == N,
                "node_SET_equals_partition_keys": set_eq}
        NOTES.setdefault(ds, {})["master_verification"] = info
        if not (len(doc_ids) == N and set_eq):
            mrec["ok"] = False; mrec["issue"] = f"MASTER_NODESET_MISMATCH docs={len(doc_ids)} N={N} set_eq={set_eq}"
            MISSING.append(master_path)


def audit_file(cid, path, category, required=True, hashit=False, hash_cap=None, loader=None):
    c = rec(cid, path, category, required=required, hashit=hashit, hash_cap=hash_cap)
    if c["ok"] and loader is not None:
        try:
            loader(path)
        except Exception as e:
            c["ok"] = False; c["issue"] = f"CONTENT_ERROR {type(e).__name__}: {e}"
            if required: MISSING.append(path)
    return c


# ---------------- SOURCE ----------------
audit_substrate("2wiki_clean", ("train", "val"), "SOURCE")
audit_substrate("musique_clean", ("train", "val"), "SOURCE")
import joblib
audit_file("frozen/C11_models", "results/L2/_ctrl/C11_models.joblib", "SOURCE", hashit=True,
           loader=lambda p: (lambda d: (d["C11a"], d["emean"], d["estd"]))(joblib.load(p)))
audit_file("frozen/C8c_reranker", "results/L2/_ctrl/C8c_reranker_fulltrain.joblib", "SOURCE", hashit=True,
           loader=lambda p: joblib.load(p))
for f in ("base_full.joblib", "c8c.joblib", "C11_models.joblib", "source_val_stats.json"):
    audit_file(f"backbone/{f}", f"results/GENERALIZATION/_g1_backbone/{f}", "SOURCE", hashit=True,
               loader=(json.load and (lambda p: json.load(open(p)))) if f.endswith(".json") else (lambda p: joblib.load(p)))
# base_full structural check
bf = "results/GENERALIZATION/_g1_backbone/base_full.joblib"
if os.path.exists(bf):
    try:
        t = joblib.load(bf); assert isinstance(t, (tuple, list)) and len(t) >= 3
        NOTES["backbone_base_full"] = {"tuple_len": len(t), "clf_in_features": int(t[0].in_features), "clf_out": int(t[0].out_features)}
    except Exception as e:
        NOTES["backbone_base_full"] = {"error": str(e)}

# ---------------- UNIVERSAL ----------------
import torch
def _head_ok(expect_out):
    def f(p):
        sd = torch.load(p, map_location="cpu")
        w0 = sd["net.0.weight"]; w2 = sd["net.2.weight"]
        assert w0.shape[1] == 1536, f"in!=1536 {w0.shape}"
        assert w2.shape[0] == expect_out, f"out!={expect_out} {w2.shape}"
        NOTES.setdefault("universal_heads", {})[os.path.basename(p)] = {"net.0.weight": list(w0.shape), "net.2.weight": list(w2.shape)}
    return f
audit_file("universal/offset", "results/L2/_heads/universal_offset_src_gteqwen.pt", "UNIVERSAL", hashit=True, loader=_head_ok(1536))
audit_file("universal/mixture", "results/L2/_heads/universal_mixture_src_gteqwen.pt", "UNIVERSAL", hashit=True, loader=_head_ok(8 * 1536))
audit_file("universal/manifest", "results/L2/_heads/_universal_heads_src_manifest.json", "UNIVERSAL", hashit=True, loader=lambda p: json.load(open(p)))

# ---------------- SQUAD ----------------
audit_substrate("squad_clean", ("train", "val"), "SQUAD", produced_remote=True)
audit_file("squad/build_manifest", "results/GENERALIZATION/_g1_build_squad_clean.json", "SQUAD", hashit=True, loader=lambda p: json.load(open(p)))
audit_file("squad/eval_json", "results/GENERALIZATION/_g1_eval_squad_clean.json", "SQUAD", hashit=True,
           loader=lambda p: (lambda d: (d["MAIN_TABLE"], d["INTERPRETATION"]))(json.load(open(p))))

# ---------------- TARGET INPUTS ----------------
audit_target_inputs("hotpotqa_clean", "HOTPOT", "data/processed/master_nodes_hotpotqa_clean.json")
audit_target_inputs("webqsp", "WEBQSP", "data/processed/master_nodes_webqsp.json")
audit_target_inputs("metaqa", "METAQA", "data/processed/master_nodes.json", master_source_filter="metaqa")

# ---------------- squad manifest integrity vs downloaded files ----------------
sm = {}
mpath = "results/GENERALIZATION/_g1_build_squad_clean.json"
if os.path.exists(mpath):
    try:
        man = json.load(open(mpath))
        sm["manifest_top_keys"] = list(man.keys())
        # independent integrity: the 10 core files per split must exist locally
        integ = {}
        for sp in ("train", "val"):
            d = f"data/l2_corpus/squad_clean/{sp}"
            present = {f: (os.path.exists(f"{d}/{f}") and os.path.getsize(f"{d}/{f}") > 0)
                       for f in CORE + ["expert_meta.npz"]}
            integ[sp] = {"all_present_nonzero": all(present.values()),
                         "missing": [f for f, ok in present.items() if not ok]}
        sm["independent_core_file_integrity"] = integ
        # echo any integrity/finalize section the manifest itself carries
        for k in ("integrity", "finalize", "E_meta", "A_scope", "files"):
            if k in man: sm[f"manifest_{k}"] = man[k]
    except Exception as e:
        sm["error"] = str(e)
NOTES["squad_manifest_vs_files"] = sm

# ---------------- verdict ----------------
MISSING_U = sorted(set(MISSING)); REMOTE_U = sorted(set(REMOTE_ONLY))
everything = (len(MISSING_U) == 0 and len(REMOTE_U) == 0)
# offline rebuild = source substrate + heads + backbone + all 4 target INPUTS present & valid
rebuild_cats = {"SOURCE", "UNIVERSAL", "HOTPOT", "WEBQSP", "METAQA"}
rebuild_ok = all(c["ok"] for c in CHECKS if c["required"] and c["category"] in rebuild_cats) and \
             all(c["ok"] for c in CHECKS if c["required"] and c["category"] == "SQUAD")
out = {
    "phase": "G1 FINAL LOCAL ARTIFACT AUDIT",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "elapsed_sec": round(time.time() - T0, 1),
    "EVERYTHING_REQUIRED_LOCAL": "YES" if everything else "NO",
    "READY_FOR_OFFLINE_REBUILD": "YES" if rebuild_ok else "NO",
    "REMOTE_ONLY_ARTIFACTS": REMOTE_U,
    "MISSING_OR_CORRUPT_LOCAL": MISSING_U,
    "n_checks": len(CHECKS), "n_required": sum(c["required"] for c in CHECKS),
    "n_ok": sum(c["ok"] for c in CHECKS),
    "SHA256_uniquely_produced": SHA,
    "NOTES": NOTES,
    "CHECKS": CHECKS,
    "sha256_scope": "G1-uniquely-produced artifacts (heads, backbone bundle, frozen C8c/C11, squad substrate arrays+manifest+eval); "
                    "multi-GB pre-existing encoder outputs (target nodes/queries) verified by existence+shape, not hashed",
    "remote_copies_disposable": "ONLY IF EVERYTHING_REQUIRED_LOCAL==YES (nothing deleted here)",
}
os.makedirs("results/GENERALIZATION", exist_ok=True)
json.dump(out, open("results/GENERALIZATION/G1_LOCAL_ARTIFACT_AUDIT.json", "w"), indent=1, default=str)
print("=" * 70)
print(f"EVERYTHING_REQUIRED_LOCAL = {out['EVERYTHING_REQUIRED_LOCAL']}")
print(f"READY_FOR_OFFLINE_REBUILD = {out['READY_FOR_OFFLINE_REBUILD']}")
print(f"REMOTE_ONLY_ARTIFACTS ({len(REMOTE_U)}): {REMOTE_U}")
print(f"MISSING_OR_CORRUPT_LOCAL ({len(MISSING_U)}): {MISSING_U}")
print(f"checks={len(CHECKS)} required={out['n_required']} ok={out['n_ok']} sha256={len(SHA)}")
print("WROTE results/GENERALIZATION/G1_LOCAL_ARTIFACT_AUDIT.json")
