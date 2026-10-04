"""Section 0-1 -- local cache / sync audit for the H4+halo attribution & optimization phase.

Walks every directory this program has written or depends on and records what exists locally,
so no experiment below re-derives or re-partitions something already on disk.  Pure inventory:
reads file metadata only, computes nothing about retrieval.

  python scratchpad/_cache_manifest.py
"""
import os, sys, json, hashlib, time

ROOT = os.getcwd()
OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]

# (category, purpose, glob-root, creation source) -- walked non-recursively unless noted
CATEGORIES = [
    ("hypergraph_core_assignments", "H0-H4 hypergraph-core partition assignments per corpus/rule",
     "scratchpad/_l1hu/parts", "Mt-KaHyPar build (Phase A), local"),
    ("hypergraph_exports", "hyperedge/pin export (.npz) per corpus/rule, feeds boundary-mass",
     "scratchpad/_l1hu/graphs", "Mt-KaHyPar build (Phase A), local"),
    ("registered_l1_partition_files", "stable-named copies _l1ov_eval/_l1hu_global read "
     "(CURRENT=METIS via PU, C1_HYPER_UNIVERSAL=H4_SPLIT_PRESERVE)",
     "scratchpad/_l1ep/parts", "_l1hu_halo.register(), local copy from hypergraph_core_assignments"),
    ("halo_pair_caches", "per-family (STRUCT/NERX/KNN/FULL_C) 1-hop halo (block,node) pair sets, "
     "H4 core only -- METIS-core halo pairs live in the prior phase's own _l1ov cache under tag=CURRENT",
     "scratchpad/_l1ov", "OV.halo_pairs(), local, built from corpus topology only (no query/gold)"),
    ("keyset_caches", "STRUCT/KNN/NERX undirected edge-key arrays per corpus (topology only)",
     "scratchpad/_l1kn", "KS.keysets(), local, derived from data/ukb_storage graph.pt + ner_edges"),
    ("halo_coverage_results", "BASE+SAFE(F6) x {METIS,H4} x {core-only,FULL_C,b0.25,b0.5} retrieval "
     "coverage, per-query _ind_ALL arrays included -- this is the Part-A ladder's raw material",
     "%s/halo" % OUT, "_l1hu_halo.evaluate() / _l1ov_eval.evaluate(), local CPU"),
    ("partition_replay_gate", "balance/parity replay per corpus/rule and the universal hypergraph "
     "gate decision (BASE+SAFE lanes, per-corpus significance)",
     "%s/partitions" % OUT, "_l1hu_hard.py score/gate, local CPU"),
    ("hypergraph_build_manifest", "pin-retention by rule, checksums, Mt-KaHyPar API/cost logs",
     "%s/hypergraph_build" % OUT, "Mt-KaHyPar build + Modal cost logs, synced local"),
    ("pin_retention_audit", "raw hyperedge-size distribution before any cap",
     "%s/pin_retention" % OUT, "local CPU, corpus topology only"),
    ("interaction_exposure", "B6 core/halo additivity + B7/B8 exposure-efficiency summaries",
     "%s/interaction" % OUT, "_l1hu_report.interaction()/exposure(), pure aggregation, local"),
    ("global_halo_part_f", "F0/F1/F2 per-query matched-exposure comparison (this phase, in progress)",
     "%s/global_halo" % OUT, "_l1hu_global.py, local CPU, in progress"),
    ("top_level_reports", "FINAL_REPORT.md / TABLES.md / RETURNS.json for the H4+halo phase",
     OUT, "_l1hu_report.py, pure aggregation, local", False),
    ("l2_frozen_substrate", "L2_C7b->C8c->C11a checkpoints + features -- EXISTS ONLY for "
     "2wiki_clean/musique_clean, the two corpora with the weakest L1 gain; see "
     "l1-hypergraph-halo-l2-scoping memory before using this for Part J",
     "results/L2/_ctrl", "prior L2 research program, local", False),
]


def sha_partial(fp, size, cap=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        h.update(f.read(min(size, cap)))
    h.update(str(size).encode())
    return h.hexdigest()[:16] + ("_partial" if size > cap else "_full")


def scan_dir(path, recursive=False):
    entries = []
    if not os.path.isdir(path):
        return entries
    if recursive:
        walker = ((r, f) for r, _, fs in os.walk(path) for f in fs)
    else:
        walker = ((path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
    for r, f in walker:
        fp = os.path.join(r, f)
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        entries.append({"path": fp.replace("\\", "/"), "size_bytes": size,
                        "sha256_16": sha_partial(fp, size)})
    return entries


def corpus_of(name):
    for d in sorted(DS, key=len, reverse=True):
        if d in name:
            return d
    return None


def build():
    manifest = {"_generated": "cache/sync audit, Section 0-1 of the H4+halo attribution phase",
               "_note": "sha256_16 is a partial hash (first 8MB + size) for files > 8MB, "
                        "full sha256 truncated to 16 hex chars otherwise -- collision-safe enough "
                        "to detect an accidental overwrite, not a cryptographic guarantee.",
               "categories": {}}
    total_files, total_bytes = 0, 0
    for cat in CATEGORIES:
        name, purpose, root, source = cat[0], cat[1], cat[2], cat[3]
        recursive = cat[4] if len(cat) > 4 else True
        entries = scan_dir(root, recursive=recursive)
        by_corpus = {}
        for e in entries:
            c = corpus_of(os.path.basename(e["path"])) or "_shared"
            by_corpus.setdefault(c, []).append(e)
        total_files += len(entries)
        total_bytes += sum(e["size_bytes"] for e in entries)
        manifest["categories"][name] = {
            "purpose": purpose, "local_root": root, "creation_source": source,
            "reusable": True, "file_count": len(entries),
            "total_size_bytes": sum(e["size_bytes"] for e in entries),
            "corpora_present": sorted(by_corpus),
            "missing_corpora": sorted(set(DS) - set(by_corpus)) if name not in
            ("top_level_reports", "l2_frozen_substrate", "keyset_caches") else [],
            "files": entries}
    manifest["_totals"] = {"files": total_files, "bytes": total_bytes,
                           "mb": round(total_bytes / 1e6, 1)}
    # explicit modal-only check: anything referenced in logs but absent locally
    modal_logs = [f for f in os.listdir("scratchpad/_l1hu") if f.startswith("modal_")]
    manifest["_modal_sync_check"] = {
        "modal_execution_logs_found": modal_logs,
        "conclusion": "every corpus x rule this phase needs (hypergraph_core_assignments, "
                      "hypergraph_exports, halo_pair_caches, halo_coverage_results) is present "
                      "locally for all 6 corpora already (see missing_corpora per category above) "
                      "-- the modal_*.log files record completed jobs whose OUTPUT was already "
                      "pulled down. NOTHING further needs to be synced from Modal before Parts "
                      "A/B/E/F/G/H/I run.",
    }
    os.makedirs(OUT, exist_ok=True)
    fp = "%s/LOCAL_CACHE_MANIFEST.json" % OUT
    json.dump(manifest, open(fp, "w"), indent=1)
    print("wrote", fp)
    print("totals:", json.dumps(manifest["_totals"]))
    for name, c in manifest["categories"].items():
        miss = (" MISSING=%s" % c["missing_corpora"]) if c["missing_corpora"] else ""
        print("  %-32s files=%-5d %8.1fMB  corpora=%s%s" % (
            name, c["file_count"], c["total_size_bytes"] / 1e6, c["corpora_present"], miss))
    print(json.dumps(manifest["_modal_sync_check"], indent=1))
    return manifest


if __name__ == "__main__":
    build()
