# -*- coding: utf-8 -*-
"""Build the ONE common manifest over all six canonical datasets, in UKB shape.

WHY THIS FILE EXISTS
--------------------
The canonical_v1 package is correct but its metadata is scattered across three frozen
records and eight per-dataset sidecars, and the six datasets do not carry the same
sidecars: webqsp has no build_info.json, no dataset_manifest.json, no query lanes and no
legacy id map, while the five text corpora do. A consumer therefore cannot ask "what does
dataset X have" without knowing which of three freezes to look in and which sidecar shape
that particular dataset happens to use.

UKB logic (src/pipeline/ukb_results.py) answers exactly this for RESULTS: one index keyed
dataset -> level -> artifact -> {path + headline metrics}, so the store can be queried
directly instead of walked. This module applies the same logic to PREPROCESSING:

    dataset -> slot -> {present, path(s), bytes, sha256, headline counts}

Every dataset carries every slot. A dataset that genuinely lacks one carries it as
{"present": false, "reason": ...} rather than omitting the key, because absence is a real
property of the corpus and a missing key is indistinguishable from an unbuilt slot.

WHAT THIS FILE IS NOT
---------------------
It is not a new authority. Every hash here is COPIED from the frozen record that pinned it
(LOCKED_5_OF_5 / LOCKED_6_OF_6 / LOCKED_6_OF_6_FAMILY_COMPLETION_V1, or _V2 once it exists:
V2 supersedes the stale frozen musique kNN by declaration), never recomputed, so
this manifest can never silently disagree with a freeze -- verify_manifest.py is the tool
that recomputes and compares. Where a fact exists in no frozen record (the query lanes, the
legacy id maps, the webqsp v1 source tables) the manifest says so with pinned_by = null,
which is the honest statement that those files are in use but were never hash-pinned.
"""
import hashlib
import io
import json
import os
import sys
import time

ROOT = "data/final_canonical"
DS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]
FAMILIES = ["structural", "ner", "knn"]
SCHEMA_VERSION = 1

RECORDS = {
    "five": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
    "six": "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
    "family": "LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json",
}
# the governing family-completion record is the newest one on disk: V2 BUILDS_ON V1 and carries
# every V1 dataset block verbatim, adding only the musique kNN supersession
if os.path.exists(ROOT + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"):
    RECORDS["family"] = "LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"
FAMILY_RECORD_NAME = RECORDS["family"][:-5]


def rj(path):
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def fstat(path):
    """Size on disk, or None. Never hashes -- hashing 5.3 GB per dataset is the verifier's job."""
    p = path.replace("\\", "/")
    if not os.path.exists(p):
        return {"path": p, "exists": False, "bytes": None}
    return {"path": p, "exists": True, "bytes": os.path.getsize(p)}


def absent(reason):
    return {"present": False, "reason": reason}


# ---------------------------------------------------------------------------
# the declared-artifact index: path -> (bytes, sha256, which record pinned it)
# ---------------------------------------------------------------------------
def declared_index(five, six, fam):
    d = {}

    def put(p, b, s, src):
        d[p.replace("\\", "/")] = {"bytes": b, "sha256": s, "pinned_by": src}

    for ds, v in five["datasets"].items():
        for a in v["artifacts"]:
            put(a["path"], a["bytes"], a["sha256"], "LOCKED_5_OF_5")
    for _, v in five["package_level_artifacts"].items():
        put(v["path"], v["bytes"], v["sha256"], "LOCKED_5_OF_5")
    for a in six["webqsp"]["artifacts"]:
        put(a["path"], a["bytes"], a["sha256"], "LOCKED_6_OF_6")
    # a family added by V1 stays attributed to V1 even when V2 governs: V2 carries V1's
    # entries verbatim, so the ORIGINAL pin is the honest answer to "which freeze pinned it";
    # only what V2 adds on top (the musique kNN supersession) is attributed to V2
    chain = [fam]
    base = (fam.get("BUILDS_ON") or {}).get("path")
    if fam.get("RECORD") != "LOCKED_6_OF_6_FAMILY_COMPLETION_V1" and base and os.path.exists(base):
        v1 = rj(base)
        if v1 and v1.get("RECORD") == "LOCKED_6_OF_6_FAMILY_COMPLETION_V1":
            chain = [v1, fam]
    for rec in chain:
        for ds, v in rec["datasets"].items():
            for _f, a in (v.get("added") or {}).items():
                key = (ROOT + "/" + a["file"]).replace("\\", "/")
                if key in d:
                    continue
                put(ROOT + "/" + a["file"], a["npz_bytes"], a["npz_sha256"],
                    rec.get("RECORD") or FAMILY_RECORD_NAME)
    return d


def slot(decl, path, **extra):
    """One artifact slot: on-disk truth plus the freeze that pinned it, if any."""
    p = path.replace("\\", "/")
    out = fstat(p)
    dd = decl.get(p)
    out["pinned_by"] = dd["pinned_by"] if dd else None
    out["declared_bytes"] = dd["bytes"] if dd else None
    out["sha256"] = dd["sha256"] if dd else None
    if dd and out["exists"] and dd["bytes"] is not None:
        out["bytes_match_declared"] = (out["bytes"] == dd["bytes"])
    else:
        out["bytes_match_declared"] = None
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------
def s_identity(ds, five, six, fam, dm, bi, sc):
    if ds == "webqsp":
        w = six["webqsp"]
        return {
            "dataset": ds,
            "canonical_version": "canonical_v1",
            "corpus_type": w["corpus_type"],
            "n_nodes": w["n_nodes"],
            "n_queries": w["n_queries"],
            "node_id_rule": "webqsp:n<position> <-> nodes.jsonl line number <-> source_rog_endpoint",
            "source_of_record": w["provenance"],
            "phase_c_tree": None,
            "phase_c_tree_note": ("NONE deliberately: data/canonical/webqsp is a DIFFERENT node "
                                  "universe (1,316,466 nodes) and shares no ids with this corpus. "
                                  "Built from data/final_canonical/webqsp/v1/*.parquet instead."),
        }
    f = five["datasets"][ds]
    return {
        "dataset": ds,
        "canonical_version": (bi or {}).get("dataset_version", "canonical_v1"),
        "corpus_type": "TEXT_CORPUS",
        "n_nodes": f["n_nodes"],
        "n_queries": f["n_queries"],
        "node_id_rule": (dm or {}).get("node_id_scheme"),
        "source_of_record": sorted((bi or {}).get("source_files", {})) or None,
        "phase_c_tree": f["phase_c_tree_for_documents"],
        "phase_c_tree_note": None,
    }


def s_documents(ds, decl, five, six, bi):
    out = {"present": True,
           "nodes_jsonl": slot(decl, "%s/%s/nodes.jsonl" % (ROOT, ds)),
           "corpus_hash": (bi or {}).get("CORPUS_HASH"),
           "node_order_hash": (bi or {}).get("NODE_ORDER_HASH"),
           "n_nodes": (six["webqsp"]["n_nodes"] if ds == "webqsp"
                       else five["datasets"][ds]["n_nodes"])}
    if ds == "webqsp":
        out["corpus_hash_note"] = ("webqsp was not built by build.py and carries no build_info.json, "
                                   "so it has no CORPUS_HASH/NODE_ORDER_HASH. Its equivalent is "
                                   "CANONICAL_V1_BUILD.json + CANONICAL_V1_VERIFICATION.json.")
        out["equivalent_records"] = [
            slot(decl, "%s/webqsp/CANONICAL_V1_BUILD.json" % ROOT),
            slot(decl, "%s/webqsp/CANONICAL_V1_VERIFICATION.json" % ROOT),
        ]
    return out


def s_queries(ds, decl, five, six, dm):
    base = "%s/%s/queries" % (ROOT, ds)
    splits = {}
    for sp in ("train", "dev", "validation", "test"):
        p = "%s/%s.jsonl" % (base, sp)
        if os.path.exists(p):
            splits[sp] = slot(decl, p)
    lanes = {}
    ldir = base + "/lanes"
    if os.path.isdir(ldir):
        for fn in sorted(os.listdir(ldir)):
            lanes[fn] = slot(decl, ldir + "/" + fn)
    ev = None
    for cand in ("eval_2000.jsonl", "eval_1998.jsonl"):
        p = "%s/%s/%s" % (ROOT, ds, cand)
        if os.path.exists(p):
            ev = slot(decl, p)
            break
    return {
        "present": True,
        "n_queries": (six["webqsp"]["n_queries"] if ds == "webqsp"
                      else five["datasets"][ds]["n_queries"]),
        "splits": splits,
        "split_counts": (dm or {}).get("query_counts"),
        "queries_without_gold_by_split": (None if ds == "webqsp" else
                                          five["datasets"][ds]["queries_without_gold_by_split"]),
        "unlabelled_splits": (None if ds == "webqsp" else
                              five["datasets"][ds]["unlabelled_splits"]),
        "lanes": lanes or absent("webqsp has no query lanes: lanes are a property of the "
                                 "legacy-continuity design for the five text corpora, and webqsp "
                                 "has no legacy substrate to stay continuous with."),
        "eval_subset": ev or absent("no fixed eval subset file was cut for this dataset"),
        "query_order": (six["webqsp"]["query_order"] if ds == "webqsp"
                        else "official splits as written by the builder; "
                             "queries/pointer_index/query_ids.json is the authoritative order"),
    }


def s_graph(ds, decl, fam):
    fe = fam["datasets"][ds]
    fams = {}
    for f in FAMILIES:
        src = fe["family_source"].get(f)
        if src is None:
            fams[f] = absent("this corpus supports no %s family; no edge was invented to make "
                             "the six uniform" % f)
            continue
        rel = "graph/%s.npz" % f if src == "frozen" else "graph2/%s.npz" % f
        e = slot(decl, "%s/%s/%s" % (ROOT, ds, rel))
        e["present"] = True
        e["source"] = src
        e["manifest"] = ("%s/%s/graph/GRAPH_MANIFEST.json" % (ROOT, ds) if src == "frozen"
                         else "%s/%s/graph2/GRAPH_MANIFEST.json" % (ROOT, ds))
        e["coverage"] = fe["coverage"].get(f)
        sup = (fe.get("superseded_frozen_families") or {}).get(f)
        if sup:
            e["supersedes_frozen"] = {
                "frozen_file": "%s/%s/%s" % (ROOT, ds, sup["frozen_file"].split("/", 1)[1]),
                "frozen_sha256": sup["frozen_sha256"],
                "still_pinned_by": sup["still_pinned_by"],
                "frozen_bytes_untouched": sup["frozen_bytes_untouched"],
                "why": sup["why_superseded"],
                "declared_in": "%s/%s" % (ROOT, sup["declared_in"]),
            }
        fams[f] = e
    out = {
        "present": True,
        "families_present": fe["families_present"],
        "families_absent": fe["families_absent"],
        "families": fams,
        "endpoint_space": ("edge endpoints are POSITIONS: line i of nodes.jsonl is canonical "
                           "position i is pointer row i, so an endpoint indexes an embedding "
                           "with no join"),
        "read_via": ("data/final_canonical/pointer_resolver_v2.py -- it unions graph/ and "
                     "graph2/; frozen wins for a family both declare UNLESS the graph2 manifest "
                     "carries an explicit supersedes.<family> block"),
        "superseded_frozen_families": sorted(fe.get("superseded_frozen_families") or {}),
        "frozen_manifest": slot(decl, "%s/%s/graph/GRAPH_MANIFEST.json" % (ROOT, ds)),
        "added_manifest": (slot(decl, "%s/%s/graph2/GRAPH_MANIFEST.json" % (ROOT, ds))
                           if os.path.exists("%s/%s/graph2/GRAPH_MANIFEST.json" % (ROOT, ds))
                           else absent("every family this dataset has was already in the "
                                       "LOCKED_5_OF_5 freeze; nothing was added later")),
    }
    for k in ("frozen_knn_self_consistency", "new_knn_self_consistency",
              "knn_pipeline_parity_vs_frozen"):
        if k in fe:
            out[k] = fe[k]
    return out


def s_encodings(ds, decl, pi):
    e = pi["datasets"].get(ds, {})
    chans = {}
    for ch in ("dense", "splade"):
        c = e.get(ch)
        if not c:
            chans[ch] = absent("no %s channel recorded in POINTER_INDEX.json" % ch)
            continue
        chans[ch] = {
            "present": True,
            "pointer_file": slot(decl, "%s/%s/pointer_index/%s.npz" % (ROOT, ds, ch)),
            "query_pointer_file": slot(decl, "%s/%s/queries/pointer_index/%s.npz" % (ROOT, ds, ch)),
            "n_pointers": c.get("n_pointers"),
            "distinct_source_rows": c.get("distinct_source_rows"),
            "shard_size": c.get("phase_c_shard_size"),
            "repaired_pointers": c.get("repaired_pointers"),
            "stores": c.get("stores"),
        }
    return {
        "present": True,
        "channels": chans,
        "n_canonical_nodes": e.get("n_canonical_nodes"),
        "reuse_source_mode": e.get("reuse_source_mode"),
        "shard_hash_record": ("data/final_canonical/WEBQSP_ENCODING_SHARD_HASHES.json"
                              if ds == "webqsp"
                              else "data/final_canonical/ENCODING_SHARD_HASHES.json"),
        "NOT_SELF_CONTAINED": ("the vectors are NOT inside data/final_canonical/. The pointer "
                               "index references data/canonical/ and both trees must travel "
                               "together."),
        "DAMAGED_SHARDS_WARNING": ("reading data/canonical/<tree>/encodings/dense/ shards "
                                   "directly still returns zeros for 142,633 repaired rows. "
                                   "Always read through pointer_resolver.CanonicalEmbeddings."),
    }


def s_ids(ds, decl, ib):
    e = (ib.get("datasets") or {}).get(ds, {})
    lm = "%s/%s/node_id_map_legacy.json" % (ROOT, ds)
    return {
        "present": True,
        "rule": e.get("rule"),
        "canonical_nodes": e.get("canonical_nodes"),
        "node_id_collisions": e.get("node_id_collisions"),
        "rule_violations": e.get("rule_violations"),
        "gold_refs": e.get("gold_refs"),
        "gold_unresolved": e.get("gold_unresolved"),
        "gold_resolution_pct": e.get("gold_resolution_pct"),
        "PASS": e.get("PASS"),
        "PASS_criterion": e.get("PASS_criterion"),
        "positive_control": e.get("POSITIVE_CONTROL"),
        "gold_comparability_warning": e.get("GOLD_IS_NOT_COMPARABLE_TO_THE_OTHER_FIVE"),
        "query_ids": slot(decl, "%s/%s/queries/pointer_index/query_ids.json" % (ROOT, ds)),
        "legacy_node_id_map": (slot(decl, lm) if os.path.exists(lm) else
                               absent("webqsp has no legacy substrate to map back to: it did not "
                                      "exist before canonical_v1")),
    }


def s_provenance(ds, decl, bi):
    if ds == "webqsp":
        return {
            "present": True,
            "builder": "scratchpad/final_canonical_build/build_kb.py (+ webqsp_v1/*)",
            "builder_sha256": None,
            "builder_command": None,
            "git_commit": None,
            "seconds": None,
            "builder_sha256_note": ("webqsp carries no build_info.json, so its builder was never "
                                    "sha-pinned. This is a real provenance gap relative to the "
                                    "five; the compensating records are CANONICAL_V1_BUILD.json, "
                                    "CANONICAL_V1_VERIFICATION.json and the WEBQSP_*_AUDIT.md set."),
            "source_files": {"data/final_canonical/webqsp/v1/*.parquet":
                             "the V1-resolved RoG WebQSP+CWQ graph -- NEVER 'full Freebase'"},
            "source_contract": slot(decl, "%s/webqsp/SOURCE_CONTRACT.json" % ROOT),
            "dataset_manifest": absent("webqsp was not built by build.py and has no "
                                       "dataset_manifest.json; ENCODING_ASSEMBLY.json and "
                                       "CANONICAL_V1_BUILD.json carry the equivalent fields"),
            "build_info": absent("see builder_sha256_note"),
            "integrity_report": absent("replaced by CANONICAL_V1_VERIFICATION.json"),
        }
    return {
        "present": True,
        "builder": (bi or {}).get("builder"),
        "builder_sha256": (bi or {}).get("builder_sha256"),
        "builder_command": (bi or {}).get("command"),
        "git_commit": (bi or {}).get("git_commit"),
        "seconds": (bi or {}).get("seconds"),
        "builder_sha256_note": None,
        "source_files": (bi or {}).get("source_files"),
        "source_contract": slot(decl, "%s/%s/SOURCE_CONTRACT.json" % (ROOT, ds)),
        "dataset_manifest": slot(decl, "%s/%s/dataset_manifest.json" % (ROOT, ds)),
        "build_info": slot(decl, "%s/%s/build_info.json" % (ROOT, ds)),
        "integrity_report": slot(decl, "%s/%s/integrity_report.json" % (ROOT, ds)),
    }


CACHE_ABSENT_REASON = ("no retrieval cache was built for this dataset. The cache is a "
                       "convenience over the substrate, not part of it -- it can be rebuilt "
                       "from the pointer index at any time.")


def s_retrieval_cache(ds, rc, decl, heavy):
    """The deep-K cache slot, with its provenance class stated rather than smoothed over.

    metaqa/musique/squad caches are pinned by LOCKED_5_OF_5, so their hashes are freeze-copied
    like every other slot. hotpotqa/2wiki/webqsp caches postdate every freeze: nothing pins
    them, their hashes come from RETRIEVAL_CACHE.json (measured from disk), and the slot says
    so in a `provenance` block -- the shape manifest_cache_slots.py first wrote on 2026-09-10,
    reproduced here so a regeneration of this manifest cannot silently drop the distinction.
    """
    c = (rc.get("caches") or {}).get(ds)
    if not c:
        return absent(CACHE_ABSENT_REASON)
    out = {"present": True, "channels": {}}
    pinned = []
    for ch, v in c.items():
        if isinstance(v, dict):
            out["channels"][ch] = {"file": v.get("file"), "bytes": v.get("bytes"),
                                   "sha256": v.get("sha256"), "K": v.get("K"),
                                   "n_queries": v.get("n_queries")}
            dd = decl.get((v.get("file") or "").replace("\\", "/"))
            if dd:
                pinned.append(dd["pinned_by"])
    if not pinned:
        res = (heavy or {}).get("results", {}) or {}
        out["provenance"] = {
            "NOT_PINNED_BY_ANY_FREEZE": True,
            "why": "this manifest copies hashes from the freeze named in pinned_by and "
                   "never recomputes them, so that it cannot silently disagree with a "
                   "freeze. These three caches postdate every freeze, so there is "
                   "nothing to copy and the hashes here were computed from disk. That "
                   "is a weaker guarantee than the other three slots carry, and it is "
                   "recorded rather than hidden.",
            "measured_by": "data/final_canonical/RETRIEVAL_CACHE.json",
            "verified_by": "data/final_canonical/HEAVY_CACHE_VERIFICATION.json",
            "authority_to_add_after_a_freeze":
                "data/final_canonical/FROZEN_CAVEAT_SUPERSESSION.json, which supersedes "
                "the frozen caveats asserting these caches do not exist, by new record "
                "rather than by editing them",
            "verification_verdicts": {
                m: (res.get("%s.%s" % (ds, m), {}) or {}).get("verdict")
                for m in ("dense", "splade")}}
        out["superseded_slot_VERBATIM"] = absent(CACHE_ABSENT_REASON)
        out["which_half_of_that_reason_still_holds"] = (
            "'The cache is a convenience over the substrate, not part of it -- it can "
            "be rebuilt from the pointer index at any time' remains TRUE and is the "
            "reason nothing in the package depends on these files. Only 'no retrieval "
            "cache was built for this dataset' is superseded.")
    return out


def s_caveats(ds, five, six, fam):
    out = []
    fe = fam["datasets"][ds]
    fsc = fe.get("frozen_knn_self_consistency") or {}
    if fsc.get("n_gt_1e-2"):
        sup = (fe.get("superseded_frozen_families") or {}).get("knn")
        nsc = fe.get("new_knn_self_consistency") or {}
        if sup:
            out.append("FROZEN kNN IS STALE AND SUPERSEDED: %d of %d frozen edges (%.2f%%) cannot "
                       "be reproduced from this dataset's own vectors (max abs err %.3g). The "
                       "frozen file stays hash-pinned in LOCKED_5_OF_5 and byte-identical, but "
                       "graph2/knn.npz supersedes it by declaration (%s): rebuilt with the "
                       "parity-validated pipeline over the pointer-resolved vectors and audited "
                       "every edge (max abs err %.3g, %d edges > 1e-2). Read the family through "
                       "pointer_resolver_v2; pointer_resolver.CanonicalGraph still returns the "
                       "frozen copy."
                       % (fsc["n_gt_1e-2"], fsc["n_edges"],
                          100.0 * fsc["n_gt_1e-2"] / fsc["n_edges"], fsc["max_abs_err"],
                          FAMILY_RECORD_NAME, nsc.get("max_abs_err", float("nan")),
                          nsc.get("n_gt_1e-2", -1)))
        else:
            out.append("FROZEN kNN IS STALE: %d of %d edges (%.2f%%) cannot be reproduced from "
                       "this dataset's own frozen vectors (max abs err %.3g). squad and metaqa "
                       "set the fp16 floor at ~1e-6, so this is a real stale artifact, not fp "
                       "noise. The file is hash-pinned in LOCKED_5_OF_5 and was deliberately "
                       "NOT edited."
                       % (fsc["n_gt_1e-2"], fsc["n_edges"],
                          100.0 * fsc["n_gt_1e-2"] / fsc["n_edges"], fsc["max_abs_err"]))
    for f in FAMILIES:
        cov = (fe["coverage"] or {}).get(f) or {}
        if cov.get("source") == "graph2" and (cov.get("node_coverage") or 1.0) < 0.35:
            out.append("%s family reaches only %.1f%% of nodes. It was built to the frozen "
                       "contract and is correct, but the original 'ner_available: false' "
                       "judgement for this corpus was a USEFULNESS call and this coverage "
                       "confirms it -- do not read the family's existence as parity with the "
                       "text corpora." % (f, 100.0 * cov["node_coverage"]))
    if ds == "webqsp":
        note = six.get("WEBQSP_IS_A_DIFFERENT_KIND_OF_CORPUS")
        if note:
            out.append(note)
        out.append("gold resolution is 62.42%, which is a property of the RoG source, not of the "
                   "bridge: the positive control on topic entities clears 99.92%. Reading this "
                   "column against the five would be wrong.")
        out.append("shard_size is 12,000 here against 40,000 for the other five, and its store "
                   "lives on a separate volume under a dense__p%04d/dense.npy pattern.")
    if ds in ("hotpotqa", "2wiki"):
        out.append("TEXTUALIZATION_REV 2 applies to this corpus. The REV 1 nodes.jsonl is kept "
                   "under _superseded_textualization_rev1/ and is NOT the substrate.")
    return out


# ---------------------------------------------------------------------------
def build():
    five, six, fam = (rj(ROOT + "/" + RECORDS[k]) for k in ("five", "six", "family"))
    if not (five and six and fam):
        sys.exit("a locked record is missing from %s" % ROOT)
    decl = declared_index(five, six, fam)
    pi = rj(ROOT + "/POINTER_INDEX.json")
    ib = rj(ROOT + "/ID_BRIDGE.json")
    rc = rj(ROOT + "/RETRIEVAL_CACHE.json") or {}
    heavy = rj(ROOT + "/HEAVY_CACHE_VERIFICATION.json") or {}

    out = {}
    for ds in DS:
        dm = rj("%s/%s/dataset_manifest.json" % (ROOT, ds))
        bi = rj("%s/%s/build_info.json" % (ROOT, ds))
        sc = rj("%s/%s/SOURCE_CONTRACT.json" % (ROOT, ds))
        out[ds] = {
            "identity": s_identity(ds, five, six, fam, dm, bi, sc),
            "documents": s_documents(ds, decl, five, six, bi),
            "queries": s_queries(ds, decl, five, six, dm),
            "graph": s_graph(ds, decl, fam),
            "encodings": s_encodings(ds, decl, pi),
            "ids": s_ids(ds, decl, ib),
            "provenance": s_provenance(ds, decl, bi),
            "retrieval_cache": s_retrieval_cache(ds, rc, decl, heavy),
            "pinned_by_records": sorted({v["pinned_by"] for k, v in decl.items()
                                         if k.startswith("%s/%s/" % (ROOT, ds))}),
            "caveats": s_caveats(ds, five, six, fam),
        }

    # every slot key must be identical across all six -- that IS the deliverable
    slotsets = {ds: sorted(out[ds]) for ds in DS}
    uniform = len({tuple(v) for v in slotsets.values()}) == 1

    tot_edges = 0
    for ds in DS:
        for f in FAMILIES:
            c = (fam["datasets"][ds]["coverage"] or {}).get(f) or {}
            tot_edges += c.get("n_edges") or 0

    return {
        "RECORD": "UKB_COMMON_MANIFEST",
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "DERIVED -- not an authority; every hash is copied from the record in pinned_by",
        "BUILT_BY": "src/dataset_canonical/ukb_manifest.py",
        "VERIFIED_BY": "src/dataset_canonical/verify_manifest.py",
        "WHAT_THIS_IS": (
            "one queryable index over all six canonical datasets, in the shape UKB uses for "
            "results (src/pipeline/ukb_results.py): dataset -> slot -> {paths, bytes, sha256, "
            "headline counts}. Every dataset carries every slot; a dataset that lacks one "
            "carries {present: false, reason: ...} so absence is readable as a property rather "
            "than a hole."),
        "WHAT_THIS_IS_NOT": (
            "not a freeze and not a replacement for one. It adds no fact of its own: hashes come "
            "from LOCKED_5_OF_5 / LOCKED_6_OF_6 / " + FAMILY_RECORD_NAME + ", counts from "
            "POINTER_INDEX / ID_BRIDGE / the graph manifests. Where pinned_by is null the file is "
            "in use but was never hash-pinned by any freeze -- that is a finding, not an error."),
        # Copying five["CORE_INVARIANT"] verbatim into a SIX-dataset manifest over-claimed.
        # The frozen record is not wrong -- it asserted query-independence about its five text
        # corpora, and each of those five carries a query_independence_test.json that proves
        # it. webqsp has no such file and its shipped lane is declared query_independent:false
        # in webqsp/status.json. So the RULE is restated unchanged for all six, and the
        # PROPERTY is scoped to the five that hold it, with webqsp's exception named rather
        # than averaged away.
        "CORE_INVARIANT": {
            "RULE": "QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN.",
            "rule_scope": "all six datasets; never violated by this package",
            "PROPERTY_query_independent_corpus":
                "the node set is a function of the source corpus alone",
            "property_holds_for": ["squad", "musique", "metaqa", "hotpotqa", "2wiki"],
            "property_evidence":
                "each of the five carries data/final_canonical/<ds>/query_independence_test.json",
            "property_does_not_hold_for": ["webqsp"],
            "webqsp_exception": (
                "webqsp's shipped corpus is the union of RoG's per-question WebQSP+CWQ "
                "subgraphs, so its node set WAS shaped by the questions. It carries no "
                "query_independence_test.json and webqsp/status.json declares "
                "query_independent: false for this lane. It is accepted as CORPUS_TYPE = "
                "STANDARD_BENCHMARK_KG_SUBGRAPH -- an upstream benchmark artifact used "
                "unmodified by the KBQA literature, not a subsetting this project performed. "
                "State it that way in the paper; do not describe webqsp as query-independent "
                "and do not describe it as full Freebase. The only lane that would be "
                "query-independent at scale, WEBQSP_FREEBASE_SCALE, is NOT STARTED."),
            "inherited_from": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.CORE_INVARIANT",
            "verbatim_five_dataset_original": five["CORE_INVARIANT"],
        },
        "DETERMINISM": five["DETERMINISM"],
        "ENCODER_CONTRACT": five["ENCODER_CONTRACT"],
        "RECORD_CHAIN": [
            {"record": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES", "scope": "the five text corpora",
             "sha256": five.get("RECORD_SHA256")},
            {"record": "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES", "scope": "adds webqsp",
             "sha256": six.get("RECORD_SHA256"), "pins_predecessor": True},
            {"record": "LOCKED_6_OF_6_FAMILY_COMPLETION_V1",
             "scope": "adds the missing ner/knn families",
             "sha256": (fam.get("BUILDS_ON", {}).get("recorded_sha256")
                        if fam.get("RECORD") == "LOCKED_6_OF_6_FAMILY_COMPLETION_V2"
                        else fam.get("RECORD_SHA256")),
             "pins_predecessor": True},
        ] + ([{"record": "LOCKED_6_OF_6_FAMILY_COMPLETION_V2",
               "scope": "supersedes the stale frozen musique kNN by declaration "
                        "(musique/graph2/knn.npz); every other block is V1 verbatim",
               "sha256": fam.get("RECORD_SHA256"), "pins_predecessor": True}]
             if fam.get("RECORD") == "LOCKED_6_OF_6_FAMILY_COMPLETION_V2" else []),
        "SLOTS": {
            "identity": "dataset, corpus type, node/query counts, node-id rule, source of record",
            "documents": "nodes.jsonl and the hashes that fix its content and its order",
            "queries": "splits, lanes, eval subset, and the authoritative query order",
            "graph": ("the three edge families unioned across the frozen graph/ and the added "
                      "graph2/; frozen wins unless graph2 declares supersedes.<family>"),
            "encodings": "the dense and splade pointer indexes and the stores they resolve into",
            "ids": "the canonical<->legacy id bridge and its gold-resolution evidence",
            "provenance": "builder, builder sha256, git commit, source files, contract",
            "retrieval_cache": "the optional deep-K cache over the substrate",
            "pinned_by_records": "which freezes pin any artifact of this dataset",
            "caveats": "dataset-specific properties a consumer must not smooth over",
        },
        "SLOT_UNIFORMITY": {
            "all_six_carry_identical_slot_keys": uniform,
            "slot_keys": slotsets[DS[0]],
            "retrieval_cache_present_for_all_six": all(
                out[d]["retrieval_cache"].get("present") for d in DS),
            "retrieval_cache_provenance_is_not_uniform": (
                "metaqa/musique/squad are freeze-pinned (LOCKED_5_OF_5); hotpotqa/2wiki/webqsp are "
                "not pinned by any freeze -- see each slot's provenance block. Uniform KEYS, "
                "non-uniform AUTHORITY, stated rather than smoothed over."),
        },
        "TOTALS": {
            "n_datasets": len(DS),
            "n_nodes": sum(out[d]["identity"]["n_nodes"] for d in DS),
            "n_queries": sum(out[d]["identity"]["n_queries"] for d in DS),
            "n_edges_all_families": tot_edges,
            "declared_artifacts": len(decl),
        },
        "datasets": out,
    }


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/UKB_COMMON_MANIFEST.json"
    man = build()
    with io.open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(man, indent=1, ensure_ascii=False))
    with io.open(dest, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    t = man["TOTALS"]
    print("wrote %s  sha256=%s" % (dest, digest))
    print("  datasets=%d nodes=%d queries=%d edges=%d declared_artifacts=%d"
          % (t["n_datasets"], t["n_nodes"], t["n_queries"], t["n_edges_all_families"],
             t["declared_artifacts"]))
    print("  identical slot keys across all six: %s"
          % man["SLOT_UNIFORMITY"]["all_six_carry_identical_slot_keys"])
    miss = [(d, s) for d in DS for s in man["SLOTS"]
            if isinstance(man["datasets"][d].get(s), dict)
            and man["datasets"][d][s].get("present") is False]
    print("  slots explicitly absent (a property, not a hole): %d" % len(miss))
    for d, s in miss:
        print("     %-9s %s" % (d, s))
    nopin = sum(1 for d in DS for s in man["datasets"][d].values()
                if isinstance(s, dict) and s.get("exists") and s.get("pinned_by") is None)
    print("  top-level slots on disk with no freeze pinning them: %d" % nopin)


if __name__ == "__main__":
    main()
