# -*- coding: utf-8 -*-
"""
FREEZE THE FREEBASE TREE -- data/final_canonical/freebase/DATASET.json
======================================================================
The per-dataset index for the normalised Freebase tree, the same kind of record the six datasets
carry (RECORD CANONICAL_DATASET): counts, every served file with its byte size and sha256, the
name census, the layout and the position semantics.  Run after every build step has finished:

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/freeze_freebase.py

The build records under scratchpad/fb4/ are copied into freebase/records/ (bytes unchanged) and
pinned.  The record is self-hashed with the repo convention (json.dumps(indent=1), CRLF and LF).
freeze_canonical.py --with-freebase then pins this DATASET.json inside CANONICAL_FREEZE.json.
"""
import glob
import io
import json
import os
import shutil
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical"))
from freeze_canonical import FC, finfo, record_hash, rj, sha_file, utc, write_record  # noqa: E402

FB = FC + "/freebase"
V3 = FC + "/freebase_v3"
SCR = "scratchpad/fb4"
N = 301977131
E = 2062430072
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
NAME_KINDS = ["ORIGINAL", "RECOVERED_ORIGINAL", "RECOVERED_EXTERNAL", "URI_SELF", "KEY_SEMANTIC", "INFERRED", "GENERATED_FLOOR"]
BUILD_RECORDS = ["V4_FLOOR_CENSUS.json", "POSITIONS.json", "SIDECAR_BUCKETS.json", "NAMES_V4_BASE.json", "FLOOR_ANALYSIS.json",
                 "NEIGHBOUR_NAMES.json", "FLOOR_NAMES.json", "NAMES_V4_FINAL.json", "FREEBASE_NAMES_V4.json", "EDGES_FORWARD.json", "EDGES_REVERSE.json",
                 "NODES_BUILD.json", "METADATA_BUILD.json", "BRIDGE.json"]
METADATA_TABLES = ["alias", "description", "key", "label_residue", "master_property", "name", "property_schema",
                   "rdf_type", "rdf_type_residue", "reverse_property", "type", "type_hints"]
t0 = time.time()


def say(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def npy_shape(p):
    a = np.load(p, mmap_mode="r")   # header only; nothing is read
    return list(a.shape), str(a.dtype)


def pin(p, extra=None):
    d = finfo(p)
    if p.endswith(".npy"):
        d["shape"], d["dtype"] = npy_shape(p)
    elif p.endswith(".parquet"):
        m = pq.ParquetFile(p)
        d["rows"] = m.metadata.num_rows
        d["columns"] = m.schema_arrow.names
    if extra:
        d.update(extra)
    say("  pinned", p, d["bytes"])
    return d


def main():
    for must in ("nodes/node_uid.npy", "nodes/kind.npy", "nodes/name_kind.npy", "graph/out_indptr.npy", "graph/in_indptr.npy",
                 "relations/relations.parquet", "metadata/type_indptr.npy", "bridge/webqsp_positions.npy"):
        if not os.path.exists(FB + "/" + must):
            raise SystemExit("freebase tree lacks " + must)
    # ---- build records: copied unchanged, then pinned -----------------------------------------------
    os.makedirs(FB + "/records", exist_ok=True)
    records = {}
    for r in BUILD_RECORDS:
        s, d = SCR + "/" + r, FB + "/records/" + r
        if os.path.exists(s):
            if not os.path.exists(d) or sha_file(s) != sha_file(d):
                shutil.copyfile(s, d)
        if os.path.exists(d):
            records[r] = finfo(d)
            R = rj(d)
            if "RECORD_SHA256" in R:
                records[r]["self_hash_ok"] = record_hash(R) == R["RECORD_SHA256"]
    names_rec = rj(FB + "/records/FREEBASE_NAMES_V4.json") if "FREEBASE_NAMES_V4.json" in records else None
    edges_f = rj(FB + "/records/EDGES_FORWARD.json") if "EDGES_FORWARD.json" in records else None
    edges_r = rj(FB + "/records/EDGES_REVERSE.json") if "EDGES_REVERSE.json" in records else None
    meta_rec = rj(FB + "/records/METADATA_BUILD.json") if "METADATA_BUILD.json" in records else None
    bridge_rec = rj(FB + "/records/BRIDGE.json") if "BRIDGE.json" in records else None

    # ---- nodes ------------------------------------------------------------------------------------------
    say("nodes")
    kind = np.load(FB + "/nodes/kind.npy")
    name_kind = np.load(FB + "/nodes/name_kind.npy")
    assert kind.shape == name_kind.shape == (N,)
    by_kind = {KINDS[i]: int(c) for i, c in enumerate(np.bincount(kind, minlength=len(KINDS)))}
    by_name_kind = {NAME_KINDS[i]: int(c) for i, c in enumerate(np.bincount(name_kind, minlength=len(NAME_KINDS)))}
    assert (name_kind >= 0).all() and (name_kind < len(NAME_KINDS)).all()
    del kind, name_kind
    shards = sorted(glob.glob(FB + "/nodes/shard_*.parquet"))
    shard_pins = [pin(p) for p in shards]
    assert sum(s["rows"] for s in shard_pins) == N
    nodes = {
        "index": {"node_uid": pin(FB + "/nodes/node_uid.npy"), "kind": pin(FB + "/nodes/kind.npy"), "name_kind": pin(FB + "/nodes/name_kind.npy")},
        "shards": shard_pins, "n_shards": len(shards), "rows_per_shard": 4000000, "row_group_size": 500000,
        "columns": shard_pins[0]["columns"],
        "column_semantics": {
            "position": "row index; shard k row r is position 4,000,000*k + r",
            "node_uid": "the frozen V3 uid (CPython hash of the id bytes under PYTHONHASHSEED=0; literal uid = hash(lexical_form \\t datatype \\t language))",
            "node_id": "namespace-stripped Freebase id (m.xxx / g.xxx / schema path), the external URI, or the literal's tagged lexical form",
            "kind": "int8 code into KINDS", "name": "display name, never empty; read name_kind before trusting it as a label",
            "name_kind": "int8 code into NAME_KINDS: 0-2 are ACTUAL names (Freebase's own, or exact-identifier recoveries), 3-4 are the node's own identifier "
                         "made readable, 5 is a structural inference, 6 is a generated floor name (evidence-based description, never a label)",
            "name_source": "the layer / source that produced the name", "name_rule": "the inference or floor rule id (null for actual names)",
            "is_original_name": "true only for names Freebase itself asserted for this node (current or historical dump)",
            "nameless_grade": "SOURCE_DECLARED_NAMELESS / EMPIRICALLY_NAMELESS / INFERRED_NAMELESS / NOT_NAMELESS from V3 semantic_kind_v2.1 (null outside the 69.8M residue)",
            "recovery_class": "V3 recovery class of the residue node (null outside the residue)",
            "floor_reason": "1 INFERENCE_REJECTED_BOOKKEEPING_RELATION, 2 INFERENCE_REJECTED_UNREADABLE_IDENTITY, 3 NO_INFERENCE_ROW, 0 not a floor node",
            "lexical_form/datatype/language": "literal identity (null for non-literals)",
            "semantic_text": "K=6 role text of semantic_v1 (null where that layer holds none)",
        },
        "KINDS": KINDS, "by_kind": by_kind,
    }
    # ---- names ------------------------------------------------------------------------------------------
    names = {"NAME_KINDS": NAME_KINDS, "census": by_name_kind,
             "actual_names": int(by_name_kind["ORIGINAL"] + by_name_kind["RECOVERED_ORIGINAL"] + by_name_kind["RECOVERED_EXTERNAL"]),
             "identifier_names": int(by_name_kind["URI_SELF"] + by_name_kind["KEY_SEMANTIC"]),
             "inferred_names": int(by_name_kind["INFERRED"]), "generated_floor_names": int(by_name_kind["GENERATED_FLOOR"]),
             "empty_names": 0,
             "STATEMENT": "Every node carries a name. Only name_kind 0-2 are Freebase labels (ORIGINAL: the node's own /type/object/name "
                          "in the current or a historical dump; RECOVERED_*: exact-identifier recoveries of the 2026-09 campaign). URI_SELF and "
                          "KEY_SEMANTIC are the node's own identifier made readable. INFERRED and GENERATED_FLOOR strings are derived from the "
                          "graph and must never be presented as authentic Freebase labels; the floor names embed the MID so they stay unique "
                          "and honest about what they are.",
             "record": records.get("FREEBASE_NAMES_V4.json")}
    if names_rec:
        names["floor_by_rule"] = names_rec["FLOOR"].get("by_rule")
        names["floor_by_family"] = names_rec["FLOOR"].get("by_family")
        final_rec = rj(FB + "/records/NAMES_V4_FINAL.json")
        names["floor_by_reason"] = final_rec["floor_by_reason"]
        assert sum(final_rec["floor_by_reason"].values()) == by_name_kind["GENERATED_FLOOR"]
        names["floor_uniqueness"] = names_rec["FLOOR"].get("uniqueness")
    # ---- graph ------------------------------------------------------------------------------------------
    say("graph")
    flipped = 0
    fl = np.load(FB + "/graph/out_flag.npy", mmap_mode="r")
    for s in range(0, E, 200_000_000):
        flipped += int(np.count_nonzero(np.asarray(fl[s:min(s + 200_000_000, E)])))
    del fl
    graph = {
        "n_edges": E, "flipped_edges": flipped,
        "out": {c: pin(FB + "/graph/out_%s.npy" % c) for c in ("indptr", "dst", "rel", "flag")},
        "in": {c: pin(FB + "/graph/in_%s.npy" % c) for c in ("indptr", "src", "rel")},
        "layout": {"out": "CSR by source position: edges of node i are rows out_indptr[i]:out_indptr[i+1] of out_dst/out_rel/out_flag, "
                          "ordered by (rel_id, dst)",
                   "in": "CSR by destination position: rows in_indptr[i]:in_indptr[i+1] of in_src/in_rel, ordered by (rel_id, src); "
                         "the same 2,062,430,072 edges transposed",
                   "flag": "bit0 = flipped: 1 only where the source dump asserted a fact solely on the reverse side of a reverse-property "
                           "pair (V3 PASS D swaps it onto the master side); 0 where the master side was asserted directly. In this source "
                           "every one of the 126,994,565 reverse-side assertions had a directly asserted master-side twin, so the flag is 0 "
                           "on all %d edges (flipped_edges above is measured, not assumed)" % E,
                   "direction": "src -> dst is Freebase's own subject -> object of the master-side property; the only re-oriented family is "
                                "the type declaration, stored on its master side as (type, type.type.instance, instance) -- a node's types "
                                "are the sources of its type.type.instance in-edges; type.object.type does not occur",
                   "dst_kind": "not stored: kind[dst] (LITERAL -> literal object, EXTERNAL_URI -> external object, else Freebase node)",
                   "rel_id": "row of relations/relations.parquet"},
        "source": "freebase_v3/canonical/edges (V3 PASS D: reverse pairs canonicalised to the master direction, duplicates removed)",
        "records": {"forward": (edges_f or {}).get("CHECKS"), "reverse": (edges_r or {}).get("CHECKS"),
                    "out_degree": (edges_f or {}).get("OUT_DEGREE"), "in_degree": (edges_r or {}).get("IN_DEGREE")},
    }
    # ---- relations --------------------------------------------------------------------------------------
    rel = pq.read_table(FB + "/relations/relations.parquet")
    roles = rel["role"].to_pylist()
    relations = {"file": pin(FB + "/relations/relations.parquet"), "n": rel.num_rows,
                 "roles": {r: roles.count(r) for r in sorted(set(roles))},
                 "with_reverse": int(sum(1 for x in rel["reverse_rel_id"].to_pylist() if x >= 0)),
                 "canonical_edge_count_sum": int(sum(rel["canonical_edge_count"].to_pylist())),
                 "row_semantics": "row i is rel_id i; relation strings sorted ascending; reverse_rel_id is the rel_id of reverse_relation or -1"}
    assert relations["canonical_edge_count_sum"] == E
    # ---- metadata ---------------------------------------------------------------------------------------
    say("metadata")
    metadata = {"tables": {t: pin(FB + "/metadata/%s.parquet" % t) for t in METADATA_TABLES},
                "type_csr": {"type_indptr": pin(FB + "/metadata/type_indptr.npy"), "type_val": pin(FB + "/metadata/type_val.npy"),
                             "type_dict": pin(FB + "/metadata/type_dict.parquet")},
                "row_semantics": "every table: position int32 first (rows sorted by position), the frozen V3 columns verbatim; "
                                 "master_property / reverse_property also carry object_position; type_val[type_indptr[i]:type_indptr[i+1]] "
                                 "are the type ids of node i (ascending), type_dict maps type_id -> type string and its SCHEMA_TYPE position",
                "source": "freebase_v3/canonical/metadata (12 tables), subjects resolved to positions by the uid rule",
                "build": (meta_rec or {}).get("CHECKS")}
    # ---- bridge -----------------------------------------------------------------------------------------
    bridge = {"webqsp_positions": pin(FB + "/bridge/webqsp_positions.npy"), "schema_index": pin(FB + "/bridge/schema_index.parquet"),
              "semantics": "webqsp_positions[j] is the Freebase position of webqsp position j's source_mid (-1: no MID or not in the universe); "
                           "schema_index resolves every SCHEMA_* node_id to its position without reproducing CPython's hash",
              "build": {k: (bridge_rec or {}).get(k) for k in ("webqsp_nodes", "webqsp_nodes_with_source_mid", "resolved_to_freebase_position", "unresolved_mids")}}
    # ---- source layer -----------------------------------------------------------------------------------
    src_layer = {}
    for f in ("canonical/CANONICAL_MANIFEST.json", "FREEBASE_V3_CLOSURE_STATUS.json", "NAME_HIERARCHY_CONTRACT_V1.json",
              "V3_ACTUAL_NAME_LAYER_FROZEN.json", "inference_admissibility_v2_amendment.json"):
        p = V3 + "/" + f
        if os.path.exists(p):
            src_layer[f] = finfo(p)
    tree_bytes = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(FB) for x in fs if x != "DATASET.json")
    rec = {
        "RECORD": "CANONICAL_DATASET", "dataset": "freebase", "frozen_utc": utc(),
        "n_nodes": N, "n_edges": E, "n_relations": relations["n"], "n_queries": 0,
        "STATEMENT": "The full CRAG_FREEBASE_CANONICAL universe (freebase_v3, 301,977,131 nodes / 2,062,430,072 cleaned edges) in the "
                     "normalised position form of the six datasets: one node table in position order with a name on every row, the "
                     "edges as forward and reverse CSR over positions, dense relation ids, the twelve metadata tables keyed by position, "
                     "and the bridge from the served webqsp positions. It has no queries and no embeddings: encoding 302M nodes is "
                     "infeasible on this machine (FREEBASE_V3_CLOSURE_STATUS) and nothing was fabricated in their place.",
        "POSITION_INVARIANT": "Position i is the rank of the node's uid: nodes/node_uid.npy is strictly increasing, so "
                              "position(uid) = searchsorted(node_uid, uid). Row i of the node shards concatenated, the i-th block of the "
                              "CSR arrays, a subject/object value in metadata/*, a value in bridge/webqsp_positions.npy and an endpoint in "
                              "graph/* are the same integer. No pointer index, no join.",
        "IDENTITY": {"uid_rule": "CPython hash() of the UTF-8 bytes of the namespace-stripped id under PYTHONHASHSEED=0 (V3 PASS B); a "
                                 "literal's uid hashes lexical_form + '\\t' + datatype + '\\t' + language",
                     "resolve_an_id": "PYTHONHASHSEED=0 required for canonical.Freebase.position(id); schema ids also resolve through "
                                      "bridge/schema_index.parquet, webqsp nodes through bridge/webqsp_positions.npy"},
        "layout": {"nodes": "nodes/node_uid.npy, nodes/kind.npy, nodes/name_kind.npy, nodes/shard_XXXXX.parquet (4,000,000 rows each)",
                   "graph": "graph/out_{indptr,dst,rel,flag}.npy, graph/in_{indptr,src,rel}.npy",
                   "relations": "relations/relations.parquet", "metadata": "metadata/<table>.parquet, metadata/type_{indptr,val}.npy, metadata/type_dict.parquet",
                   "bridge": "bridge/webqsp_positions.npy, bridge/schema_index.parquet", "records": "records/*.json (the build and audit records, pinned)",
                   "loader": "data/final_canonical/canonical.py (Freebase)"},
        "nodes": nodes, "names": names, "graph": graph, "relations": relations, "metadata": metadata, "bridge": bridge,
        "records": records, "source_layer": src_layer,
        "not_included": {"embeddings": "none (infeasible: 302M nodes on CPU; recorded, not fabricated)", "queries": "none (Freebase is a KB, not a benchmark)",
                         "retrieval_cache": "none"},
        "tree_bytes": int(tree_bytes), "seconds": round(time.time() - t0, 1),
    }
    write_record(FB + "/DATASET.json", rec)
    say("wrote %s/DATASET.json  RECORD_SHA256 %s  (%.1f GB tree)" % (FB, rec["RECORD_SHA256"], tree_bytes / 1e9))


if __name__ == "__main__":
    main()
