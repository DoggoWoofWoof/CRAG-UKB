# -*- coding: utf-8 -*-
"""FREEBASE_V3_CLOSURE_STATUS_V2: the Freebase layer is TERMINAL.

BUILDS_ON FREEBASE_V3_CLOSURE_STATUS (2026-09-11, CLOSED_FOR_COMPUTATION), which is not edited. What changed since:

  * the layer is SERVED: data/final_canonical/freebase/ (the seventh tree of CANONICAL_FREEZE.json) carries every node with a
    display name and its provenance, the whole graph as forward + reverse CSR over positions, the metadata by position and the
    webqsp bridge -- built from freebase_v3/, verified (verify_freebase.py --full 155/0; verify_canonical.py --full 679/0);
  * under the owner's authorisation of 2026-09-13 ("delete the superseded freebase_v3 sub-layers and free up space and
    completely finish the freebase so that we can freeze it completely and go for edge creation or embedding or
    partitioning") the DATA files of the sub-layers the served tree supersedes were deleted, sha-matched against the census
    taken first (_history/logs/FREEBASE_V3_SUBLAYER_DELETION.json); every record and report stayed in place;
  * every open item of the V1 record receives a terminal disposition below. None of them is "resolved" by computation: they
    are closed as INFEASIBLE_HERE, BLOCKED, NOT_RUN (no approval given), NOT_APPLIED (proposal recorded), SERVED_AS_BUILT
    (default recorded, no contract commitment) or REFUSED -- each with what it would take to reopen it under a new record.

It computes nothing new and edits nothing. Run:
  python src/dataset_canonical/freebase_v3_closure_status_v2.py
"""
import datetime
import hashlib
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_canonical import record_hash  # noqa: E402

FC = "data/final_canonical"
V3 = FC + "/freebase_v3"
FB = FC + "/freebase"
OUT = V3 + "/FREEBASE_V3_CLOSURE_STATUS_V2.json"
DEL = FC + "/_history/logs/FREEBASE_V3_SUBLAYER_DELETION.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def cite(p):
    r = rj(p)
    self_hash = None
    for k in ("RECORD_SHA256", "record_sha256", "RECORD_HASH", "MANIFEST_HASH"):
        if r.get(k):
            self_hash = {k: r[k]}
            break
    return {"file": p, "bytes": os.path.getsize(p), "sha256_of_bytes": sha_file(p), "self_hash": self_hash,
            "status": r.get("STATUS") or r.get("status")}, r


def du(path):
    n, b = 0, 0
    for r, _, fs in os.walk(path):
        for x in fs:
            n += 1
            b += os.path.getsize(os.path.join(r, x))
    return {"files": n, "bytes": b, "gb": round(b / 1e9, 2)}


def main():
    v1_cite, v1 = cite(V3 + "/FREEBASE_V3_CLOSURE_STATUS.json")
    ds_cite, ds = cite(FB + "/DATASET.json")
    vf_cite, vf = cite(FB + "/VERIFICATION.json")
    vt_cite, vt = cite(FC + "/VERIFICATION.json")
    del_cite, dl = cite(DEL)
    assert dl["status"] == "APPLIED" and not dl["mismatched_not_deleted"] and not dl["missing_at_apply"], dl["status"]
    assert vf["PASS"] and vf["mode"] == "full" and vt["PASS"] and vt["mode"] == "full" and "freebase" in vt["datasets"]
    import pandas as pd
    rel = pd.read_parquet(FB + "/relations/relations.parquet")
    tti = rel[rel["relation"] == "type.type.instance"].iloc[0]
    r2 = None
    p = os.environ.get("R2_CENSUS")
    if p and os.path.exists(p):
        r2 = rj(p)

    remaining = {}
    for d in sorted(os.listdir(V3)):
        q = V3 + "/" + d
        if os.path.isdir(q):
            remaining[d + "/"] = du(q)
    remaining["*.json records at the root"] = {"files": sum(1 for x in os.listdir(V3) if x.endswith(".json")),
                                              "bytes": sum(os.path.getsize(V3 + "/" + x) for x in os.listdir(V3) if x.endswith(".json"))}
    v1_items = {it["item"]: it for it in v1["OPEN_ITEMS"]}

    def item(name, disposition, why, reopen, served_effect):
        v = v1_items.get(name)
        return {"item": name, "v1_class": v["class"] if v else "(not in V1; raised by the 2026-09-12 floor analysis)",
                "disposition": disposition, "why": why, "what_would_reopen_it": reopen, "served_effect": served_effect}

    dispositions = [
        item("external authorities behind a bot challenge, a missing API key, or a retired service", "BLOCKED (unchanged)",
             "292 queued rows (stanford 106 bot-challenge, thetvdb 153 API key, giantbomb 22 API key, netflix 8 retired, tvrage 3 defunct); "
             "no key or access path was supplied; working around a bot check is out of bounds. BLOCKED is not CLOSED.",
             "an API key or an allowed access path from the owner, then a new names record + node-shard rebuild for the rows it resolves",
             "those rows carry their floor names (name_kind GENERATED_FLOOR); nothing is presented as an authentic label"),
        item("archived Freebase topic-page expansion over the unattempted residue", "NOT_RUN -- no approval given",
             "an external acquisition (30,570 unattempted MIDs, ~17 h at the completed rate); the owner's 2026-09-13 directive did not "
             "approve it and the standing rule requires explicit approval before a single request is made",
             "explicit approval; then the expansion under its own record, a new names record and a node-shard rebuild",
             "the residue keeps its floor names"),
        item("URL site-chrome strip", "NOT_APPLIED -- proposal recorded, names served as found",
             "5,302 URI_SELF rows would get a cleaner label; a content change to served names is the owner's call and was not made",
             "acceptance; then a new names record (the strip is a pure string rule, no acquisition) and a node-shard rebuild",
             "URI_SELF names are the URIs as found in the source (74,454,117 rows)"),
        item("untyped terminal classification", "NOT_APPLIED -- proposal stands in its re-derived form",
             "the original form was refuted by measurement and the re-derived form covers 14.2% of the residue; no overlay column was "
             "ever written and none is served",
             "a decision to adopt the re-derived classification, under a new record",
             "no served column depends on it; floor_reason / nameless_grade are served as frozen"),
        item("type-mirror edges: keep both layers as built vs pre-register an ablation", "SERVED_AS_BUILT -- default recorded, no contract commitment",
             "the served graph carries every edge including the type declarations: relation type.type.instance, rel_id %d, %s edges "
             "(%.2f%% of %s). No ablation is pre-registered here; any ablation is a filter on that one rel_id and needs no rebuild"
             % (int(tti["rel_id"]), "{:,}".format(int(tti["canonical_edge_count"])),
                100.0 * int(tti["canonical_edge_count"]) / ds["n_edges"], "{:,}".format(ds["n_edges"])),
             "a pre-registration record naming ALL_NODE vs ALL_NODE - TYPE_MIRROR_EDGES before any experiment that compares them",
             "graph/out_rel == %d selects the type declarations; a node's types are the SOURCES of its type.type.instance in-edges" % int(tti["rel_id"])),
        item("58,144 freeq pseudo-titles", "REFUSED, permanently (unchanged)",
             "Freeq job variables are not names; the rows carry R6_KEY_OPAQUE floor names that embed the MID",
             "nothing", "R6_KEY_OPAQUE 630,101 rows"),
        item("weakly connected components of the 301,977,131-node graph", "INFEASIBLE_HERE (unchanged)",
             "a union-find over 2,062,430,072 edges does not fit 16 GB of RAM; not attempted, not estimated",
             "a machine with ~40 GB of RAM or an out-of-core implementation, under its own record",
             "no served statistic claims connectivity"),
        item("dense/SPLADE retrieval encodings over the full Freebase universe", "INFEASIBLE_HERE (unchanged) -- recorded in DATASET.json not_included",
             "302M nodes on a CPU whose measured ceiling already puts 24.9M dense rows at a FLOP wall of months; nothing was fabricated. "
             "The served text per node is name (+ semantic_text where the K=6 role text exists) if an encoder is ever run elsewhere",
             "a GPU machine and a budget decision by the owner (Modal is not available); embeddings would be added as a new "
             "position-ordered channel under a new DATASET.json, never by editing this one",
             "freebase/DATASET.json not_included.embeddings"),
        item("MusicBrainz NGS lookups for the R2_KEY_IDENTITY floor rows", "NOT_RUN -- no approval given",
             ("R2_KEY_IDENTITY = 1,531,606 floor names built from a decodable external identity"
              + ("; census of the served names by identity family: %s" % json.dumps(r2["by_identity_family"]) if r2 else "")
              + ". Resolving the MusicBrainz ids to titles is an external acquisition (<= 1 req/s by the standing rule) that the "
                "2026-09-13 directive did not approve"),
             "explicit approval; then exact-identifier lookups under their own record, a new names record and a node-shard rebuild",
             "those rows carry '<Head> m.x (<identity>)' floor names (name_rule R2_KEY_IDENTITY)"),
    ]

    rec = {
        "RECORD": "FREEBASE_V3_CLOSURE_STATUS_V2",
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "STATUS": "TERMINAL -- the layer is served as data/final_canonical/freebase/ and pinned by CANONICAL_FREEZE.json; the "
                  "superseded sub-layer data is deleted under the owner's authorisation; every open item has a terminal disposition; "
                  "nothing remains that this repository can compute on its own",
        "BUILDS_ON": v1_cite,
        "AUTHORISATION": "user, 2026-09-13: 'delete the superseded freebase_v3 sub-layers and free up space and completely finish the "
                         "freebase so that we can freeze it completely and go for edge creation or embedding or partitioning'",
        "SERVED_AS": {
            "tree": FB + "/", "DATASET.json": ds_cite, "n_nodes": ds["n_nodes"], "n_edges": ds["n_edges"], "n_relations": ds["n_relations"],
            "names": {"every_node_named": True, "census": ds["names"]["census"], "actual_names": ds["names"]["actual_names"],
                      "identifier_names": ds["names"]["identifier_names"], "inferred_names": ds["names"]["inferred_names"],
                      "generated_floor_names": ds["names"]["generated_floor_names"]},
            "tree_bytes": ds["tree_bytes"], "not_included": ds["not_included"],
            "verification": {"freebase": vf_cite, "freebase_checks": [vf["checks"], vf["failed"]],
                             "package": vt_cite, "package_checks": [vt["checks"], vt["failed"]],
                             "package_freeze_at_this_record": vt["freeze_RECORD_SHA256"],
                             "note": "the freeze that pins this record supersedes that one; freebase/DATASET.json is unchanged by it"},
            "loader": "data/final_canonical/canonical.py (class Freebase)",
        },
        "SUBLAYER_DATA_DELETED": {
            "log": del_cite, "files": dl["deleted"], "bytes": dl["bytes_deleted"], "gb": round(dl["bytes_deleted"] / 1e9, 2),
            "sha_matched": "every file was re-hashed at apply time and deleted only on an exact match with the census; %d of %d matched a "
                           "digest pinned by a V3 record (overlay_v1, inferred_name_v1, inference_overlay_v1, pass_d entirely)" % (dl["pinned_total"], dl["files_total"]),
            "layers": {L: {"files": a.get("files", 0), "bytes": a.get("bytes", 0), "pinned": a.get("pinned", 0), "records_kept": a.get("kept_records", 0),
                           "what": a.get("LAYERS_WHAT")} for L, a in dl["layers"].items()},
            "records_kept_in_place": "every *.json under those directories (OVERLAY_MANIFEST.json, the 201 PASS A and 200 PASS B "
                                     "per-member reports, pass_d/reverse_map.json); empty directories removed",
            "why_superseded": "the served tree carries their content by position: name / name_kind / name_source / name_rule / "
                              "is_original_name / nameless_grade / recovery_class / floor_reason (overlay_v1, inferred_name_v1, "
                              "inference_overlay_v1), semantic_text (semantic_v1); pass_a-d and the probes were build intermediates "
                              "of canonical/, which stays",
        },
        "REMAINING_UNDER_freebase_v3": remaining,
        "STILL_PROTECTED": ["_acquisition/raw (the raw mirror, never deleted; and the published-dump recovery sources)",
                            "_acquisition/idir (the IDIR zip: audit oracle, never merged; oracle/ and metadata/ derived from it)",
                            "_acquisition/facc1", "canonical/ (the frozen V3 graph the served tree was built from; CANONICAL_MANIFEST.json "
                            "is pinned by freebase/DATASET.json source_layer)", "inference_admissibility_v1/", "every *.json record"],
        "OPEN_ITEMS_DISPOSITION": dispositions,
        "READY_FOR_THE_NEXT_PHASES": {
            "edge creation": "positions, forward + reverse CSR, relations.parquet (rel_id, relation, role, counts), names and types are "
                             "served; a new edge family would be a new file under graph/ with its own manifest, never a change to out_*/in_*",
            "embedding": "not possible on this machine (see the INFEASIBLE_HERE item); on a GPU machine the text per node is name "
                         "(+ semantic_text); rows are positions, so any channel is a position-ordered array like the six datasets'",
            "partitioning": "the CSR is memory-mappable (in_indptr 2.4 GB, out_indptr 2.4 GB, dst/src 8.2 GB each); a partitioner over "
                            "2.06B edges must be out-of-core on 16 GB of RAM -- a design decision, not a data gap",
        },
        "STANDING_RULES_THAT_BIND_THIS_LAYER": v1["STANDING_RULES_THAT_BIND_THIS_LAYER"],
        "RULE_AMENDMENT": "the V1 rule 'CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 is not modified' is satisfied by supersession, not by "
                          "retention: its content is served in freebase/nodes/ and its manifest record is kept; the parquet data was "
                          "deleted under the 2026-09-13 authorisation. Everything else in the list is unchanged.",
        "HOW_TO_REOPEN": v1["HOW_TO_REOPEN"],
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote %s  RECORD_SHA256 %s" % (OUT, rec["RECORD_SHA256"]))
    print("remaining:", json.dumps(remaining))


if __name__ == "__main__":
    main()
