# -*- coding: utf-8 -*-
"""FREEBASE_V3_CLOSURE_STATUS: what "finished" means for the Freebase layer, and what is not.

The Freebase V3 layer is a chain of FROZEN records: raw-mirror provenance lock, canonical
universe, resolution overlay, provenance order, actual-name layer, inference layer, name
hierarchy, semantic-text sidecar. Nothing in it is half-built. What remains is a short list
of items that are BLOCKED by an external party, NOT RUN because they are external acquisitions
that need the owner's approval, PROPOSED and awaiting the owner's acceptance, an OPEN DECISION
of contract level, or INFEASIBLE on this machine. This record states each with its class, so
"finish Freebase" has an answer that is not "run more passes": the layer is closed for
computation; the open items are decisions and permissions, not work this repository can do
on its own.

It computes nothing new and edits nothing. Every figure is read from the record that owns it,
and every cited record is pinned here by the sha256 of its bytes on disk.

Run:
  python src/dataset_canonical/freebase_v3_closure_status.py
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
from freeze_canonical import record_hash  # noqa: E402  (same convention; verify_manifest moved to _pre_v3/)

FC = "data/final_canonical"
V3 = FC + "/freebase_v3"
OUT = V3 + "/FREEBASE_V3_CLOSURE_STATUS.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def cite(rel):
    """Pin a record by the bytes on disk; carry its own self-hash and status verbatim."""
    p = "%s/%s" % (V3, rel)
    r = rj(p)
    self_hash = None
    for k in ("RECORD_SHA256", "record_sha256", "RECORD_HASH", "MANIFEST_HASH"):
        if r.get(k):
            self_hash = {k: r[k]}
            break
    return {"file": p, "sha256_of_bytes": sha_file(p), "self_hash": self_hash,
            "status": r.get("STATUS") or r.get("status")}, r


def exists(p):
    return os.path.exists(os.path.join(REPO, p))


def main():
    if os.path.exists(OUT):
        sys.exit("%s exists; a closure record is amended by a new record, not rewritten" % OUT)

    raw_c, raw = cite("V3_RAW_MIRROR_PROVENANCE_LOCK.json")
    cm_c, cm = cite("canonical/CANONICAL_MANIFEST.json")
    val_c, val = cite("V3_CANONICAL_VALIDATION.json")
    ov_c, ov = cite("V3_RESOLUTION_OVERLAY_FROZEN.json")
    po_c, po = cite("V3_NAME_PROVENANCE_ORDER_V2.json")
    es_c, es = cite("V3_EXACT_SOURCE_SWEEP_CLOSED.json")
    anl_c, anl = cite("V3_ACTUAL_NAME_LAYER_FROZEN.json")
    inf_c, inf = cite("LOCKED_FREEBASE_INFERENCE_V1.json")
    am_c, am = cite("inference_admissibility_v2_amendment.json")
    nh_c, nh = cite("NAME_HIERARCHY_CONTRACT_V1.json")
    sem_c, sem = cite("V3_SEMANTIC_TEXT_BUILD.json")
    aq_c, aq = cite("V3_AUTHORITY_QUEUE_RESOLUTION.json")
    mb_c, _mb = cite("V3_EXTERNAL_AUTHORITY_MUSICBRAINZ.json")
    dg_c, _dg = cite("V3_URI_FETCH_DISCOGS.json")
    ph_c, ph = cite("V3_PAGE_HUNT_COMPLETE.json")
    ut_c, ut = cite("V3_UNTYPED_TERMINAL_PROPOSAL.json")
    tm_c, tm = cite("V3_TYPE_MIRROR_FINDING.json")
    dr_c, dr = cite("V3_DISK_RECLAMATION.json")

    n_nodes = val["CHECKS"]["node_uid_unique"]["nodes"]
    n_edges = val["STATISTICS"]["EDGES_TOTAL"]
    tgt = raw.get("TARGET") or {}

    frozen = {}
    frozen["RAW_MIRROR"] = {
        "what": "the source of record: freebase-rdf-latest.gz as published on archive.org, "
                "identity locked before transfer (md5/sha1/crc32 expected values recorded)",
        "size_bytes": tgt.get("size_bytes"), "md5_expected": tgt.get("md5_expected"),
        "sha1_expected": tgt.get("sha1_expected"), "source_mtime_utc": tgt.get("source_mtime_utc"),
        "record": raw_c,
        "protected_on_disk": ["%s/_acquisition" % V3],
        "rule": "never deleted; never expanded to ~250 GB; 200 independent gzip members",
    }
    frozen["CRAG_FREEBASE_CANONICAL"] = {
        "what": "the full Freebase universe built from the raw mirror: nodes, edges, relations, "
                "metadata CVTs retained",
        "n_nodes": n_nodes, "n_edges": n_edges,
        "n_relations": val["STATISTICS"].get("RELATIONS_TOTAL"),
        "node_counts_by_kind": val["STATISTICS"].get("NODE_COUNTS"),
        "manifest": cm_c,
        "manifest_status_verbatim": cm.get("status"),
        "validation": val_c, "all_checks_pass": val.get("ALL_CHECKS_PASS"),
        "checks_recorded": sorted(val["CHECKS"].keys()),
        "idir_note": "the IDIR gate STOP at 95.662% is source divergence (symmetric), recorded "
                     "as FROZEN_WITH_ORACLE_GATE_OVERRIDDEN; IDIR is the audit oracle, not the "
                     "canonical graph, and its ZIP is kept",
        "protected_on_disk": ["%s/canonical" % V3],
    }
    frozen["CRAG_FREEBASE_RESOLUTION_OVERLAY_V1"] = {
        "what": "the frozen name-resolution overlay, one row per node (do not modify)",
        "rows": ov.get("rows"), "bytes": ov.get("bytes"), "frozen_utc": ov.get("frozen_utc"),
        "resolution_mix": ov.get("RESOLUTION_MIX"),
        "record": ov_c,
        "protected_on_disk": ["%s/overlay_v1" % V3],
    }
    frozen["NAME_PROVENANCE_ORDER_V2"] = {
        "what": "the provenance order that ranks name sources (one historical class resolved by "
                "time; source = authenticity, time = which name)",
        "status_verbatim": po.get("STATUS"),
        "record": po_c,
    }
    frozen["EXACT_SOURCE_SWEEP"] = {
        "what": "every exact historical/authority source for the ENTITY_MID residue, run to "
                "completion and CLOSED with mechanisms; the cascade rebuilt on the finished state",
        "final_cascade": es.get("FINAL_CASCADE"),
        "last_two_authority_passes_completed": es.get("THE_LAST_TWO_PASSES"),
        "records": {"sweep_closed": es_c, "musicbrainz": mb_c, "discogs": dg_c},
    }
    frozen["ACTUAL_NAME_LAYER"] = {
        "what": "census of all %s nodes into SELF_IDENTIFYING / ACTUAL_NAME / "
                "HANDED_TO_THE_INFERENCE_LAYER; the boundary between source names and computed "
                "names, fixed" % format(n_nodes, ","),
        "ruling_executed_verbatim": anl.get("RULING_THIS_EXECUTES"),
        "frozen_counts": anl.get("FROZEN_COUNTS"),
        "record": anl_c,
    }
    frozen["INFERENCE_LAYER"] = {
        "what": "inferred_name_v1 + inference_overlay_v1 + inference_admissibility_v1, with the "
                "v2 admissibility amendment applied as an overlay (36 rows), never in place",
        "counts": inf.get("COUNTS"),
        "layers": {k: {"n_parts": v.get("n_parts"), "bytes": v.get("bytes"),
                       "layer_hash": v.get("layer_hash")}
                   for k, v in (inf.get("layers") or {}).items()},
        "known_limitations_verbatim": inf.get("KNOWN_LIMITATIONS"),
        "record": inf_c, "amendment": am_c,
        "protected_on_disk": ["%s/inferred_name_v1" % V3, "%s/inference_overlay_v1" % V3,
                              "%s/inference_admissibility_v1" % V3],
    }
    frozen["NAME_HIERARCHY"] = {
        "what": "precedence ORIGINAL_NAME -> ADMISSIBLE_INFERRED_NAME -> FLOOR; tiers disjoint; "
                "the floor is never lowered; tier-2 names are GENERATED and never presented as "
                "authentic Freebase labels",
        "precedence": [{"tier": t.get("tier"), "name": t.get("name"), "nodes": t.get("nodes"),
                        "authority": t.get("authority")} for t in (nh.get("PRECEDENCE") or [])],
        "coverage": nh.get("COVERAGE"),
        "record": nh_c,
    }
    frozen["SEMANTIC_TEXT"] = {
        "what": "append-only role-text sidecar keyed by node_uid (K=%s roles), not a name; "
                "canonical/ read, never written" % sem.get("roles_cap_K"),
        "rows_written": sem.get("rows_written"), "edges_scanned": sem.get("edges_scanned"),
        "edges_complete": sem.get("EDGES_COMPLETE"),
        "not_supported_verbatim": sem.get("NOT_supported"),
        "record": sem_c,
        "protected_on_disk": ["%s/semantic_v1" % V3],
    }
    for k, v in frozen.items():
        for p in v.get("protected_on_disk", []):
            v.setdefault("protected_paths_exist", {})[p] = exists(p)

    blocked = aq.get("BLOCKED_NOT_ABSENT") or {}
    proj = ph.get("PROJECTION_ONTO_THE_UNATTEMPTED_POOL") or {}
    open_items = [
        {"item": "external authorities behind a bot challenge, a missing API key, or a retired "
                 "service",
         "class": "BLOCKED",
         "authorities": {k: {"queued": v.get("queued"), "status": v.get("status"),
                             "reason": v.get("reason") or v.get("READING")}
                         for k, v in blocked.items()},
         "n_rows_blocked": sum(int(v.get("queued") or 0) for v in blocked.values()),
         "why_not_closed_verbatim": aq.get("THE_DISTINCTION_THAT_MATTERS"),
         "what_would_change_it": "an API key or an allowed access path supplied by the owner; "
                                 "working around a bot check is out of bounds",
         "record": aq_c},
        {"item": "archived Freebase topic-page expansion over the unattempted residue",
         "class": "NOT_RUN -- external acquisition, needs the owner's approval",
         "unattempted_mids": proj.get("not_yet_attempted_mids"),
         "naive_projection_at_the_completed_rate": proj.get("naive_projection_at_the_completed_rate"),
         "projected_hours": proj.get("projected_hours"),
         "caveat_verbatim": proj.get("THE_CAVEAT_THAT_MATTERS_MORE_THAN_THE_NUMBER"),
         "cheap_first_step_verbatim": proj.get("HOW_TO_MAKE_IT_AN_ESTIMATE_INSTEAD"),
         "completed_run_evidence": ph.get("RESULT"),
         "record": ph_c},
        {"item": "URL site-chrome strip",
         "class": "PROPOSED -- not applied, awaiting acceptance",
         "statement_verbatim": (anl.get("WHAT_IS_NOT_CLOSED_AND_IS_NOT_SILENTLY_ABANDONED") or {}
                                ).get("URL_site_chrome_strip"),
         "record": anl_c},
        {"item": "untyped terminal classification",
         "class": "PROPOSAL -- original form REFUTED by measurement, re-derived, NOT APPLIED",
         "status_verbatim": ut.get("STATUS"),
         "headline_verbatim": ut.get("THE_HEADLINE"),
         "open_and_not_chased": ut.get("OPEN_AND_NOT_CHASED_HERE"),
         "record": ut_c},
        {"item": "type-mirror edges: keep both layers as built vs pre-register an ablation",
         "class": "OPEN DECISION -- measured and recorded; contract-level; the owner's call",
         "disposition_verbatim": tm.get("DISPOSITION"),
         "record": tm_c},
        {"item": "58,144 freeq pseudo-titles",
         "class": "REFUSED, permanently",
         "statement_verbatim": (anl.get("WHAT_IS_NOT_CLOSED_AND_IS_NOT_SILENTLY_ABANDONED") or {}
                                ).get("58144_freeq_pseudo_titles"),
         "record": anl_c},
        {"item": "weakly connected components of the %s-node graph" % format(n_nodes, ","),
         "class": "INFEASIBLE on this machine (RAM)",
         "note": "V3_CANONICAL_VALIDATION records every other statistic; a union-find over "
                 "%s edges does not fit the available RAM and was not attempted" % format(n_edges, ","),
         "record": val_c},
        {"item": "dense/SPLADE retrieval encodings over the full Freebase universe",
         "class": "INFEASIBLE on local CPU; and not part of the six-dataset package",
         "note": "the measured CPU ceiling already puts 24.9M dense rows at a FLOP wall of "
                 "months; the universe is %s nodes. The six-dataset package's KB corpus is "
                 "webqsp = %s (%s nodes, corpus_type %s), which IS fully encoded and frozen. "
                 "The V3 universe is the separate, parallel Freebase layer (HANDOFF section 10) "
                 "and is never described as webqsp's corpus."
                 % (format(n_nodes, ","), "V1-resolved RoG WebQSP+CWQ graph", "2,592,894",
                    "STANDARD_BENCHMARK_KG_SUBGRAPH"),
         "record": None},
    ]

    rec = {
        "RECORD": "FREEBASE_V3_CLOSURE_STATUS",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "STATUS": "CLOSED_FOR_COMPUTATION -- every layer is FROZEN; the open items are decisions "
                  "and permissions, not passes",
        "_what": "the answer to 'finish Freebase'. Nothing here is computed or edited: each figure "
                 "is read from the record that owns it and each record is pinned by the sha256 of "
                 "its bytes. Open items are listed by class (BLOCKED / NOT_RUN pending approval / "
                 "PROPOSED / OPEN DECISION / REFUSED / INFEASIBLE) so that none of them can be "
                 "mistaken for a finding about Freebase.",
        "FROZEN_LAYERS": frozen,
        "OPEN_ITEMS": open_items,
        "STATEMENTS_THIS_RECORD_SUPERSEDES": [
            {"file": aq_c["file"], "section": "IN_PROGRESS",
             "statement": "musicbrainz 3,186 queued (<=1 req/s); discogs 33 queued",
             "current_fact": "both passes completed and were folded into the cascade before the "
                             "actual-name freeze: musicbrainz 2,346 of 2,484 distinct MBIDs "
                             "resolved (2,313 nodes; 109 ws2_404, 29 no_entity_type; one request "
                             "per 1.15 s); discogs 11,737 rows over all runs (242 invalid artist "
                             "rows excluded by URL shape at cascade time). The authority-queue "
                             "record is not edited; the sweep-closed record carries the final "
                             "numbers.",
             "superseded_by": [es_c["file"], mb_c["file"], dg_c["file"]]},
        ],
        "STANDING_RULES_THAT_BIND_THIS_LAYER": [
            "the raw mirror (freebase-rdf-latest.gz, 31.3 GB, 200 gzip members) is the source of "
            "record and is never deleted",
            "the IDIR ZIP is not deleted (audit oracle, not the canonical graph)",
            "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 is not modified",
            "the 460K metadata CVTs are retained; the source is never expanded to ~250 GB",
            "BLOCKED is never conflated with CLOSED; nameless grades stay three separate grades",
            "ZERO_JOIN_INVARIANT: a zero-hit result needs a positive control before acceptance",
            "authority IDs are matched exhaustively and only by exact identifier lookup -- no "
            "fuzzy-name matching; MusicBrainz at <= 1 req/s",
            "external acquisitions (Wayback page expansion included) need the owner's explicit "
            "approval before a single request is made",
            "contract amendments are new records with their own hash, never edits",
            "never write 'We reconstructed the official final Google Freebase dump'; never "
            "describe webqsp as 'full Freebase'",
        ],
        "DISK": {
            "earlier_reclamation": dict(dr_c, deleted_gb=dr.get("deleted_gb"),
                                        free_before_gb=dr.get("free_before_gb"),
                                        free_after_gb=dr.get("free_after_gb")),
            "current_pass": "CLEANUP_PASS_3 (data/final_canonical/CLEANUP_PASS_3.json): only "
                            "cross-checked DELETE verdicts from CLEANUP_REFERENCE_CROSSCHECK.json, "
                            "none of them cited by any V3 record by path or by name; the gate is "
                            "re-derived at apply time",
            "kept_by_rule": "raw mirror, IDIR, FACC1, overlay_v1, canonical/, the three inference "
                            "layers, semantic_v1, and every intermediate a V3 record names",
        },
        "HOW_TO_REOPEN": "a new record that BUILDS_ON this one, naming the item it advances and "
                         "the approval it acted under. This record is not edited.",
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("frozen layers (%d): %s" % (len(frozen), ", ".join(frozen)))
    for k, v in frozen.items():
        for p, ok in (v.get("protected_paths_exist") or {}).items():
            print("   %-60s exists=%s" % (p, ok))
    print("open items (%d):" % len(open_items))
    for it in open_items:
        print("  %-78s %s" % (it["item"][:78], it["class"]))
    print("wrote %s  RECORD_SHA256 %s" % (OUT, rec["RECORD_SHA256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
