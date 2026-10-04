# -*- coding: utf-8 -*-
"""SUPERSEDED_STATEMENTS_INDEX: every stale status sentence that is still on disk, and why.

This package amends by new record, never by edit: a LOCKED_* record is self-hashed and pinned
by everything that BUILDS_ON it, and many status files are hash-pinned as artifacts of a
freeze. The cost of that rule is that superseded sentences stay readable in place -- "webqsp
is BLOCKED", "kNN not built", "still open on the Freebase side" -- and a reader who lands on
one has no way to know it was overtaken. This index is that way. Each entry quotes the
statement VERBATIM from where it sits (resolved by JSON pointer or line anchor at run time,
so the index cannot describe a sentence that is not there), names the record that replaced
it (pinned by the sha256 of its bytes), states the current fact, and says why the original
was not edited -- with the pin or self-hash that would have broken if it had been.

Nothing here edits anything. Run it again whenever another statement goes stale; the previous
index is chained by hash, not overwritten silently.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/superseded_statements_index.py
"""
import datetime
import hashlib
import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import declared_index, record_hash  # noqa: E402

FC = "data/final_canonical"
V3 = FC + "/freebase_v3"
OUT = FC + "/SUPERSEDED_STATEMENTS_INDEX.json"

SIX = FC + "/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json"
FIVE = FC + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json"
FAM1 = FC + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json"
FAM2 = FC + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"
APPR = FC + "/_APPROVALS.json"
UKB = FC + "/UKB_COMMON_MANIFEST.json"
MUS_G2 = FC + "/musique/graph2/GRAPH_MANIFEST.json"
INF = V3 + "/LOCKED_FREEBASE_INFERENCE_V1.json"
NH = V3 + "/NAME_HIERARCHY_CONTRACT_V1.json"
CLOSURE = V3 + "/FREEBASE_V3_CLOSURE_STATUS.json"
SWEEP = V3 + "/V3_EXACT_SOURCE_SWEEP_CLOSED.json"
MB = V3 + "/V3_EXTERNAL_AUTHORITY_MUSICBRAINZ.json"
DG = V3 + "/V3_URI_FETCH_DISCOGS.json"
RAW_LOCK = V3 + "/V3_RAW_MIRROR_PROVENANCE_LOCK.json"
RAW_ACQ = V3 + "/V3_RAW_MIRROR_ACQUISITION.json"
ACQ = V3 + "/V3_ACQUISITION_RECORD.json"
CANON_VAL = V3 + "/V3_CANONICAL_VALIDATION.json"
WQ_BUILD = FC + "/webqsp/CANONICAL_V1_BUILD.json"

# ---------------------------------------------------------------- why-not-edited classes
FROZEN_SELF_HASHED = ("FROZEN_SELF_HASHED_RECORD: the file is a frozen record whose RECORD_SHA256 is "
                      "computed over its own content and pinned by later records' BUILDS_ON; an edit "
                      "would invalidate it and every record chained on it")
HASH_PINNED_ARTIFACT = ("HASH_PINNED_ARTIFACT: a freeze declares this file's sha256 as an artifact; an "
                        "edit would make the full verification report it as bad")
HISTORICAL_BUILD_LOG = ("HISTORICAL_BUILD_LOG_RETAINED_VERBATIM: nothing pins this file, but it is the "
                        "Track-B build manifest as of the moment it was written and its status strings "
                        "are the record of how the block was reached; the package convention is that "
                        "a status is superseded by a later record, not rewritten in place")
LIVING_DOC_ANNOTATED = ("LIVING_DOCUMENT_ANNOTATED_IN_PLACE: HANDOFF.md is prose, not a record; the "
                        "sentence was kept and struck through with a dated SUPERSEDED note in the "
                        "document's own convention, and the 2026-09-11 amendments section lists it")

WEBQSP_NOW = ("webqsp was UNBLOCKED by explicit user decision on 2026-09-05 (_APPROVALS.json: "
              "WEBQSP_UNBLOCKED_ROG_STANDARD_KG, approvals.webqsp.status APPROVED_ROG_STANDARD_KG, "
              "dataset_name WEBQSP_ROG_STANDARD, CORPUS_TYPE STANDARD_BENCHMARK_KG_SUBGRAPH), built on "
              "2026-09-08 (webqsp/CANONICAL_V1_BUILD.json) and FROZEN the same day in "
              "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json: 2,592,894 nodes, 4,737 queries, 8,309,195 "
              "structural edges over 7,058 relations; provenance 'V1-resolved RoG WebQSP+CWQ graph -- "
              "NEVER full Freebase'")

FAM_NOW = ("LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json (FROZEN) added the families the frozen tree "
           "lacked, under graph2/ with graph/ byte-identical: metaqa ner 5,949 edges (coverage "
           "0.1314), hotpotqa knn 11,946,289, 2wiki knn 13,643,063, webqsp ner 2,994,802 (coverage "
           "0.3067) and webqsp knn 6,470,520. Same builder contract as the frozen families; "
           "'not a uniformity fix' -- each family's coverage is recorded and must be read. "
           "pointer_resolver_v2.families(ds) unions the trees; pointer_resolver (v1) still reads "
           "graph/ only")

ENTRIES = [
    # ------------------------------------------------------------ MANIFEST.json (Track-B log)
    dict(id="MANIFEST.WEBQSP_STATUS", file=FC + "/MANIFEST.json", pointer="/WEBQSP_STATUS",
         expect="WEBQSP_BLOCKED_ON_FREEBASE_SOURCE",
         superseded_by=[APPR, SIX], what_replaced_it=WEBQSP_NOW, why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.CANONICAL_V1_STATUS", file=FC + "/MANIFEST.json", pointer="/CANONICAL_V1_STATUS",
         expect="BUILT_5_OF_6", superseded_by=[SIX],
         what_replaced_it="6 of 6 built and FROZEN on 2026-09-08 (LOCKED_6_OF_6_BENCHMARK_SUBSTRATES, "
                          "13,997,071 nodes / 877,119 queries / 112,192,284 frozen edges)",
         why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.FULL_CANONICAL_V1_STATUS", file=FC + "/MANIFEST.json",
         pointer="/FULL_CANONICAL_V1_STATUS", expect="READY_5_OF_6_SOURCE_CONTRACTS",
         superseded_by=[APPR, SIX],
         what_replaced_it="_APPROVALS.json: READY_6_OF_6_SOURCE_CONTRACTS; then FROZEN 6 of 6",
         why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.PENDING_FULL_CANONICAL", file=FC + "/MANIFEST.json",
         pointer="/PENDING_FULL_CANONICAL", expect="webqsp", superseded_by=[SIX],
         what_replaced_it="nothing is pending; " + WEBQSP_NOW, why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.corpus_totals.webqsp", file=FC + "/MANIFEST.json",
         pointer="/corpus_totals/webqsp", expect="BLOCKED -- no corpus", superseded_by=[SIX],
         what_replaced_it="2,592,894 nodes (corpus_type STANDARD_BENCHMARK_KG_SUBGRAPH)",
         why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.phase_status.PHASE_4", file=FC + "/MANIFEST.json",
         pointer="/phase_status/PHASE_4_WEBQSP_HOTPOT_BUILD", expect="webqsp NOT BUILT",
         superseded_by=[WQ_BUILD, SIX],
         still_true_part="the hotpotqa and 2wiki figures (5,233,329 and 5,989,847) and the fact that "
                         "scratchpad/final_canonical_build/build_kb.py refuses webqsp: the webqsp "
                         "build ran in its own lane (webqsp/CANONICAL_V1_BUILD.json, source "
                         "webqsp/v1), not through that builder",
         what_replaced_it=WEBQSP_NOW, why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.phase_status.PHASE_5", file=FC + "/MANIFEST.json",
         pointer="/phase_status/PHASE_5_INTEGRITY_AND_LOCK", expect="LOCK NOT DONE",
         superseded_by=[APPR, SIX],
         still_true_part="the rule itself held: no 6-of-6 record was written while webqsp was "
                         "blocked. The terminal record is named LOCKED_6_OF_6_BENCHMARK_SUBSTRATES "
                         "(not plain LOCKED_6_OF_6) exactly because five datasets are full "
                         "canonical universes and webqsp is a standard benchmark KG subgraph "
                         "(_APPROVALS.json _do_not_mark_locked_until)",
         what_replaced_it="the lock was done on 2026-09-08 after the block was lifted",
         why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.approval_gate_note", file=FC + "/MANIFEST.json", pointer="/approval_gate_note",
         expect="must not be written while webqsp is BLOCKED_PENDING_FREEBASE_SOURCE",
         superseded_by=[APPR, SIX],
         still_true_part="per-dataset approval read from _APPROVALS.json; the all-or-nothing file "
                         "is obsolete and was never created",
         what_replaced_it="the condition in the last sentence no longer holds: " + WEBQSP_NOW,
         why_not_edited=HISTORICAL_BUILD_LOG),
    dict(id="MANIFEST.blocked.webqsp.status", file=FC + "/MANIFEST.json",
         pointer="/blocked/webqsp/status", expect="BLOCKED_PENDING_FREEBASE_SOURCE",
         superseded_by=[APPR, SIX], what_replaced_it=WEBQSP_NOW, why_not_edited=HISTORICAL_BUILD_LOG),
    # ------------------------------------------------------------ webqsp/status.json (pinned)
    dict(id="webqsp.status.status", file=FC + "/webqsp/status.json", pointer="/status",
         expect="BLOCKED_PENDING_FREEBASE_SOURCE", superseded_by=[APPR, SIX],
         what_replaced_it=WEBQSP_NOW, why_not_edited=HASH_PINNED_ARTIFACT),
    dict(id="webqsp.status.WEBQSP_STATUS", file=FC + "/webqsp/status.json", pointer="/WEBQSP_STATUS",
         expect="WEBQSP_BLOCKED_ON_FREEBASE_SOURCE", superseded_by=[APPR, SIX],
         what_replaced_it=WEBQSP_NOW, why_not_edited=HASH_PINNED_ARTIFACT),
    dict(id="webqsp.status.current_substrate_is_final_canonical", file=FC + "/webqsp/status.json",
         pointer="/current_substrate_is_final_canonical", expect="false", superseded_by=[SIX],
         still_true_part="the 781,485-node experimental substrate this sentence is about is still "
                         "not canonical; it is legacy",
         what_replaced_it="the final canonical webqsp substrate is data/final_canonical/webqsp "
                          "(WEBQSP_ROG_STANDARD, 2,592,894 nodes), frozen in "
                          "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES",
         why_not_edited=HASH_PINNED_ARTIFACT),
    # ------------------------------------------------------------ webqsp verification (pinned)
    dict(id="webqsp.CANONICAL_V1_VERIFICATION.graph.knn.why",
         file=FC + "/webqsp/CANONICAL_V1_VERIFICATION.json", pointer="/graph/knn/why",
         expect="not built", superseded_by=[FAM1],
         still_true_part="true of the frozen graph/ tree, which is byte-identical",
         what_replaced_it="webqsp/graph2/knn.npz, 6,470,520 edges, coverage 1.0 -- " + FAM_NOW,
         why_not_edited=HASH_PINNED_ARTIFACT),
    dict(id="webqsp.CANONICAL_V1_VERIFICATION.graph.ner.why",
         file=FC + "/webqsp/CANONICAL_V1_VERIFICATION.json", pointer="/graph/ner/why",
         expect="no document prose to run NER over", superseded_by=[FAM1],
         still_true_part="no edge was invented; the family was built by the same spaCy contract "
                         "over the resolved node text, and two thirds of webqsp nodes receive no "
                         "NER edge (coverage 0.3067) -- read the coverage, not the edge count",
         what_replaced_it="webqsp/graph2/ner.npz, 2,994,802 edges -- " + FAM_NOW,
         why_not_edited=HASH_PINNED_ARTIFACT),
    # ------------------------------------------------------------ LOCKED_6_OF_6 (frozen)
    dict(id="LOCKED_6_OF_6.webqsp.edge_families_absent.knn", file=SIX,
         pointer="/webqsp/edge_families_absent/knn", expect="not built", superseded_by=[FAM1],
         still_true_part="true of the frozen tree the record describes",
         what_replaced_it="webqsp/graph2/knn.npz, 6,470,520 edges -- " + FAM_NOW,
         why_not_edited=FROZEN_SELF_HASHED),
    dict(id="LOCKED_6_OF_6.webqsp.edge_families_absent.ner", file=SIX,
         pointer="/webqsp/edge_families_absent/ner", expect="No edge was invented",
         superseded_by=[FAM1],
         still_true_part="'No edge was invented to match the text datasets' remains true",
         what_replaced_it="webqsp/graph2/ner.npz, 2,994,802 edges, coverage 0.3067 -- " + FAM_NOW,
         why_not_edited=FROZEN_SELF_HASHED),
    dict(id="LOCKED_6_OF_6.KNOWN_LIMITATIONS.0", file=SIX, pointer="/KNOWN_LIMITATIONS/0",
         expect="WebQSP carries the structural family ONLY", superseded_by=[FAM1],
         still_true_part="'No edge was invented to make the six uniform' remains true; graph/ is "
                         "byte-identical",
         what_replaced_it=FAM_NOW, why_not_edited=FROZEN_SELF_HASHED),
    # ------------------------------------------------------------ EDGE_FAMILY_MATRIX (pinned)
    dict(id="EDGE_FAMILY_MATRIX.matrix", file=FC + "/EDGE_FAMILY_MATRIX.json", pointer="/matrix",
         expect='"ner": null', superseded_by=[FAM1, FAM2, UKB],
         still_true_part="every non-null count is the frozen family's count and is still exact for "
                         "graph/; musique knn 266,488 is the frozen file, still pinned by "
                         "LOCKED_5_OF_5",
         what_replaced_it="served topology has all 18 families: the nulls are filled by graph2/ "
                          "(FAMILY_COMPLETION_V1) and musique knn is SERVED from graph2/knn.npz "
                          "(265,366 edges, FAMILY_COMPLETION_V2 supersession); "
                          "UKB_COMMON_MANIFEST.json carries the served per-family counts",
         why_not_edited=HASH_PINNED_ARTIFACT),
    # ------------------------------------------------------------ V3_SOURCE_ASSESSMENT (pinned)
    dict(id="webqsp.V3_SOURCE_ASSESSMENT.RECOMMENDED_SEQUENCE.0",
         file=FC + "/webqsp/V3_SOURCE_ASSESSMENT.json", pointer="/RECOMMENDED_SEQUENCE/0",
         expect="Do NOT download the 31.3 GB dump yet", superseded_by=[RAW_LOCK, RAW_ACQ, ACQ, CANON_VAL],
         still_true_part="the ORDER it recommended is the order that was followed: IDIR first, "
                         "then the mirror",
         what_replaced_it="the mirror was acquired under explicit user approval with provenance "
                          "locked before transfer (md5 38d8e9cb..., sha1 d1950f3c..., 31,305,093,084 "
                          "bytes), checksummed, and streamed into CRAG_FREEBASE_CANONICAL "
                          "(301,977,131 nodes / 2,062,430,072 edges, ALL_CHECKS_PASS)",
         why_not_edited=HASH_PINNED_ARTIFACT),
    dict(id="webqsp.V3_SOURCE_ASSESSMENT.no_downloads_performed",
         file=FC + "/webqsp/V3_SOURCE_ASSESSMENT.json", pointer="/no_downloads_performed",
         expect="true", superseded_by=[RAW_ACQ, ACQ],
         still_true_part="true at the time of writing, which is the claim it makes",
         what_replaced_it="the IDIR package and the raw mirror were both downloaded afterwards, each "
                          "under its own approval and provenance record",
         why_not_edited=HASH_PINNED_ARTIFACT),
    # ------------------------------------------------------------ FAMILY_COMPLETION_V1 (frozen)
    dict(id="FAMILY_COMPLETION_V1.musique.family_source.knn", file=FAM1,
         pointer="/datasets/musique/family_source/knn", expect="frozen", superseded_by=[FAM2, MUS_G2],
         still_true_part="V1's frozen_knn_self_consistency measurement (4,402 edges > 1e-2, max "
                         "0.50) stands and is the reason for the supersession",
         what_replaced_it="'graph2': musique/graph2/knn.npz (265,366 edges, every edge audited "
                          "against the pointer-resolved vectors, max abs err 5.96e-07) supersedes "
                          "the stale frozen file by declaration in "
                          "musique/graph2/GRAPH_MANIFEST.json#supersedes.knn; the frozen file stays "
                          "byte-identical and pinned by LOCKED_5_OF_5. Precedence rule: frozen wins "
                          "unless graph2 declares supersedes.<family>",
         why_not_edited=FROZEN_SELF_HASHED),
    # ------------------------------------------------------------ Freebase authority queue
    dict(id="freebase_v3.V3_AUTHORITY_QUEUE_RESOLUTION.IN_PROGRESS",
         file=V3 + "/V3_AUTHORITY_QUEUE_RESOLUTION.json", pointer="/IN_PROGRESS",
         expect="musicbrainz", superseded_by=[SWEEP, MB, DG, CLOSURE],
         what_replaced_it="both passes completed and were folded into the cascade before the "
                          "actual-name freeze: musicbrainz 2,346 of 2,484 distinct MBIDs resolved "
                          "(2,313 nodes; one request per 1.15 s), discogs 11,737 rows over all runs "
                          "with 242 invalid artist rows excluded by URL shape at cascade time; "
                          "final cascade 534,532 distinct named (3.0013% of the ENTITY_MID residue)",
         why_not_edited=FROZEN_SELF_HASHED),
    # ------------------------------------------------------------ HANDOFF.md (living document)
    dict(id="HANDOFF.section10.still_open", file=FC + "/HANDOFF.md",
         line_anchor="Still open on the Freebase side**: the freeze record and the name-hierarchy "
                     "contract",
         superseded_by=[INF, NH, CLOSURE],
         what_replaced_it="both were written and FROZEN on 2026-09-08: LOCKED_FREEBASE_INFERENCE_V1 "
                          "(62,022,081 admissible inferred names; 72.6685% of nodes named by tier 1 "
                          "or 2, 27.3315% resolve to the floor) and NAME_HIERARCHY_CONTRACT_V1 "
                          "(ORIGINAL_NAME -> ADMISSIBLE_INFERRED_NAME -> FLOOR). "
                          "FREEBASE_V3_CLOSURE_STATUS lists what is still open, by class",
         why_not_edited=LIVING_DOC_ANNOTATED),
    dict(id="HANDOFF.section2c.six_not_uniform", file=FC + "/HANDOFF.md",
         line_anchor="metaqa has no NER graph; hotpotqa, 2wiki and webqsp have no kNN graph; webqsp "
                     "additionally has no NER graph",
         superseded_by=[FAM1, FAM2],
         still_true_part="exact for the frozen graph/ tree and for pointer_resolver (v1); 'No edge "
                         "was invented to make the six look alike' remains true",
         what_replaced_it=FAM_NOW, why_not_edited=LIVING_DOC_ANNOTATED),
    dict(id="HANDOFF.section3.pitfall3_families", file=FC + "/HANDOFF.md",
         line_anchor="hotpotqa, 2wiki and webqsp have no kNN graph; webqsp has *only* structural",
         superseded_by=[FAM1, FAM2],
         still_true_part="the advice -- call families(ds) rather than assuming -- is unchanged; "
                         "it now applies to pointer_resolver_v2.families(ds)",
         what_replaced_it=FAM_NOW, why_not_edited=LIVING_DOC_ANNOTATED),
]

PRIOR_SUPERSESSION_RECORDS = [
    (FC + "/FROZEN_CAVEAT_SUPERSESSION.json",
     "the three frozen caveats asserting the retrieval caches cover three of six datasets"),
    (FAM2, "the stale frozen musique kNN (LOCKED_5_OF_5-pinned, byte-identical) by declaration"),
    (MUS_G2, "graph2 manifest carrying the supersedes.knn block the precedence rule honours"),
    (V3 + "/inference_admissibility_v2_amendment.json",
     "36 rows of inference_admissibility_v1, applied as an overlay"),
    (V3 + "/V3_NAME_PROVENANCE_ORDER_V2.json", "V3_NAME_PROVENANCE_ORDER_V1"),
    (V3 + "/V3_PAGE_HUNT_COMPLETE.json", "the mid-run page-hunt rate records (SUPERSESSION_CHAIN)"),
    (CLOSURE, "the authority queue's IN_PROGRESS block (STATEMENTS_THIS_RECORD_SUPERSEDES)"),
]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_pointer(doc, pointer):
    cur = doc
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(tok)]
        else:
            cur = cur[tok]
    return cur


def self_hash_check(rec):
    """Stamped vs recomputed self-hash, for records that carry one.

    Two conventions exist on disk: the package records stamp RECORD_SHA256 (verify_manifest.
    record_hash: indent=1, RECORD_SHA256 and RECORD_SHA256_LF popped, CRLF or LF); the
    freebase_v3 records stamp record_sha256 (indent=1, record_sha256 popped, LF). Both are
    recomputed here so that "frozen" is checked, not asserted.
    """
    if rec.get("RECORD_SHA256"):
        stamped = rec["RECORD_SHA256"]
        out = {"stamped_field": "RECORD_SHA256", "stamped": stamped,
               "recomputed_crlf": record_hash(rec, crlf=True),
               "recomputed_lf": record_hash(rec, crlf=False)}
        out["self_hash_reproduces"] = stamped in (out["recomputed_crlf"], out["recomputed_lf"])
        return out
    if rec.get("record_sha256"):
        stamped = rec["record_sha256"]
        body = dict(rec)
        body.pop("record_sha256")
        lf = json.dumps(body, indent=1)
        out = {"stamped_field": "record_sha256", "stamped": stamped,
               "recomputed_lf": hashlib.sha256(lf.encode("utf-8")).hexdigest(),
               "recomputed_crlf": hashlib.sha256(
                   lf.replace(chr(10), chr(13) + chr(10)).encode("utf-8")).hexdigest()}
        out["self_hash_reproduces"] = stamped in (out["recomputed_crlf"], out["recomputed_lf"])
        return out
    return None


def pin_record(p, decl):
    r = None
    try:
        r = rj(p)
    except Exception:
        pass
    out = {"path": p, "exists": os.path.exists(p),
           "sha256_of_bytes": sha_file(p) if os.path.exists(p) else None}
    if isinstance(r, dict):
        for k in ("RECORD_SHA256", "record_sha256", "RECORD_HASH", "MANIFEST_HASH"):
            if r.get(k):
                out["self_hash"] = {k: r[k]}
                break
        out["status"] = r.get("STATUS") or r.get("status")
        out["created"] = r.get("created_utc") or r.get("generated_utc") or r.get("frozen_utc")
    d = decl.get(p)
    if d:
        out["hash_pinned_by"] = {"record": d[2], "sha256": d[1], "bytes": d[0],
                                 "on_disk_matches_pin": d[1] == out["sha256_of_bytes"]}
    return out


def norm_ws(s):
    return re.sub(r"\s+", " ", s)


def main():
    decl = declared_index()[0]
    decl = {k.replace(chr(92), "/"): v for k, v in decl.items()}
    entries = []
    files = {}
    fails = []
    for e in ENTRIES:
        p = e["file"]
        if p not in files:
            files[p] = {"sha256_of_bytes": sha_file(p)}
            d = decl.get(p)
            if d:
                files[p]["hash_pinned_by"] = {"record": d[2], "sha256": d[1], "bytes": d[0],
                                              "on_disk_matches_pin": d[1] == files[p]["sha256_of_bytes"]}
                if p.endswith("HANDOFF.md") and d[1] != files[p]["sha256_of_bytes"]:
                    files[p]["hash_pinned_by"]["expected_drift"] = (
                        "LOCKED_5_OF_5 hashes the 5-of-5-era HANDOFF.md (24,471 bytes); the "
                        "document must change whenever the package is extended, so this pin is "
                        "reported as DOCS_DRIFT by design (HANDOFF.md section 1 box; "
                        "verify_manifest counts it under expected_drift, not bad)")
            if p.endswith(".json"):
                rec = rj(p)
                files[p]["record_name"] = rec.get("RECORD") or rec.get("name") or rec.get("schema")
                files[p]["self_hash"] = self_hash_check(rec)
                files[p]["_doc"] = rec
            else:
                files[p]["_doc"] = io.open(p, encoding="utf-8").read()
        doc = files[p]["_doc"]
        out = {"id": e["id"], "file": p}
        if "pointer" in e:
            try:
                val = resolve_pointer(doc, e["pointer"])
            except (KeyError, IndexError, TypeError):
                fails.append("%s: pointer %s does not resolve" % (e["id"], e["pointer"]))
                continue
            text = json.dumps(val, ensure_ascii=False)
            if e["expect"] not in text:
                fails.append("%s: statement at %s does not contain %r" % (e["id"], e["pointer"], e["expect"]))
                continue
            out["json_pointer"] = e["pointer"]
            out["superseded_statement_VERBATIM"] = val
        else:
            anchor = e["line_anchor"]
            hit = norm_ws(anchor) in norm_ws(doc)
            if not hit:
                fails.append("%s: line anchor not found in %s" % (e["id"], p))
                continue
            lines = doc.split("\n")
            ln = next((i + 1 for i, l in enumerate(lines) if anchor[:40] in l), None)
            out["line_anchor"] = anchor
            out["line_number_at_index_time"] = ln
            out["superseded_statement_VERBATIM"] = anchor
            out["struck_through_in_place"] = ("~~" in (lines[ln - 1] if ln else "")) or any(
                "~~" in lines[i] for i in range(max(0, (ln or 1) - 3), min(len(lines), (ln or 1) + 3)))
        if e.get("still_true_part"):
            out["still_true_part"] = e["still_true_part"]
        out["what_replaced_it"] = e["what_replaced_it"]
        out["superseded_by"] = [pin_record(q, decl) for q in e["superseded_by"]]
        for q in out["superseded_by"]:
            if not q["exists"]:
                fails.append("%s: superseding record missing: %s" % (e["id"], q["path"]))
        out["why_not_edited"] = e["why_not_edited"]
        entries.append(out)
    for p in files:
        files[p].pop("_doc", None)
    if fails:
        for f in fails:
            print("FAIL", f)
        sys.exit("index refused: %d statement(s) could not be verified in place" % len(fails))

    prev = None
    if os.path.exists(OUT):
        old = rj(OUT)
        prev = {"path": OUT, "sha256_of_bytes": sha_file(OUT), "created_utc": old.get("created_utc"),
                "RECORD_SHA256": old.get("RECORD_SHA256"), "n_entries": len(old.get("entries") or [])}

    by_class = {}
    for x in entries:
        by_class.setdefault(x["why_not_edited"].split(":")[0], []).append(x["id"])
    rec = {
        "RECORD": "SUPERSEDED_STATEMENTS_INDEX",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_what": "every status sentence in this package that is still readable on disk but has been "
                 "overtaken by a later record. Quoted verbatim from where it sits (resolved at index "
                 "time -- an entry whose statement is not there refuses to be written), with the "
                 "record that replaced it pinned by sha256, the current fact, and the reason the "
                 "original was not edited.",
        "HOW_TO_USE": "before quoting a status string from any file listed under `files`, look its "
                      "pointer up here. If it is listed, the sentence is history, not state; read "
                      "`what_replaced_it` and the pinned record. If it is not listed and the file is "
                      "frozen or hash-pinned, it is current as far as this package knows.",
        "CONVENTION": "amendment by new record, never by edit (HANDOFF.md section 11). Self-hashed "
                      "records are checked here by recomputing their own hash "
                      "(verify_manifest.record_hash); hash-pinned artifacts are checked against the "
                      "freeze that pins them (verify_manifest.declared_index).",
        "SUPERSEDES_PREVIOUS_INDEX": prev,
        "n_entries": len(entries),
        "entries_by_reason": by_class,
        "files": files,
        "entries": entries,
        "PRIOR_SUPERSESSION_RECORDS": [dict(pin_record(p, decl), supersedes=what)
                                       for p, what in PRIOR_SUPERSESSION_RECORDS],
        "NOT_LISTED_ON_PURPOSE": [
            "webqsp/status.json lane_status_2026_09_06.lanes.WEBQSP_ROG_MID.status = "
            "UNRESOLVED_SOURCE: still accurate. The NSM lead was tested and returned "
            "SIBLING_EXTRACTION (NSM_UNION_RESULT.json VERDICT 'NOT RESOLVED'); the accepted lane is "
            "WEBQSP_ROG_STANDARD (RoG union REPRODUCED, V1-resolved names), not ROG_MID.",
            "frozen GRAPH_MANIFEST.json families.<fam>.present=false with reason 'no graph_<fam>.tsv "
            "in the <ds> tree': a true statement about the frozen tree, which is byte-identical.",
            "the corpus table in HANDOFF.md section 1: its edge columns are the FROZEN families and "
            "sum to LOCKED_6_OF_6 n_edges by design (HANDOFF_CLAIMS_CHECK checks exactly that); a "
            "note under the table points at the served counts.",
            "LOCKED_5_OF_5 and LOCKED_6_OF_6 retrieval-cache caveats: already superseded by "
            "FROZEN_CAVEAT_SUPERSESSION.json (listed under PRIOR_SUPERSESSION_RECORDS).",
        ],
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("%d entries over %d files" % (len(entries), len(files)))
    for p, v in files.items():
        pin = v.get("hash_pinned_by")
        sh = v.get("self_hash")
        print("  %-62s pinned_by=%-22s pin_ok=%s self_hash_ok=%s"
              % (p.replace(FC + "/", ""), pin["record"] if pin else "-",
                 pin["on_disk_matches_pin"] if pin else "-",
                 sh["self_hash_reproduces"] if sh else "-"))
    for k, v in by_class.items():
        print("  %-40s %d" % (k, len(v)))
    print("wrote %s  RECORD_SHA256 %s" % (OUT, rec["RECORD_SHA256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
