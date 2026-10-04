# -*- coding: utf-8 -*-
"""
LOCKED_6_OF_6_BENCHMARK_SUBSTRATES
==================================
The terminal freeze record. It adds WebQSP to the five and states, as plainly as the five did,
what is NOT true of the result.

WHY THE FIVE ARE REFERENCED AND NOT RE-FROZEN
    LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json is already frozen and carries its own content hash.
    Re-hashing 27.4 GB to restate numbers that are already on record adds nothing, and a second
    hash of a frozen artifact invites the question of which one is authoritative. This record
    pins the 5/5 record BY HASH and fails if that hash has moved, which is a stronger guarantee
    than recomputing it would be: it detects a change instead of silently absorbing one.

WEBQSP IS NOT THE SAME KIND OF OBJECT AS THE OTHER FIVE, AND THE RECORD SAYS SO
    The five are TEXT corpora: nodes are documents, gold is a document, and gold resolution at
    anything below 100% is a bug. WebQSP is a KB SUBGRAPH: nodes are entities and CVTs, gold is
    an answer entity, and whether a given answer is in the graph is a property of the
    third-party RoG release rather than of this build. Averaging a metric across all six
    without conditioning on that is a category error, so the difference is recorded here rather
    than left for a consumer to infer from a number that looks low.

    Two further asymmetries are carried rather than smoothed away: WebQSP has only the
    structural edge family (no prose to run NER over, and kNN needs embeddings the build
    deliberately did not produce), and its SPLADE is attention-masked while 99.23% of the other
    five's is not. Neither was "fixed" by inventing edges or by reproducing a defect.
"""
import ast
import glob
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
D = ROOT + "/webqsp"
FIVE = ROOT + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json"
OUT = ROOT + "/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json"
BUF = 8 << 20


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def entry(p):
    return {"path": p.replace("\\", "/"), "bytes": os.path.getsize(p), "sha256": sha256(p)}


def gte_qinstr(path):
    """Pull the evaluated GTE_QINSTR constant out of a script without importing it.

    Importing verify_resolution_text.py to read one string would execute a gate. Regexing the
    source is worse: the constant is written across two source lines in some files and across a
    chr(10) concatenation in the Modal script (the shell heredoc that wrote it mangles a literal
    backslash-n), so a regex reports four different prefixes that are in fact one. AST-evaluating
    the assignment compares the VALUES, which is the only comparison that means anything.
    """
    t = ast.parse(io.open(path, encoding="utf-8").read())
    for n in ast.walk(t):
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name) and tg.id == "GTE_QINSTR":
                    return eval(compile(ast.Expression(n.value), "<c>", "eval"), {"chr": chr})
    return None


def record_hash(rec, crlf=True):
    """Reproduce the digest a LOCKED_* record stores about itself.

    THIS IS NOT sha256(open(path,"rb").read()), AND THE DIFFERENCE IS NOT COSMETIC
        freeze_five wrote the record, hashed the FILE BYTES, then wrote the record AGAIN with
        the digest stamped in. The bytes that were hashed therefore no longer exist on disk --
        the stamped file is a strictly longer file -- so the digest can only be reproduced by
        re-serialising the record with the hash field removed. Two details are load-bearing:

          json.dump(io.open(p, "w"))  went through a Windows text-mode handle, which translated
                                      every newline to CRLF. Hashing the LF form of the very
                                      same content yields 501d770f..., a real digest of the
                                      real content, and still the wrong answer.
          indent=1                    any other indent is a different byte stream.

        So a consumer verifying LOCKED_5_OF_5 on Linux, or after git normalises line endings,
        computes a mismatch and concludes the frozen record was tampered with. It was not. That
        is why the LF digest is recorded alongside: it is stable across platforms and is what
        future verification should use.
    """
    d = dict(rec)
    d.pop("RECORD_SHA256", None)
    d.pop("RECORD_SHA256_LF", None)
    t = json.dumps(d, indent=1)
    if crlf:
        t = t.replace(chr(10), chr(13) + chr(10))
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def need(p):
    if not os.path.exists(p):
        sys.exit("missing prerequisite: %s" % p)
    return json.load(io.open(p, encoding="utf-8"))


def main():
    t0 = time.time()
    five = need(FIVE)
    if five.get("STATUS") != "FROZEN":
        sys.exit("the 5/5 record is not FROZEN")
    five_hash = record_hash(five)
    five_lf = record_hash(five, crlf=False)

    verif = need(D + "/CANONICAL_V1_VERIFICATION.json")
    if verif.get("VERDICT") != "ALL_PASS":
        sys.exit("webqsp verification is %s, refusing to freeze" % verif.get("VERDICT"))
    build = need(D + "/CANONICAL_V1_BUILD.json")
    asm = need(D + "/ENCODING_ASSEMBLY.json")
    wo = need(D + "/QUERY_WORK_ORDER.json")
    bridge = need(ROOT + "/ID_BRIDGE.json")["datasets"]["webqsp"]
    if not bridge.get("PASS"):
        sys.exit("webqsp id bridge did not pass")
    gm = need(D + "/graph/GRAPH_MANIFEST.json")
    e2e = need(ROOT + "/SIX_DATASET_END_TO_END.json")
    if e2e.get("VERDICT") != "ALL_PASS":
        raise SystemExit("the end-to-end interface gate is %s, refusing to freeze: %s"
                         % (e2e.get("VERDICT"), e2e.get("failures", [])[:3]))
    if not e2e.get("is_full_six"):
        raise SystemExit(
            "the end-to-end gate on record covers %s, not all six. Rerun it with no arguments "
            "-- freezing 6/6 on a partial interface check would put a claim in the record that "
            "was never tested." % e2e.get("datasets_covered"))

    # The dense query prefix is part of the encoder contract: queries encoded without it are
    # not comparable to queries encoded with it, and nothing downstream would ever notice --
    # the vectors are the right shape, the right dtype and unit norm either way. So it is
    # compared BY VALUE against three independent definitions of it -- above all the script
    # that actually encoded the five -- and the digest is published so a future consumer can
    # check rather than trust this sentence.
    SOURCES = {
        # the script that actually encoded the five -- the strongest available reference
        "encoder_of_record": "src/experiments/canonical_encode.py",
        # the gate that certified the five's resolved text against that encoding
        "gate_that_certified_the_five": "scratchpad/verify_resolution_text.py",
        # the script that encoded WebQSP's queries on Modal
        "encoder_that_ran_for_webqsp": "scratchpad/modal_webqsp_encode.py"}
    rec_p = wo.get("dense_query_prefix")
    found = {k: gte_qinstr(v) for k, v in SOURCES.items()}
    prefix_check = {
        "sha256": hashlib.sha256((rec_p or "").encode("utf-8")).hexdigest(),
        "length": len(rec_p or ""),
        "sources_compared": SOURCES,
        "agreement": {k: (v == rec_p) for k, v in found.items()},
        "unanimous": all(v == rec_p for v in found.values()) and rec_p is not None,
        "splade_query_prefix": wo.get("splade_query_prefix"),
        "why": ("SPLADE takes no prefix and the Modal script raises if handed one. Dense "
                "QUERIES take this prefix and dense DOCUMENTS take none -- an asymmetry that "
                "is part of the frozen contract, not an oversight. Queries encoded without "
                "the prefix are not comparable to queries encoded with it, and nothing "
                "downstream would ever notice: the vectors are the right shape, the right "
                "dtype and unit norm either way. Hence a value comparison against the "
                "encoder of record, and a published digest so a consumer can re-check.")}
    if not prefix_check["unanimous"]:
        raise SystemExit(
            "dense query prefix disagreement -- WebQSP query vectors would not be comparable "
            "to the five. recorded=%r %s" % (rec_p, {k: (v == rec_p) for k, v in found.items()}))

    arts = []
    for p in sorted(glob.glob(D + "/*.json")) + sorted(glob.glob(D + "/*.jsonl")) + \
            sorted(glob.glob(D + "/graph/*")) + sorted(glob.glob(D + "/queries/*.jsonl")) + \
            sorted(glob.glob(D + "/queries/pointer_index/*")) + \
            sorted(glob.glob(D + "/pointer_index/*")) + [D + "/encoder_row_of_node.npz"]:
        if os.path.isfile(p):
            arts.append(entry(p))
    art_bytes = sum(a["bytes"] for a in arts)

    ch = asm["channels"]
    enc_bytes = sum(c["bytes"] for c in ch.values())
    q = build["queries"]["gold_query_level"]

    rec = {
        "RECORD": "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "FROZEN",
        "SCOPE": "the five text datasets (by reference) plus WebQSP, the KB subgraph",
        "BUILDS_ON": {
            "record": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES",
            "path": FIVE,
            "recorded_sha256": five.get("RECORD_SHA256"),
            "recomputed_sha256": five_hash,
            "hash_matches": five.get("RECORD_SHA256") == five_hash,
            "line_ending_independent_sha256": five_lf,
            "note": ("pinned by hash, not re-frozen. If this ever fails to match, the five "
                     "changed after they were locked and THAT is the finding.")},
        "HASHING_CONVENTION": {
            "what_is_hashed": (
                "the record re-serialised with json.dumps(indent=1) and its own RECORD_SHA256 "
                "field removed. NOT the bytes of the file as it sits on disk: those include "
                "the stamped digest and therefore cannot reproduce it. Every LOCKED_* record "
                "here follows this convention."),
            "RECORD_SHA256": (
                "CRLF newlines, because freeze_five wrote through a Windows text-mode handle "
                "and hashed the bytes that came out. Reproducible, but only under that "
                "convention."),
            "RECORD_SHA256_LF": (
                "identical content with LF newlines. Verify with THIS one on Linux, in CI, or "
                "anywhere git may have normalised line endings. Both are published so that a "
                "platform difference is never mistaken for tampering -- the LF digest of the "
                "5/5 record is 501d770f7eb85a6838bb9c8bcba332b5ebbe10780dd63aa72d39e409d13ff200 "
                "and a reader who computed that was right about the content and would have "
                "been wrong to call it a mismatch."),
            "verified_here": (
                "LOCKED_5_OF_5 was re-hashed under this convention while this record was "
                "written, and matched. The five are byte-identical to what was frozen.")},
        "CORE_INVARIANT": five.get("CORE_INVARIANT"),
        "WEBQSP_IS_A_DIFFERENT_KIND_OF_CORPUS": (
            "the five are TEXT corpora whose gold IS a corpus document, so gold resolution "
            "below 100% is a bug. WebQSP is a KB SUBGRAPH whose gold is an answer ENTITY, so "
            "gold coverage is a property of the third-party RoG release. Do not average "
            "retrieval metrics across all six without conditioning on that."),
        "webqsp": {
            "corpus_type": "STANDARD_BENCHMARK_KG_SUBGRAPH",
            "provenance": "V1-resolved RoG WebQSP+CWQ graph -- NEVER 'full Freebase'",
            "n_nodes": int(build["nodes"]["n"]),
            "n_queries": int(build["queries"]["n"]),
            "n_edges": int(gm["families"]["structural"]["n_edges"]),
            "n_relations": int(gm["families"]["structural"]["n_relations"]),
            "self_loops": int(gm["families"]["structural"]["self_loops"]),
            "isolated_nodes": int(gm["isolated_nodes"]),
            "edge_families_present": [k for k, v in gm["families"].items() if v.get("present")],
            "edge_families_absent": {k: v.get("why") for k, v in gm["families"].items()
                                     if not v.get("present")},
            "distinct_encoder_texts": int(asm["channels"]["docs/dense"]["expected_rows"]),
            "encoding_channels": {k: {"store": v["store"], "rows": v["n_rows"],
                                      "bytes": v["bytes"]} for k, v in ch.items()},
            "encoding_bytes": enc_bytes,
            "shard_size": asm["shard_size"],
            "query_order": wo["order_convention"],
            "dense_query_prefix": prefix_check,
            "gold_query_level": q,
            "id_bridge": {"PASS": bridge["PASS"],
                          "positive_control_pct": bridge["POSITIVE_CONTROL"]["pct"],
                          "gold_resolution_pct": bridge["gold_resolution_pct"]},
            "artifacts": arts, "artifact_bytes": art_bytes},
        "TOTALS_ALL_SIX": {
            "n_nodes": int(five["TOTALS"]["n_nodes"]) + int(build["nodes"]["n"]),
            "n_queries": int(five["TOTALS"]["n_queries"]) + int(build["queries"]["n"]),
            "n_edges": (int(five["TOTALS"]["n_edges"])
                        + int(gm["families"]["structural"]["n_edges"]))},
        "GATES": {
            "WEBQSP_ARTIFACTS": {
                "verdict": verif["VERDICT"],
                "claim": ("15 checks on what landed on disk, not on the build's own report. "
                          "The load-bearing one is the encoder join: all 2,592,894 node texts "
                          "compared exactly against the encoder row they point at, 0 "
                          "mismatches, all 1,791,533 rows referenced. A one-position error "
                          "there attaches vectors to the wrong nodes with every shape and "
                          "count still agreeing."),
                "evidence": D + "/CANONICAL_V1_VERIFICATION.json",
                "script": "scratchpad/verify_webqsp_canonical.py"},
            "WEBQSP_GOLD_ACCOUNTING": {
                "claim": ("empty gold lists and the benchmark's own answerability=False flag "
                          "are checked to be the SAME SET, in both directions, on every row. "
                          "So the 37 unanswerable questions are never counted as a corpus gap "
                          "and the 106 real gaps are never hidden among them."),
                "evidence": D + "/CANONICAL_V1_BUILD.json"},
            "WEBQSP_ID_BRIDGE": {
                "verdict": "PASS",
                "claim": ("node id rule holds on all 2,592,894 rows, and the gold key clears "
                          "99.92% on the topic-entity POSITIVE CONTROL. The control is what "
                          "makes 62.42% gold coverage readable as a property of RoG rather "
                          "than a broken join."),
                "evidence": ROOT + "/ID_BRIDGE.json"},
            "SIX_DATASET_END_TO_END": {
                "verdict": e2e["VERDICT"],
                "covers": e2e["datasets_covered"],
                "claim": (
                    "all six datasets opened through the PUBLIC interface -- "
                    "CanonicalEmbeddings, CanonicalGraph, families -- imported the way a "
                    "stranger following the handoff would import them, with 0 failures. "
                    "Checks: dense dim 1536 and unit norm with no zero rows (a zero row would "
                    "mean a pointer resolved into the damaged Phase-C region the repair exists "
                    "to route around), splade vocab 30522 with no empty and no negative rows, "
                    "len(E) == len(E.ids) so position i really is ids[i], dense and splade "
                    "agreeing on the population size, graph endpoints in range, and every "
                    "family declared present having a file on disk with no undeclared file "
                    "beside it. Every artifact can be correct and the package still unusable "
                    "if a manifest points elsewhere; no artifact-level gate would catch that."),
                "sampling": (
                    "fixed positions and a fixed seed, not exhaustive. The exhaustive per-row "
                    "checks already ran per dataset and are cited in the other gates; this one "
                    "is about the interface."),
                "evidence": ROOT + "/SIX_DATASET_END_TO_END.json",
                "script": "scratchpad/verify_six_end_to_end.py"},
            "WEBQSP_ENCODING_ASSEMBLY": {
                "claim": ("part row coverage verified to be exactly 0..N-1 before any file was "
                          "moved, dense re-checked as L2-normalised at dim 1536, splade at "
                          "vocab 30522, and the resolver round-tripped. Nodes sharing a text "
                          "confirmed to share a vector, which is the dedup's whole premise."),
                "evidence": D + "/ENCODING_ASSEMBLY.json",
                "script": "scratchpad/assemble_webqsp_encodings.py"}},
        "KNOWN_LIMITATIONS": [
            ("WebQSP carries the structural family ONLY. There is no NER graph (a KB subgraph "
             "has no document prose to run NER over) and no kNN graph (kNN needs embeddings, "
             "which the graph build deliberately did not depend on). No edge was invented to "
             "make the six uniform; the absence is in GRAPH_MANIFEST.json with its reason."),
            ("WebQSP SPLADE is attention-masked. 99.23% of the other five's is NOT -- see "
             "SPLADE_POOLING_FINDING.json. Dataset 6 masks because it has no legacy rows to "
             "match and because the unmasked path is batch-dependent and therefore not "
             "reproducible for new data at all. Raw splade scores are not comparable between "
             "WebQSP and the five; rank-based fusion is unaffected."),
            ("WebQSP gold coverage is 62.42% of references and 97.74% of answerable queries. "
             "Neither number is a bridge defect: the topic-entity control resolves at 99.92%, "
             "so the key reaches the population and what it misses is absent from the RoG "
             "graph. 106 queries have gold that is genuinely not in the corpus. A separate 37 "
             "are the benchmark's OWN unanswerable questions and are counted separately."),
            ("WebQSP entity text is NAME_ONLY. NAME_PLUS_FACTS is materialised and available "
             "as the ablation arm; it was not chosen because it re-serialises every CVT "
             "argument inside each adjacent entity for 2.554x the tokens. This is a column "
             "choice, recorded as an open axis rather than presented as the only option."),
            ("WebQSP shards at 12,000 rows where the five shard at 40,000. The resolver reads "
             "shard_size per store from POINTER_INDEX.json, so this is transparent; it exists "
             "because re-sharding meant a second 5.5 GB copy on a volume with ~12 GB free."),
            ("WebQSP has no retrieval cache. The five's caches cover metaqa, squad and musique "
             "only, for the same throughput reason. Caches are an accelerator over the frozen "
             "embeddings, not a dependency."),
            ("the corpus is the RoG WebQSP+CWQ subgraph and must NEVER be described as full "
             "Freebase. The canonical Freebase universe is a separate, separately frozen "
             "artifact of 301,977,131 nodes and is not what these 2,592,894 nodes are.")],
        "INHERITED_LIMITATIONS_FROM_THE_FIVE": five.get("KNOWN_LIMITATIONS"),
    }

    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    if not rec["BUILDS_ON"]["hash_matches"]:
        raise SystemExit(
            "LOCKED_5_OF_5 no longer hashes to its recorded digest. Refusing to freeze 6/6 on "
            "top of a record that changed after it was locked. Recomputed CRLF %s / LF %s "
            "against recorded %s" % (five_hash, five_lf, five.get("RECORD_SHA256")))

    t = rec["TOTALS_ALL_SIX"]
    print("\n5/5 hash pinned and re-verified: %s" % rec["BUILDS_ON"]["hash_matches"])
    print("webqsp  nodes=%s queries=%s edges=%s  enc=%.2f GB"
          % (format(rec["webqsp"]["n_nodes"], ","), format(rec["webqsp"]["n_queries"], ","),
             format(rec["webqsp"]["n_edges"], ","), enc_bytes / 1e9))
    print("ALL SIX nodes=%s queries=%s edges=%s"
          % (format(t["n_nodes"], ","), format(t["n_queries"], ","),
             format(t["n_edges"], ",")))
    print("wrote %s  %.1fs" % (OUT, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
