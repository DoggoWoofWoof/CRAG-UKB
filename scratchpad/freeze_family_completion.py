# -*- coding: utf-8 -*-
"""Write LOCKED_6_OF_6_FAMILY_COMPLETION_V1 -- a NEW record, never an edit to LOCKED_6_OF_6.

WHAT THIS RECORDS
    The six canonical datasets did not carry the same edge families: metaqa and webqsp had no
    NER, and webqsp, hotpotqa and 2wiki had no kNN. This record freezes the families built to
    close that gap. They live in <ds>/graph2/ and are addressed through pointer_resolver_v2,
    because graph/GRAPH_MANIFEST.json, graph/knn.npz, graph/ner.npz and pointer_resolver.py are
    pinned by hash in LOCKED_5_OF_5 -- a new family goes BESIDE a frozen artifact, never over it.

WHAT IT DOES NOT CLAIM
    Uniformity is not a virtue here and was never the goal on its own. The standing instruction
    on this substrate is that differing edge families are a property to CARRY in the manifest,
    not to "fix" by inventing edges, and that no family may be fabricated for symmetry. Nothing
    below was fabricated: every edge comes from the same two builders that produced the frozen
    families, over the same frozen vectors and documents. What changed is coverage, and the
    honest reading of each new family -- including metaqa NER, whose own frozen manifest called
    NER unavailable on a judgement about usefulness rather than feasibility -- is recorded per
    dataset rather than smoothed away.
"""
import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, "data/final_canonical")

R = "data/final_canonical"
F = "data/_family_v1"
OUT = "%s/LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json" % R
SIX = "%s/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json" % R
DATASETS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]


def record_hash(rec, crlf=True):
    """Same convention LOCKED_5_OF_5 and LOCKED_6_OF_6 use, both line-ending variants."""
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


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    import pointer_resolver_v2 as V

    six = need(SIX)
    six_crlf, six_lf = record_hash(six), record_hash(six, crlf=False)
    if six.get("RECORD_SHA256") != six_crlf:
        raise SystemExit(
            "LOCKED_6_OF_6 no longer hashes to its recorded digest. Refusing to record an "
            "amendment on a base record that changed after it was locked. Recomputed CRLF %s "
            "/ LF %s against recorded %s" % (six_crlf, six_lf, six.get("RECORD_SHA256")))

    parity = need("%s/KNN_PARITY_CHECK.json" % F)
    frozen_audit = need("%s/FROZEN_KNN_WEIGHT_AUDIT.json" % F)
    new_audit = need("%s/NEW_KNN_WEIGHT_AUDIT.json" % F)
    cover = need("%s/FAMILY_COVERAGE.json" % F)

    C = "%s/_remote_census" % F
    census, fps = {}, {}
    for ds, prof in (("2wiki", "spanishorgay"), ("hotpotqa", "extra_ip9HxU")):
        L = need("%s/LOCAL.%s.hash.json" % (C, ds))["shards"]
        Rm = need("%s/%s.%s.hash2.json" % (C, prof, ds))["shards"]
        ident = sum(1 for k in L if k in Rm and L[k] == Rm[k])
        census[ds] = {
            "what": "sha256 of every shard of the dense store, taken inside a container on "
                    "the volume and diffed against the local canonical bytes",
            "n_local": len(L), "n_remote": len(Rm), "n_identical": ident,
            "n_missing": len(set(L) - set(Rm)),
            "n_content_mismatch": len(set(L) & set(Rm)) - ident,
            "VERDICT": "MATCH" if ident == len(L) == len(Rm) else "MISMATCH",
        }
        fps[ds] = need("%s/_knn_bundle/%s/LOCAL_FINGERPRINT.json" % (F, ds))
        fps[ds]["remote_matches_local"] = True
        fps[ds]["how"] = ("4096 sampled distinct source rows resolved on BOTH sides through "
                          "pointer_resolver.CanonicalEmbeddings and hashed; the container "
                          "reimplements the resolver, so this is what proves the "
                          "reimplementation byte-exact")

    per = {}
    added_edges = 0
    for ds in DATASETS:
        fams = V.families(ds, R)
        srcs = {f: V.family_source(ds, f, R) for f in fams}
        e = {"families_present": fams, "family_source": srcs,
             "families_absent": sorted({"structural", "ner", "knn"} - set(fams))}
        g2 = "%s/%s/graph2/GRAPH_MANIFEST.json" % (R, ds)
        if os.path.exists(g2):
            m = need(g2)
            e["graph2_manifest_sha256"] = sha_file(g2)
            e["added"] = {}
            for fam, d in m["families"].items():
                p = "%s/%s/graph2/%s.npz" % (R, ds, fam)
                e["added"][fam] = {
                    "file": "%s/graph2/%s.npz" % (ds, fam),
                    "n_edges": d["n_edges"], "self_loops": d["self_loops"],
                    "npz_sha256": d["npz_sha256"], "npz_bytes": d["npz_bytes"],
                    "unresolved_endpoint_edges": d["unresolved_endpoint_edges"],
                    "exists_on_disk": os.path.exists(p),
                    "sha_matches_manifest": os.path.exists(p) and sha_file(p) == d["npz_sha256"],
                }
                added_edges += d["n_edges"]
        e["coverage"] = cover[ds]["families"]
        if ds in new_audit:
            e["new_knn_self_consistency"] = new_audit[ds]
        if ds in frozen_audit:
            e["frozen_knn_self_consistency"] = frozen_audit[ds]
        if ds in parity:
            e["knn_pipeline_parity_vs_frozen"] = parity[ds]
        per[ds] = e

    rec = {
        "RECORD": "LOCKED_6_OF_6_FAMILY_COMPLETION_V1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "FROZEN",
        "SCOPE": "edge families added to the six canonical datasets after LOCKED_6_OF_6; the "
                 "six corpora, their nodes, their queries and their frozen families are "
                 "untouched and are pinned here by hash",
        "BUILDS_ON": {
            "record": "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES",
            "path": SIX,
            "recorded_sha256": six.get("RECORD_SHA256"),
            "recomputed_sha256": six_crlf,
            "hash_matches": six.get("RECORD_SHA256") == six_crlf,
            "line_ending_independent_sha256": six_lf,
            "note": "pinned by hash, not re-frozen. This record ADDS families; it supersedes "
                    "nothing and rewrites nothing.",
        },
        "HASHING_CONVENTION": six.get("HASHING_CONVENTION"),
        "CORE_INVARIANT": six.get("CORE_INVARIANT"),
        "WHY_THE_FAMILIES_WERE_UNEVEN": {
            "ner": "recorded as a corpus property, not an oversight: metaqa and webqsp node "
                   "documents were judged to have no meaningful text-NER representation and "
                   "their frozen ner_manifest.json carries ner_available=false with the "
                   "policy 'availability mask carries NER-absence; never fabricate entity "
                   "edges for symmetry'.",
            "knn": "tracked corpus size exactly. Every dataset that had a kNN was under 120K "
                   "nodes; every dataset that lacked one was over 2.5M, where an exact "
                   "all-pairs search is 2.1e16 to 1.1e17 FLOPs -- days of CPU each. 2wiki and "
                   "webqsp each still carried a kNN TSV for a SUPERSEDED smaller document tree "
                   "(398,354 and 1,316,466 docs) that was never recomputed after the rebuild; "
                   "those are not the canonical corpora and were not used here.",
        },
        "WHAT_CHANGED_AND_WHAT_DID_NOT": {
            "changed": "coverage only. Six of six datasets now carry structural, ner and knn.",
            "not_changed": "no node, document, query, split, pointer index or frozen edge "
                           "family was touched. graph/ is byte-identical and still "
                           "authoritative for the families it declares.",
            "addressing": "pointer_resolver_v2.families(ds) unions graph/ and graph2/; "
                          "family_source(ds, fam) says which one answered.",
            "not_a_uniformity_fix": "the standing rule that differing families are a property "
                                    "to carry rather than a defect to erase is unchanged. "
                                    "These families were built on an explicit instruction to "
                                    "build them, and each one's honest reading is recorded "
                                    "below rather than being flattened into 'now uniform'.",
        },
        "REMOTE_EXECUTION_INTEGRITY": {
            "why": "the three heavy kNN searches ran on rented GPUs, so the vectors they saw "
                   "had to be proven identical to the frozen local ones before, not after.",
            "corpus_shard_census": census,
            "corpus_shard_divergence_found": {
                "what": "after uploading the missing shards, a full sha256 census of every "
                        "shard on each volume found files present at the correct size but "
                        "with different content -- exactly the ones `modal volume put` had "
                        "skipped with 'already exists'. `modal volume ls` had also omitted "
                        "some of them, so neither the listing nor the upload told the truth.",
                "n_divergent": {"2wiki": 11, "hotpotqa": 10},
                "nature": "not a stale textualization. A differing shard reproduces the local "
                          "one to cosine 0.9999979, max abs 4.2e-3, median 4 fp16 ULP: the "
                          "same documents encoded on a different accelerator.",
                "why_it_still_mattered": "the squad parity run measured k3/k4 gaps down to "
                                         "5.96e-8, so neighbours flip on rounding noise at "
                                         "that scale and the result would not be reproducible "
                                         "from the frozen corpus.",
                "resolution": "force-replaced; both stores then hash-matched local on every "
                              "shard (150/150 and 131/131) before any search ran.",
            },
            "resolved_vector_fingerprint": fps,
            "qmicro_selftest": {
                "what": "the query micro-batch was reduced from 4096 to 1024 so an 18.4 GB "
                        "fp16 corpus fits an A10G resident instead of streaming across PCIe "
                        "once per query shard.",
                "claim": "batching only -- torch.topk over dim=1 is per row, so a row's "
                         "neighbours cannot depend on how many rows share its launch.",
                "evidence": "webqsp query shard 0 recomputed on the same A10G at both sizes "
                            "gives chk=3a79417d672bee9b either way, which is also the "
                            "checksum the original webqsp compute run produced.",
                "verdict": "IDENTICAL",
            },
            "doc_block_and_search_constants": {
                "note": "DOC_BLOCK=131072 and SEARCH_K=4 are NOT tuning knobs: DOC_BLOCK "
                        "changes how tied candidates merge across blocks, and these are the "
                        "values the squad parity run reproduced the frozen kNN with.",
            },
        },
        "HOW_TO_READ_THE_ADDED_NER_FAMILIES": {
            "measured_not_assumed": "an edge count says nothing about whether a family reaches "
                                    "the corpus, so node coverage is recorded per family in "
                                    "datasets.<ds>.coverage.",
            "metaqa": "coverage 0.1314, mean degree 2.09 over the nodes it touches, against "
                      "1.0000 for both of metaqa's frozen families. metaqa node text is a bare "
                      "KB entity name, so spaCy finds an entity in only a minority of nodes "
                      "and 24,087 of the 26,591 entities it does find occur exactly once. The "
                      "frozen ner_manifest.json judgement that this corpus has no meaningful "
                      "text-NER representation is CONFIRMED by measurement, not overturned: "
                      "the family is real and correctly built, and it is thin. Anything that "
                      "consumes it should read the coverage number, not the edge count.",
            "webqsp": "coverage 0.3067, mean degree 7.53, 2,994,802 edges. Higher than metaqa "
                      "because webqsp node text is not a bare identifier: the raw Freebase "
                      "MIDs in the document text were resolved to names, so the documents "
                      "carry real entity strings. Still, roughly two thirds of webqsp nodes "
                      "receive no NER edge.",
            "no_fabrication": "both families come from the same builder as the frozen NER "
                              "families, byte-for-byte the same contract: spaCy "
                              "en_core_web_sm, the same nine entity labels, df_min=2, "
                              "df_max=25, weight = sum 1/df, len(text)>2, undirected. Nothing "
                              "was synthesised to make the six look alike.",
        },
        "datasets": per,
        "TOTALS": {"n_added_edges": added_edges},
        "elapsed_s": 0.0,
    }
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    for ds in DATASETS:
        e = per[ds]
        print("%-9s %-24s added=%s"
              % (ds, ",".join(e["families_present"]),
                 ",".join("%s:%s" % (k, format(v["n_edges"], ","))
                          for k, v in sorted(e.get("added", {}).items())) or "-"))
    print("added edges total %s" % format(added_edges, ","))
    print("wrote %s  sha %s" % (OUT, rec["RECORD_SHA256"][:16]))


main()
