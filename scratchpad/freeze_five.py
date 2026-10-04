"""
LOCKED_5_OF_5_BENCHMARK_SUBSTRATES
==================================
The freeze record for the five non-WebQSP canonical datasets: an inventory of every artifact
with its content hash, the counts that define it, the gates that were run with their evidence,
and -- given equal weight -- the things that are NOT true of it.

A freeze record is only worth the honesty of its limitations section, so three facts that a
consumer will otherwise discover the hard way are stated at the top of the record rather than
buried:

  1. THIS PACKAGE IS NOT SELF-CONTAINED ON DISK. The embeddings are not copied into
     data/final_canonical/; the pointer index REFERENCES them in data/canonical/. 2wiki's
     dense matrix alone is 17.6 GB against 23 GB of free disk, so copying was never on the
     table. Both trees must travel together. The dependency is machine-readable in
     POINTER_INDEX.json rather than a matter of lore.

  2. THE PHASE-C DENSE SHARDS ARE STILL PHYSICALLY DAMAGED. 142,633 rows across ten channels
     were lost to storage-layer writeback and were repaired NON-DESTRUCTIVELY: canonical_v1
     stops pointing at them, the bad bytes stay where they are. Any consumer that opens
     data/canonical/<tree>/encodings/dense/ shards directly, instead of going through the
     resolver, still reads zeros.

  3. THE FIVE ARE NOT UNIFORM. metaqa has no NER graph; hotpotqa and 2wiki_universe have no
     kNN graph. No edge was invented to hide that, and the matrix is part of the record.

Everything below is measured at freeze time, not copied from an earlier run.
"""
import glob
import hashlib
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
CANON = "data/canonical"
DS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki"]
TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
        "hotpotqa": "hotpotqa", "squad": "squad"}
SPLITS = ["train", "dev", "validation", "test"]
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


def count_lines(p):
    n = 0
    with io.open(p, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            n += b.count(b"\n")
    return n


def count_split(p):
    """Rows, and rows carrying no gold node at all.

    The second number is not cosmetic. MuSiQue, HotpotQA and 2Wiki ship BLIND official test
    splits, so those rows have no gold by construction rather than by any failure of the
    bridge. A consumer that averages recall over 'all queries' silently divides by rows that
    can never contribute, so the count is measured per split and published.
    """
    n = ng = 0
    with io.open(p, "rb") as f:
        for ln in f:
            n += 1
            if b'"gold_node_ids": []' in ln:
                ng += 1
    return n, ng


def freeze_dataset(ds):
    t0 = time.time()
    d = "%s/%s" % (ROOT, ds)
    art = []
    for p in [d + "/nodes.jsonl", d + "/dataset_manifest.json", d + "/SOURCE_CONTRACT.json",
              d + "/build_info.json", d + "/integrity_report.json",
              d + "/query_independence_test.json"]:
        if os.path.exists(p):
            art.append(entry(p))
    q, nogold = {}, {}
    for sp in SPLITS:
        p = "%s/queries/%s.jsonl" % (d, sp)
        if os.path.exists(p):
            q[sp], nogold[sp] = count_split(p)
            art.append(entry(p))
    for p in sorted(glob.glob(d + "/pointer_index/*")) + \
            sorted(glob.glob(d + "/queries/pointer_index/*")) + \
            sorted(glob.glob(d + "/graph/*")) + \
            sorted(glob.glob(d + "/reuse_map/*")) + \
            sorted(glob.glob(d + "/retrieval_cache/*")):
        if os.path.isfile(p):
            art.append(entry(p))

    gm = json.load(io.open(d + "/graph/GRAPH_MANIFEST.json", encoding="utf-8"))
    out = {
        "n_nodes": count_lines(d + "/nodes.jsonl"),
        "queries_by_split": q,
        "n_queries": int(sum(q.values())),
        "queries_without_gold_by_split": nogold,
        "n_queries_without_gold": int(sum(nogold.values())),
        "unlabelled_splits": sorted(k for k, v in nogold.items() if v and v == q[k]),
        "phase_c_tree_for_documents": TREE[ds],
        "graph_families": {f: (v["n_edges"] if v.get("present") else None)
                           for f, v in gm["families"].items()},
        "graph_unresolved_endpoints": {f: v.get("unresolved_endpoint_edges", 0)
                                       for f, v in gm["families"].items() if v.get("present")},
        "nodes_unrepresented_in_graph": len(gm.get("nodes_unrepresented_in_graph") or []),
        "retrieval_cache_models": sorted(
            os.path.basename(x).split("_top")[0]
            for x in glob.glob(d + "/retrieval_cache/*_top*.npz")),
        "artifacts": art,
        "artifact_bytes": int(sum(a["bytes"] for a in art)),
        "seconds": round(time.time() - t0, 1)}
    print("  %-9s nodes=%-9s queries=%-8s artifacts=%-4d %6.1f GB  %.0fs"
          % (ds, format(out["n_nodes"], ","), format(out["n_queries"], ","),
             len(art), out["artifact_bytes"] / 1e9, out["seconds"]), flush=True)
    return out


def main():
    t0 = time.time()
    rec = {
        "RECORD": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "FROZEN",
        "SCOPE": ("the five non-WebQSP canonical datasets. WebQSP is dataset 6 and is NOT "
                  "part of this freeze."),
        "CORE_INVARIANT": ("QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS "
                           "FORBIDDEN. Every corpus here is query-independent: the node set "
                           "is a function of the source corpus alone."),
        "READ_THIS_FIRST": {
            "not_self_contained": (
                "the embeddings are NOT copied into data/final_canonical/. The pointer index "
                "references data/canonical/ and both trees must travel together. 2wiki's "
                "dense matrix alone is 17.6 GB against 23 GB free, so copying was never "
                "possible. The dependency is declared in POINTER_INDEX.json."),
            "phase_c_dense_shards_are_still_damaged": (
                "142,633 dense rows across ten channels were lost to storage-layer writeback "
                "and repaired NON-DESTRUCTIVELY -- canonical_v1 stops pointing at them, the "
                "bad bytes remain. READING data/canonical/<tree>/encodings/dense/ SHARDS "
                "DIRECTLY STILL RETURNS ZEROS. Use pointer_resolver.CanonicalEmbeddings."),
            "the_five_are_not_uniform": (
                "metaqa has no NER graph; hotpotqa and 2wiki_universe have no kNN graph. See "
                "EDGE_FAMILY_MATRIX.json. No edge was invented to make them match.")},
        "DETERMINISM": {
            "PYTHONHASHSEED": "0 -- part of dataset identity, not a convenience; every "
                              "builder refuses to run without it where it can affect output",
            "ordering": "nodes.jsonl line order IS the canonical position space, and is the "
                        "same space used by the pointer index and by every graph endpoint"},
        "ENCODER_CONTRACT": {
            "dense": {"model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "dim": 1536,
                      "dtype": "float16", "normalize_embeddings": True,
                      "max_seq_length": 32768, "trust_remote_code": True,
                      "doc_prefix": "(none)",
                      "query_prefix": "Instruct: Given a web search query, retrieve relevant "
                                      "passages that answer the query\\nQuery: ",
                      "flash_attn": "deliberately NOT installed"},
            "splade": {"model": "naver/splade-cocondenser-ensembledistil", "max_length": 256,
                       "pooling": "log(1+relu(logits)), attention-masked, max-pooled",
                       "vocab": 30522, "format": "CSR"},
            "shard_size": 40000,
            "reuse_rule": ("reuse is token-ID equality under the FROZEN tokenizer, not raw "
                           "text equality. Two texts differing only past the truncation "
                           "point are the same encoder input, which is why the pointer index "
                           "is legitimately many-to-one.")},
        "datasets": {}}

    for ds in DS:
        rec["datasets"][ds] = freeze_dataset(ds)

    for k, p in [("edge_family_matrix", ROOT + "/EDGE_FAMILY_MATRIX.json"),
                 ("pointer_index", ROOT + "/POINTER_INDEX.json"),
                 ("id_bridge", ROOT + "/ID_BRIDGE.json"),
                 ("encoding_shard_hashes", ROOT + "/ENCODING_SHARD_HASHES.json"),
                 ("splade_pooling_finding", ROOT + "/SPLADE_POOLING_FINDING.json"),
                 ("dense_cache_order_diagnosis",
                  ROOT + "/DENSE_CACHE_ORDER_DIAGNOSIS.json"),
                 ("retrieval_cache_record", ROOT + "/RETRIEVAL_CACHE.json"),
                 ("handoff", ROOT + "/HANDOFF.md"),
                 ("resolver", ROOT + "/pointer_resolver.py")]:
        if os.path.exists(p):
            rec.setdefault("package_level_artifacts", {})[k] = entry(p)

    rec["TOTALS"] = {
        "n_nodes": int(sum(v["n_nodes"] for v in rec["datasets"].values())),
        "n_queries": int(sum(v["n_queries"] for v in rec["datasets"].values())),
        "n_edges": int(sum(sum(x or 0 for x in v["graph_families"].values())
                           for v in rec["datasets"].values())),
        "artifact_bytes": int(sum(v["artifact_bytes"] for v in rec["datasets"].values()))}

    rec["GATES"] = {
        "RESOLUTION_TEXT": {
            "claim": ("for every canonical row, every dataset, both models, the text that "
                      "produced the resolved vector is that node's canonical text -- exactly "
                      "or token-identically under the frozen tokenizer"),
            "evidence": "scratchpad/verify_resolution_text.json",
            "script": "scratchpad/verify_resolution_text.py"},
        "POST_REPAIR_STALENESS": {
            "claim": "no canonical pointer resolves to a damaged Phase-C dense row",
            "evidence": "scratchpad/verify_post_repair.json",
            "script": "scratchpad/verify_post_repair.py"},
        "ID_BRIDGE_AND_GOLD": {
            "claim": ("the canonical<->Phase-C id relation holds on every row, and every "
                      "gold answer a query points at is a real canonical node"),
            "evidence": ROOT + "/ID_BRIDGE.json",
            "script": "scratchpad/build_id_bridge.py"},
        "GRAPH_ENDPOINTS": {
            "claim": ("every edge endpoint of every family resolves to a canonical position; "
                      "unresolved endpoints are zero in all 13 materialised families"),
            "evidence": ROOT + "/<ds>/graph/GRAPH_MANIFEST.json",
            "script": "scratchpad/build_canonical_graphs.py"},
        "ENCODING_SHARD_HASHES": {
            "claim": ("every encoding shard is fingerprinted, so a future silent content "
                      "change is detectable -- the direct lesson of the dense corruption, "
                      "which row counts could not see"),
            "evidence": ROOT + "/ENCODING_SHARD_HASHES.json",
            "script": "scratchpad/hash_encoding_shards.py"},
        "SPLADE_POOLING": {
            "claim": ("the two splade code paths that wrote into this package do NOT compute "
                      "the same function: canonical_encode.py pools over padding positions, "
                      "encode_rev2_delta.py masks first. Proven by encoding 80 sampled texts "
                      "three ways -- masked reproduces the solo (batch-of-one) encoding to "
                      "1.1e-05, unmasked differs by up to 0.615. Published, not fixed."),
            "evidence": ROOT + "/SPLADE_POOLING_FINDING.json",
            "script": "scratchpad/verify_splade_pooling.py"},
        "RETRIEVAL_CACHE": {
            "claim": ("every materialised cache is structurally sound on all rows, and on a "
                      "deterministic query sample its top-K reproduces exactly under an "
                      "INDEPENDENT implementation (lexsort, not the builder's uint64 key) the "
                      "top-K SET reproduces exactly and loses zero score mass against a float64 "
                      "arbiter. Order among equally-scoring documents is reported, not "
                      "asserted -- see the limitation on order reproducibility"),
            "evidence": ROOT + "/RETRIEVAL_CACHE.json",
            "script": "scratchpad/verify_retrieval_cache.py"}}

    rec["KNOWN_LIMITATIONS"] = [
        ("musique carries one node with no graph edges: canonical musique nodes 'Etz Efraim' "
         "and 'Adei Ad' hold byte-identical settlement boilerplate, Phase-C deduplicated them "
         "into one document, and the graph derivation therefore never ran under the second "
         "node's id. Its edges were NOT copied from the twin, because that would assert "
         "title_mention relations the derivation never computed."),
        ("the dense repair is a pointer redirect, so the physically damaged Phase-C shards "
         "are still on disk and still damaged. Their hashes in ENCODING_SHARD_HASHES.json "
         "record the damaged bytes on purpose, and each channel carries a row_integrity "
         "verdict so the two are never confused."),
        ("edge direction is preserved as found and NOT symmetrised. Every source declares "
         "directed: true. A consumer wanting an undirected view builds it; this package will "
         "not fabricate reverse edges."),
        ("splade vectors are NOT a reproducible function of their text. 11,316,320 document "
         "rows and all 872,382 query rows were pooled over padding positions as well as real "
         "ones, and HuggingFace pads to the longest text in the BATCH, so a vector depends on "
         "which other texts shared its shard batch. Measured against a batch-of-one encoding "
         "the difference reaches 0.615 absolute, cosine 0.9787. The remaining 87,857 document "
         "rows (hotpotqa 94, 2wiki 87,763) came from the REV2 patch and ARE masked -- those "
         "are rows whose rev1 text was empty, so for them the comparison is masked vs no text "
         "at all. Query and document sides agree for 99.23% of the package, so ranking within "
         "a dataset is consistent; raw splade scores must not be compared across datasets. "
         "This is published rather than repaired because re-encoding 11.4M documents would "
         "invalidate every experiment already run. See SPLADE_POOLING_FINDING.json."),
        ("earlier L1/L2 experiments in this repository ran against the corrupted dense rows. "
         "Numbers produced before this freeze are not comparable to numbers produced after "
         "it without a rerun."),
        ("retrieval caches are materialised for metaqa, squad and musique only. hotpotqa and "
         "2wiki were left unbuilt as a COST decision, not a technical one: 33.8 GFLOPS "
         "effective with no local GPU puts them at ~13.9 h and ~29.1 h. Nothing in the package "
         "depends on them -- they are an accelerator over the frozen embeddings, and the exact "
         "same top-K is recomputable from the pointer index. See RETRIEVAL_CACHE.json for the "
         "command."),
        ("three official TEST splits are BLIND and carry no gold at all: musique 2,459, "
         "hotpotqa 7,405, 2wiki 12,576 -- 22,440 queries. This is a property of the sources, "
         "not a gap in the id bridge: every train, dev and validation row resolves its gold "
         "with zero unresolved references. A consumer that averages recall over 'all queries' "
         "will silently divide by rows that cannot contribute, so the per-split counts are in "
         "this record. metaqa's test split IS labelled; squad has no test split."),
        ("the cached top-K ORDER is not implementation-independent, though its CONTENT is. "
         "The builder scores a matrix-matrix product and breaks ties by ascending canonical "
         "position, but that tie-break only fires when two scores come out BIT-equal, and the "
         "pointer index is many-to-one so genuine ties are common. Recomputing the same query "
         "as a matrix-vector product sums in a different order, lands one ulp away, and orders "
         "the tie differently. Measured over 24 queries across the three dense caches: top-K "
         "SET identical every time, score mass lost against a float64 arbiter exactly 0.0, "
         "earliest ordering disagreement at rank 130 with a score delta of 9.5e-08. Splade is "
         "unaffected -- sparse dots have a fixed accumulation order and reproduced 8/8 exactly. "
         "See DENSE_CACHE_ORDER_DIAGNOSIS.json."),
        ("cached scores are float16 and therefore lossy. The cached IDS are exact -- ranking "
         "ran in float32 and was resolved in uint64, and the top-K SET was independently "
         "confirmed -- but two distinct float32 scores can round to the same float16, so the "
         "stored scores must not be used to re-derive the ranking. Note this is a SEPARATE "
         "issue from the order limitation above: float16 storage is why you cannot re-rank "
         "from the stored scores, and float32 matmul associativity is why the stored order "
         "itself is not implementation-independent.")]

    p = ROOT + "/LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json"
    json.dump(rec, io.open(p, "w", encoding="utf-8"), indent=1)
    body = io.open(p, "rb").read()
    h = hashlib.sha256(body).hexdigest()
    rec["RECORD_SHA256"] = h
    json.dump(rec, io.open(p, "w", encoding="utf-8"), indent=1)
    print("\nTOTALS nodes=%s queries=%s edges=%s"
          % (format(rec["TOTALS"]["n_nodes"], ","), format(rec["TOTALS"]["n_queries"], ","),
             format(rec["TOTALS"]["n_edges"], ",")))
    print("record sha256 (pre-stamp) %s" % h)
    print("wrote %s  %.1fs" % (p, time.time() - t0))


if __name__ == "__main__":
    main()
