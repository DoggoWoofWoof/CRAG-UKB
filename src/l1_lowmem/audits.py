"""Web audits for the H4_SK low-memory validation (spec sections 7 and 8) -- records only, nothing executed.

    python -u src/l1_lowmem/audits.py   -> results/L1_LOWMEM/AUDIT_MTKAHYPAR_OUTOFCORE.json, AUDIT_ZOLTAN_PHG.json
Every quote below was read from the cited source on the recorded date; the verdicts are restricted to what those sources say.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import OUT, wj, log  # noqa: E402


def mtkahypar():
    return {
        "RECORD": "AUDIT_MTKAHYPAR_OUTOFCORE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "web audit (README, mt-kahypar/io/hypergraph_io.cpp, GitHub forks listing + API, issue search); no execution",
        "question": "Does the official Mt-KaHyPar or any public fork offer TRUE out-of-core / external-memory / distributed-memory hypergraph "
                    "partitioning, i.e. an internal hypergraph that is NOT materialised in RAM?",
        "official": {
            "repo": "https://github.com/kahypar/mt-kahypar",
            "self_description": "shared-memory multilevel graph and hypergraph partitioner ... implemented in C++17, parallelized using the TBB library",
            "README_memory_statements": "none -- no out-of-core / external-memory / disk-based / distributed mode is documented; machine requirement = "
                                        "'A 64-bit Linux, MacOS, or Windows operating system'; presets default / quality / highest_quality / deterministic / "
                                        "deterministic_quality / large_k (large_k = many blocks, 'can significantly worsen the partitioning quality')",
            "file_parsing": {
                "file": "mt-kahypar/io/hypergraph_io.cpp",
                "evidence": [
                    "FileHandle handle = mmap_file(filename); ... munmap_file(handle);  (readHypergraphFile / readGraphFile)",
                    "comment: 'Sequential pass over all lines to determine ranges in the input file that are read in parallel.'",
                    "line_ranges = split_lines(mapped_file, pos, current_line, length, num_hyperedges)",
                    "copy_to_global_list(local_edges, hyperedges) -> HyperedgeVector& hyperedges, vec<HyperedgeWeight>& hyperedges_weight, "
                    "vec<HypernodeWeight>& hypernodes_weight are fully materialised in RAM"],
                "verdict": "mmap is used ONLY to parse the hMETIS text in parallel; the complete hyperedge / pin / weight vectors and then the multilevel "
                           "hierarchy live in RAM -> NOT an out-of-core algorithm (mmap-based file parsing is not out-of-core)"},
            "issues_search": "issues?q=memory OR out-of-core OR distributed OR mmap -> no issue or PR about external-memory or distributed partitioning "
                             "(hits: determinism #189/#198, MSVC #229, coarsening speed #218, flow refinement #213)"},
        "forks": {
            "count_reported_by_github": 48,
            "listing_source": "github.com/kahypar/mt-kahypar/network/members + api.github.com/repos/kahypar/mt-kahypar/forks (first page)",
            "descriptions": "every listed fork keeps the upstream 'shared-memory' description except Mykograph/negative-edge-weights "
                            "('Implementation of negative edge weights.') and szchenshixi/mt-kahypar (default branch '64bit-weight'); "
                            "dsalwasser/mt-kahypar (author of dKaMinPar) is a plain master fork with no distributed branch visible",
            "verdict": "no public fork advertises out-of-core, external-memory, streaming or distributed-memory hypergraph partitioning"},
        "related_but_not_applicable": [
            "dKaMinPar (distributed deep multilevel GRAPH partitioning, same group): graphs only, not hypergraphs, not the frozen H4_SK contract",
            "KaHyPar (sequential): same in-RAM model"],
        "VERDICT": "NO_TRUE_OUT_OF_CORE -- official Mt-KaHyPar and its forks materialise the whole hypergraph (and the coarsening hierarchy) in RAM; "
                   "only the input parsing is mmap-based. The frozen Mt-KaHyPar recipe on the big three still needs the RAM measured in the "
                   "external-lane memory record; no fork changes that.",
        "sources": ["https://github.com/kahypar/mt-kahypar",
                    "https://raw.githubusercontent.com/kahypar/mt-kahypar/master/README.md",
                    "https://raw.githubusercontent.com/kahypar/mt-kahypar/master/mt-kahypar/io/hypergraph_io.cpp",
                    "https://github.com/kahypar/mt-kahypar/network/members",
                    "https://api.github.com/repos/kahypar/mt-kahypar/forks?per_page=100&sort=newest",
                    "https://github.com/kahypar/mt-kahypar/issues?q=memory+OR+out-of-core+OR+distributed+OR+mmap"]}


def zoltan():
    return {
        "RECORD": "AUDIT_ZOLTAN_PHG", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "web audit of the official Zoltan user guide (Sandia); NOT executed -- spec section 7: execute only if FREIGHT fails feasibility or quality",
        "question": "Can Zoltan-PHG ingest the IDENTICAL canonical H4_SK hypergraph in distributed memory and partition it under the same contract "
                    "family (k = max(1, N//100), imbalance 0.03, connectivity / KM1 objective, H4 hyperedge weights, unit vertex weights)?",
        "algorithm": {"doc": "ug_alg_phg.html", "quote": "This is the built-in parallel hypergraph partitioner in Zoltan.",
                      "model": "MPI distributed memory; multilevel (PHG_MULTILEVEL default 1 for LB_APPROACH = partition / repartition); internal 2D "
                               "layout (PHG_RANDOMIZE_INPUT: 'Randomize layout of vertices and hyperedges in internal parallel 2D layout?', default 0); "
                               "coarsening PHG_COARSENING_METHOD default agg; refinement PHG_REFINEMENT_METHOD default fm; "
                               "PHG_PROCESSOR_REDUCTION_LIMIT default 0.5 ('Redistribute coarsened hypergraph to this fraction of processors')"},
        "objective": {"parameter": "PHG_CUT_OBJECTIVE", "default": "connectivity",
                      "quote": "weights each cut edge by the number of participating parts minus one (aka the lambda-1 cut metric)",
                      "verdict": "same objective family as the frozen KM1 (sum over nets of weight x (lambda - 1)) -> OBJECTIVE_COMPATIBLE"},
        "k_and_imbalance": {
            "NUM_GLOBAL_PARTS": "'The total number of parts to be generated by a call to Zoltan_LB_Partition'; default = number of processors -> must be set to k",
            "IMBALANCE_TOL": "'The amount of load imbalance the partitioning algorithm should deem acceptable.' (1.2 = 20 % OK); default 1.1 -> set 1.03",
            "NUM_LOCAL_PARTS": "default 1 per processor; superseded when NUM_GLOBAL_PARTS is set",
            "caveat": "Zoltan's tolerance is max part weight / average part weight; Mt-KaHyPar's eps 0.03 bounds max block by ceil((1+eps) N/k) -- the two "
                      "agree only up to rounding; the exact bound must be recorded on execution"},
        "distributed_ingestion": {
            "required_queries": ["ZOLTAN_NUM_OBJ_FN", "ZOLTAN_OBJ_LIST_FN", "ZOLTAN_HG_SIZE_CS_FN", "ZOLTAN_HG_CS_FN"],
            "optional_queries": ["ZOLTAN_HG_SIZE_EDGE_WTS_FN", "ZOLTAN_HG_EDGE_WTS_FN", "ZOLTAN_NUM_FIXED_OBJ_FN", "ZOLTAN_FIXED_OBJ_LIST_FN"],
            "formats": {"ZOLTAN_COMPRESSED_VERTEX": "vertex global IDs in vtxedge_GID, hyperedge global IDs in pin_GIDs "
                                                    "(= exactly the per-node net-list rows of H4_SK_STREAM_V1, global net IDs)",
                        "ZOLTAN_COMPRESSED_EDGE": "hyperedge GIDs in vtxedge_GID, vertex GIDs in pin_GIDs (= the hMETIS rows)"},
            "distribution_rule_quote": "When a hypergraph is distributed across multiple processes, Zoltan expects that all processes share a consistent "
                                       "global numbering scheme for hyperedges and vertices. Also, no two processes should return the same pin (matrix "
                                       "non-zero) in this query function. (Pin ownership is unique.)",
            "edge_weight_rule_quote": "if more than one process supplies edge weights for the same hyperedge, the different edge weights will be resolved "
                                      "according to the value of the PHG_EDGE_WEIGHT_OPERATION parameter (ADD, MAX, ERROR; default max)",
            "mapping_to_H4_SK_STREAM_V1": "rank r owns a contiguous range of shards (canonical positions): it returns those positions as its objects "
                                          "(unit weights), returns its shard rows in ZOLTAN_COMPRESSED_VERTEX format with the global net IDs, so every "
                                          "pin is returned by exactly one rank (the node owner) -> pin ownership unique; each net's H4 weight is supplied "
                                          "by one designated rank (or by every rank holding a pin with PHG_EDGE_WEIGHT_OPERATION = ERROR as a consistency "
                                          "check). The mathematical hypergraph is never sharded; only its representation is.",
            "structure_guard": "PHG_EDGE_SIZE_THRESHOLD default 0.25 OMITS hyperedges larger than 25 % of the vertices -> must be set to 1.0 (H4_SK has no "
                               "such nets, but the identity gate must not rely on that); CHECK_HYPERGRAPH = 1 for validation runs; the structure digest "
                               "must be recomputed from what the ranks actually returned",
            "verdict": "DISTRIBUTED_INGESTION_FEASIBLE from the existing shards without any new monolithic file; per-rank memory ~ its shard pins + "
                       "PHG's 2D redistribution + coarsening hierarchy (not measured here)"},
        "determinism": {"finding": "no seed / determinism parameter is documented in ug_param.html or ug_alg.html; the 2D layout and coarsening depend on "
                                   "the number of MPI ranks -> results are not expected to reproduce across rank counts; this is a contract difference "
                                   "from Mt-KaHyPar DETERMINISTIC_QUALITY seed 0 (bit-reproducible) and must be recorded, not hidden"},
        "differences_vs_canonical_recipe": ["multilevel agglomerative coarsening + FM refinement instead of Mt-KaHyPar's deterministic multilevel + flow refinement",
                                            "LB_APPROACH must be PARTITION (default REPARTITION assumes an existing distribution)",
                                            "REMAP default 1 renumbers parts (labels only; block labels are never compared)"],
        "VERDICT": "FEASIBLE_NOT_EXECUTED -- Zoltan-PHG can ingest the identical H4_SK from the H4_SK_STREAM_V1 shards in distributed memory with the same "
                   "objective family and contract values (k, 1.03, connectivity, H4 weights, unit vertex weights); held in reserve per spec section 7.",
        "sources": ["https://sandialabs.github.io/Zoltan/ug_html/ug_alg_phg.html", "https://sandialabs.github.io/Zoltan/ug_html/ug_query_lb.html",
                    "https://sandialabs.github.io/Zoltan/ug_html/ug_alg.html", "https://sandialabs.github.io/Zoltan/ug_html/ug_param.html"]}


if __name__ == "__main__":
    wj(os.path.join(OUT, "AUDIT_MTKAHYPAR_OUTOFCORE.json"), mtkahypar())
    wj(os.path.join(OUT, "AUDIT_ZOLTAN_PHG.json"), zoltan())
    log("audits written to", OUT)
