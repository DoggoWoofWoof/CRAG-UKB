"""W0 Task 9 (assemble C) + Task 10 (partitions) at WORLD scale.

Topology-C = structural_native/derived  ∪  NER shares-entity (text corpora)  ∪
Qwen semantic k=3 kNN  — the exact frozen composition (variant_C reuses A's
synthetic_qwen_edges). Endpoints throughout are canonical_doc_id; node index = the
documents.jsonl canonical order (aligns with dense/splade shards).

Partitioning = the FROZEN builder src/core/indexers.build_partition_map VERBATIM:
  pymetis.part_graph(n_nodes // target_per_partition, adjacency=[neighbors(i)])
with target_per_partition = 100 (frozen variant_C: 2wiki 65865 -> 658 parts,
mean 100.1 / min 97 / max 103). P_MAIN=50 stays -> ~5k scope/query (scale-invariant).

Writes: data/canonical/<ds>/partition_map_C.json   (canonical_doc_id -> part_id)
        data/canonical/<ds>/partition_manifest_C.json
Usage:  python scratchpad/_w0_world_partition.py <ds> [--target 100]
"""
import os, sys, json, time, hashlib
import numpy as np

TARGET = 100

def load_id_index(ds):
    id_to_idx = {}
    with open(f"data/canonical/{ds}/documents.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            id_to_idx[json.loads(line)["canonical_doc_id"]] = i
    return id_to_idx

def add_tsv(path, id_to_idx, adj, stats, fam):
    if not os.path.exists(path):
        stats[fam] = "absent"; return
    n_edge = 0; n_skip = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            a = id_to_idx.get(p[0]); b = id_to_idx.get(p[1])
            if a is None or b is None:
                n_skip += 1; continue
            if a == b:
                continue
            adj[a].add(b); adj[b].add(a); n_edge += 1
    stats[fam] = dict(edges_added=n_edge, endpoints_unmapped=n_skip)

def build(ds, target=TARGET):
    import pymetis
    t0 = time.time()
    id_to_idx = load_id_index(ds)
    n = len(id_to_idx)
    adj = [set() for _ in range(n)]
    stats = {}
    base = f"data/canonical/{ds}"
    add_tsv(f"{base}/graph_structural.tsv", id_to_idx, adj, stats, "structural")
    add_tsv(f"{base}/graph_ner.tsv",        id_to_idx, adj, stats, "ner")
    add_tsv(f"{base}/graph_knn.tsv",        id_to_idx, adj, stats, "qwen_knn")
    n_undirected = sum(len(s) for s in adj) // 2
    isolated = sum(1 for s in adj if not s)
    print(f"[{ds}] n={n} undirected_edges={n_undirected} isolated={isolated} "
          f"families={ {k:(v if isinstance(v,str) else v['edges_added']) for k,v in stats.items()} }", flush=True)

    adjacency_list = [sorted(adj[i]) for i in range(n)]      # exact frozen format
    n_parts = max(1, n // target)
    n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
    membership = list(membership)

    counts = np.bincount(np.array(membership), minlength=n_parts)
    idx_to_id = [None] * n
    for cid, i in id_to_idx.items():
        idx_to_id[i] = cid
    part_map = {idx_to_id[i]: int(membership[i]) for i in range(n)}

    outp = f"{base}/partition_map_C.json"; tmp = outp + ".tmp"
    json.dump(part_map, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, outp)
    h = hashlib.sha256(open(outp, "rb").read()).hexdigest()

    man = dict(dataset=ds, topology="C", target_per_partition=target, n_nodes=n,
               n_parts=int(n_parts), n_cuts=int(n_cuts),
               n_edges_undirected=int(n_undirected), isolated_nodes=int(isolated),
               size_min=int(counts.min()), size_max=int(counts.max()),
               size_mean=round(float(counts.mean()), 3), size_median=int(np.median(counts)),
               edge_cut_ratio=round(float(n_cuts) / max(1, n_undirected), 4),
               edge_families=stats, partitioner="pymetis.part_graph (frozen build_partition_map)",
               P_MAIN=50, note="P_MAIN=50 -> ~5k scope/query, scale-invariant L1->L2 interface",
               partition_map_sha256=h, secs=round(time.time() - t0, 1))
    json.dump(man, open(f"{base}/partition_manifest_C.json", "w"), indent=2)
    print(f"[{ds}] DONE parts={n_parts} size[min{counts.min()}/mean{counts.mean():.1f}/max{counts.max()}] "
          f"cuts={n_cuts} cut_ratio={man['edge_cut_ratio']} secs={time.time()-t0:.1f} -> {outp}", flush=True)

if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    target = TARGET
    if "--target" in args:
        i = args.index("--target"); target = int(args[i + 1]); del args[i:i + 2]
    for ds in args:
        build(ds, target)
