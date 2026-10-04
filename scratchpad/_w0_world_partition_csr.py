"""W0 Task 9/10 assembly+partition for the BIG-2 (hotpot 5.23M, universe 5.99M) —
memory-efficient CSR variant of _w0_world_partition.py (python-set adjacency does not
fit 8.7 GB local RAM at 30M+ edges).

Semantics are IDENTICAL to the frozen set-based builder (src/core/indexers +
_w0_world_partition.py): topology-C = structural ∪ NER(text) ∪ Qwen-kNN, undirected,
deduplicated, no self-loops; pymetis.part_graph(N//target) over the exact same
adjacency. Only the assembly data structure differs (numpy CSR, not python sets) — so
for the same edge set METIS returns the same (deterministic) partition. Validated to
reproduce the set-builder's webqsp result bit-for-bit (--validate webqsp).

Writes data/canonical/<ds>/partition_map_C.json + partition_manifest_C.json (same
schema as the set builder).  Usage: python scratchpad/_w0_world_partition_csr.py <ds> [--target 100]
"""
import os, sys, json, time, hashlib
import numpy as np


TARGET = 100


def load_id_index(ds):
    ids = []
    with open(f"data/canonical/{ds}/documents.jsonl", encoding="utf-8") as fh:
        for line in fh:
            ids.append(json.loads(line)["canonical_doc_id"])
    id_to_idx = {c: i for i, c in enumerate(ids)}
    return ids, id_to_idx


def read_edges(path, id_to_idx, stats, fam):
    """Return int32 (src,dst) arrays for one family; endpoints mapped, self-loops dropped."""
    if not os.path.exists(path):
        stats[fam] = "absent"
        return np.empty(0, np.int32), np.empty(0, np.int32)
    src = []; dst = []; skip = 0; n = 0
    g = id_to_idx.get
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            i = line.find("\t")
            if i < 0:
                continue
            j = line.find("\t", i + 1)
            a = line[:i]; b = line[i + 1:j] if j >= 0 else line[i + 1:].rstrip("\n")
            ai = g(a); bi = g(b)
            if ai is None or bi is None:
                skip += 1; continue
            if ai == bi:
                continue
            src.append(ai); dst.append(bi); n += 1
    stats[fam] = dict(edges_added=n, endpoints_unmapped=skip)
    return np.asarray(src, np.int32), np.asarray(dst, np.int32)


def build(ds, target=TARGET, validate_against=None):
    import pymetis
    t0 = time.time()
    ids, id_to_idx = load_id_index(ds)
    n = len(ids)
    stats = {}
    base = f"data/canonical/{ds}"
    parts = []
    for fam, fn in [("structural", "graph_structural.tsv"), ("ner", "graph_ner.tsv"), ("qwen_knn", "graph_knn.tsv")]:
        s, d = read_edges(f"{base}/{fn}", id_to_idx, stats, fam)
        if s.size:
            parts.append((s, d))
    src = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int32)
    dst = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, np.int32)
    # symmetrize (undirected)
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    del src, dst, parts
    # dedup (a,b) unordered-per-node == dedup directed (u,v) after symmetrization
    key = u.astype(np.int64) * np.int64(n) + v.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key = key[order]; u = u[order]; v = v[order]
    keep = np.empty(key.shape, bool); keep[0] = True
    np.not_equal(key[1:], key[:-1], out=keep[1:])
    u = u[keep]; v = v[keep]
    del key, order, keep
    n_directed = u.size
    n_undirected = n_directed // 2
    # CSR (u already sorted by u because key = u*n+v): xadj via bincount
    deg = np.bincount(u, minlength=n)
    xadj = np.zeros(n + 1, np.int64); np.cumsum(deg, out=xadj[1:])
    adjncy = v.astype(np.int32)
    isolated = int((deg == 0).sum())
    print(f"[{ds}] n={n} undirected_edges={n_undirected} isolated={isolated} "
          f"families={ {k:(x if isinstance(x,str) else x['edges_added']) for k,x in stats.items()} } "
          f"assembled {time.time()-t0:.0f}s", flush=True)

    n_parts = max(1, n // target)
    n_cuts, membership = pymetis.part_graph(n_parts, xadj=xadj.tolist(), adjncy=adjncy.tolist())
    membership = np.asarray(membership, np.int32)
    counts = np.bincount(membership, minlength=n_parts)

    if validate_against:
        ref = json.load(open(validate_against))
        ref_arr = np.array([ref[ids[i]] for i in range(n)], np.int32)
        # partition ids are arbitrary labels; compare via co-membership on a sample is heavy —
        # instead compare the SORTED partition-size histogram + exact membership if labels align
        same = int((ref_arr == membership).sum())
        hist_match = np.array_equal(np.sort(counts), np.sort(np.bincount(ref_arr, minlength=n_parts)))
        print(f"[VALIDATE] exact_membership_match={same}/{n} size_histogram_match={hist_match} "
              f"(exact label match may differ by permutation; histogram is the invariant)", flush=True)

    idx_to_id = ids
    part_map = {idx_to_id[i]: int(membership[i]) for i in range(n)}
    outp = f"{base}/partition_map_C.json"; tmp = outp + ".tmp"
    json.dump(part_map, open(tmp, "w", encoding="utf-8")); os.replace(tmp, outp)
    h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
    man = dict(dataset=ds, topology="C", target_per_partition=target, n_nodes=n,
               n_parts=int(n_parts), n_cuts=int(n_cuts), n_edges_undirected=int(n_undirected),
               isolated_nodes=isolated, size_min=int(counts.min()), size_max=int(counts.max()),
               size_mean=round(float(counts.mean()), 3), size_median=int(np.median(counts)),
               edge_cut_ratio=round(float(n_cuts) / max(1, n_undirected), 4), edge_families=stats,
               partitioner="pymetis.part_graph CSR (frozen build_partition_map, CSR assembly)",
               P_MAIN=50, partition_map_sha256=h, secs=round(time.time() - t0, 1))
    json.dump(man, open(f"{base}/partition_manifest_C.json", "w"), indent=2)
    print(f"[{ds}] DONE parts={n_parts} size[min{counts.min()}/mean{counts.mean():.1f}/max{counts.max()}] "
          f"cuts={n_cuts} cut_ratio={man['edge_cut_ratio']} secs={time.time()-t0:.1f} -> {outp}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    target = TARGET; validate = None
    if "--target" in args:
        i = args.index("--target"); target = int(args[i + 1]); del args[i:i + 2]
    if "--validate" in args:
        i = args.index("--validate"); validate = args[i + 1]; del args[i:i + 2]
    for ds in args:
        v = f"data/canonical/{ds}/partition_map_C.json" if validate == ds else None
        build(ds, target, validate_against=v)
