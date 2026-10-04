"""FREEBASE_SCALE stage 1 -- exact statistics of the served Freebase graph, read-only and streaming.

What it answers (nothing here partitions, embeds or routes):
  * |V|, |E| (directed, as served), self loops, directed non-self edges D, unique undirected pairs F
    (== the STRUCT key set the six datasets' H4 build uses: min*N+max, self loops dropped, unique),
    so D/F is the multiplicity the undirected view collapses;
  * out / in / undirected degree distributions (exact below 4,096, log2 bins above), the top hubs,
    and how much of the pin mass sits in high-degree anchors;
  * the H4_SPLIT_PRESERVE size of the STRUCT-only hypergraph (hyperedge = closed neighbourhood of an
    anchor, so pins = N_anchor + 2F exactly), and for a K grid (cap = round(N/K)) how many hyperedges
    the split rule has to chop and how many pins they hold;
  * the relation histogram (type-mirror share = rel 779840 `type.type.instance`).

The frozen tree is opened with mmap and never written.  The node range is cut into chunks of about
CHUNK_EDGES directed edge endpoints (a single hub bigger than that is a chunk of its own); a chunk
merges each node's out and in neighbour lists, drops self loops and unique-sorts (u*N+v), which is
the same key contract as CanonicalDataset.ukeys.

  python scratchpad/_fbx_stats.py smoke <out_dir> [n_chunks]      # first n chunks, output outside the repo
  python scratchpad/_fbx_stats.py run [workers]                   # results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json (write-once)
"""
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("FBX_ROOT") or os.getcwd()
TREE = "data/final_canonical/freebase"
OUT = "results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json"
CHUNK_EDGES = int(os.environ.get("FBX_CHUNK", 32_000_000))
EXACT_MAX = 4096
MIRROR_REL = 779840
KGRID = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 3019771]      # 3,019,771 = N // 100, the frozen native k
HUB_T = [100, 1000, 10000, 100000, 1000000]
NBIN = 34
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def L(name):
    return np.load(os.path.join(TREE, "graph", name), mmap_mode="r")


def log2bin(x):
    b = np.zeros(len(x), np.int64)
    p = x > 0
    b[p] = np.floor(np.log2(x[p].astype(np.float64))).astype(np.int64) + 1
    return np.minimum(b, NBIN - 1)


def plan(N):
    """chunk boundaries by cumulative (out + in) endpoints, computed slice by slice (never the whole 2.4 GB x 2 at once)"""
    oi, ii = L("out_indptr.npy"), L("in_indptr.npy")
    tot = int(oi[N]) + int(ii[N])
    targets = np.arange(CHUNK_EDGES, tot, CHUNK_EDGES, dtype=np.int64)
    cuts = [0]
    S = 20_000_000
    for s in range(0, N, S):
        e = min(N + 1, s + S + 1)
        c = np.asarray(oi[s:e], dtype=np.int64) + np.asarray(ii[s:e], dtype=np.int64)
        lo = np.searchsorted(targets, c[0], side="left")
        hi = np.searchsorted(targets, c[-1], side="right")
        for t in targets[lo:hi]:
            cuts.append(s + int(np.searchsorted(c, t, side="left")))
    cuts.append(N)
    cuts = sorted(set(min(max(int(x), 0), N) for x in cuts))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def work(ab):
    a, b = ab
    N = int(NN)
    oi, ii = L("out_indptr.npy"), L("in_indptr.npy")
    od, isrc = L("out_dst.npy"), L("in_src.npy")
    orl = L("out_rel.npy")
    o = np.asarray(oi[a:b + 1]); i = np.asarray(ii[a:b + 1])
    o0, o1, i0, i1 = int(o[0]), int(o[-1]), int(i[0]), int(i[-1])
    nn = b - a
    odeg, ideg = np.diff(o), np.diff(i)
    ids = np.arange(a, b, dtype=np.int64)
    dst = np.asarray(od[o0:o1]).astype(np.int64)
    src = np.asarray(isrc[i0:i1]).astype(np.int64)
    uo = np.repeat(ids, odeg)
    ui = np.repeat(ids, ideg)
    mo, mi = dst != uo, src != ui
    self_loops = int((~mo).sum())
    self_loops_in = int((~mi).sum())
    keys = np.concatenate([uo[mo] * N + dst[mo], ui[mi] * N + src[mi]])
    del uo, ui, dst, src
    keys.sort()
    if len(keys):
        keep = np.empty(len(keys), bool)
        keep[0] = True
        np.not_equal(keys[1:], keys[:-1], out=keep[1:])
        keys = keys[keep]
    und = np.bincount(keys // N - a, minlength=nn).astype(np.int64)
    del keys
    rel = np.asarray(orl[o0:o1])
    rc = np.bincount(rel, minlength=NREL).astype(np.int64)
    del rel
    anch = und >= 1
    res = {
        "a": a, "b": b, "nodes": nn, "out_edges": o1 - o0, "in_edges": i1 - i0,
        "self_loops": self_loops, "self_loops_in": self_loops_in,
        "directed_nonself": (o1 - o0) - self_loops,
        "und_sum": int(und.sum()), "n_anchor": int(anch.sum()), "n_isolated": int((~anch).sum()),
        "n_out0": int((odeg == 0).sum()), "n_in0": int((ideg == 0).sum()),
        "hist_und": np.bincount(log2bin(und), minlength=NBIN).astype(np.int64).tolist(),
        "hist_out": np.bincount(log2bin(odeg), minlength=NBIN).astype(np.int64).tolist(),
        "hist_in": np.bincount(log2bin(ideg), minlength=NBIN).astype(np.int64).tolist(),
        "exact_und": np.bincount(np.minimum(und, EXACT_MAX), minlength=EXACT_MAX + 1).astype(np.int64).tolist(),
        "exact_out": np.bincount(np.minimum(odeg, EXACT_MAX), minlength=EXACT_MAX + 1).astype(np.int64).tolist(),
        "exact_in": np.bincount(np.minimum(ideg, EXACT_MAX), minlength=EXACT_MAX + 1).astype(np.int64).tolist(),
        "hub_thr": {str(t): [int((und > t).sum()), int(und[und > t].sum())] for t in HUB_T},
        "oversize": {str(k): [int((und + 1 > max(1, int(round(N / k)))).sum()),
                              int((und[und + 1 > max(1, int(round(N / k)))] + 1).sum())] for k in KGRID},
        "rel_counts": rc,
        "mirror_edges": int(rc[MIRROR_REL]) if MIRROR_REL < len(rc) else 0,
        "top": [],
    }
    top = np.argsort(-und)[:25]
    res["top"] = [[int(a + t), int(und[t]), int(odeg[t]), int(ideg[t])] for t in top if und[t] > 0]
    return res


def _init(n, nrel):
    global NN, NREL
    NN, NREL = n, nrel
    os.chdir(ROOT)


def quantile_exact(hist, total, q):
    """smallest value x with cumulative >= q*total (exact below EXACT_MAX; the last bin means '>= EXACT_MAX')"""
    c = np.cumsum(hist)
    x = int(np.searchsorted(c, q * total, side="left"))
    return x if x < EXACT_MAX else ">=%d" % EXACT_MAX


def aggregate(rs, N, E, nrel):
    S = lambda k: int(sum(r[k] for r in rs))
    A = lambda k: np.sum([np.array(r[k], dtype=np.int64) for r in rs], axis=0)
    und_sum = S("und_sum")
    n_anchor = S("n_anchor")
    D = S("directed_nonself")
    F = und_sum // 2
    assert und_sum % 2 == 0, "undirected degree sum must be even"
    assert S("nodes") == N and S("out_edges") == E and S("in_edges") == E, (S("nodes"), S("out_edges"), S("in_edges"))
    rc = np.sum([r["rel_counts"] for r in rs], axis=0)
    top = sorted([t for r in rs for t in r["top"]], key=lambda t: -t[1])[:25]
    pins = n_anchor + und_sum
    out = {
        "nodes": N, "directed_edges_as_served": E,
        "self_loops": S("self_loops"), "self_loops_in_side": S("self_loops_in"),
        "directed_nonself_edges": D, "unique_undirected_pairs_F": F,
        "mean_multiplicity_D_over_F": round(D / F, 4) if F else None,
        "anchors_deg_ge_1": n_anchor, "isolated_nodes": S("n_isolated"),
        "nodes_without_out_edges": S("n_out0"), "nodes_without_in_edges": S("n_in0"),
        "mean_und_degree": round(und_sum / N, 4),
        "H4_STRUCT_only": {
            "hyperedges_precap": n_anchor, "pins_precap": pins,
            "formula": "pins = anchors + sum(und_degree) = anchors + 2F (closed neighbourhood per anchor; the split rule keeps every pin and adds the anchor once per extra chunk)",
            "pins_int32_eidx_bytes": pins * 4,
        },
        "degree_quantiles_exact": {},
        "hist_log2_bin0_is_zero_binb_is_2^(b-1)_to_2^b-1": {"und": A("hist_und").tolist(), "out": A("hist_out").tolist(), "in": A("hist_in").tolist()},
        "hub_pin_mass": {},
        "oversize_hyperedges_by_K": {},
        "top_hubs_pos_und_out_in": top,
        "relations_with_edges": int((rc > 0).sum()), "relations_total": nrel,
        "mirror_rel_779840_edges": int(rc[MIRROR_REL]),
        "mirror_share_of_edges": round(int(rc[MIRROR_REL]) / E, 6),
        "top_relations_id_count": [[int(r), int(rc[r])] for r in np.argsort(-rc)[:12]],
    }
    for nm, key in (("und", "exact_und"), ("out", "exact_out"), ("in", "exact_in")):
        h = A(key)
        out["degree_quantiles_exact"][nm] = {("p%g" % (100 * q)): quantile_exact(h, N, q) for q in (0.5, 0.9, 0.99, 0.999)}
    for t in HUB_T:
        c = sum(r["hub_thr"][str(t)][0] for r in rs)
        m = sum(r["hub_thr"][str(t)][1] for r in rs)
        out["hub_pin_mass"]["und_degree_>%d" % t] = {"anchors": int(c), "pins": int(m), "share_of_und_degree_sum": round(m / und_sum, 6)}
    for k in KGRID:
        c = sum(r["oversize"][str(k)][0] for r in rs)
        m = sum(r["oversize"][str(k)][1] for r in rs)
        out["oversize_hyperedges_by_K"][str(k)] = {"cap_round_N_over_K": int(round(N / k)), "oversize_anchors": int(c), "pins_in_oversize": int(m),
                                                   "share_of_pins": round(m / pins, 6)}
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    os.chdir(ROOT)
    DS = json.load(io.open(os.path.join(TREE, "DATASET.json"), encoding="utf-8"))
    N, E, nrel = int(DS["n_nodes"]), int(DS["n_edges"]), int(DS["n_relations"])
    assert L("out_indptr.npy").shape[0] == N + 1 and L("out_dst.npy").shape[0] == E
    chunks = plan(N)
    log("N %d  E %d  chunks %d (about %d endpoints each)" % (N, E, len(chunks), CHUNK_EDGES))
    if mode == "smoke":
        out_dir, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
        chunks = chunks[:n]
        workers = 1
    elif mode == "run":
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        if os.path.exists(OUT):
            raise SystemExit("refusing: %s exists (write-once)" % OUT)
    else:
        raise SystemExit(__doc__)
    rs = []
    t0 = time.time()
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init, initargs=(N, nrel)) as pool:
            for j, r in enumerate(pool.imap_unordered(work, chunks)):
                rs.append(r)
                if j % 10 == 0 or j == len(chunks) - 1:
                    log("chunk %d/%d  nodes %d  und_sum %d" % (j + 1, len(chunks), r["nodes"], r["und_sum"]))
    else:
        _init(N, nrel)
        for j, ab in enumerate(chunks):
            r = work(ab)
            rs.append(r)
            log("chunk %d/%d  [%d,%d)  endpoints %d  und_sum %d  (%.1fs)" % (j + 1, len(chunks), ab[0], ab[1], r["out_edges"] + r["in_edges"], r["und_sum"], time.time() - t0))
    if mode == "smoke":
        S = lambda k: int(sum(r[k] for r in rs))
        log("smoke: nodes %d out %d in %d self %d und_sum %d anchors %d mirror %d" % (S("nodes"), S("out_edges"), S("in_edges"), S("self_loops"), S("und_sum"), S("n_anchor"), S("mirror_edges")))
        os.makedirs(out_dir, exist_ok=True)
        return
    agg = aggregate(rs, N, E, nrel)
    rec = {
        "stage": "FBX_STATS", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DEVELOPMENT (systems statistics only; nothing partitioned, embedded or routed)",
        "input": {"tree": TREE, "DATASET_json_RECORD_SHA256": DS["RECORD_SHA256"],
                  "graph_files_verified_by": "results/FREEBASE_SCALE/FBX_HOST_VERIFY__v1.json (verify_freebase --full on this host)"},
        "method": {"chunk_endpoints": CHUNK_EDGES, "chunks": len(chunks), "workers": workers,
                   "und_rule": "per node: unique(out_dst U in_src) minus itself == the STRUCT undirected key set (CanonicalDataset.ukeys)"},
        "wall_seconds": round(time.time() - t0, 1),
        "code": {"scratchpad/_fbx_stats.py": sha_file(os.path.join(ROOT, "scratchpad/_fbx_stats.py"))},
        "stats": agg,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    log("wrote", OUT, sha_file(OUT))


if __name__ == "__main__":
    main()
