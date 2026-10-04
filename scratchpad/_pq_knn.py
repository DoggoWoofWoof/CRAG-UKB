"""FBX_SCALE Track C -- compressed-KNN calibration on WebQSP: canonical exact KNN versus KNN over product-quantised canonical vectors (addendum 6).

The deployed Freebase regime this calibrates: encode a chunk -> quantise IMMEDIATELY -> discard the float batch, so a search can only ever see DECODED vectors on both sides (query = decoded
code, database = decoded code).  The calibration therefore searches decoded vectors against decoded vectors, exhaustively (the IVF/nprobe approximation is a separate, later measurement against
this exhaustive-PQ truth: here it would only add a second error source to the one being validated).

  python -u scratchpad/_pq_knn.py TEST                       # synthetic end-to-end check (CPU, no data needed)
  python -u scratchpad/_pq_knn.py PREP                       # distinct source rows of the canonical dense docs (bitwise-equal vectors), pos2u / first_pos, write-once record
  python -u scratchpad/_pq_knn.py EXACT                      # canonical exact search (search_k 4, fp32, TF32 off) + merge; must reproduce graph/knn.npz
  python -u scratchpad/_pq_knn.py PQ <m>                     # OPQ+PQ m x 8 bit: train on a fixed sample, encode ALL distinct rows, search decoded-vs-decoded, merge, metrics, keys file

Definitions (frozen by addendum 6, written before any PQ result exists):
  distinct row   = a class of positions with bitwise-identical fp16 vectors; rows numbered in ascending order of their first position.
  search space   = the distinct rows; SEARCH_K = 4 including self; drop self; each position inherits its row's neighbours mapped to the lowest position of each neighbour (the canonical merge).
  similarity     = cosine of the (decoded) vectors, computed in fp32 after an fp32 renormalisation of the fp16 storage.
  PQ             = faiss OPQMatrix(1536, m) (niter 20) then ProductQuantizer(1536, m, 8); training sample = 131,072 distinct rows drawn with RandomState(0); everything else unseen by training.
  keys family    = undirected pair keys min*N+max, self loops dropped, KNN = pairs \\ STRUCT  (the frozen l1_canonical contract); STRUCT is copied unchanged.
Bulk output data/freebase_scale/pq/ (git-ignored); records results/FREEBASE_SCALE/pq/.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DENSE = os.path.join(REPO, "data", "final_canonical", "webqsp", "embeddings", "dense", "docs")
KNN_NPZ = os.path.join(REPO, "data", "final_canonical", "webqsp", "graph", "knn.npz")
KEYS = os.path.join(REPO, "data", "l1_canonical", "webqsp", "keys.npz")
POINTER = os.path.join(REPO, "data", "final_canonical", "_history", "materialize", "webqsp__pointer_index__dense.npz")
POINTER_SHA = "34300695707b20edff239c9269e87f14e1aaab65f1395d157680933a022f650e"      # the materialisation pointer index named by the dense docs manifest (position -> source row)
WORK = os.path.join(REPO, "data", "freebase_scale", "pq")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "pq")
DIM = 1536
SEARCH_K = 4
DOC_BLOCK = 131072
QSHARD = 65536
QMICRO = 4096
TRAIN_ROWS = 131072
OPQ_NITER = 20                  # OPQ alternations (faiss default 50); fixed by addendum 6
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def wj(path, obj):
    assert not os.path.exists(path), "write-once: %s exists" % path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def save_npy(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, path)


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


# ---------------------------------------------------------------------------------------------------------------- distinct rows
def load_dense(root=DENSE):
    man = json.load(io.open(os.path.join(root, "manifest.json"), encoding="utf-8"))
    n = int(man["n_rows"])
    X = np.empty((n, DIM), np.float16)
    r = 0
    for s in man["shards"]:
        a = np.load(os.path.join(root, s["file"]), mmap_mode="r")
        X[r:r + len(a)] = a
        r += len(a)
    assert r == n
    return X


def row_hashes(X, chunk=262144):
    """two independent 64-bit multiplicative hashes of the raw fp16 bytes of every row (bitwise equality)."""
    rng = np.random.RandomState(12345)
    r1 = rng.randint(1, 2 ** 62, size=DIM // 4, dtype=np.int64).astype(np.uint64) | np.uint64(1)
    r2 = rng.randint(1, 2 ** 62, size=DIM // 4, dtype=np.int64).astype(np.uint64) | np.uint64(1)
    h1 = np.empty(len(X), np.uint64)
    h2 = np.empty(len(X), np.uint64)
    with np.errstate(over="ignore"):
        for a in range(0, len(X), chunk):
            v = np.ascontiguousarray(X[a:a + chunk]).view(np.uint64)
            h1[a:a + chunk] = (v * r1).sum(axis=1, dtype=np.uint64)
            h2[a:a + chunk] = ((v ^ (v >> np.uint64(29))) * r2).sum(axis=1, dtype=np.uint64)
    return h1, h2


def distinct_rows(h1, h2):
    """pos2u (int32, row of every position), first (int32, lowest position of every row); rows numbered by ascending first position."""
    n = len(h1)
    o = np.lexsort((np.arange(n), h2, h1))
    new = np.ones(n, bool)
    new[1:] = (h1[o][1:] != h1[o][:-1]) | (h2[o][1:] != h2[o][:-1])
    grp = np.cumsum(new) - 1
    fpos = o[np.flatnonzero(new)]                       # lowest position of each group (o is ascending in position within a group)
    order = np.argsort(fpos, kind="stable")
    rank = np.empty(len(fpos), np.int64)
    rank[order] = np.arange(len(fpos))
    pos2u = np.empty(n, np.int32)
    pos2u[o] = rank[grp].astype(np.int32)
    return pos2u, fpos[order].astype(np.int32)


def rows_from_labels(lab):
    """rows = the classes of equal labels; pos2u (int32, row of every position), first (int32, lowest position of every row); rows numbered by ascending first position."""
    lab = np.asarray(lab)
    n = len(lab)
    o = np.lexsort((np.arange(n), lab))
    new = np.ones(n, bool)
    new[1:] = lab[o][1:] != lab[o][:-1]
    grp = np.cumsum(new) - 1
    fpos = o[np.flatnonzero(new)]
    order = np.argsort(fpos, kind="stable")
    rank = np.empty(len(fpos), np.int64)
    rank[order] = np.arange(len(fpos))
    pos2u = np.empty(n, np.int32)
    pos2u[o] = rank[grp].astype(np.int32)
    return pos2u, fpos[order].astype(np.int32)


# ---------------------------------------------------------------------------------------------------------------- search + merge (the canonical contract)
def search4(Xd, dev, qshards=None, ckpt=None):
    """exact top-SEARCH_K (incl. self) by cosine over the rows of Xd (fp16 storage, fp32 renormalise, fp32 accumulate, TF32 off); per-query-shard checkpoints when ckpt is a directory."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nu = len(Xd)
    nrm = torch.nn.functional.normalize
    nsh = (nu + QSHARD - 1) // QSHARD
    if dev != "cpu":
        while torch.cuda.mem_get_info()[0] < nu * DIM * 2 + 8 * 2 ** 30:
            log("waiting for %.1f GB of free GPU memory (free %.1f GB)" % ((nu * DIM * 2 + 8 * 2 ** 30) / 2 ** 30, torch.cuda.mem_get_info()[0] / 2 ** 30))
            time.sleep(120)
    Xg = torch.from_numpy(Xd).to(dev)
    nbr = np.full((nu, SEARCH_K), -1, np.int32)
    sc = np.zeros((nu, SEARCH_K), np.float32)
    for sh in range(nsh):
        if qshards is not None and sh not in qshards:
            continue
        r0, r1 = sh * QSHARD, min((sh + 1) * QSHARD, nu)
        cp = os.path.join(ckpt, "q%05d.npz" % sh) if ckpt else None
        if cp and os.path.exists(cp):
            d = np.load(cp)
            nbr[r0:r1], sc[r0:r1] = d["nbr"], d["score"]
            continue
        t = time.time()
        q = r1 - r0
        best_sc = torch.full((q, SEARCH_K), -1e30, device=dev)
        best_id = torch.full((q, SEARCH_K), -1, dtype=torch.long, device=dev)
        for b in range(0, nu, DOC_BLOCK):
            e = min(b + DOC_BLOCK, nu)
            Db = nrm(Xg[b:e].float(), dim=1)
            k = min(SEARCH_K, e - b)
            for qm in range(0, q, QMICRO):
                qe = min(qm + QMICRO, q)
                Qs = nrm(Xg[r0 + qm:r0 + qe].float(), dim=1)
                tile = Qs @ Db.T
                v, jj = torch.topk(tile, k, dim=1)
                cs = torch.cat([best_sc[qm:qe], v], 1)
                ci = torch.cat([best_id[qm:qe], jj + b], 1)
                nsc, o = torch.topk(cs, SEARCH_K, dim=1)
                best_sc[qm:qe] = nsc
                best_id[qm:qe] = torch.gather(ci, 1, o)
                del Qs, tile
            del Db
        nbr[r0:r1], sc[r0:r1] = best_id.cpu().numpy().astype(np.int32), best_sc.cpu().numpy()
        if cp:
            os.makedirs(ckpt, exist_ok=True)
            tmp = cp + ".tmp.npz"
            np.savez(tmp, nbr=nbr[r0:r1], score=sc[r0:r1], row_start=r0)
            os.replace(tmp, cp)
        log("  search shard %d/%d (%d rows) %.0fs" % (sh + 1, nsh, q, time.time() - t))
    return nbr, sc


def merge(nbr, sc, pos2u, first):
    """the canonical merge (modal_canonical_knn.py, stage merge): (src, dst, weight) of the undirected edge list over positions."""
    pos2u = pos2u.astype(np.int64)
    first = first.astype(np.int64)
    n = len(pos2u)
    pos = np.arange(n, dtype=np.int64)
    src_l, dst_l, w_l = [], [], []
    for c in range(SEARCH_K):
        v = nbr[pos2u, c].astype(np.int64)
        w = sc[pos2u, c]
        keep = (v >= 0) & (v != pos2u)
        j = first[v[keep]]
        i = pos[keep]
        ok = i != j
        src_l.append(i[ok])
        dst_l.append(j[ok])
        w_l.append(w[keep][ok])
    i, j, w = np.concatenate(src_l), np.concatenate(dst_l), np.concatenate(w_l)
    a, b = np.minimum(i, j), np.maximum(i, j)
    key = (a << 32) | b
    o = np.lexsort((-w, key))
    key, a, b, w = key[o], a[o], b[o], w[o]
    keep = np.ones(len(key), bool)
    keep[1:] = key[1:] != key[:-1]
    return a[keep].astype(np.int32), b[keep].astype(np.int32), w[keep].astype(np.float32)


def pair_keys(src, dst, N):
    a, b = src.astype(np.int64), dst.astype(np.int64)
    m = a != b
    return np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))


def neighbour_recall(nb_ref, nb_new, n_keep=3):
    """per distinct row: self dropped, first n_keep others; returns (mean recall@3, recall of the nearest, rows with a full set)."""
    n = len(nb_ref)
    sref, snew = nb_ref.astype(np.int64), nb_new.astype(np.int64)
    idx = np.arange(n)[:, None]

    def take(nb):
        ns = nb != idx
        # stable: first n_keep non-self entries per row
        rank = np.cumsum(ns, axis=1) * ns
        res = np.full((n, n_keep), -1, np.int64)
        for c in range(1, n_keep + 1):
            has = (rank == c)
            r, cc = np.nonzero(has)
            res[r, c - 1] = nb[r, cc]
        return res
    ref, new = take(sref), take(snew)
    hit = (ref[:, :, None] == new[:, None, :]).any(axis=2) & (ref >= 0)
    full = (ref >= 0).all(axis=1)
    r3 = float(hit[full].sum() / (full.sum() * n_keep))
    r1 = float((ref[full, 0][:, None] == new[full]).any(axis=1).mean())
    return r3, r1, int(full.sum())


# ---------------------------------------------------------------------------------------------------------------- PQ
def pq_train(Xs, m, nthreads=None):
    import faiss
    if nthreads:
        faiss.omp_set_num_threads(nthreads)
    opq = faiss.OPQMatrix(DIM, m)
    opq.verbose = False
    opq.niter = OPQ_NITER
    opq.train(Xs)
    pq = faiss.ProductQuantizer(DIM, m, 8)
    pq.train(opq.apply(Xs))
    return opq, pq


def pq_encode(opq, pq, Xd, chunk=131072):
    """codes for every row; rows are the fp16 storage renormalised in fp32 exactly as the search will."""
    m = pq.M
    codes = np.empty((len(Xd), m), np.uint8)
    for a in range(0, len(Xd), chunk):
        x = Xd[a:a + chunk].astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        codes[a:a + chunk] = pq.compute_codes(opq.apply(np.ascontiguousarray(x)))
    return codes


def pq_decode(pq, codes, chunk=131072):
    """decoded vectors (in the OPQ-rotated space; orthogonal, so cosines equal the original space's) as fp16 storage, like every other vector the search reads."""
    out = np.empty((len(codes), DIM), np.float16)
    for a in range(0, len(codes), chunk):
        out[a:a + chunk] = pq.decode(np.ascontiguousarray(codes[a:a + chunk])).astype(np.float16)
    return out


def recon_cosine(opq, pq, Xd, codes, n=100000, seed=1):
    rng = np.random.RandomState(seed)
    ids = np.sort(rng.choice(len(Xd), size=min(n, len(Xd)), replace=False))
    x = Xd[ids].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    r = pq.decode(np.ascontiguousarray(codes[ids]))
    y = opq.apply(x)
    c = (y * r).sum(1) / (np.linalg.norm(y, axis=1) * np.maximum(np.linalg.norm(r, axis=1), 1e-12))
    return {"mean": float(c.mean()), "p05": float(np.percentile(c, 5)), "min": float(c.min()), "n": int(len(ids))}


# ---------------------------------------------------------------------------------------------------------------- stages
def cmd_prep():
    out = os.path.join(RECS, "WEBQSP_DISTINCT_ROWS__v2.json")
    assert not os.path.exists(out), "write-once"
    t = time.time()
    assert sha_file(POINTER) == POINTER_SHA, "pointer index differs from the one the dense docs manifest names"
    zp = np.load(POINTER)
    assert int(zp["src"].max()) == 0 and int(zp["src"].min()) == 0, "more than one source store"
    X = load_dense()
    N = len(X)
    assert len(zp["row"]) == N
    pos2u, first = rows_from_labels(zp["row"])
    D = len(first)
    log("N %d -> %d pointer rows (%.0fs)" % (N, D, time.time() - t))
    # every position's vector is bitwise-equal to its row representative's (the manifest's sha1 proof, rechecked)
    bad = 0
    for a in range(0, N, 262144):
        seg = slice(a, min(a + 262144, N))
        bad += int((X[seg].view(np.uint16) != X[first[pos2u[seg]]].view(np.uint16)).any(axis=1).sum())
    assert bad == 0, "%d positions differ from their row representative" % bad
    # the bitwise classes of the same vectors (the first attempt's definition): how many rows does the frozen family keep apart although their vectors are bitwise equal?
    h1, h2 = row_hashes(X)
    pu_b, first_b = distinct_rows(h1, h2)
    man = json.load(io.open(os.path.join(REPO, "data", "final_canonical", "webqsp", "graph", "GRAPH_MANIFEST.json"), encoding="utf-8"))["families"]["knn"]["source"]["source_manifest"]
    assert D == man["n_distinct_source_rows"], "distinct-row count %d differs from the frozen KNN's %d" % (D, man["n_distinct_source_rows"])
    save_npy(os.path.join(WORK, "pos2u.npy"), pos2u)
    save_npy(os.path.join(WORK, "first_pos.npy"), first)
    wj(out, {"RECORD": "PQ_CALIB_WEBQSP_DISTINCT_ROWS", "version": 2, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_positions": N, "n_distinct_rows": D,
             "frozen_knn_manifest_n_distinct_source_rows": man["n_distinct_source_rows"], "equals_frozen_count": True, "positions_equal_to_their_row_representative": True,
             "definition": "rows = the classes of the frozen materialisation pointer index (data/final_canonical/_history/materialize/webqsp__pointer_index__dense.npz, sha256 pinned; src all 0), i.e. the FROZEN KNN's distinct source rows; numbered by ascending first position",
             "pointer_index_sha256": POINTER_SHA,
             "bitwise_classes_of_the_same_vectors": int(len(first_b)), "rows_kept_apart_by_the_pointer_index_although_bitwise_equal": int(D - len(first_b)),
             "supersedes": "WEBQSP_DISTINCT_ROWS__v1.json (first attempt: rows = bitwise classes, 1,791,525, which failed the count assertion against the frozen 1,791,533)",
             "files": {"pos2u": {"file": rel(os.path.join(WORK, "pos2u.npy")), "sha256": sha_file(os.path.join(WORK, "pos2u.npy"))},
                       "first_pos": {"file": rel(os.path.join(WORK, "first_pos.npy")), "sha256": sha_file(os.path.join(WORK, "first_pos.npy"))}},
             "seconds": round(time.time() - t, 1), "code_sha256": sha_file(os.path.abspath(__file__))})
    log("PREP done: %d rows (bitwise classes %d)" % (D, len(first_b)))


def dev_name():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_prep():
    pos2u = np.load(os.path.join(WORK, "pos2u.npy"))
    first = np.load(os.path.join(WORK, "first_pos.npy"))
    return pos2u, first


def cmd_exact():
    out = os.path.join(RECS, "WEBQSP_EXACT_KNN__v1.json")
    assert not os.path.exists(out), "write-once"
    t = time.time()
    pos2u, first = load_prep()
    X = load_dense()
    Xd = np.ascontiguousarray(X[first])
    del X
    dev = dev_name()
    log("exact search on %s over %d rows" % (dev, len(Xd)))
    nbr, sc = search4(Xd, dev, ckpt=os.path.join(WORK, "exact"))
    save_npy(os.path.join(WORK, "exact_nbr.npy"), nbr)
    save_npy(os.path.join(WORK, "exact_score.npy"), sc)
    a, b, w = merge(nbr, sc, pos2u, first)
    z = np.load(KNN_NPZ)
    N = len(pos2u)
    ka, kz = pair_keys(a, b, N), pair_keys(z["src"], z["dst"], N)
    same = bool(len(ka) == len(kz) and (ka == kz).all())
    miss, extra = int(len(np.setdiff1d(kz, ka, assume_unique=True))), int(len(np.setdiff1d(ka, kz, assume_unique=True)))
    wmax = None
    if same:
        za = np.lexsort((z["dst"], z["src"]))
        oa = np.lexsort((b, a))
        wmax = float(np.abs(w[oa] - z["weight"][za]).max())
    np.savez(os.path.join(WORK, "exact_edges.npz"), src=a, dst=b, weight=w)
    wj(out, {"RECORD": "PQ_CALIB_WEBQSP_EXACT_KNN", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": dev, "rows": int(len(Xd)), "edges_recomputed": int(len(a)), "edges_frozen": int(len(z["src"])),
             "pair_sets_equal": same, "frozen_pairs_missing_from_recompute": miss, "recomputed_pairs_not_in_frozen": extra, "max_abs_weight_diff_when_equal": wmax,
             "files": {"nbr": sha_arr(nbr), "score": sha_arr(sc)}, "seconds": round(time.time() - t, 1), "code_sha256": sha_file(os.path.abspath(__file__))})
    log("EXACT: pair sets equal %s (missing %d, extra %d, max |dw| %s)" % (same, miss, extra, wmax))


def cmd_pq(m):
    import faiss
    m = int(m)
    out = os.path.join(RECS, "WEBQSP_PQ%d_KNN__v1.json" % m)
    assert not os.path.exists(out), "write-once"
    t = time.time()
    pos2u, first = load_prep()
    N = len(pos2u)
    X = load_dense()
    Xd = np.ascontiguousarray(X[first])
    del X
    nu = len(Xd)
    rs = np.random.RandomState(0)
    tr = np.sort(rs.choice(nu, size=TRAIN_ROWS, replace=False))
    Xs = Xd[tr].astype(np.float32)
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    t1 = time.time()
    opq, pq = pq_train(Xs, m, nthreads=6)
    log("PQ%d trained on %d rows in %.0fs" % (m, len(Xs), time.time() - t1))
    del Xs
    codes = pq_encode(opq, pq, Xd)
    d = os.path.join(WORK, "pq%d" % m)
    os.makedirs(d, exist_ok=True)
    A = faiss.vector_to_array(opq.A).reshape(DIM, DIM)
    cb = faiss.vector_to_array(pq.centroids).reshape(m, 256, DIM // m)
    save_npy(os.path.join(d, "codes.npy"), codes)
    save_npy(os.path.join(d, "opq_A.npy"), A)
    save_npy(os.path.join(d, "pq_centroids.npy"), cb)
    rc = recon_cosine(opq, pq, Xd, codes)
    log("PQ%d encoded %d rows, %d B/row; reconstruction cosine mean %.4f p05 %.4f" % (m, nu, m, rc["mean"], rc["p05"]))
    Rd = pq_decode(pq, codes)
    del Xd
    dev = dev_name()
    nbr, sc = search4(Rd, dev, ckpt=os.path.join(d, "search"))
    del Rd
    a, b, w = merge(nbr, sc, pos2u, first)
    while not os.path.exists(os.path.join(WORK, "exact_nbr.npy")) or not os.path.exists(os.path.join(RECS, "WEBQSP_EXACT_KNN__v1.json")):
        log("waiting for the exact search (EXACT) to finish")
        time.sleep(120)
    nbx = np.load(os.path.join(WORK, "exact_nbr.npy"))
    r3, r1, nfull = neighbour_recall(nbx, nbr)
    z = np.load(KEYS)
    Ns = int(z["N"][0])
    assert Ns == N
    S = z["STRUCT"]
    K_ex = z["KNN"]                                      # the FROZEN canonical family (l1_canonical keys.npz) is the reference; the recompute's agreement with it is in WEBQSP_EXACT_KNN__v1.json
    K_pq = np.setdiff1d(pair_keys(a, b, N), S, assume_unique=True)
    inter = int(len(np.intersect1d(K_pq, K_ex, assume_unique=True)))
    kp = os.path.join(d, "keys_pq%d.npz" % m)
    np.savez(kp, N=np.array([N]), STRUCT=S, KNN=K_pq)
    np.savez(os.path.join(d, "edges_pq%d.npz" % m), src=a, dst=b, weight=w)
    wj(out, {"RECORD": "PQ_CALIB_WEBQSP_PQ_KNN", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "m": m, "bits": 8, "bytes_per_row": m, "opq": "OPQMatrix(1536, %d)" % m,
             "train_rows": TRAIN_ROWS, "train_seed": 0, "device": dev, "faiss": faiss.__version__, "distinct_rows": int(nu), "reconstruction_cosine": rc,
             "neighbour_recall_at_3_rowwise": round(r3, 6), "nearest_recall_rowwise": round(r1, 6), "rows_scored": nfull,
             "knn_family": {"pairs_exact": int(len(K_ex)), "pairs_pq": int(len(K_pq)), "pairs_common": inter, "recall_of_exact_pairs": round(inter / len(K_ex), 6), "precision_vs_exact": round(inter / len(K_pq), 6)},
             "files": {"codes": {"file": rel(os.path.join(d, "codes.npy")), "sha256": sha_file(os.path.join(d, "codes.npy"))}, "opq_A": {"sha256": sha_arr(A)}, "pq_centroids": {"sha256": sha_arr(cb)},
                       "keys": {"file": rel(kp), "sha256": sha_file(kp), "STRUCT": int(len(S)), "KNN": int(len(K_pq))}},
             "seconds": round(time.time() - t, 1), "code_sha256": sha_file(os.path.abspath(__file__))})
    log("PQ%d: recall@3 %.4f, nearest %.4f, KNN pairs %d (exact %d, common %d = %.4f)" % (m, r3, r1, len(K_pq), len(K_ex), inter, inter / len(K_ex)))


# ---------------------------------------------------------------------------------------------------------------- test
def _test():
    import tempfile
    rng = np.random.RandomState(3)
    n, d = 6000, 1536
    base = rng.randn(n, d).astype(np.float32)
    base[1000:1050] = base[0:50] + 0.05 * rng.randn(50, d).astype(np.float32)        # near neighbours
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    X = base.astype(np.float16)
    dup = rng.randint(0, n, 500)
    X = np.concatenate([X, X[dup]])                                                      # exact duplicates
    perm = rng.permutation(len(X))
    X = X[perm]
    N = len(X)
    h1, h2 = row_hashes(X)
    pos2u, first = distinct_rows(h1, h2)
    assert len(first) <= n and int(pos2u.max()) == len(first) - 1 and bool((first[1:] > first[:-1]).all())
    assert bool((X[first[pos2u]].view(np.uint16) == X.view(np.uint16)).all())
    # row numbering by ascending first position
    assert bool((pos2u[first] == np.arange(len(first))).all())
    pu2, f2 = rows_from_labels(pos2u * 7 + 3)                                            # the label path gives the same rows for any relabelling
    assert bool((pu2 == pos2u).all()) and bool((f2 == first).all())
    Xd = np.ascontiguousarray(X[first])
    nbr, sc = search4(Xd, "cpu")
    # brute-force reference
    xn = Xd.astype(np.float32)
    xn /= np.linalg.norm(xn, axis=1, keepdims=True)
    S = xn @ xn.T
    ref = np.argsort(-S, axis=1, kind="stable")[:, :SEARCH_K]
    assert (np.take_along_axis(S, nbr.astype(np.int64), 1) - np.take_along_axis(S, ref, 1)).max() < 1e-5
    a, b, w = merge(nbr, sc, pos2u, first)
    assert (a < b).all() and len(np.unique(pair_keys(a, b, N))) == len(a)
    # PQ on the same data
    m = 16
    rt = np.random.RandomState(0)
    tr = np.sort(rt.choice(len(Xd), size=5000, replace=False))
    Xs = xn[tr]
    opq, pq = pq_train(Xs, m, nthreads=2)
    codes = pq_encode(opq, pq, Xd)
    assert codes.shape == (len(Xd), m)
    Rd = pq_decode(pq, codes)
    rc = recon_cosine(opq, pq, Xd, codes, n=1000)
    nbr2, sc2 = search4(Rd, "cpu")
    r3, r1, nf = neighbour_recall(nbr, nbr2)
    a2, b2, w2 = merge(nbr2, sc2, pos2u, first)
    assert 0.0 < r3 <= 1.0 and rc["mean"] > 0.5
    # the recall measure is 1 against itself
    assert neighbour_recall(nbr, nbr)[0] == 1.0
    print("TEST PASS: %d positions, %d distinct, exact search = brute force, merge ok (%d edges), PQ%d recon cos %.3f, recall@3 vs exact %.3f (self 1.0)" % (N, len(first), len(a), m, rc["mean"], r3))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "TEST":
        _test()
    elif cmd == "PREP":
        cmd_prep()
    elif cmd == "EXACT":
        cmd_exact()
    elif cmd == "PQ":
        cmd_pq(a[1])
    else:
        raise SystemExit(__doc__)
