"""FBX_SCALE Track C -- the Freebase NAME-DEDUP table: which of the 301,977,131 node names are distinct, and the pointer from every position to its distinct row.

Why: the canonical dense encoding of WebQSP is over DISTINCT source rows (1,791,533 of 2,592,894 positions; every position inherits its row's vector / neighbours, the lowest position of a
name is its representative).  The same rule for Freebase means the encoder runs once per distinct name, not once per node, and the KNN is computed over distinct rows.  This script builds the
dedup table exactly and deterministically; it reads the `name` column only and writes nothing that depends on a model.

  python -u scratchpad/_fbx_name_census.py TEST                         # synthetic check against a dict reference
  python -u scratchpad/_fbx_name_census.py CENSUS first last procs      # per row-group unit: blake2b-128 of the UTF-8 name, sorted by hash, write-once records
  python -u scratchpad/_fbx_name_census.py BUCKET                       # 64 hash buckets: group equal names, first position of every group
  python -u scratchpad/_fbx_name_census.py FINALIZE                     # rank the groups by first position -> ptr.npy / first_pos.npy + the write-once record

Definitions (frozen here): name key = blake2b(name.encode('utf-8', 'surrogatepass'), digest_size=16) read as two little-endian uint64 (hi, lo); two positions share a distinct row iff the keys
are equal; distinct rows are numbered 0..D-1 in ascending order of their FIRST position; ptr[p] is the row of position p; first_pos[r] is the lowest position of row r (the node whose name is
encoded).  Bulk output data/freebase_scale/names/ (git-ignored); records results/FREEBASE_SCALE/names/.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

import _fbx_ner_build as NB

REPO = NB.REPO
TREE = NB.TREE
ROOT = os.path.join(REPO, "data", "freebase_scale", "names")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "names")
HB = 6
LEN_EDGES = [0, 1, 2, 3, 4, 5, 9, 17, 33, 65, 129, 257]        # character-length histogram bins: [e_i, e_{i+1}), last open
sha256, wj, save_npy = NB.sha256, NB.wj, NB.save_npy


# ---------------------------------------------------------------------------------------------------------------- kernels
def name_keys(names):
    """(hi, lo) uint64 arrays of the frozen name key."""
    b = [hashlib.blake2b(("" if x is None else x).encode("utf-8", "surrogatepass"), digest_size=16).digest() for x in names]
    a = np.frombuffer(b"".join(b), dtype="<u8").reshape(-1, 2) if b else np.zeros((0, 2), "<u8")
    return a[:, 0].astype(np.uint64), a[:, 1].astype(np.uint64)


def census_arrays(names, pos0):
    """sorted by (hi, stable): hi, lo, pos int32 (pos0 + row index), off (bucket offsets), length histogram."""
    hi, lo = name_keys(names)
    o = np.argsort(hi, kind="stable")
    hi, lo = hi[o], lo[o]
    pos = (pos0 + o).astype(np.int32)
    ln = np.fromiter((len(x) if x is not None else 0 for x in names), dtype=np.int64, count=len(names))
    hist = np.bincount(np.searchsorted(np.array(LEN_EDGES), ln, side="right") - 1, minlength=len(LEN_EDGES)).tolist()
    return hi, lo, pos, NB.bucket_offsets(hi, HB), hist, int(ln.sum())


def group_bucket(hi, lo, pos):
    """hi/lo/pos hold WHOLE keys (one hash bucket): returns pos (sorted by key then position), grp (group index per row), first (first position per group), size (rows per group)."""
    n = len(hi)
    if n == 0:
        z = np.zeros(0, np.int32)
        return z, z, z, z
    o = np.lexsort((pos, lo, hi))
    hi, lo, pos = hi[o], lo[o], pos[o]
    new = np.ones(n, bool)
    new[1:] = (hi[1:] != hi[:-1]) | (lo[1:] != lo[:-1])
    starts = np.flatnonzero(new)
    grp = (np.cumsum(new) - 1).astype(np.int32)
    size = np.diff(np.concatenate([starts, [n]])).astype(np.int32)
    return pos.astype(np.int32), grp, pos[starts].astype(np.int32), size


def rank_groups(firsts):
    """firsts: list of per-bucket int32 first-position arrays -> (first_sorted, [rank arrays]) with ranks in ascending order of the first position."""
    allf = np.concatenate(firsts) if firsts else np.zeros(0, np.int32)
    fs = np.sort(allf)
    assert len(fs) < 2 or bool((fs[1:] != fs[:-1]).all()), "two groups share a first position"
    return fs, [np.searchsorted(fs, f).astype(np.int32) for f in firsts]


def build_ptr(N, poss, grps, ranks):
    ptr = np.full(N, -1, np.int32)
    for p, g, r in zip(poss, grps, ranks):
        if len(p):
            ptr[p] = r[g]
    return ptr


def reference(names):
    """dict reference: (ptr, first_pos) under the same definition."""
    seen, first, ptr = {}, [], []
    for i, x in enumerate(names):
        k = hashlib.blake2b(("" if x is None else x).encode("utf-8", "surrogatepass"), digest_size=16).digest()
        if k not in seen:
            seen[k] = len(first)
            first.append(i)
        ptr.append(seen[k])
    return np.array(ptr, np.int32), np.array(first, np.int32)


# ---------------------------------------------------------------------------------------------------------------- stage CENSUS
def census_unit(unit):
    import pyarrow.parquet as pq
    s, g, r0, n = unit
    base = os.path.join(ROOT, "census", "s%03dg%d" % (s, g))
    rec = base + ".json"
    if os.path.exists(rec):
        return {"unit": [s, g], "skipped": True}
    t0 = time.time()
    tb = pq.ParquetFile(os.path.join(TREE, "nodes", "shard_%05d.parquet" % s)).read_row_group(g, columns=["position", "name"])
    positions = tb.column("position").to_numpy()
    names = tb.column("name").to_pylist()
    assert len(names) == n and len(positions) == n
    assert bool((positions == positions[0] + np.arange(n)).all()), "positions not consecutive"
    hi, lo, pos, off, hist, nchar = census_arrays(names, int(positions[0]))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    files = {}
    for nm, a in (("hi", hi), ("lo", lo), ("pos", pos), ("off", off)):
        p = "%s.%s.npy" % (base, nm)
        save_npy(p, a)
        files[nm] = {"file": os.path.relpath(p, REPO).replace("\\", "/"), "sha256": sha256(p)}
    wj(rec, {"RECORD": "FBX_NAME_CENSUS_UNIT", "shard": s, "row_group": g, "rows": n, "position_first": int(positions[0]), "position_last": int(positions[-1]), "chars": nchar,
             "char_length_hist": hist, "char_length_bins": LEN_EDGES, "hash_buckets": 1 << HB, "seconds": round(time.time() - t0, 1), "files": files,
             "code_sha256": sha256(os.path.abspath(__file__)), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return {"unit": [s, g], "seconds": round(time.time() - t0, 1)}


def cmd_census(first, last, procs):
    us = NB.units()
    sel = us[first:last + 1]
    print("CENSUS units %d..%d of %d (%d rows), %d process(es)" % (first, min(last, len(us) - 1), len(us), sum(u[3] for u in sel), procs), flush=True)
    t0 = time.time()
    if procs <= 1:
        for u in sel:
            print(census_unit(u), "%.0fs" % (time.time() - t0), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            for r in pool.imap_unordered(census_unit, sel):
                print(r, "%.0fs" % (time.time() - t0), flush=True)


# ---------------------------------------------------------------------------------------------------------------- stage BUCKET / FINALIZE
def _unit_names(root):
    return sorted(f[:-5] for f in os.listdir(os.path.join(root, "census")) if f.endswith(".json"))


def cmd_bucket(root=ROOT, recs=RECS):
    names = _unit_names(root)
    assert names, "no census unit"
    for b in range(1 << HB):
        rec = os.path.join(recs, "bucket_b%02d.json" % b)
        if os.path.exists(rec):
            continue
        t0 = time.time()
        H_, L_, P_ = [], [], []
        for nm in names:
            base = os.path.join(root, "census", nm)
            off = np.load(base + ".off.npy")
            a, z = int(off[b]), int(off[b + 1])
            if z > a:
                H_.append(np.load(base + ".hi.npy", mmap_mode="r")[a:z])
                L_.append(np.load(base + ".lo.npy", mmap_mode="r")[a:z])
                P_.append(np.load(base + ".pos.npy", mmap_mode="r")[a:z])
        hi = np.concatenate(H_) if H_ else np.zeros(0, np.uint64)
        lo = np.concatenate(L_) if L_ else np.zeros(0, np.uint64)
        pos = np.concatenate(P_) if P_ else np.zeros(0, np.int32)
        p, g, f, s = group_bucket(hi, lo, pos)
        os.makedirs(os.path.join(root, "bucket"), exist_ok=True)
        files = {}
        for nm, a in (("pos", p), ("grp", g), ("first", f), ("size", s)):
            q = os.path.join(root, "bucket", "b%02d.%s.npy" % (b, nm))
            save_npy(q, a)
            files[nm] = {"file": os.path.relpath(q, REPO).replace("\\", "/"), "sha256": sha256(q)}
        wj(rec, {"RECORD": "FBX_NAME_BUCKET", "bucket": b, "rows": int(len(p)), "groups": int(len(f)), "units": len(names), "seconds": round(time.time() - t0, 1), "files": files,
                 "code_sha256": sha256(os.path.abspath(__file__)), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print("bucket %02d: %d rows, %d groups, %.0fs" % (b, len(p), len(f), time.time() - t0), flush=True)


SIZE_EDGES = [1, 2, 3, 10, 100, 1000, 10000, 100000]


def finalize_arrays(N, root):
    poss, grps, firsts, sizes = [], [], [], []
    for b in range(1 << HB):
        for lst, nm in ((poss, "pos"), (grps, "grp"), (firsts, "first"), (sizes, "size")):
            lst.append(np.load(os.path.join(root, "bucket", "b%02d.%s.npy" % (b, nm))))
    fs, ranks = rank_groups(firsts)
    ptr = build_ptr(N, poss, grps, ranks)
    assert int(ptr.min()) >= 0, "a position has no group"
    D = len(fs)
    assert int(ptr.max()) == D - 1
    assert bool((ptr[fs] == np.arange(D)).all()), "first position of a row is not in that row"
    mult = np.bincount(ptr, minlength=D).astype(np.int64)
    return ptr, fs, mult


def cmd_finalize(root=ROOT, recs=RECS, N=None):
    out = os.path.join(recs, "FBX_NAME_DEDUP__v1.json")
    assert not os.path.exists(out), "write-once: %s exists" % out
    N = N or int(np.load(os.path.join(TREE, "nodes", "kind.npy"), mmap_mode="r").shape[0])
    units = [json.load(io.open(os.path.join(root, "census", n + ".json"), encoding="utf-8")) for n in _unit_names(root)]
    assert sum(u["rows"] for u in units) == N, "census units do not cover N"
    t0 = time.time()
    ptr, fs, mult = finalize_arrays(N, root)
    D = int(len(fs))
    pp, fp = os.path.join(root, "ptr.npy"), os.path.join(root, "first_pos.npy")
    save_npy(pp, ptr)
    save_npy(fp, fs)
    hist = np.bincount(np.searchsorted(np.array(SIZE_EDGES), mult, side="right") - 1, minlength=len(SIZE_EDGES)).tolist()
    top = np.argsort(-mult, kind="stable")[:25]
    top_rows = [{"row": int(r), "copies": int(mult[r]), "first_position": int(fs[r])} for r in top]
    chars = sum(u["chars"] for u in units)
    lh = np.sum([u["char_length_hist"] for u in units], axis=0).tolist()
    rec = {"RECORD": "FBX_NAME_DEDUP", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "definition": "name key = blake2b-128 of the UTF-8 name bytes; distinct rows numbered in ascending order of first position; ptr[p] = row of position p; first_pos[r] = lowest position of row r",
           "n_positions": N, "n_distinct_names": D, "distinct_fraction": round(D / N, 6), "duplicate_positions": N - D, "max_copies": int(mult.max()),
           "copies_histogram": {"edges_lower_bound": SIZE_EDGES, "rows_with_that_many_copies_or_more_up_to_next_edge": hist},
           "top_rows_by_copies": top_rows, "total_characters_all_positions": int(chars), "char_length_bins": LEN_EDGES, "char_length_hist_all_positions": lh,
           "files": {"ptr": {"file": os.path.relpath(pp, REPO).replace("\\", "/"), "dtype": "int32", "shape": [N], "sha256": sha256(pp), "bytes": os.path.getsize(pp)},
                     "first_pos": {"file": os.path.relpath(fp, REPO).replace("\\", "/"), "dtype": "int32", "shape": [D], "sha256": sha256(fp), "bytes": os.path.getsize(fp)}},
           "units": len(units), "seconds": round(time.time() - t0, 1), "code_sha256": sha256(os.path.abspath(__file__))}
    wj(out, rec)
    print(json.dumps({k: rec[k] for k in ("n_positions", "n_distinct_names", "distinct_fraction", "max_copies", "top_rows_by_copies")}, indent=1)[:1800])


# ---------------------------------------------------------------------------------------------------------------- test
def _test():
    import tempfile
    rng = np.random.RandomState(7)
    pool = ["name %d" % i for i in range(900)] + ["", "  padded  ", "été", "😀", "Name 1", "x" * 300]
    names = [pool[i] if rng.rand() < 0.97 else None for i in rng.randint(0, len(pool), 6000)]
    N = len(names)
    refp, reff = reference(names)
    root = tempfile.mkdtemp(prefix="fbx_names_")
    os.makedirs(os.path.join(root, "census"))
    cuts = [0, 1777, 1778, 4000, N]
    for i, (a, z) in enumerate(zip(cuts[:-1], cuts[1:])):
        hi, lo, pos, off, hist, nchar = census_arrays(names[a:z], a)
        assert off[-1] == z - a and sum(hist) == z - a
        base = os.path.join(root, "census", "s000g%d" % i)
        for nm, arr in (("hi", hi), ("lo", lo), ("pos", pos), ("off", off)):
            np.save("%s.%s.npy" % (base, nm), arr)
        json.dump({"rows": z - a, "chars": nchar, "char_length_hist": hist}, open(base + ".json", "w"))
    recs = os.path.join(root, "recs")
    cmd_bucket(root, recs)
    ptr, fs, mult = finalize_arrays(N, root)
    assert len(fs) == len(reff) and bool((fs == reff).all()) and bool((ptr == refp).all()), "ptr / first_pos differ from the dict reference"
    assert int(mult.sum()) == N
    print("TEST PASS: %d names, %d distinct, ptr and first_pos equal the dict reference; buckets %d" % (N, len(fs), 1 << HB))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "TEST":
        _test()
    elif cmd == "CENSUS":
        cmd_census(int(a[1]), int(a[2]), int(a[3]))
    elif cmd == "BUCKET":
        cmd_bucket()
    elif cmd == "FINALIZE":
        cmd_finalize()
    else:
        raise SystemExit(__doc__)
