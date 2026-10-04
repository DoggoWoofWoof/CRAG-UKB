# -*- coding: utf-8 -*-
"""Step 2 of the normalised Freebase tree: relations table + edges as position-indexed CSR.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/build_edges.py forward   [--workers 4]
    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/build_edges.py reverse   [--passes 2]

forward  reads the 264 frozen canonical edge parts, maps every endpoint uid to its position
         (binary search in nodes/node_uid.npy), every rel_uid to its dense rel_id, and writes
             relations/relations.parquet     row i == rel_id i (sorted by relation string)
             graph/out_indptr.npy int64[N+1]  graph/out_dst.npy int32[E]
             graph/out_rel.npy    int32[E]    graph/out_flag.npy int8[E]   (bit0 = flipped)
         Within a node's out-block entries are sorted by (rel_id, dst), so a relation's range
         inside any adjacency list -- including the 47.5M-edge hubs -- is a binary search.
reverse  transposes the forward CSR (no second parquet decode):
             graph/in_indptr.npy int64[N+1]   graph/in_src.npy int32[E]   graph/in_rel.npy int32[E]
         Within a node's in-block entries are sorted by (rel_id, src).

The dst_kind column of the frozen layer (0 = Freebase-namespace node, 1 = external URI,
2 = literal; PASS B constants K_NS/K_EXT/K_LIT) is not copied: it is re-derived from
nodes/kind.npy and every row is checked against it, the mismatch count is recorded and must be 0.

External sort: 64 buckets by source position on disk (13 bytes/edge), one bucket sorted in RAM
at a time, output appended sequentially into pre-sized .npy memmaps.  No edge is ever fabricated,
dropped or re-oriented: E in == E out is asserted, and per-relation counts are recorded.
"""
import glob
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = "data/final_canonical/freebase"
TMP = "scratchpad/fb4/edges_tmp"
N = 301977131
E = 2062430072
NB = 64
BUCKET = -(-N // NB)  # 4,718,393
KIND_EXTERNAL_URI, KIND_LITERAL = 5, 6
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# relations: dense ids
# ----------------------------------------------------------------------------------------------
def build_relations():
    os.makedirs(OUT + "/relations", exist_ok=True)
    r = pq.read_table(FB + "/canonical/relations.parquet")
    assert r.num_rows == 784928, r.num_rows
    rel = np.array(r["relation"].to_pylist(), dtype=object)
    order = np.argsort(rel, kind="stable")
    assert len(set(rel.tolist())) == len(rel), "relation strings not unique"
    r = r.take(pa.array(order))
    rel_id = np.arange(r.num_rows, dtype=np.int32)
    names = r["relation"].to_pylist()
    idx = {n: i for i, n in enumerate(names)}
    rev = r["reverse_relation"].to_pylist()
    rev_id = np.array([idx.get(x, -1) if x is not None else -1 for x in rev], dtype=np.int32)
    r = r.add_column(0, "rel_id", pa.array(rel_id)).append_column("reverse_rel_id", pa.array(rev_id))
    return r


# ----------------------------------------------------------------------------------------------
# partition helpers
# ----------------------------------------------------------------------------------------------
class BucketWriter:
    def __init__(self, tag):
        self.fh = {}
        self.tag = tag
        os.makedirs(TMP, exist_ok=True)

    def _f(self, b, col):
        k = (b, col)
        if k not in self.fh:
            self.fh[k] = open(TMP + "/%s_b%02d_%s.bin" % (self.tag, b, col), "ab")
        return self.fh[k]

    def write(self, key, cols):
        """key int32 positions deciding the bucket; cols dict name -> array (same length)."""
        b = key // BUCKET
        order = np.argsort(b, kind="stable")
        bs = b[order]
        edges = np.flatnonzero(np.diff(bs)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(bs)]])
        for s, e in zip(starts, ends):
            bb = int(bs[s])
            sel = order[s:e]
            for name, arr in cols.items():
                arr[sel].tofile(self._f(bb, name))

    def close(self):
        for f in self.fh.values():
            f.close()


def forward_worker(args):
    wid, parts = args
    uids = np.load(OUT + "/nodes/node_uid.npy", mmap_mode="r")
    kind = np.load(OUT + "/nodes/kind.npy")
    rels = pq.read_table(OUT + "/relations/relations.parquet", columns=["rel_id", "rel_uid"])
    ru = rels["rel_uid"].to_numpy()
    ro = np.argsort(ru)
    ru_sorted, rid_sorted = ru[ro], rels["rel_id"].to_numpy()[ro]
    bw = BucketWriter("w%d" % wid)
    n = 0
    rel_count = np.zeros(len(ru), dtype=np.int64)
    dk_mismatch = 0
    flipped = 0
    for i, f in enumerate(parts):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["src", "rel", "dst", "dst_kind", "flipped"])
            s, r, d = t["src"].to_numpy(), t["rel"].to_numpy(), t["dst"].to_numpy()
            dk, fl = t["dst_kind"].to_numpy(), t["flipped"].to_numpy()
            sp = np.searchsorted(uids, s)
            sp[sp >= N] = 0
            assert np.array_equal(uids[sp], s), "dangling src uid in " + f
            dp = np.searchsorted(uids, d)
            dp[dp >= N] = 0
            assert np.array_equal(uids[dp], d), "dangling dst uid in " + f
            rp = np.searchsorted(ru_sorted, r)
            rp[rp >= len(ru)] = 0
            assert np.array_equal(ru_sorted[rp], r), "unknown rel uid in " + f
            rid = rid_sorted[rp]
            kd = kind[dp]
            expect = np.where(kd == KIND_LITERAL, 2, np.where(kd == KIND_EXTERNAL_URI, 1, 0)).astype(np.int8)
            dk_mismatch += int((expect != dk).sum())
            flipped += int((fl != 0).sum())
            rel_count += np.bincount(rid, minlength=len(ru))
            bw.write(sp.astype(np.int32), {"src": sp.astype(np.int32), "dst": dp.astype(np.int32),
                                           "rel": rid.astype(np.int32), "flag": (fl != 0).astype(np.int8)})
            n += len(s)
        if i % 10 == 0:
            log("worker", wid, "part", i, "/", len(parts), os.path.basename(f), "rows", n)
    bw.close()
    return {"rows": n, "rel_count": rel_count, "dst_kind_mismatch": dk_mismatch, "flipped": flipped}


class NpyAppender:
    """Sequential writer for a .npy file of known final shape (no memmap: bounded working set)."""

    def __init__(self, path, dtype, n):
        self.fh = open(path, "wb")
        np.lib.format.write_array_header_1_0(self.fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                                       "fortran_order": False, "shape": (n,)})
        self.dtype = np.dtype(dtype)
        self.n = n
        self.written = 0

    def append(self, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        arr.tofile(self.fh)
        self.written += len(arr)

    def close(self):
        assert self.written == self.n, (self.written, self.n)
        self.fh.close()


class CsrEmitter:
    """Pass 2: per bucket, sort by (key, rel, other) and append sequentially into .npy files.
    Buckets may arrive in several partition passes; the bucket files are deleted as soon as they are consumed."""

    def __init__(self, out_prefix, other_col, with_flag):
        os.makedirs(OUT + "/graph", exist_ok=True)
        self.out_prefix, self.other_col, self.with_flag = out_prefix, other_col, with_flag
        self.indptr = np.zeros(N + 1, dtype=np.int64)
        self.o_other = NpyAppender(OUT + "/graph/%s_%s.npy" % (out_prefix, other_col), np.int32, E)
        self.o_rel = NpyAppender(OUT + "/graph/%s_rel.npy" % out_prefix, np.int32, E)
        self.o_flag = NpyAppender(OUT + "/graph/%s_flag.npy" % out_prefix, np.int8, E) if with_flag else None
        self.base = 0
        self.next_bucket = 0
        self.flag_sum = 0

    def bucket(self, b, key_name, tag):
        assert b == self.next_bucket, (b, self.next_bucket)
        fs_all = []

        def load(col, dt):
            fs = sorted(glob.glob(TMP + "/*_b%02d_%s.bin" % (b, col)))
            fs_all.extend(fs)
            if not fs:
                return np.zeros(0, dtype=dt)
            return np.concatenate([np.fromfile(f, dtype=dt) for f in fs])
        key = load(key_name, np.int32)
        other = load(self.other_col, np.int32)
        rel = load("rel", np.int32)
        flag = load("flag", np.int8) if self.with_flag else None
        assert len(key) == len(other) == len(rel)
        lo, hi = b * BUCKET, min((b + 1) * BUCKET, N)
        assert len(key) == 0 or (key.min() >= lo and key.max() < hi)
        order = np.lexsort((other, rel, key))
        m = len(order)
        self.o_other.append(other[order])
        self.o_rel.append(rel[order])
        if self.with_flag:
            self.o_flag.append(flag[order])
            self.flag_sum += int(np.count_nonzero(flag))
        deg = np.bincount(key - lo, minlength=hi - lo)
        self.indptr[lo + 1:hi + 1] = self.base + np.cumsum(deg)
        self.base += m
        del key, other, rel, flag, order
        for f in fs_all:
            os.remove(f)
        self.next_bucket = b + 1
        log(tag, "bucket", b, "rows", m, "cum", self.base)

    def close(self):
        assert self.next_bucket == NB and self.base == E, (self.next_bucket, self.base, E)
        self.o_other.close(); self.o_rel.close()
        if self.with_flag:
            self.o_flag.close()
        assert self.indptr[-1] == E
        np.save(OUT + "/graph/%s_indptr.npy" % self.out_prefix, self.indptr)
        for nm in [self.other_col, "rel"] + (["flag"] if self.with_flag else []):
            a = np.load(OUT + "/graph/%s_%s.npy" % (self.out_prefix, nm), mmap_mode="r")
            assert a.shape == (E,), (nm, a.shape)
        return self.indptr


def emit(tag, out_prefix, other_col, key_name, with_flag):
    em = CsrEmitter(out_prefix, other_col, with_flag)
    for b in range(NB):
        em.bucket(b, key_name, tag)
    return em.close()


def cmd_forward(workers):
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    rels = build_relations()
    parts = sorted(glob.glob(FB + "/canonical/edges/*.parquet"))
    assert len(parts) == 264, len(parts)
    pq.write_table(rels, OUT + "/relations/relations.parquet", compression="zstd")
    log("relations", rels.num_rows)
    jobs = [(w, parts[w::workers]) for w in range(workers)]
    with Pool(workers) as pool:
        res = pool.map(forward_worker, jobs)
    rows = sum(r["rows"] for r in res)
    assert rows == E, (rows, E)
    rel_count = sum(r["rel_count"] for r in res)
    dk_mismatch = sum(r["dst_kind_mismatch"] for r in res)
    flipped = sum(r["flipped"] for r in res)
    assert dk_mismatch == 0, dk_mismatch
    log("partition done: rows", rows, "dst_kind mismatches", dk_mismatch, "flipped", flipped)
    rels = pq.read_table(OUT + "/relations/relations.parquet")
    rels = rels.append_column("canonical_edge_count", pa.array(rel_count))
    assert int(rel_count.sum()) == E
    pq.write_table(rels, OUT + "/relations/relations.parquet", compression="zstd")
    indptr = emit("forward", "out", "dst", "src", True)
    shutil.rmtree(TMP)
    deg = np.diff(indptr)
    rec = {
        "RECORD": "FREEBASE_EDGES_FORWARD",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "edges": int(rows), "nodes": N, "relations": int(rels.num_rows),
        "CHECKS": {"edges_in_equals_edges_out": rows == E, "dst_kind_recomputed_mismatches": dk_mismatch,
                   "dangling_endpoints": 0, "unknown_rel_uid": 0, "indptr_last_equals_E": True,
                   "rel_count_sums_to_E": int(rel_count.sum()) == E},
        "flipped_edges": flipped,
        "OUT_DEGREE": {"nodes_with_out_edges": int((deg > 0).sum()), "max": int(deg.max()),
                       "p50": float(np.percentile(deg[deg > 0], 50)), "p99": float(np.percentile(deg[deg > 0], 99))},
        "layout": {"out_indptr": "int64[N+1]", "out_dst": "int32[E] position", "out_rel": "int32[E] rel_id",
                   "out_flag": "int8[E] bit0 = flipped (source asserted only the reverse direction; PASS D)",
                   "order_within_block": "(rel_id, dst) ascending"},
        "relations_layout": "relations/relations.parquet row i == rel_id i; columns rel_id, rel_uid, relation, reverse_relation, role, raw_edge_count, reverse_rel_id, canonical_edge_count",
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(rec, open("scratchpad/fb4/EDGES_FORWARD.json", "w"), indent=1)
    log("done", json.dumps(rec["CHECKS"]), json.dumps(rec["OUT_DEGREE"]))


def cmd_reverse(passes=2):
    """Transpose the forward CSR in `passes` sweeps over the out-CSR (each sweep partitions only the dst buckets
    it owns), so the on-disk temp never exceeds E*12/passes bytes; buckets are emitted and deleted in order."""
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    indptr = np.load(OUT + "/graph/out_indptr.npy")
    dst = np.load(OUT + "/graph/out_dst.npy", mmap_mode="r")
    rel = np.load(OUT + "/graph/out_rel.npy", mmap_mode="r")
    assert indptr[-1] == E == len(dst) == len(rel)
    em = CsrEmitter("in", "src", False)
    CH = 16_000_000
    per = -(-NB // passes)
    seen = 0
    for ps in range(passes):
        b_lo, b_hi = ps * per, min((ps + 1) * per, NB)
        p_lo, p_hi = b_lo * BUCKET, min(b_hi * BUCKET, N)
        bw = BucketWriter("t%d" % ps)
        for s in range(0, E, CH):
            e = min(s + CH, E)
            d = np.asarray(dst[s:e])
            r = np.asarray(rel[s:e])
            # sources of edges s..e: node i owns [indptr[i], indptr[i+1])
            first = np.searchsorted(indptr, s, side="right") - 1
            last = np.searchsorted(indptr, e - 1, side="right") - 1
            ids = np.arange(first, last + 1, dtype=np.int32)
            cnt = np.minimum(indptr[first + 1:last + 2], e) - np.maximum(indptr[first:last + 1], s)
            src = np.repeat(ids, cnt)
            assert len(src) == e - s
            m = (d >= p_lo) & (d < p_hi)
            if m.any():
                bw.write(d[m], {"dst": d[m], "src": src[m], "rel": r[m]})
                seen += int(m.sum())
            if (s // CH) % 20 == 0:
                log("reverse pass", ps, "partition", s, "/", E, "kept so far", seen)
        bw.close()
        for b in range(b_lo, b_hi):
            em.bucket(b, "dst", "reverse")
    assert seen == E, (seen, E)
    in_indptr = em.close()
    shutil.rmtree(TMP)
    deg = np.diff(in_indptr)
    rec = {
        "RECORD": "FREEBASE_EDGES_REVERSE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "edges": E, "passes": passes,
        "CHECKS": {"indptr_last_equals_E": True, "in_degree_sum_equals_E": int(deg.sum()) == E, "edges_partitioned_equals_E": seen == E},
        "IN_DEGREE": {"nodes_with_in_edges": int((deg > 0).sum()), "max": int(deg.max()),
                      "p50": float(np.percentile(deg[deg > 0], 50)), "p99": float(np.percentile(deg[deg > 0], 99))},
        "layout": {"in_indptr": "int64[N+1]", "in_src": "int32[E] position", "in_rel": "int32[E] rel_id",
                   "order_within_block": "(rel_id, src) ascending", "flag": "not duplicated; look up the (src, rel, dst) row in the out-CSR"},
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(rec, open("scratchpad/fb4/EDGES_REVERSE.json", "w"), indent=1)
    log("done", json.dumps(rec["CHECKS"]), json.dumps(rec["IN_DEGREE"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "forward":
        w = 4
        if "--workers" in sys.argv:
            w = int(sys.argv[sys.argv.index("--workers") + 1])
        cmd_forward(w)
    elif cmd == "reverse":
        p = 2
        if "--passes" in sys.argv:
            p = int(sys.argv[sys.argv.index("--passes") + 1])
        cmd_reverse(p)
    else:
        sys.exit("forward | reverse")
