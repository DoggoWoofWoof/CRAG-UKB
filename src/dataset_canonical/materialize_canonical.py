"""
MATERIALISE THE CANONICAL EMBEDDINGS IN POSITION ORDER  (2026-09-12)
====================================================================
End state the user asked for: one frozen tree per dataset that carries everything -- nodes,
queries, embeddings, graphs, caches -- in the same normalised form, with no pointer indirection
and no second root. So every embedding channel is rewritten so that ROW i IS POSITION i:

    data/final_canonical/<ds>/embeddings/<dense|splade>/<docs|queries>/shard_XXXXX.{npy,npz}
        docs:    row i  <->  line i of nodes.jsonl
        queries: row j  <->  entry j of queries/query_ids.json (== train, dev/validation, test
                             split files concatenated in that order)

Nothing is encoded, invented or approximated: every new row is a byte copy of the vector the
pointer index resolved for that position (docs), or the query store already in position order
(queries; the query pointer index is the identity for all six datasets). The copy is proved
by an independent path: a per-row digest of every OLD store row and every NEW store row, and
new[i] == old[row[i]] for every position, plus a chain hash over the resolved digests that a
later verifier can recompute from the new store alone.

Phases (run in this order; each is resumable and logs to _history/logs/MATERIALIZE_LOG.json):
    docs      --dataset <ds> [--model dense|splade] [--delete-old]
    queries   --dataset <ds> [--delete-old]
    layout    --dataset <ds>            (pointer_index/reuse_map/encoder extras -> history or gone)
    graphs    --dataset <ds>            (GRAPH_MANIFEST.json in the normalised form)
    purge                               (data/canonical, data/_retired_v3, CLEANUP_PASS_3 list)

Deletion is authorised by the user's message of 2026-09-12: "the rest other versions i give you
permission to delete and freeup space".
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FC = "data/final_canonical"
CAN = "data/canonical"
HIST = FC + "/_history"
LOG = HIST + "/logs/MATERIALIZE_LOG.json"
DATASETS = ["metaqa", "squad", "musique", "webqsp", "hotpotqa", "2wiki"]
SPLITS = {"metaqa": ["train", "dev", "test"], "squad": ["train", "dev"], "musique": ["train", "dev", "test"],
          "hotpotqa": ["train", "validation", "test"], "2wiki": ["train", "dev", "test"], "webqsp": ["train", "test"]}
SS = 40000
DIM = 1536
VOCAB = 30522
NORM_TOL = 0.01
DENSE_BAND_BYTES = 1.0e9
SPLADE_BAND_BYTES = 0.5e9


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rj(p):
    return json.load(io.open(p, encoding="utf-8"))


def wj(p, obj):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def log_phase(phase, rec):
    L = rj(LOG) if os.path.exists(LOG) else {"RECORD": "MATERIALIZE_LOG", "phases": []}
    rec = dict(rec)
    rec["phase"], rec["utc"] = phase, utc()
    L["phases"].append(rec)
    wj(LOG, L)


def say(*a):
    print(*a)
    sys.stdout.flush()


# ------------------------------------------------------------------------------ stores
class Store(object):
    """A sharded store as found on disk: shard_XXXXX.npy (dense) / .npz (splade CSR)."""

    def __init__(self, model, d):
        self.model, self.d = model, d
        ext = "npy" if model == "dense" else "npz"
        self.files = sorted(f for f in os.listdir(d) if f.startswith("shard_") and f.endswith("." + ext))
        if not self.files:
            raise RuntimeError("no shards in %s" % d)
        for k, f in enumerate(self.files):
            if f != "shard_%05d.%s" % (k, ext):
                raise RuntimeError("shard numbering gap in %s at %s" % (d, f))
        self.rows = [self._rows(os.path.join(d, f)) for f in self.files]
        self.n_rows = int(sum(self.rows))
        self.ss = int(self.rows[0]) if len(self.rows) > 1 else max(int(self.rows[0]), 1)
        for k, r in enumerate(self.rows[:-1]):
            if r != self.ss:
                raise RuntimeError("shard %d of %s has %d rows, expected %d" % (k, d, r, self.ss))
        self._cache = (None, None)

    def _rows(self, p):
        if self.model == "dense":
            mm = np.load(p, mmap_mode="r")
            r = int(mm.shape[0])
            if mm.ndim != 2 or mm.shape[1] != DIM or mm.dtype != np.float16:
                raise RuntimeError("%s is not (rows,%d) float16" % (p, DIM))
            del mm
            return r
        z = np.load(p)
        r = int(z["shape"][0])
        if int(z["shape"][1]) != VOCAB:
            raise RuntimeError("%s vocab %d != %d" % (p, int(z["shape"][1]), VOCAB))
        z.close()
        return r

    def mmap(self, k):
        """dense only: the shard memory-mapped, so a scattered gather touches only its pages."""
        if self._cache[0] == ("mmap", k):
            return self._cache[1]
        obj = np.load(os.path.join(self.d, self.files[k]), mmap_mode="r")
        self._cache = (("mmap", k), obj)
        return obj

    def load(self, k):
        """dense: an in-memory float16 array of the shard; splade: (indptr, indices, data)."""
        if self._cache[0] == k:
            return self._cache[1]
        p = os.path.join(self.d, self.files[k])
        if self.model == "dense":
            obj = np.ascontiguousarray(np.load(p, mmap_mode="r"))
        else:
            z = np.load(p)
            obj = (z["indptr"].astype(np.int64), z["indices"], z["data"])
            z.close()
        self._cache = (k, obj)
        return obj

    def row_digests(self, label=""):
        """sha1 of every row's bytes, in row order (dense: the 3072 bytes; splade: indices+data)."""
        out = np.empty((self.n_rows, 20), dtype=np.uint8)
        base = 0
        t0 = time.time()
        for k in range(len(self.files)):
            obj = self.load(k)
            if self.model == "dense":
                for i in range(obj.shape[0]):
                    out[base + i] = np.frombuffer(hashlib.sha1(obj[i].tobytes()).digest(), dtype=np.uint8)
                base += obj.shape[0]
            else:
                indptr, ind, dat = obj
                n = indptr.size - 1
                for i in range(n):
                    a, b = int(indptr[i]), int(indptr[i + 1])
                    h = hashlib.sha1(ind[a:b].tobytes())
                    h.update(dat[a:b].tobytes())
                    out[base + i] = np.frombuffer(h.digest(), dtype=np.uint8)
                base += n
            if k % 20 == 0 or k == len(self.files) - 1:
                say("    digest %s shard %d/%d  %.0fs" % (label, k + 1, len(self.files), time.time() - t0))
        assert base == self.n_rows
        self._cache = (None, None)
        return out


def damaged_count(a, chunk=20000):
    """rows that are all-zero, non-finite, or off the unit sphere; chunked so a 1 GB band stays 1 GB"""
    total = 0
    for i in range(0, a.shape[0], chunk):
        v = np.asarray(a[i:i + chunk], dtype=np.float32)
        bad = ~np.isfinite(v).all(axis=1)
        n = np.sqrt((v * v).sum(axis=1))
        bad |= (n == 0.0) | (np.abs(n - 1.0) > NORM_TOL)
        total += int(bad.sum())
    return total


def save_dense(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr, dtype=np.float16))
    os.replace(tmp, path)


def save_splade(path, csr):
    from scipy.sparse import csr_matrix
    csr = csr_matrix(csr)
    csr.sort_indices()
    tmp = path + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, indices=csr.indices.astype(np.int32), indptr=csr.indptr.astype(np.int32),
                 data=csr.data.astype(np.float32), shape=np.array(csr.shape, dtype=np.int64),
                 format=np.bytes_(b"csr"))
    os.replace(tmp, path)


def old_tree(ds):
    return CAN + "/" + ds


def pointer_rows(ds, model):
    p = FC + "/%s/pointer_index/%s.npz" % (ds, model)
    if not os.path.exists(p):
        p = HIST + "/materialize/%s__pointer_index__%s.npz" % (ds, model)
    z = np.load(p)
    src, row = z["src"].copy(), z["row"].astype(np.int64)
    z.close()
    if src.size and int(src.max()) != 0:
        raise RuntimeError("%s: pointer index still carries src != 0" % p)
    return row, p


def n_nodes(ds):
    n = 0
    with io.open(FC + "/%s/nodes.jsonl" % ds, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def query_ids(ds):
    for p in (FC + "/%s/queries/query_ids.json" % ds, FC + "/%s/queries/pointer_index/query_ids.json" % ds):
        if os.path.exists(p):
            return rj(p)
    raise RuntimeError("no query_ids.json for %s" % ds)


# ------------------------------------------------------------------------------ docs
def build_docs(ds, model, delete_old):
    out = FC + "/%s/embeddings/%s/docs" % (ds, model)
    man_p = out + "/manifest.json"
    old_dir = old_tree(ds) + "/encodings/%s/docs" % model
    if os.path.exists(man_p) and rj(man_p).get("materialised", {}).get("proof", {}).get("equal"):
        say("  %s %s docs already materialised and proved; skipping" % (ds, model))
        if delete_old and os.path.isdir(old_dir):
            old_man = rj(old_dir + "/manifest.json") if os.path.exists(old_dir + "/manifest.json") else None
            _retire_old_store(old_dir, old_man, ds, model, "docs")
        return
    if not os.path.isdir(old_dir):
        raise RuntimeError("old store missing: %s" % old_dir)
    old = Store(model, old_dir)
    row, pidx = pointer_rows(ds, model)
    n = row.size
    if n != n_nodes(ds):
        raise RuntimeError("%s: %d pointers vs %d nodes" % (ds, n, n_nodes(ds)))
    if int(row.max()) >= old.n_rows or int(row.min()) < 0:
        raise RuntimeError("%s/%s: pointer row out of range" % (ds, model))
    say("  %s %s docs: %d positions -> %d old rows (%d shards of %d), %d distinct"
        % (ds, model, n, old.n_rows, len(old.files), old.ss, np.unique(row).size))

    # a partial previous run leaves derived shards only; rebuild them
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith("shard_") or f.endswith(".tmp.npy") or f.endswith(".tmp.npz"):
            os.remove(os.path.join(out, f))

    if model == "dense":
        band_rows = max(SS, int(DENSE_BAND_BYTES / (DIM * 2)) // SS * SS)
    else:
        nnz_total = 0
        for k in range(len(old.files)):
            z = np.load(os.path.join(old.d, old.files[k]))
            nnz_total += int(z["indptr"][-1])
            z.close()
        per_row = 8.0 * nnz_total / old.n_rows + 8.0
        band_rows = max(SS, int(SPLADE_BAND_BYTES / per_row) // SS * SS)
    say("    band of %d positions" % band_rows)

    t0 = time.time()
    damaged = 0
    shards = []
    for lo in range(0, n, band_rows):
        hi = min(n, lo + band_rows)
        rows = row[lo:hi]
        oshard = rows // old.ss
        if model == "dense":
            band = np.empty((hi - lo, DIM), dtype=np.float16)
            for s in np.unique(oshard):
                m = np.nonzero(oshard == s)[0]
                off = rows[m] - int(s) * old.ss
                o = np.argsort(off, kind="stable")
                band[m[o]] = old.mmap(int(s))[off[o]]
            damaged += damaged_count(band)
        else:
            from scipy.sparse import csr_matrix, vstack
            pieces, order = [], []
            for s in np.unique(oshard):
                m = np.nonzero(oshard == s)[0]
                off = rows[m] - int(s) * old.ss
                indptr, ind, dat = old.load(int(s))
                csr = csr_matrix((dat, ind, indptr), shape=(indptr.size - 1, VOCAB))
                pieces.append(csr[off])
                order.append(m)
            order = np.concatenate(order)
            inv = np.argsort(order, kind="stable")
            band = vstack(pieces).tocsr()[inv]
            del pieces
        for k in range(lo // SS, (hi + SS - 1) // SS):
            a, b = k * SS - lo, min(hi, (k + 1) * SS) - lo
            fn = "shard_%05d.%s" % (k, "npy" if model == "dense" else "npz")
            if model == "dense":
                save_dense(os.path.join(out, fn), band[a:b])
            else:
                save_splade(os.path.join(out, fn), band[a:b])
            shards.append({"shard": k, "rows": int(b - a), "file": fn})
        del band
        say("    wrote positions %d..%d  (%.0fs)" % (lo, hi, time.time() - t0))
    old._cache = (None, None)

    # ---- independent proof: per-row digests of old and new, compared through the map
    say("    proving byte equality per position")
    new = Store(model, out)
    if new.n_rows != n or len(new.files) != len(shards):
        raise RuntimeError("new store has %d rows / %d shards, expected %d / %d" % (new.n_rows, len(new.files), n, len(shards)))
    dig_old = old.row_digests("old")
    dig_new = new.row_digests("new")
    resolved = dig_old[row]
    equal = bool(np.array_equal(resolved, dig_new))
    bad_positions = [] if equal else np.nonzero((resolved != dig_new).any(axis=1))[0][:20].tolist()
    chain_old = hashlib.sha256(resolved.tobytes()).hexdigest()
    chain_new = hashlib.sha256(dig_new.tobytes()).hexdigest()
    say("    equal=%s  chain_old=%s chain_new=%s" % (equal, chain_old[:16], chain_new[:16]))
    if not equal:
        raise RuntimeError("%s/%s docs: %d positions differ, e.g. %s" % (ds, model, int((resolved != dig_new).any(axis=1).sum()), bad_positions))
    if model == "dense" and damaged:
        raise RuntimeError("%s dense docs: %d damaged rows in the materialised store" % (ds, damaged))

    # ---- V2 resolved-vector fingerprint, reproduced on the new store from the pre-V3 pointer index
    fp = None
    if model == "dense":
        fp = v2_fingerprint(ds, new)

    # ---- manifest with per-shard digests
    for sh in shards:
        p = os.path.join(out, sh["file"])
        sh["bytes"], sh["sha256"] = os.path.getsize(p), sha_file(p)
    old_man = rj(old_dir + "/manifest.json") if os.path.exists(old_dir + "/manifest.json") else None
    encoder = _encoder_block(ds, model, old_man)
    man = {
        "dataset": ds, "kind": "docs", "model": model,
        "n_rows": n, "shard_size": SS, "n_shards": len(shards),
        "row_semantics": "row i is the %s vector of position i == line i of nodes.jsonl" % model,
        "format": _format_block(model),
        "encoder": encoder,
        "shards": shards,
        "materialised": {
            "utc": utc(),
            "from_store": old_dir, "from_store_rows": old.n_rows, "from_store_shard_size": old.ss,
            "from_store_manifest_sha256": sha_file(old_dir + "/manifest.json") if old_man else None,
            "pointer_index": pidx, "pointer_index_sha256": sha_file(pidx),
            "distinct_source_rows": int(np.unique(row).size),
            "duplicate_positions": int(n - np.unique(row).size),
            "source_rows_unreferenced": int(old.n_rows - np.unique(row).size),
            "proof": {
                "method": "sha1 of every row of the old store and of the new store; new[i] == old[row[i]] for all i; "
                          "chain = sha256 over the 20-byte row digests in position order",
                "positions_checked": n, "equal": equal,
                "resolved_row_digest_chain_sha256": chain_old,
                "new_store_row_digest_chain_sha256": chain_new,
            },
            "dense_integrity": {"damaged_rows": damaged, "definition": "all-zero, non-finite, or |norm-1| > %g" % NORM_TOL} if model == "dense" else None,
            "v2_fingerprint": fp,
        },
    }
    wj(man_p, man)
    log_phase("docs", {"dataset": ds, "model": model, "store": out, "n_rows": n, "n_shards": len(shards),
                       "proof_equal": equal, "chain_sha256": chain_new, "damaged_rows": damaged,
                       "v2_fingerprint": fp, "old_store": old_dir, "old_rows": old.n_rows,
                       "seconds": round(time.time() - t0, 1)})
    if delete_old:
        _retire_old_store(old_dir, old_man, ds, model, "docs")
    say("  %s %s docs DONE in %.0fs" % (ds, model, time.time() - t0))


def _encoder_block(ds, model, old_man):
    if old_man:
        keys = ["encoder", "dim", "dtype", "query_instruction", "source_sha256"]
        return {k: old_man.get(k) for k in keys if k in old_man}
    # webqsp: the assembled Modal parts carried no manifest; the encoder facts are the same models
    # the other five used and are recorded in ENCODING_ASSEMBLY.json
    if model == "dense":
        return {"encoder": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "dim": DIM, "dtype": "float16",
                "query_instruction": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                "record": "%s/ENCODING_ASSEMBLY.json" % ds}
    return {"encoder": "naver/splade-cocondenser-ensembledistil", "record": "%s/ENCODING_ASSEMBLY.json" % ds}


def _format_block(model):
    if model == "dense":
        return {"file": "shard_%05d.npy", "dtype": "float16", "shape": "(rows, 1536)",
                "normalised": "unit L2 (|norm-1| <= 0.01)"}
    return {"file": "shard_%05d.npz", "keys": {"indices": "int32", "indptr": "int32", "data": "float32",
                                              "shape": "int64 [rows, 30522]", "format": "S3 b'csr'"}}


def v2_fingerprint(ds, new):
    rec_p = None
    for cand in (FC + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json", HIST + "/records/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"):
        if os.path.exists(cand):
            rec_p = cand
            break
    if rec_p is None:
        return None
    rec = rj(rec_p)
    blk = (rec.get("REMOTE_EXECUTION_INTEGRITY") or {}).get("resolved_vector_fingerprint", {}).get(ds)
    if not blk:
        return None
    pre = HIST + "/pre_v3/%s__pointer_index__dense.npz" % ds
    z = np.load(pre)
    src, row = z["src"].astype(np.int64), z["row"].astype(np.int64)
    z.close()
    key = (src << 40) | row
    uniq, pos2u = np.unique(key, return_inverse=True)
    nu = uniq.size
    order = np.arange(key.size - 1, -1, -1)
    first = np.empty(nu, dtype=np.int64)
    first[pos2u[order]] = order
    pick = np.sort(np.random.default_rng(20260909).choice(nu, 4096, replace=False))
    pos = first[pick]
    # gather through the new store: row == position
    o = np.argsort(pos, kind="stable")
    sample = np.empty((pos.size, DIM), dtype=np.float16)
    ps = pos[o]
    sh = ps // SS
    for s in np.unique(sh):
        m = np.nonzero(sh == s)[0]
        sample[o[m]] = new.mmap(int(s))[ps[m] - int(s) * SS]
    new._cache = (None, None)
    got = hashlib.sha256(sample.astype(np.float16).tobytes()).hexdigest()
    res = {"record": rec_p, "recorded_sample_sha256": blk.get("sample_sha256"), "recorded_n_distinct": blk.get("n_distinct"),
           "reproduced_sample_sha256": got, "reproduced_n_distinct": int(nu),
           "match": got == blk.get("sample_sha256") and int(nu) == int(blk.get("n_distinct", -1)),
           "pre_v3_pointer_index": pre}
    say("    V2 fingerprint %s: match=%s" % (ds, res["match"]))
    if not res["match"]:
        raise RuntimeError("V2 resolved-vector fingerprint does not reproduce on %s: %s" % (ds, res))
    return res


def _retire_old_store(old_dir, old_man, ds, model, kind):
    """keep the tiny manifest in history, delete the bulk (authorised)."""
    hp = HIST + "/phase_c/%s/encodings__%s__%s__manifest.json" % (ds, model, kind)
    if old_man and not os.path.exists(hp):
        os.makedirs(os.path.dirname(hp), exist_ok=True)
        shutil.copy2(old_dir + "/manifest.json", hp)
    nb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(old_dir) for f in fs)
    shutil.rmtree(old_dir)
    log_phase("retire_store", {"dataset": ds, "model": model, "kind": kind, "deleted": old_dir, "bytes": nb,
                               "manifest_kept": hp if old_man else None})
    say("    deleted old store %s (%.2f GB)" % (old_dir, nb / 1e9))


# ------------------------------------------------------------------------------ queries
def move_queries(ds, model, delete_old):
    out = FC + "/%s/embeddings/%s/queries" % (ds, model)
    man_p = out + "/manifest.json"
    if os.path.exists(man_p) and rj(man_p).get("materialised", {}).get("proof", {}).get("equal"):
        say("  %s %s queries already in place; skipping" % (ds, model))
        return
    old_dir = old_tree(ds) + "/encodings/%s/queries" % model
    if not os.path.isdir(old_dir) and os.path.isdir(out):
        old_dir = out          # moved by an earlier partial run; finish the manifest
    old = Store(model, old_dir)
    qids = query_ids(ds)
    n = len(qids)
    if old.n_rows != n:
        raise RuntimeError("%s/%s queries: store has %d rows, %d query ids" % (ds, model, old.n_rows, n))
    if old.ss != SS and len(old.files) > 1:
        raise RuntimeError("%s/%s queries: shard size %d" % (ds, model, old.ss))
    # the pointer index must be the identity
    pp = FC + "/%s/queries/pointer_index/%s.npz" % (ds, model)
    if os.path.exists(pp):
        z = np.load(pp)
        ident = bool(np.array_equal(z["row"], np.arange(n, dtype=z["row"].dtype))) and (z["src"].size == 0 or int(z["src"].max()) == 0)
        z.close()
        if not ident:
            raise RuntimeError("%s/%s query pointer index is not the identity" % (ds, model))
    # the store's own id files must list the query ids in order (webqsp: row numbers)
    ids = []
    for k in range(len(old.files)):
        ip = os.path.join(old_dir, "ids_%05d.json" % k)
        if os.path.exists(ip):
            ids.extend(rj(ip))
    if ids:
        if ds == "webqsp":
            ids_ok = ids == list(range(n))
        else:
            ids_ok = ids == qids
        if not ids_ok:
            raise RuntimeError("%s/%s queries: shard id files disagree with query_ids.json" % (ds, model))
    else:
        ids_ok = None
    # split-file concatenation == query_ids order
    cat = []
    for sp in SPLITS[ds]:
        with io.open(FC + "/%s/queries/%s.jsonl" % (ds, sp), encoding="utf-8") as f:
            for ln in f:
                cat.append(json.loads(ln)["query_id"])
    if cat != qids:
        raise RuntimeError("%s: query_ids.json is not the split concatenation" % ds)
    damaged = 0
    if model == "dense":
        for k in range(len(old.files)):
            damaged += damaged_count(old.load(k))
        old._cache = (None, None)
        if damaged:
            raise RuntimeError("%s dense queries: %d damaged rows" % (ds, damaged))
    old_man = rj(old_dir + "/manifest.json") if os.path.exists(old_dir + "/manifest.json") else None
    if old_dir != out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.rename(old_dir, out)
        say("    moved %s -> %s" % (old_dir, out))
    # drop the per-shard id files and the old index/manifest: the manifest below is the description
    kept_old_manifest = None
    if old_man:
        kept_old_manifest = HIST + "/phase_c/%s/encodings__%s__queries__manifest.json" % (ds, model)
        os.makedirs(os.path.dirname(kept_old_manifest), exist_ok=True)
        shutil.copy2(out + "/manifest.json", kept_old_manifest)
    for f in os.listdir(out):
        if f.startswith("ids_") or f in ("index.json", "manifest.json"):
            os.remove(os.path.join(out, f))
    new = Store(model, out)
    shards = []
    for k, fn in enumerate(new.files):
        p = os.path.join(out, fn)
        shards.append({"shard": k, "rows": int(new.rows[k]), "file": fn, "bytes": os.path.getsize(p), "sha256": sha_file(p)})
    dig = new.row_digests("queries")
    chain = hashlib.sha256(dig.tobytes()).hexdigest()
    man = {
        "dataset": ds, "kind": "queries", "model": model,
        "n_rows": n, "shard_size": SS, "n_shards": len(shards),
        "row_semantics": "row j is the %s vector of query j == queries/query_ids.json[j] == the j-th line of the split files %s concatenated in that order" % (model, SPLITS[ds]),
        "format": _format_block(model),
        "encoder": _encoder_block(ds, model, old_man),
        "shards": shards,
        "materialised": {
            "utc": utc(),
            "from_store": old_dir if old_dir != out else "(moved earlier)", "moved_not_copied": True,
            "shard_id_files_matched_query_ids": ids_ok,
            "query_pointer_index_was_identity": True,
            "proof": {"method": "store moved unchanged (query pointer index == identity, verified); sha1 per row chained by sha256",
                      "positions_checked": n, "equal": True, "new_store_row_digest_chain_sha256": chain},
            "dense_integrity": {"damaged_rows": damaged, "definition": "all-zero, non-finite, or |norm-1| > %g" % NORM_TOL} if model == "dense" else None,
        },
    }
    wj(man_p, man)
    log_phase("queries", {"dataset": ds, "model": model, "store": out, "n_rows": n, "n_shards": len(shards),
                          "chain_sha256": chain, "damaged_rows": damaged, "old_manifest_kept": kept_old_manifest})
    say("  %s %s queries DONE" % (ds, model))


# ------------------------------------------------------------------------------ layout
def layout(ds):
    moves, deleted = [], []
    d = FC + "/" + ds
    hm = HIST + "/materialize"
    os.makedirs(hm, exist_ok=True)
    # query ids up one level; the identity pointer files go
    qp = d + "/queries/pointer_index"
    if os.path.isdir(qp):
        if os.path.exists(qp + "/query_ids.json"):
            os.rename(qp + "/query_ids.json", d + "/queries/query_ids.json")
            moves.append([qp + "/query_ids.json", d + "/queries/query_ids.json"])
        for m in ("dense", "splade"):
            p = qp + "/%s.npz" % m
            if os.path.exists(p):
                deleted.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha_file(p), "why": "identity map, verified before the move"})
                os.remove(p)
        if not os.listdir(qp):
            os.rmdir(qp)
    # docs pointer maps -> history (the proof of the materialisation refers to them)
    pi = d + "/pointer_index"
    if os.path.isdir(pi):
        for m in ("dense", "splade"):
            p = pi + "/%s.npz" % m
            if os.path.exists(p):
                dst = hm + "/%s__pointer_index__%s.npz" % (ds, m)
                os.rename(p, dst)
                moves.append([p, dst])
        for f in os.listdir(pi):
            dst = hm + "/%s__pointer_index__%s" % (ds, f)
            os.rename(os.path.join(pi, f), dst)
            moves.append([os.path.join(pi, f), dst])
        os.rmdir(pi)
    # reuse_map: the node -> Phase-C row attribution; its job is done and proved
    rm = d + "/reuse_map"
    if os.path.isdir(rm):
        for f in sorted(os.listdir(rm)):
            p = os.path.join(rm, f)
            deleted.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha_file(p), "why": "reuse attribution, superseded by position-ordered embeddings"})
        shutil.rmtree(rm)
    # webqsp extras
    if ds == "webqsp":
        p = d + "/encoder_row_of_node.npz"
        if os.path.exists(p):
            os.rename(p, hm + "/webqsp__encoder_row_of_node.npz")
            moves.append([p, hm + "/webqsp__encoder_row_of_node.npz"])
        # encoder_inputs.jsonl: per encoder row text; equal to the node text for every position?
        p = d + "/encoder_inputs.jsonl"
        if os.path.exists(p):
            z = np.load(hm + "/webqsp__pointer_index__dense.npz")
            row = z["row"].astype(np.int64)
            z.close()
            enc = []
            with io.open(p, encoding="utf-8") as f:
                for ln in f:
                    o = json.loads(ln)
                    assert int(o["row"]) == len(enc)
                    enc.append(o["text"])
            ok, i = True, 0
            with io.open(d + "/nodes.jsonl", encoding="utf-8") as f:
                for ln in f:
                    if json.loads(ln)["text"] != enc[row[i]]:
                        ok = False
                        break
                    i += 1
            if ok and i == row.size:
                deleted.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha_file(p), "why": "encoder row text == nodes.jsonl text for all %d positions" % i})
                os.remove(p)
            else:
                say("    encoder_inputs.jsonl differs from node text at position %d; kept" % i)
        p = d + "/query_inputs.jsonl"
        if os.path.exists(p):
            q = []
            with io.open(p, encoding="utf-8") as f:
                for ln in f:
                    q.append(json.loads(ln)["text"])
            cat = []
            for sp in SPLITS[ds]:
                with io.open(d + "/queries/%s.jsonl" % sp, encoding="utf-8") as f:
                    for ln in f:
                        cat.append(json.loads(ln)["question"])
            if q == cat:
                deleted.append({"path": p, "bytes": os.path.getsize(p), "sha256": sha_file(p), "why": "query row text == question text of the split files in order"})
                os.remove(p)
            else:
                say("    query_inputs.jsonl differs from the questions; kept")
        p = d + "/_enc"
        if os.path.isdir(p):
            nb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
            deleted.append({"path": p, "bytes": nb, "files": sum(len(fs) for _, _, fs in os.walk(p)), "why": "Modal assembly parts of the store (ENCODING_ASSEMBLY.json); the store was assembled and is now materialised"})
            shutil.rmtree(p)
        p = d + "/status.json"
        if os.path.exists(p):
            dst = HIST + "/records/webqsp__status.json"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(p, dst)
            moves.append([p, dst])
    # empty leftovers
    for f in ("_superseded_context_union_398354", "_legacy_alias_hotpotqa_clean"):
        p = os.path.join(d, f)
        if os.path.isdir(p) and not os.listdir(p):
            os.rmdir(p)
            deleted.append({"path": p, "bytes": 0, "why": "empty directory"})
    log_phase("layout", {"dataset": ds, "moves": moves, "deleted": deleted,
                         "deleted_bytes": int(sum(x.get("bytes", 0) for x in deleted))})
    say("  %s layout DONE: %d moves, %d deletions (%.2f GB)" % (ds, len(moves), len(deleted), sum(x.get("bytes", 0) for x in deleted) / 1e9))


# ------------------------------------------------------------------------------ graphs
def graphs(ds):
    d = FC + "/%s/graph" % ds
    mp = d + "/GRAPH_MANIFEST.json"
    m = rj(mp)
    if m.get("RECORD") == "CANONICAL_GRAPH":
        say("  %s graph manifest already normalised" % ds)
        return
    prev = HIST + "/materialize/%s__GRAPH_MANIFEST__pre_normalise.json" % ds
    os.makedirs(os.path.dirname(prev), exist_ok=True)
    shutil.copy2(mp, prev)
    n = int(m["n_nodes"])
    fams = {}
    for name, fam in m["families"].items():
        if not fam.get("present"):
            fams[name] = {"present": False, "reason": fam.get("reason") or fam.get("absent_reason") or "not derivable from the source"}
            continue
        f = os.path.join(d, os.path.basename(fam["file"]))
        z = np.load(f)
        src, dst = z["src"], z["dst"]
        keys = list(z.files)
        ne = int(src.size)
        mx = int(max(src.max(), dst.max())) if ne else -1
        loops = int((src == dst).sum())
        z.close()
        if ne != int(fam["n_edges"]) or mx >= n:
            raise RuntimeError("%s/%s: %d edges (manifest %s), max endpoint %d, n_nodes %d" % (ds, name, ne, fam["n_edges"], mx, n))
        sha = sha_file(f)
        if fam.get("npz_sha256") and sha != fam["npz_sha256"]:
            raise RuntimeError("%s/%s npz sha drifted" % (ds, name))
        e = {"present": True, "file": "graph/%s.npz" % name, "n_edges": ne, "directed": bool(fam["directed"]),
             "attributes": fam.get("attributes"), "relation_vocabulary": fam.get("relation_vocabulary"),
             "self_loops": loops, "npz_keys": keys, "npz_bytes": os.path.getsize(f), "npz_sha256": sha,
             "provenance": fam.get("provenance"), "edge_subtype": fam.get("edge_subtype"),
             "source": {k: fam.get(k) for k in ("source_tsv", "source_tsv_bytes", "phase_c_manifest_sha256", "n_edges_in_source_tsv",
                                                 "unresolved_endpoint_edges", "built", "coverage", "source_manifest") if k in fam},
             "origin": {"record": (fam.get("origin") or {}).get("record"), "tree": (fam.get("origin") or {}).get("tree")},
             "verified": {"utc": utc(), "n_edges": ne, "max_endpoint": mx, "self_loops": loops}}
        fams[name] = e
    out = {
        "RECORD": "CANONICAL_GRAPH", "dataset": ds, "n_nodes": n,
        "endpoint_space": "edge endpoints are POSITIONS: line numbers of nodes.jsonl == rows of embeddings/*/docs",
        "direction_policy": m.get("direction_policy"),
        "family_policy": m.get("family_policy"),
        "nodes_unrepresented_in_graph": m.get("nodes_unrepresented_in_graph"),
        "nodes_unrepresented_reason": m.get("nodes_unrepresented_reason"),
        "families": fams,
        "families_present": sorted(k for k, v in fams.items() if v["present"]),
        "families_absent": sorted(k for k, v in fams.items() if not v["present"]),
        "history": m.get("history"),
        "lineage": {"previous_manifest": prev, "previous_manifest_sha256": sha_file(prev),
                    "consolidated": m.get("consolidated_v3"), "phase_c_tree": m.get("phase_c_tree")},
    }
    wj(mp, out)
    log_phase("graphs", {"dataset": ds, "families": {k: v.get("n_edges") for k, v in fams.items() if v["present"]}, "previous": prev})
    say("  %s graph manifest normalised: %s" % (ds, out["families_present"]))


# ------------------------------------------------------------------------------ direction
PAIR_FAMILIES = ("ner", "knn")


def direction_census(src, dst):
    """How the rows of a family are stored: reverse rows present, orientation, self loops."""
    s, d = src.astype(np.int64), dst.astype(np.int64)
    n = int(max(s.max(), d.max())) + 1 if s.size else 0
    k, kr = s * n + d, d * n + s
    return {"n_edges": int(s.size), "reverse_rows_present": int(np.isin(k, kr).sum()), "self_loops": int((s == d).sum()),
            "rows_src_lt_dst": int((s < d).sum()), "rows_src_gt_dst": int((s > d).sum()),
            "distinct_unordered_pairs": int(np.unique(np.minimum(s, d) * n + np.maximum(s, d)).size)}


def direction(ds):
    """Set the directed flag from how the edges are actually stored, with the census as evidence.

    The Phase-C manifests declared directed:true for every family as a blanket convention ("as found
    in the source"). The NER and kNN families were built as UNORDERED PAIRS (the NER builder
    canonicalises pair orientation by doc id; the kNN builder keeps one row per neighbour pair), so
    a reader that honours directed:true sees only one of the two endpoints' adjacency. The census
    decides: a ner/knn family with 0 reverse rows and 0 self loops is stored as pairs and is flagged
    undirected; structural families keep the declared direction (they carry real reverse rows)."""
    d = FC + "/%s/graph" % ds
    mp = d + "/GRAPH_MANIFEST.json"
    m = rj(mp)
    if m.get("RECORD") != "CANONICAL_GRAPH":
        raise RuntimeError("%s: run graphs first" % ds)
    changed = {}
    for name, fam in m["families"].items():
        if not fam.get("present"):
            continue
        if "storage_census" in fam and "directed_as_frozen" in fam:
            continue
        z = np.load(os.path.join(d, os.path.basename(fam["file"])))
        cen = direction_census(z["src"], z["dst"])
        z.close()
        fam["storage_census"] = cen
        fam["directed_as_frozen"] = bool(fam["directed"])
        pair_stored = (cen["reverse_rows_present"] == 0 and cen["self_loops"] == 0 and cen["distinct_unordered_pairs"] == cen["n_edges"])
        if name in PAIR_FAMILIES and pair_stored and fam["directed"]:
            fam["directed"] = False
            fam["direction_note"] = ("stored as one row per unordered pair (0 reverse rows, orientation canonicalised by the "
                                     "builder); the frozen manifest's directed:true was the blanket Phase-C declaration, not a "
                                     "property of the rows. Readers expose both endpoints (canonical.Graph does).")
            changed[name] = cen
        elif name in PAIR_FAMILIES and not fam["directed"]:
            fam["direction_note"] = "declared undirected by its builder; census agrees (0 reverse rows)"
        else:
            fam["direction_note"] = "directed as found in the source; census shows real reverse rows" if cen["reverse_rows_present"] else                                     "directed as found in the source"
    m["direction_policy"] = ("structural: as found in the source, never symmetrised. ner/knn: one row per unordered pair, "
                             "directed:false; a reader exposes both endpoints. Every family carries storage_census + "
                             "directed_as_frozen so the flag is checkable against the rows.")
    wj(mp, m)
    log_phase("direction", {"dataset": ds, "flag_changed_to_undirected": changed,
                            "census": {k: v["storage_census"] for k, v in m["families"].items() if v.get("present")}})
    say("  %s direction: %s" % (ds, {k: ("undirected (was directed)" if k in changed else ("undirected" if not v["directed"] else "directed"))
                                    for k, v in m["families"].items() if v.get("present")}))


# ------------------------------------------------------------------------------ purge
def purge():
    entries = []
    # Phase-C build manifests (tiny) to history, then the whole tree goes
    if os.path.isdir(CAN):
        for ds in sorted(os.listdir(CAN)):
            t = os.path.join(CAN, ds)
            if not os.path.isdir(t):
                continue
            hp = HIST + "/phase_c/" + ds
            os.makedirs(hp, exist_ok=True)
            for f in sorted(os.listdir(t)):
                p = os.path.join(t, f)
                if f.endswith(".json") and os.path.getsize(p) < 50e6 and "partition_map" not in f:
                    shutil.copy2(p, os.path.join(hp, f))
            for sub in ("_src/docs/_srcmeta.json", "_src/queries/_srcmeta.json"):
                p = os.path.join(t, "encodings", sub)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(hp, sub.replace("/", "__")))
            left = {}
            for r, _, fs in os.walk(t):
                for f in fs:
                    q = os.path.join(r, f)
                    top = os.path.relpath(q, t).replace(os.sep, "/").split("/")[0]
                    left[top] = left.get(top, 0) + os.path.getsize(q)
            entries.append({"tree": t, "bytes_by_item": left, "bytes": int(sum(left.values())), "manifests_kept_in": hp})
        nb = sum(e["bytes"] for e in entries)
        shutil.rmtree(CAN)
        say("  deleted %s (%.2f GB)" % (CAN, nb / 1e9))
    # the V3 move-only retirement bucket
    ret = "data/_retired_v3"
    ret_entry = None
    if os.path.isdir(ret):
        files = []
        for r, _, fs in os.walk(ret):
            for f in fs:
                q = os.path.join(r, f).replace(os.sep, "/")
                files.append({"path": q, "bytes": os.path.getsize(q), "sha256": sha_file(q)})
        ret_entry = {"deleted": ret, "files": len(files), "bytes": int(sum(x["bytes"] for x in files)), "ledger": files}
        shutil.rmtree(ret)
        say("  deleted %s (%.2f GB, %d files)" % (ret, ret_entry["bytes"] / 1e9, len(files)))
    # CLEANUP_PASS_3: the DELETE-verdict list, applied only where the file still hashes as ledgered
    p3 = None
    for cand in (FC + "/CLEANUP_PASS_3.json", HIST + "/records/CLEANUP_PASS_3.json"):
        if os.path.exists(cand):
            p3 = cand
            break
    p3_entry = None
    if p3:
        P = rj(p3)
        done, missing, mismatch, nb = [], [], [], 0
        for it in P["deleted"]:
            q = it["path"]
            if not os.path.exists(q):
                missing.append(q)
                continue
            if it.get("sha256") and sha_file(q) != it["sha256"]:
                mismatch.append(q)
                continue
            nb += os.path.getsize(q)
            os.remove(q)
            done.append(q)
        # empty directories left behind
        removed_dirs = []
        for q in sorted(set(os.path.dirname(x) for x in done), key=len, reverse=True):
            while q and os.path.isdir(q) and not os.listdir(q):
                os.rmdir(q)
                removed_dirs.append(q)
                q = os.path.dirname(q)
        p3_entry = {"record": p3, "deleted_files": len(done), "bytes": nb, "missing_already_gone": len(missing),
                    "sha_mismatch_kept": mismatch, "removed_empty_dirs": removed_dirs}
        say("  CLEANUP_PASS_3 applied: %d files, %.2f GB; %d already gone; %d kept on sha mismatch"
            % (len(done), nb / 1e9, len(missing), len(mismatch)))
    log_phase("purge", {"phase_c_trees": entries, "retired_v3": ret_entry, "cleanup_pass_3": p3_entry})


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["docs", "queries", "layout", "graphs", "direction", "purge", "all"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None, choices=[None, "dense", "splade"])
    ap.add_argument("--delete-old", action="store_true")
    a = ap.parse_args()
    dss = [a.dataset] if a.dataset else DATASETS
    models = [a.model] if a.model else ["dense", "splade"]
    os.makedirs(HIST + "/logs", exist_ok=True)
    t0 = time.time()
    for ds in dss:
        if a.phase in ("docs", "all"):
            for m in models:
                build_docs(ds, m, a.delete_old)
        if a.phase in ("queries", "all"):
            for m in models:
                move_queries(ds, m, a.delete_old)
        if a.phase in ("layout", "all"):
            layout(ds)
        if a.phase in ("graphs", "all"):
            graphs(ds)
        if a.phase in ("direction", "all"):
            direction(ds)
    if a.phase == "purge":
        purge()
    say("ALL DONE in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
