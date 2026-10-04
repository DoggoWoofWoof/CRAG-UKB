"""CanonicalDataset -- the canonical-layout L1 adapter.

One object gives L1 everything it consumes, read straight from data/final_canonical/<ds>/ through
the pinned loader (canonical.py), indexed in canonical POSITION space:

    ds = CanonicalDataset("hotpotqa")
    ds.n_nodes, ds.n_queries, ds.eval_split, ds.split_ranges
    ds.eval_rows()                 query rows of the declared EVAL split (webqsp: the stride-2 carve)
    ds.gold(rows), ds.hops(rows)   gold POSITIONS per query / hop count (-1 when the dataset has none)
    ds.node_embeddings[i]          dense doc vector(s), float16, memory-mapped shards (zero copy)
    ds.query_embeddings[j]
    ds.dense_topk(k), ds.splade_topk(k)   int32 [n_queries, k] views of the frozen top-1000 caches
    ds.family("structural")        (src, dst, rel|weight) arrays, memory-mapped when the npz member is stored
    ds.keysets()                   (N, STRUCT, KNN, NERX) sorted undirected int64 keys u*N+v, u<v
    ds.ner_raw_keys()              NER keys before residualisation
    ds.struct_csr(directed)        CSR over positions (directed out-edges, or undirected deduped)
    ds.partition(tag)              position -> block id, when that L1 artefact exists
    ds.pins()                      every digest a consumer should record

Zero-copy discipline: embeddings and the retrieval caches are np.memmap views of the served
files; nothing is duplicated on disk. The only derived files are small L1-owned artefacts under
data/l1_canonical/<ds>/ (query index, keysets), each stamped with the DATASET.json RECORD_SHA256
it was built from and rebuilt when that changes.

Legacy paths (data/ukb_storage, data/processed/master_nodes_*, scratchpad/ablation_qwen,
scratchpad/_l1ep/parts, results/.../runs/cache_*.npz) are never read here.
"""
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FINAL_CANONICAL = os.path.join(REPO, "data", "final_canonical")
DERIVED = os.path.join(REPO, "data", "l1_canonical")
if FINAL_CANONICAL not in sys.path:
    sys.path.insert(0, FINAL_CANONICAL)
import canonical  # noqa: E402  (the pinned loader; wrapped, never edited)

DATASETS = list(canonical.DATASETS)
CACHE_DEPTH = 1000

# the frozen L1 algorithm files whose digests make up the L1 contract hash (numerics are imported
# from these, never re-implemented; a changed digest means a changed contract)
CONTRACT_FILES = [
    "scratchpad/_ta_prepartition.py", "scratchpad/_l1ps_cache.py", "scratchpad/_l1ps_router.py",
    "scratchpad/_l1hu_build.py", "scratchpad/_l1hu_local_worker.py", "scratchpad/_l1ep_part.py",
    "scratchpad/_l1ep_c.py", "scratchpad/_l1ep_pu.py", "scratchpad/_l1kn_sub.py",
    "scratchpad/_l1ov_core.py", "scratchpad/_l1ov_eval.py", "scratchpad/_l1hu_s.py",
    "src/l1_canonical/adapter.py", "src/l1_canonical/hypergraph.py", "src/l1_canonical/partition.py",
    "src/l1_canonical/replay_cache.py", "src/l1_canonical/l1_eval.py",
]
CONTRACT = {"K0": 60, "K": 100, "P_MAIN": 50, "SEED_K": 5, "BEAM": 64, "MAX_HOPS": 3, "DEG_CAP": 300,
            "SMAX": 256, "TOPP": 200, "EVAL_CAP": 2000, "MIN_VAL": 1000,
            "PARTITION": {"rule": "H4_SPLIT_PRESERVE", "families": ["STRUCT", "KNN"], "target_block_size": 100,
                          "partitioner": "mtkahypar", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1",
                          "epsilon": 0.03, "seed": 0}}


def sha_file(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _rj(p):
    return json.load(io.open(p, encoding="utf-8"))


def _npy_header(f):
    v = np.lib.format.read_magic(f)
    if v == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if v == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    from numpy.lib import _format_impl
    return _format_impl._read_array_header(f, v)


class NpzView(object):
    """np.load-like read of an .npz: a STORED member is memory-mapped (nothing decompressed, nothing
    copied); a DEFLATED member is decompressed into RAM on first access."""

    def __init__(self, path):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self.files = [n[:-4] for n in self._zip.namelist() if n.endswith(".npy")]
        self._loaded = {}

    def __contains__(self, k):
        return k in self.files

    def stored(self, k):
        return self._zip.getinfo(k + ".npy").compress_type == 0

    def __getitem__(self, k):
        if k in self._loaded:
            return self._loaded[k]
        zi = self._zip.getinfo(k + ".npy")
        if zi.compress_type != 0:
            with self._zip.open(zi) as fh:
                arr = np.load(io.BytesIO(fh.read()))
        else:
            with open(self.path, "rb") as f:
                f.seek(zi.header_offset)
                lh = f.read(30)
                fn_len = int.from_bytes(lh[26:28], "little")
                ex_len = int.from_bytes(lh[28:30], "little")
                f.seek(zi.header_offset + 30 + fn_len + ex_len)
                shape, fortran, dtype = _npy_header(f)
                off = f.tell()
            arr = np.memmap(self.path, dtype=dtype, mode="r", offset=off, shape=tuple(shape),
                            order="F" if fortran else "C")
        self._loaded[k] = arr
        return arr


class ShardedRows(object):
    """Row access over the position-ordered dense shards of one channel, memory-mapped lazily.
    Indexing returns float16 exactly as served (convert with .astype(np.float32) where the frozen
    code did its own normalisation)."""

    def __init__(self, emb):
        self.emb = emb
        self.n = int(emb.n_rows)
        self.shard_size = int(emb.shard_size)
        self.n_shards = int(emb.n_shards)
        self.dim = int(emb.manifest.get("dim") or canonical.DIM)
        self._maps = [None] * self.n_shards

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.n, self.dim)

    @property
    def dtype(self):
        return np.dtype(np.float16)

    def shard(self, k):
        m = self._maps[k]
        if m is None:
            m = np.load(self.emb._path(k), mmap_mode="r")
            self._maps[k] = m
        return m

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if i < 0:
                i += self.n
            if not 0 <= i < self.n:
                raise IndexError(i)
            return self.shard(i // self.shard_size)[i % self.shard_size]
        if isinstance(idx, slice):
            a, b, st = idx.indices(self.n)
            if st != 1:
                return self[np.arange(a, b, st)]
            if b <= a:
                return np.empty((0, self.dim), np.float16)
            sa, sb = a // self.shard_size, (b - 1) // self.shard_size
            if sa == sb:
                return self.shard(sa)[a % self.shard_size:(b - 1) % self.shard_size + 1]
            parts = [self.shard(sa)[a % self.shard_size:]]
            for s in range(sa + 1, sb):
                parts.append(self.shard(s))
            parts.append(self.shard(sb)[:(b - 1) % self.shard_size + 1])
            return np.concatenate(parts, axis=0)
        rows = np.asarray(idx)
        if rows.dtype == bool:
            rows = np.nonzero(rows)[0]
        rows = rows.astype(np.int64, copy=False)
        if rows.size == 0:
            return np.empty((0, self.dim), np.float16)
        if rows.min() < 0 or rows.max() >= self.n:
            raise IndexError("row out of range")
        out = np.empty((rows.size, self.dim), np.float16)
        sh = rows // self.shard_size
        for s in np.unique(sh):
            m = sh == s
            out[m] = self.shard(int(s))[rows[m] % self.shard_size]
        return out

    def blocks(self, block=50000):
        """(a, b, rows[a:b]) in position order -- the way the frozen cache builder walks nodes.npy."""
        for a in range(0, self.n, block):
            b = min(a + block, self.n)
            yield a, b, self[a:b]


_ID_RE = re.compile(r'"node_id":\s*"((?:[^"\\]|\\.)*)"')


class CanonicalDataset(object):
    def __init__(self, name, root=None, derived=None):
        if name not in DATASETS:
            raise ValueError("unknown dataset %r; one of %s" % (name, DATASETS))
        self.name = name
        self.ds = canonical.Dataset(name, root)
        self.dir = self.ds.dir
        self.root = self.ds.root
        self.derived_dir = os.path.join(derived or DERIVED, name)
        self.manifest = self.ds.manifest
        if not self.manifest or self.manifest.get("RECORD") != "CANONICAL_DATASET":
            raise RuntimeError("%s/DATASET.json is not a CANONICAL_DATASET record" % name)
        self.record_sha = self.manifest["RECORD_SHA256"]
        self.n_nodes = int(self.manifest["n_nodes"])
        self.n_queries = int(self.manifest["n_queries"])
        q = self.manifest["queries"]
        self.splits = list(q["splits_in_row_order"])
        self.split_ranges, a = {}, 0
        for sp in self.splits:
            n = int(q["by_split"][sp]["n"])
            self.split_ranges[sp] = (a, a + n)
            a += n
        if a != self.n_queries:
            raise RuntimeError("%s: split rows %d != n_queries %d" % (name, a, self.n_queries))
        self._freeze = None
        self._qidx = None
        self._keys = None
        self._ner_raw = None
        self._emb = {}
        self._caches = {}
        self._fams = {}
        self._csr = {}

    # ------------------------------------------------------------------ records
    @property
    def freeze(self):
        if self._freeze is None:
            self._freeze = _rj(os.path.join(self.root, "CANONICAL_FREEZE.json"))
        return self._freeze

    @property
    def eval_split(self):
        return self.freeze["EVAL_SPLITS"][self.name]["split"]

    @property
    def graph_manifest(self):
        return self.ds.graph_manifest

    def pins(self):
        """Digests a consumer records with anything derived from this dataset."""
        m = self.manifest
        g = m["graph"]["families"]
        return {
            "dataset": self.name,
            "DATASET_json_RECORD_SHA256": self.record_sha,
            "CANONICAL_FREEZE_RECORD_SHA256": self.freeze["RECORD_SHA256"],
            "nodes_jsonl_sha256": m["nodes"]["sha256"],
            "query_ids_sha256": m["queries"]["query_ids"]["sha256"],
            "split_files_sha256": {sp: m["queries"]["by_split"][sp]["sha256"] for sp in self.splits},
            "graph_files_sha256": {f: g[f].get("sha256") for f in g if g[f].get("present")},
            "embedding_manifests_sha256": {mdl: {kind: m["embeddings"][mdl][kind]["manifest"]["sha256"]
                                                for kind in ("docs", "queries")} for mdl in ("dense", "splade")},
            "retrieval_cache_sha256": {mdl: m["retrieval_cache"][mdl].get("sha256") for mdl in ("dense", "splade")},
            "eval_split": self.eval_split,
        }

    # ------------------------------------------------------------------ queries / gold
    @property
    def query_ids(self):
        return self.ds.query_ids

    def _query_index_path(self):
        return os.path.join(self.derived_dir, "query_index.npz")

    def query_index(self, rebuild=False, log=None):
        """gold positions, hops and split codes for every query row; built once from the split files
        and nodes.jsonl, stamped with the DATASET.json record digest."""
        if self._qidx is not None:
            return self._qidx
        p = self._query_index_path()
        if os.path.exists(p) and not rebuild:
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            if meta.get("DATASET_json_RECORD_SHA256") == self.record_sha:
                self._qidx = {k: z[k] for k in ("gold_ptr", "gold_pos", "hop", "split_code", "n_gold_ids")}
                self._qidx["meta"] = meta
                return self._qidx
        self._qidx = self._build_query_index(log or (lambda *a: None))
        return self._qidx

    def _build_query_index(self, log):
        t0 = time.time()
        gold_ids, hops, split_code, n_gold_ids = [], [], [], []
        need = set()
        for si, sp in enumerate(self.splits):
            n = 0
            for d in self.ds.queries(sp):
                g = d.get("gold_node_ids") or []
                gold_ids.append(g)
                n_gold_ids.append(len(g))
                need.update(g)
                h = d.get("hop")
                hops.append(int(h) if isinstance(h, (int, np.integer)) and not isinstance(h, bool) else -1)
                split_code.append(si)
                n += 1
            log("  %s/%s: %d rows read (%.0fs)" % (self.name, sp, n, time.time() - t0))
        if len(gold_ids) != self.n_queries:
            raise RuntimeError("%s: read %d query rows, manifest says %d" % (self.name, len(gold_ids), self.n_queries))
        pos = {}
        if self.name == "webqsp":
            for g in need:
                if not g.startswith("webqsp:n"):
                    raise RuntimeError("webqsp gold id without the position rule: %r" % g)
                pos[g] = int(g[8:])
        else:
            with io.open(os.path.join(self.dir, "nodes.jsonl"), encoding="utf-8") as f:
                for i, ln in enumerate(f):
                    m = _ID_RE.search(ln)
                    if m is None:
                        raise RuntimeError("%s nodes.jsonl line %d has no node_id" % (self.name, i))
                    nid = m.group(1)
                    if "\\" in nid:
                        nid = json.loads('"' + nid + '"')
                    if nid in need:
                        if nid in pos:
                            raise RuntimeError("duplicate node_id %r" % nid)
                        pos[nid] = i
            if i + 1 != self.n_nodes:
                raise RuntimeError("%s nodes.jsonl has %d lines, manifest says %d" % (self.name, i + 1, self.n_nodes))
        log("  %s: %d distinct gold ids, %d resolved (%.0fs)" % (self.name, len(need), len(pos), time.time() - t0))
        missing = need - set(pos)
        if missing:
            raise RuntimeError("%s: %d gold ids do not resolve, e.g. %s" % (self.name, len(missing), sorted(missing)[:3]))
        gold_ptr = np.zeros(self.n_queries + 1, np.int64)
        gold_ptr[1:] = np.cumsum([len(g) for g in gold_ids])
        gold_pos = np.empty(int(gold_ptr[-1]), np.int32)
        k = 0
        for g in gold_ids:
            for x in g:
                gold_pos[k] = pos[x]
                k += 1
        if self.name == "webqsp":
            # the served rows carry gold_positions too: they must agree with the id rule
            k = 0
            for sp in self.splits:
                for d in self.ds.queries(sp):
                    gp = d.get("gold_positions") or []
                    if list(gp) != [int(v) for v in gold_pos[gold_ptr[k]:gold_ptr[k + 1]]]:
                        raise RuntimeError("webqsp row %d: gold_positions disagree with gold_node_ids" % k)
                    k += 1
        meta = {"DATASET_json_RECORD_SHA256": self.record_sha, "nodes_jsonl_sha256": self.manifest["nodes"]["sha256"],
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
                "splits_in_row_order": self.splits, "n_queries": self.n_queries, "n_gold_refs": int(gold_ptr[-1]),
                "hop_field": "hop (int) or -1", "distinct_gold_ids": len(need)}
        os.makedirs(self.derived_dir, exist_ok=True)
        out = {"gold_ptr": gold_ptr, "gold_pos": gold_pos, "hop": np.array(hops, np.int8),
               "split_code": np.array(split_code, np.int8), "n_gold_ids": np.array(n_gold_ids, np.int32)}
        np.savez(self._query_index_path(), meta_json=json.dumps(meta), **out)
        out["meta"] = meta
        return out

    def split_rows(self, split):
        a, b = self.split_ranges[split]
        return np.arange(a, b, dtype=np.int64)

    def eval_rows(self):
        """Query rows of the declared EVAL split (never a test split). webqsp: every 2nd train
        query by sorted query_id, verified against the carve digest the freeze records."""
        sp = self.eval_split
        if sp in self.split_ranges:
            return self.split_rows(sp)
        if self.name == "webqsp" and sp == "train_holdout":
            a, b = self.split_ranges["train"]
            ids = self.query_ids
            tr = sorted(ids[a:b])
            stride = int(self.freeze.get("webqsp_holdout_stride", 2))
            hold = tr[::stride]
            sha = hashlib.sha256(",".join(hold).encode()).hexdigest()
            want = self._carve_sha()
            if want and sha != want:
                raise RuntimeError("webqsp train_holdout carve digest %s != recorded %s" % (sha[:16], want[:16]))
            row_of = {q: a + i for i, q in enumerate(ids[a:b])}
            return np.array(sorted(row_of[q] for q in hold), np.int64)
        raise RuntimeError("%s: eval split %r is not a served split" % (self.name, sp))

    def _carve_sha(self):
        def walk(o):
            if isinstance(o, dict):
                if "carve_ids_sha256" in o:
                    return o["carve_ids_sha256"]
                for v in o.values():
                    r = walk(v)
                    if r:
                        return r
            elif isinstance(o, list):
                for v in o:
                    r = walk(v)
                    if r:
                        return r
            return None
        return walk(self.freeze)

    def gold(self, rows):
        qi = self.query_index()
        return [qi["gold_pos"][qi["gold_ptr"][r]:qi["gold_ptr"][r + 1]] for r in np.asarray(rows, np.int64)]

    def hops(self, rows):
        return self.query_index()["hop"][np.asarray(rows, np.int64)]

    # ------------------------------------------------------------------ embeddings
    def embeddings(self, model, kind):
        key = (model, kind)
        if key not in self._emb:
            e = self.ds.embeddings(model, kind)
            self._emb[key] = ShardedRows(e) if model == "dense" else e
        return self._emb[key]

    @property
    def node_embeddings(self):
        return self.embeddings("dense", "docs")

    @property
    def query_embeddings(self):
        return self.embeddings("dense", "queries")

    # ------------------------------------------------------------------ retrieval caches
    def cache(self, model):
        if model not in self._caches:
            self._caches[model] = NpzView(os.path.join(self.dir, "retrieval_cache", "%s_top1000.npz" % model))
        return self._caches[model]

    def topk(self, model, k=CACHE_DEPTH, rows=None):
        """int32 positions, [len(rows) or n_queries, k]; a view of the served cache (k <= 1000,
        shallower k is a prefix by K_SEMANTICS)."""
        if not 1 <= k <= CACHE_DEPTH:
            raise ValueError("k must be in [1, %d]" % CACHE_DEPTH)
        ids = self.cache(model)["ids"]
        if rows is None:
            return ids[:, :k]
        return ids[np.asarray(rows, np.int64)][:, :k]

    def dense_topk(self, k=CACHE_DEPTH, rows=None):
        return self.topk("dense", k, rows)

    def splade_topk(self, k=CACHE_DEPTH, rows=None):
        return self.topk("splade", k, rows)

    def scores(self, model, rows=None):
        s = self.cache(model)["scores"]
        return s if rows is None else s[np.asarray(rows, np.int64)]

    # ------------------------------------------------------------------ graphs
    def families(self):
        return self.ds.families()

    def family(self, fam):
        """(src, dst, third) where third is rel (structural) or weight (ner/knn), plus the manifest
        entry; arrays are memory maps when the npz member is stored uncompressed."""
        if fam not in self._fams:
            ent = self.graph_manifest["families"].get(fam)
            if not ent or not ent.get("present"):
                raise ValueError("%s has no %r family" % (self.name, fam))
            z = NpzView(os.path.join(self.dir, "graph", os.path.basename(ent["file"])))
            third = z["rel"] if "rel" in z else (z["weight"] if "weight" in z else None)
            self._fams[fam] = (z["src"], z["dst"], third, ent)
        return self._fams[fam]

    @staticmethod
    def ukeys(u, v, N):
        """sorted unique undirected keys min*N+max, self loops dropped (== _l1kn_sub._ukeys)."""
        u = np.asarray(u, np.int64)
        v = np.asarray(v, np.int64)
        m = u != v
        u, v = u[m], v[m]
        return np.unique(np.minimum(u, v) * np.int64(N) + np.maximum(u, v))

    def _keys_path(self):
        return os.path.join(self.derived_dir, "keys.npz")

    def keysets(self, rebuild=False, log=None):
        """(N, STRUCT, KNN, NERX): the frozen key-set contract of _l1kn_sub.keysets over the served
        families -- KNN and NERX are residualised against STRUCT exactly as before."""
        if self._keys is not None:
            return self._keys
        p = self._keys_path()
        if os.path.exists(p) and not rebuild:
            z = np.load(p)
            meta = json.loads(str(z["meta_json"]))
            if meta.get("DATASET_json_RECORD_SHA256") == self.record_sha:
                self._keys = (int(z["N"][0]), z["STRUCT"], z["KNN"], z["NERX"])
                self._ner_raw = z["NER_RAW"]
                return self._keys
        log = log or (lambda *a: None)
        t0 = time.time()
        N = self.n_nodes
        s, d, _, _ = self.family("structural")
        STRUCT = self.ukeys(s, d, N)
        s, d, _, _ = self.family("knn")
        A = self.ukeys(s, d, N)
        KNN = np.setdiff1d(A, STRUCT, assume_unique=True)
        del A
        s, d, _, _ = self.family("ner")
        NER_RAW = self.ukeys(s, d, N)
        NERX = np.setdiff1d(NER_RAW, STRUCT, assume_unique=True)
        meta = {"DATASET_json_RECORD_SHA256": self.record_sha, "N": N,
                "STRUCT": int(len(STRUCT)), "KNN": int(len(KNN)), "NERX": int(len(NERX)), "NER_RAW": int(len(NER_RAW)),
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
                "rule": "undirected keys min*N+max, self loops dropped, KNN = KNN \\ STRUCT, NERX = NER \\ STRUCT (== _l1kn_sub)"}
        os.makedirs(self.derived_dir, exist_ok=True)
        np.savez(p, N=np.array([N]), STRUCT=STRUCT, KNN=KNN, NERX=NERX, NER_RAW=NER_RAW, meta_json=json.dumps(meta))
        log("  %s keys: STRUCT=%d KNN=%d NERX=%d NER_RAW=%d (%.0fs)" % (self.name, len(STRUCT), len(KNN), len(NERX), len(NER_RAW), time.time() - t0))
        self._keys = (N, STRUCT, KNN, NERX)
        self._ner_raw = NER_RAW
        return self._keys

    def ner_raw_keys(self):
        self.keysets()
        return self.n_nodes, self._ner_raw

    @staticmethod
    def csr_from_keys(keys, N):
        """symmetric CSR (xadj, adjncy) from undirected keys (== _l1ep_part._csr_from_keys)."""
        u = (keys // np.int64(N)).astype(np.int64)
        v = (keys % np.int64(N)).astype(np.int64)
        r = np.concatenate([u, v])
        c = np.concatenate([v, u])
        o = np.lexsort((c, r))
        r, c = r[o], c[o]
        deg = np.bincount(r, minlength=N).astype(np.int64)
        xadj = np.zeros(N + 1, np.int64)
        xadj[1:] = np.cumsum(deg)
        return xadj, c

    def struct_csr(self, directed):
        """directed=True: out-edge CSR of the structural rows as served (duplicates and self loops
        kept, the way master_nodes.neighbors was iterated); directed=False: the undirected, deduped,
        loop-free adjacency the frozen expansion contract walks (sorted neighbours)."""
        key = bool(directed)
        if key in self._csr:
            return self._csr[key]
        N = self.n_nodes
        if directed:
            s, d, _, _ = self.family("structural")
            s = np.asarray(s, np.int64)
            d = np.asarray(d, np.int64)
            o = np.argsort(s, kind="stable")
            deg = np.bincount(s, minlength=N).astype(np.int64)
            xadj = np.zeros(N + 1, np.int64)
            xadj[1:] = np.cumsum(deg)
            self._csr[key] = (xadj, d[o].astype(np.int32))
        else:
            _, STRUCT, _, _ = self.keysets()
            xadj, adj = self.csr_from_keys(STRUCT, N)
            self._csr[key] = (xadj, adj.astype(np.int32))
        return self._csr[key]

    # ------------------------------------------------------------------ L1-owned artefacts
    def partition_path(self, tag="H4_SK"):
        return os.path.join(self.derived_dir, "parts", "%s.npy" % tag)

    def partition(self, tag="H4_SK"):
        p = self.partition_path(tag)
        if not os.path.exists(p):
            return None
        hard = np.load(p)
        if len(hard) != self.n_nodes:
            raise RuntimeError("%s partition %s has %d rows, corpus has %d" % (self.name, tag, len(hard), self.n_nodes))
        return hard.astype(np.int64)

    def cache_path(self):
        return os.path.join(self.derived_dir, "replay_cache.npz")


def contract_hash(repo=REPO):
    """sha256 over (name, digest) of every frozen algorithm file that exists, plus the constants."""
    h = hashlib.sha256()
    parts = {}
    for rel in CONTRACT_FILES:
        p = os.path.join(repo, rel)
        if os.path.exists(p):
            parts[rel] = sha_file(p)
    for rel in sorted(parts):
        h.update(("%s:%s\n" % (rel, parts[rel])).encode())
    h.update(json.dumps(CONTRACT, sort_keys=True).encode())
    return {"L1_CONTRACT_SHA256": h.hexdigest(), "files": parts, "constants": CONTRACT}


if __name__ == "__main__":
    for name in (sys.argv[1:] or DATASETS):
        d = CanonicalDataset(name)
        print(name, d.n_nodes, d.n_queries, d.splits, d.eval_split, len(d.eval_rows()))
