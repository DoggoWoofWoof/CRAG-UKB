"""The frozen L1 replay cache (Dense + SPLADE -> RRF -> partition candidates -> H4_SK -> structural
expansion) over canonical positions -- scratchpad/_l1ps_cache.build with its INPUTS injected.

Numerics are imported from the frozen modules (_ta_prepartition.partition_ranking / rrf_partitions
/ residual / expand_dir / node_rrf and every frozen constant); the traversal body below is the
legacy body line for line.  Two input providers exist:

    LegacyInputs(ds)     the legacy universes exactly as _l1ps_cache.py read them (ukb_storage,
                         _ta topology cache, variant_C METIS partition) -- used ONLY to prove the
                         fork reproduces results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_<ds>.npz
    CanonicalInputs(ds)  CanonicalDataset: EVAL_SPLITS population, served top-1000 caches, served
                         unit-L2 embeddings (zero copy), served structural family, H4_SK partition

    python src/l1_canonical/replay_cache.py parity [legacy_ds ...]     -> results/L1_CANONICAL/REPLAY_CACHE_PARITY.json
    python src/l1_canonical/replay_cache.py census                      -> bytes/query x queries, free-space floor
    python src/l1_canonical/replay_cache.py build <ds> [ds ...]         -> data/l1_canonical/<ds>/replay_cache.npz

The canonical cache is frozen against: dataset hash (DATASET.json RECORD_SHA256), query hash
(query_ids.json + split file digests + the sampled row list), node-order hash (nodes.jsonl
sha256), partition hash (H4_SK.npy sha256 + its manifest), and the L1 contract hash.  Array names
and dtypes are the legacy cache's, so _l1ep_c.rebuild / _l1ps_router.evaluate read it unchanged.
No legacy cache is ever reused as an input.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO, os.path.join(REPO, "scratchpad")):
    if p not in sys.path:
        sys.path.insert(0, p)
import _ta_prepartition as TA  # noqa: E402  (frozen numerics; imported, never copied)
from src.l1_canonical.adapter import CanonicalDataset, ShardedRows, contract_hash, sha_file  # noqa: E402

# frozen constants of _l1ps_cache.py
BEAM_MAIN = 64
TOPP = 200
SMAX = 256
EVAL_CAP = 2000
MIN_VAL = 1000
FP16_THRESHOLD_BYTES = 1.2e9          # legacy: BIG = ndocs * 1536 * 4 > 1.2e9
LEGACY_RUNS = os.path.join(REPO, "results", "GENERALIZATION", "G2_L1_PARTITION_SEARCH", "runs")
FREE_SPACE_FLOOR_GB = 5.0
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


class F32View:
    """rows of a float16 store as float32 on access (legacy F32View, generalised to any indexable)."""

    def __init__(self, A):
        self.A = A

    def __getitem__(self, k):
        return np.asarray(self.A[k], np.float32)

    def __len__(self):
        return len(self.A)


# ======================================================================== input providers
class LegacyInputs(object):
    """What _l1ps_cache.build read, from where it read it.  Parity use only."""

    def __init__(self, ds, log=log):
        self.name, self.tag = ds, "legacy"
        self.hard, self.mem, self.npart, self.adj, self.deg, self.id2row = TA.load_topology(ds, log)
        EMB = os.path.join(REPO, "data", "ukb_storage", ds, "gte_qwen") + os.sep
        j = json.load(open(EMB + "query_ids_all.json"))
        self.ids, self.golds = j["ids"], j["golds"]
        self.hops_all = j.get("hops", [None] * len(j["ids"]))
        si = j["split_indices"]
        self.rows = list(si["val"]) if si.get("val") else list(si["all"])
        self.sample_rule = "val"
        if len(self.rows) < MIN_VAL and si.get("train"):
            self.rows = self.rows + list(si["train"])
            self.sample_rule = "val+train (val<1000; TEST never touched)"
        self.val_set = set(si.get("val", []))
        self._dense = np.load(EMB + "dense_top200_all.npy", mmap_mode="r")
        self._splade = np.load(EMB + "splade_top200_all.npy", mmap_mode="r")
        self._q = np.load(EMB + "queries_all.npy", mmap_mode="r")
        self._nodes = np.load(EMB + "nodes.npy", mmap_mode="r")
        self.ndocs = len(self.hard)
        self.BIG = self.ndocs * 1536 * 4 > FP16_THRESHOLD_BYTES

    def hop(self, r):
        return self.hops_all[r]

    def gold_rows(self, r):
        return [self.id2row[x] for x in self.golds[r] if x in self.id2row]

    def d200(self, rows):
        return np.stack([np.asarray(self._dense[i]) for i in rows]).astype(np.int64)

    def s200(self, rows):
        return np.stack([np.asarray(self._splade[i]) for i in rows]).astype(np.int64)

    def Q(self, rows):
        return np.stack([np.asarray(self._q[i], np.float32) for i in rows])

    def Xn(self, log=log):
        nodes, ndocs = self._nodes, self.ndocs
        if self.BIG:
            X16 = np.empty((ndocs, nodes.shape[1]), np.float16)
            for a in range(0, ndocs, 50000):
                b = min(a + 50000, ndocs)
                blk = np.array(nodes[a:b], np.float32)
                blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
                X16[a:b] = blk.astype(np.float16)
                del blk
            return F32View(X16), "legacy fp16 path (normalised fp32 -> fp16 in RAM)"
        Xn = np.array(nodes, np.float32)
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
        return Xn, "fp32 exact"

    def row_ids(self, rows):
        return [self.ids[r] for r in rows]

    def pins(self):
        return {"provider": "LegacyInputs", "dataset": self.name, "note": "legacy universes; parity use only"}


class CanonicalInputs(object):
    """CanonicalDataset -> the same interface, in canonical position space."""

    def __init__(self, ds, tag="H4_SK", log=log):
        self.name, self.tag = ds, tag
        self.d = d = CanonicalDataset(ds)
        hard = d.partition(tag)
        if hard is None:
            raise RuntimeError("%s: no partition %s (data/l1_canonical/%s/parts/%s.npy) -- partition first" % (ds, tag, ds, tag))
        pm_path = os.path.join(d.derived_dir, "parts", "%s.json" % tag)
        self.part_manifest = json.load(io.open(pm_path, encoding="utf-8"))
        if self.part_manifest["inputs"]["DATASET_json_RECORD_SHA256"] != d.record_sha:
            raise RuntimeError("%s: partition %s was built from another DATASET.json record" % (ds, tag))
        if self.part_manifest["output"]["sha256"] != sha_file(d.partition_path(tag)):
            raise RuntimeError("%s: partition file digest changed since its manifest" % ds)
        self.hard = hard
        self.npart = int(self.part_manifest["contract"]["k"])
        if hard.max() >= self.npart or hard.min() < 0:
            raise RuntimeError("%s: block ids outside [0, k)" % ds)
        N = d.n_nodes
        # mem_idx: own block + blocks of the DIRECTED out-neighbours (legacy load_topology semantics)
        xo, ao = d.struct_csr(directed=True)
        deg_out = np.diff(xo)
        r = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
        p = np.concatenate([hard, hard[ao.astype(np.int64)]])
        keys = np.unique(r * np.int64(self.npart) + p)
        del r, p
        rr = keys // self.npart
        mlen = np.bincount(rr, minlength=N).astype(np.int64)
        mem_ptr = np.zeros(N + 1, np.int64)
        mem_ptr[1:] = np.cumsum(mlen)
        self.mem = (mem_ptr, (keys % self.npart).astype(np.int32))
        del keys, rr
        d._csr.clear()
        # expansion adjacency: undirected, deduped, loop-free, sorted neighbours
        adjp, adji = d.struct_csr(directed=False)
        self.adj = (adjp, adji)
        self.deg = np.diff(adjp).astype(np.int32)
        d._csr.clear()
        self.rows = [int(x) for x in d.eval_rows()]
        self.sample_rule = "EVAL_SPLITS %s" % d.eval_split
        self.val_set = set(self.rows)
        if len(self.rows) < MIN_VAL and "train" in d.split_ranges and d.eval_split != "train_holdout":
            self.rows = self.rows + [int(x) for x in d.split_rows("train")]
            self.sample_rule += " + train (eval<1000; TEST never touched)"
        self.ndocs = N
        self.BIG = N * 1536 * 4 > FP16_THRESHOLD_BYTES
        self._hop = d.query_index()["hop"]

    def hop(self, r):
        h = int(self._hop[r])
        return None if h < 0 else h

    def gold_rows(self, r):
        return [int(x) for x in self.d.gold([r])[0]]

    def d200(self, rows):
        return np.asarray(self.d.dense_topk(TA.TOP200, rows), np.int64)

    def s200(self, rows):
        return np.asarray(self.d.splade_topk(TA.TOP200, rows), np.int64)

    def Q(self, rows):
        return self.d.query_embeddings[np.asarray(rows, np.int64)].astype(np.float32)

    def Xn(self, log=log):
        E = self.d.node_embeddings
        if self.BIG:
            # served vectors are unit-L2 float16 by the freeze (K_SEMANTICS); the legacy fp16 path
            # re-normalised into an 18 GB RAM copy -- here the shards stay memory-mapped (zero copy)
            return F32View(E), "canonical fp16 path (served unit-L2 shards, memory-mapped, no copy)"
        Xn = np.empty((self.ndocs, E.dim), np.float32)
        for a, b, blk in E.blocks(50000):
            Xn[a:b] = blk
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
        return Xn, "fp32 exact"

    def row_ids(self, rows):
        ids = self.d.query_ids
        return [ids[r] for r in rows]

    def pins(self):
        d = self.d
        return {"provider": "CanonicalInputs", **d.pins(),
                "partition": {"tag": self.tag, "file": self.part_manifest["output"]["file"], "sha256": self.part_manifest["output"]["sha256"],
                              "k": self.npart, "hypergraph_content_digest": self.part_manifest["inputs"]["hypergraph_content_digest"],
                              "worker_sha256": self.part_manifest["contract"]["worker_sha256"]}}


# ======================================================================== the frozen body
def build(I, out, log=log, extra_meta=None):
    """_l1ps_cache.build with the inputs coming from I.  Writes the cache to `out`, returns meta."""
    DS = I.name
    log("=== L1 REPLAY CACHE %s [%s] ===" % (DS, I.tag))
    hard, mem, npart, (adjp, adji), deg = I.hard, I.mem, I.npart, I.adj, I.deg
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
    ndocs = len(hard)
    BIG = I.BIG
    est_gb = (ndocs * 1536 * (2 if BIG else 4) + adji.nbytes + ndocs * 24) / 1e9
    log("docs=%d npart=%d predicted_peak~%.2fGB fp16_path=%s" % (ndocs, npart, est_gb, BIG))

    np.random.seed(0)                                   # legacy: module-level seed, first RNG use is here
    rows = list(I.rows)
    sample_rule = I.sample_rule
    val_set = I.val_set
    if DS == "metaqa":
        buck = {1: [], 2: [], 3: []}
        for r in rows:
            if I.hop(r) in buck:
                buck[I.hop(r)].append(r)
        per = EVAL_CAP // 3
        sel = []
        for h in (1, 2, 3):
            b = list(buck[h])
            np.random.shuffle(b)
            sel += b[:per]
        rows = sorted(sel)
        sample_rule += ", hop-balanced (identical to _ta_resid.py / _ta_comb.py)"
    else:
        np.random.shuffle(rows)
        rows = sorted(rows[:EVAL_CAP])

    keep, gold_rows = [], []
    for r in rows:
        g = I.gold_rows(r)
        if g:
            keep.append(r)
            gold_rows.append(g)
    rows = keep
    nq = len(rows)
    log("dev queries=%d (%s) golds/q=%.2f" % (nq, sample_rule, np.mean([len(g) for g in gold_rows])))

    d200 = I.d200(rows)
    s200 = I.s200(rows)
    dK, sK = d200[:, :TA.K_LOCK], s200[:, :TA.K_LOCK]

    log("node embeddings ...")
    Xn, xn_path = I.Xn(log)
    Qm = I.Q(rows)
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)

    # ---- canonical BASE partition ranking (STEP 0)
    t = time.time()
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    t_base = time.time() - t
    base_sel = [set(int(x) for x in base_rank[qi][:TA.P_MAIN]) for qi in range(nq)]
    base_scope = np.array([int(part_sizes[sorted(s)].sum()) for s in base_sel], np.int64)
    allb = anyb = 0
    for qi in range(nq):
        c = [int(hard[g]) in base_sel[qi] for g in gold_rows[qi]]
        allb += all(c)
        anyb += any(c)
    BASE_ALL, BASE_ANY = allb / nq, anyb / nq
    log("BASE_ALL=%.4f BASE_ANY=%.4f scope=%.1f (%.1fs)" % (BASE_ALL, BASE_ANY, base_scope.mean(), t_base))

    # ---- retrieval continuations (canonical, unfiltered)
    fused200 = [TA.node_rrf(d200[qi], s200[qi]) for qi in range(nq)]
    ret_rrf = np.full((nq, TA.TOP200), -1, np.int32)
    for qi in range(nq):
        f = fused200[qi][:TA.TOP200]
        ret_rrf[qi, :len(f)] = f

    # ---- frozen structural expansion (RRF seeds, beam 64), UNFILTERED + provenance
    t = time.time()
    esc = 0
    s_node = np.full((nq, SMAX), -1, np.int32)
    s_hop = np.zeros((nq, SMAX), np.int8)
    s_sdir = np.zeros((nq, SMAX), np.float32)
    s_cnt = np.zeros((nq, SMAX), np.int32)
    seeds_arr = np.full((nq, TA.SEED_K), -1, np.int32)
    for qi in range(nq):
        sd = fused200[qi][:TA.SEED_K]
        seeds_arr[qi, :len(sd)] = sd
        r_q = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        a, vm, e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, BEAM_MAIN)
        esc += e
        k = min(len(a), SMAX)
        for jj in range(k):
            v = int(a[jj])
            m = vm.get(v, [0, 0.0, 0])
            s_node[qi, jj] = v
            s_hop[qi, jj] = m[0]
            s_sdir[qi, jj] = m[1]
            s_cnt[qi, jj] = m[2]
        if (qi + 1) % 500 == 0:
            log("   struct %d/%d edges=%s" % (qi + 1, nq, "{:,}".format(esc)))
    t_struct = time.time() - t
    log("struct done: avail/q=%.1f edges=%s %.1fs" % (float((s_node >= 0).sum(1).mean()), "{:,}".format(esc), t_struct))

    # ---- gold partitions (evaluation labels only)
    gp, gc, gptr = [], [], [0]
    for qi in range(nq):
        cnt = {}
        for g in gold_rows[qi]:
            p = int(hard[g])
            cnt[p] = cnt.get(p, 0) + 1
        for p in sorted(cnt):
            gp.append(p)
            gc.append(cnt[p])
        gptr.append(len(gp))

    meta = {"dataset": DS, "n_dev_queries": nq, "sample_rule": sample_rule, "n_docs": int(ndocs),
            "npart": int(npart), "fp16_path": bool(BIG), "node_embedding_path": xn_path, "predicted_peak_gb": round(est_gb, 2),
            "BASE_ALL_P50": round(BASE_ALL, 4), "BASE_ANY_P50": round(BASE_ANY, 4),
            "BASE_SCOPE_NODES": round(float(base_scope.mean()), 1),
            "BASE_PARTITION_PARITY": {"status": "NO_REFERENCE" if I.tag != "legacy" else "LEGACY_REBUILD"},
            "CONTRACT": {"K0": TA.K0, "K": TA.K_LOCK, "P_MAIN": TA.P_MAIN, "SEED_K": TA.SEED_K,
                         "BEAM": BEAM_MAIN, "MAX_HOPS": TA.MAX_HOPS, "DEG_CAP": TA.DEG_CAP,
                         "SMAX": SMAX, "TOPP": TOPP},
            "COST": {"BASE_sec": round(t_base, 1), "STRUCT_sec": round(t_struct, 1),
                     "STRUCT_edges": int(esc), "STRUCT_x_BASE": round(t_struct / max(t_base, 1e-9), 1)},
            "mean_partition_size": round(float(part_sizes.mean()), 2),
            "row_query_ids": I.row_ids(rows), "row_query_ids_sha256": hashlib.sha256(",".join(I.row_ids(rows)).encode()).hexdigest(),
            "input_provider": I.tag, "pins": I.pins(), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder": "src/l1_canonical/replay_cache.py (frozen _l1ps_cache body, _ta_prepartition numerics)"}
    if extra_meta:
        meta.update(extra_meta)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, rows=np.array(rows, np.int64),
        base_rank=base_rank[:, :TOPP].astype(np.int32),
        base_scope=base_scope,
        gold_part=np.array(gp, np.int32), gold_cnt=np.array(gc, np.int32),
        gold_ptr=np.array(gptr, np.int64),
        s_node=s_node, s_hop=s_hop, s_sdir=s_sdir, s_cnt=s_cnt, seeds=seeds_arr,
        ret_rrf=ret_rrf, ret_dense=d200[:, :TA.TOP200].astype(np.int32),
        ret_splade=s200[:, :TA.TOP200].astype(np.int32),
        hard=hard.astype(np.int32), part_sizes=part_sizes.astype(np.int64),
        hops=np.array([-1 if I.hop(r) is None else int(I.hop(r)) for r in rows], np.int8),
        in_val=np.array([1 if r in val_set else 0 for r in rows], np.int8),
        meta_json=json.dumps(meta))
    log("wrote %s  (%.1f MB)" % (out, os.path.getsize(out) / 1e6))
    return meta


# ======================================================================== parity vs legacy caches
CACHE_KEYS = ["rows", "base_rank", "base_scope", "gold_part", "gold_cnt", "gold_ptr", "s_node", "s_hop", "s_sdir", "s_cnt",
              "seeds", "ret_rrf", "ret_dense", "ret_splade", "hard", "part_sizes", "hops", "in_val"]


def parity(names, scratch):
    res = {}
    for ds in names:
        ref = os.path.join(LEGACY_RUNS, "cache_%s.npz" % ds)
        if not os.path.exists(ref):
            res[ds] = {"status": "SKIPPED", "reason": "no legacy cache"}
            continue
        out = os.path.join(scratch, "parity_cache_%s.npz" % ds)
        I = LegacyInputs(ds)
        meta = build(I, out)
        z, zr = np.load(out, allow_pickle=True), np.load(ref, allow_pickle=True)
        mr = json.loads(str(zr["meta_json"]))
        same, detail = {}, {}
        for k in CACHE_KEYS:
            a, b = z[k], zr[k]
            eq = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            same[k] = bool(eq)
            if not eq and a.shape == b.shape and a.dtype.kind == "f":
                detail[k] = {"max_abs_diff": float(np.abs(a.astype(np.float64) - b).max()), "n_diff": int((a != b).sum())}
            elif not eq:
                detail[k] = {"shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)],
                             "n_diff": int((a != b).sum()) if a.shape == b.shape else None}
        metas = {k: [meta.get(k), mr.get(k)] for k in ("n_dev_queries", "BASE_ALL_P50", "BASE_ANY_P50", "BASE_SCOPE_NODES", "npart", "n_docs")}
        ok = all(same.values()) and all(v[0] == v[1] for v in metas.values())
        res[ds] = {"status": "IDENTICAL" if ok else "DIFFERENT", "arrays_equal": same, "diff_detail": detail, "meta": metas,
                   "legacy_cache_sha256": sha_file(ref), "legacy_keys": sorted(zr.files), "rebuilt_keys": sorted(z.files)}
        log("%-15s %s  %s" % (ds, res[ds]["status"], {k: v for k, v in same.items() if not v} or "all %d arrays equal" % len(CACHE_KEYS)))
        z.close()
        zr.close()
        os.remove(out)
    return res


# ======================================================================== disk census
def census(names):
    """expected cache bytes per query x queries, per dataset, against the free-space floor."""
    per_q = (TOPP * 4 + 8 + SMAX * (4 + 1 + 4 + 4) + TA.SEED_K * 4 + 3 * TA.TOP200 * 4 + 1 + 1 + 8)   # uncompressed per query
    out = {"bytes_per_query_uncompressed": per_q, "datasets": {}}
    tot = 0
    for ds in names:
        d = CanonicalDataset(ds)
        nq = min(EVAL_CAP, len(d.eval_rows()))
        fixed = d.n_nodes * (4 + 8)                     # hard int32 + part_sizes int64 ~ N
        b = per_q * nq + fixed
        tot += b
        out["datasets"][ds] = {"queries": int(nq), "n_nodes": d.n_nodes, "bytes_uncompressed": int(b), "MB": round(b / 1e6, 1)}
    du = shutil.disk_usage(REPO)
    out.update({"total_uncompressed_MB": round(tot / 1e6, 1), "disk_free_GB": round(du.free / 1e9, 2), "floor_GB": FREE_SPACE_FLOOR_GB,
                "ok": du.free / 1e9 - tot / 1e9 > FREE_SPACE_FLOOR_GB})
    return out


def build_canonical(ds, tag="H4_SK"):
    c = census([ds])
    if not c["ok"]:
        raise RuntimeError("free space %.1f GB would drop under the %.1f GB floor" % (c["disk_free_GB"], FREE_SPACE_FLOOR_GB))
    I = CanonicalInputs(ds, tag)
    out = I.d.cache_path()
    ch = contract_hash()
    meta = build(I, out, extra_meta={"FROZEN_AGAINST": {
        "dataset_hash": I.d.record_sha, "nodes_jsonl_sha256(node_order)": I.d.manifest["nodes"]["sha256"],
        "query_hash": {"query_ids_sha256": I.d.manifest["queries"]["query_ids"]["sha256"],
                       "split_files_sha256": {sp: I.d.manifest["queries"]["by_split"][sp]["sha256"] for sp in I.d.splits}},
        "partition_hash": I.part_manifest["output"]["sha256"], "L1_CONTRACT_SHA256": ch["L1_CONTRACT_SHA256"],
        "retrieval_cache_sha256": I.d.pins()["retrieval_cache_sha256"]}})
    man = {"dataset": ds, "cache": os.path.relpath(out, REPO).replace("\\", "/"), "bytes": os.path.getsize(out), "sha256": sha_file(out),
           "meta": {k: v for k, v in meta.items() if k != "row_query_ids"}, "contract_files": ch["files"]}
    with io.open(out[:-4] + ".json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "parity":
        scratch = os.environ.get("L1_SCRATCH") or os.path.join(REPO, "data", "l1_canonical", "_parity")
        os.makedirs(scratch, exist_ok=True)
        names = a[1:] or ["metaqa", "musique_clean", "squad_clean", "2wiki_clean", "hotpotqa_clean", "webqsp"]
        res = parity(names, scratch)
        p = os.path.join(REPO, "results", "L1_CANONICAL", "REPLAY_CACHE_PARITY.json")
        prev = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {"datasets": {}}
        prev["datasets"].update(res)
        prev.update({"RECORD": "L1_CANONICAL_REPLAY_CACHE_PARITY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "what": "src/l1_canonical/replay_cache.build on LegacyInputs reproduces the legacy runs/cache_<ds>.npz arrays",
                     "frozen_numerics_sha256": {"scratchpad/_ta_prepartition.py": sha_file(os.path.join(REPO, "scratchpad", "_ta_prepartition.py")),
                                                "scratchpad/_l1ps_cache.py": sha_file(os.path.join(REPO, "scratchpad", "_l1ps_cache.py"))},
                     "ALL_IDENTICAL": all(v.get("status") == "IDENTICAL" for v in prev["datasets"].values())})
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(prev, indent=1))
        log("wrote", p)
    elif a and a[0] == "census":
        print(json.dumps(census(a[1:] or list(CanonicalDataset.__init__.__globals__["DATASETS"])), indent=1))
    elif a and a[0] == "build":
        for ds in a[1:]:
            build_canonical(ds)
    else:
        print(__doc__)
