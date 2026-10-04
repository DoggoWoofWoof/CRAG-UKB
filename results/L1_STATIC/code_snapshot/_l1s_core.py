"""L1_STATIC lane -- static, parameter-free, traversal-free block ranking (research question: how much of
the 0.64 -> 0.97 gap can a purely static L1 router recover without walking the graph).

Read-only over the frozen replay caches, the canonical embeddings (node + query, dense fp16 + SPLADE CSR),
the served retrieval caches (ids + scores) and the structural family.  Nothing here touches a CONTRACT_FILE,
a canonical manifest, a partition, a production cache or an L1 record.  NO graph traversal in any arm:
block signatures are precomputed once per block from static information (membership, embeddings, relation
incidence, cut weights); query time = existing dense/SPLADE retrieval + direct query<->signature evidence.

Populations: DEV_A (development) / DEV_B (confirmation) exactly as in _l1x90_core (sha1(query_id) parity).
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1x90_core as X  # noqa: E402
import _ta_prepartition as TA  # noqa: E402  (frozen numerics: partition_ranking / rrf_partitions; imported, never edited)

REPO = X.REPO
OUT = os.path.join(REPO, "results", "L1_STATIC")
os.makedirs(OUT, exist_ok=True)
K0, K_LOCK, P_MAIN = TA.K0, TA.K_LOCK, TA.P_MAIN
log = X.log


class Data:
    """frozen replay cache + canonical embeddings/scores for its rows + static block signatures."""

    def __init__(self, name, dense_fp32=True):
        self.C = C = X.Cache(name)
        self.name, self.cd, self.nq, self.N, self.npart = name, C.cd, C.nq, C.N, C.npart
        self.rows = np.asarray(C.rows, np.int64)
        self.hard = C.hard
        self.sizes = np.bincount(self.hard, minlength=self.npart).astype(np.int64)
        assert (self.sizes == C.part_sizes).all()
        self.mem_hard = (np.arange(self.N + 1, dtype=np.int64), self.hard.astype(np.int64))   # own block only
        self.mem = self.legacy_mem()                                                          # own block + directed out-neighbour blocks (BASE)
        # --- queries (canonical fp16 -> fp32, unit norm as the frozen builder did)
        self.Q = self.cd.query_embeddings[self.rows].astype(np.float32)
        self.Q /= (np.linalg.norm(self.Q, axis=1, keepdims=True) + 1e-9)
        self.Qs = self.cd.embeddings("splade", "queries").read(self.rows).tocsr()
        # --- served retrieval (ids + scores; the replay cache keeps only the first 200 ids)
        self.d_ids = np.asarray(self.cd.dense_topk(1000, self.rows)).astype(np.int64)
        self.d_sc = np.asarray(self.cd.scores("dense", self.rows)).astype(np.float32)
        self.s_ids = np.asarray(self.cd.splade_topk(1000, self.rows)).astype(np.int64)
        self.s_sc = np.asarray(self.cd.scores("splade", self.rows)).astype(np.float32)
        assert (self.d_ids[:, :200] == C.ret_dense).all() and (self.s_ids[:, :200] == C.ret_splade).all(), "served cache != replay cache"
        # --- node embeddings
        self._E16 = self.cd.node_embeddings
        self.E = self._E16[:].astype(np.float32) if dense_fp32 else None
        self._Es = None
        self._sig = {}
        # --- BASE
        self.base_rank = C.base_rank.astype(np.int64)
        self.base_sel = X.fuse(C, self.base_rank, self.base_rank, "T")
        self.r_base, self.base_all, self.base_any = X.evaluate(C, self.base_sel, "BASE")
        self.A = C.split == "A"
        self.B = C.split == "B"

    def override_partition(self, hard, tag):
        """evaluate everything on another partition of the same corpus (gold blocks recomputed from the adapter's
        gold nodes; BASE recomputed with the frozen numerics).  Scratch partitions only; nothing under data/."""
        hard = np.asarray(hard, np.int64)
        assert hard.shape == (self.N,)
        C = self.C
        self.hard = C.hard = hard
        self.npart = C.npart = int(hard.max()) + 1
        self.sizes = np.bincount(hard, minlength=self.npart).astype(np.int64)
        C.part_sizes = self.sizes
        C.gb = [set(int(x) for x in hard[g]) for g in C.gold_nodes]
        C._blockmat = None
        self._sig = {}
        self.mem_hard = (np.arange(self.N + 1, dtype=np.int64), hard)
        self.mem = self.legacy_mem()
        PR_d = canon_channel_rank([self.d_ids[i, :K_LOCK] for i in range(self.nq)], self.mem, self.npart)
        PR_s = canon_channel_rank([self.s_ids[i, :K_LOCK] for i in range(self.nq)], self.mem, self.npart)
        self.base_rank = C.base_rank = rrf_ranks([PR_d, PR_s])
        self.base_sel = X.fuse(C, self.base_rank, self.base_rank, "T")
        self.r_base, self.base_all, self.base_any = X.evaluate(C, self.base_sel, "BASE[%s]" % tag)
        self.partition_tag = tag

    def legacy_mem(self):
        """the frozen builder's membership: own block + blocks of the DIRECTED structural out-neighbours
        (legacy load_topology semantics) -- a static, precomputed node -> blocks map, no query-time traversal."""
        xo, ao = self.cd.struct_csr(directed=True)
        N, hard = self.N, self.hard.astype(np.int64)
        deg_out = np.diff(xo)
        r = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
        p = np.concatenate([hard, hard[np.asarray(ao, np.int64)]])
        keys = np.unique(r * np.int64(self.npart) + p)
        rr = keys // self.npart
        mlen = np.bincount(rr, minlength=N).astype(np.int64)
        mem_ptr = np.zeros(N + 1, np.int64)
        mem_ptr[1:] = np.cumsum(mlen)
        self.cd._csr.clear()
        return (mem_ptr, (keys % self.npart).astype(np.int64))

    # ------------------------------------------------------------------ node SPLADE (CSR)
    @property
    def Es(self):
        if self._Es is None:
            e = self.cd.embeddings("splade", "docs")
            self._Es = sp.vstack([e.shard(k) for k in range(e.n_shards)]).tocsr()
            assert self._Es.shape[0] == self.N
        return self._Es

    # ------------------------------------------------------------------ static block signatures
    def blockmat(self):
        return self.C.blockmat()

    def centroids(self):
        """mu_b = L2-normalised mean of the block's node vectors (npart, dim)."""
        if "mu" not in self._sig:
            B = self.blockmat()
            S = np.zeros((self.npart, self.E.shape[1] if self.E is not None else 1536), np.float64)
            if self.E is not None:
                S = np.asarray(B.T @ self.E, np.float64)
            else:
                for a, b, blk in self._E16.blocks(20000):
                    S += np.asarray(B[a:b].T @ blk.astype(np.float32), np.float64)
            mu = S / np.maximum(self.sizes, 1)[:, None]
            self._sig["mu_raw"] = mu.astype(np.float32)
            self._sig["mu"] = (mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        return self._sig["mu"]

    def splade_pool(self, how="mean"):
        """pooled SPLADE signature per block (npart, V): mean (sum / size) or max over members."""
        key = "sp_" + how
        if key not in self._sig:
            B = self.blockmat()
            if how == "mean":
                P = (B.T @ self.Es).tocsr()
                P = sp.diags(1.0 / np.maximum(self.sizes, 1)) @ P
            elif how == "sum":
                P = (B.T @ self.Es).tocsr()
            elif how == "max":
                Es = self.Es.tocoo()
                # max over members of each block per term: sort by (block, term) and take the max in each group
                blk = self.hard[Es.row]
                key2 = blk.astype(np.int64) * Es.shape[1] + Es.col.astype(np.int64)
                o = np.argsort(key2, kind="stable")
                k2 = key2[o]
                v = Es.data[o]
                starts = np.r_[0, np.nonzero(np.diff(k2))[0] + 1]
                mx = np.maximum.reduceat(v, starts)
                kk = k2[starts]
                P = sp.csr_matrix((mx.astype(np.float32), (kk // Es.shape[1], kk % Es.shape[1])), shape=(self.npart, Es.shape[1]))
            else:
                raise ValueError(how)
            self._sig[key] = P.tocsr()
        return self._sig[key]

    def struct_edges(self):
        """(src, dst, rel) of the structural family as served (directed, with duplicates)."""
        s, d, r, ent = self.cd.family("structural")
        return np.asarray(s, np.int64), np.asarray(d, np.int64), (np.asarray(r, np.int64) if r is not None else None), ent

    def relation_vocab(self):
        ent = self.cd.graph_manifest["families"]["structural"]
        return list(ent.get("relation_vocabulary") or [])

    def rel_hist(self):
        """R_b[r] = number of structural edge endpoints of relation r inside block b (in + out), plus
        R_bnd[b, r] = endpoints of relation-r edges that cross the block boundary."""
        if "R" not in self._sig:
            s, d, r, _ = self.struct_edges()
            nrel = int(r.max()) + 1 if r is not None and len(r) else 1
            if r is None:
                r = np.zeros(len(s), np.int64)
            bs, bd = self.hard[s], self.hard[d]
            R = np.zeros((self.npart, nrel), np.float64)
            np.add.at(R, (bs, r), 1.0)
            np.add.at(R, (bd, r), 1.0)
            cross = bs != bd
            Rb = np.zeros((self.npart, nrel), np.float64)
            np.add.at(Rb, (bs[cross], r[cross]), 1.0)
            np.add.at(Rb, (bd[cross], r[cross]), 1.0)
            self._sig["R"], self._sig["R_bnd"] = R, Rb
        return self._sig["R"], self._sig["R_bnd"]

    def cut_weights(self, fam="struct"):
        """W[b, b'] = number of undirected edges (deduped, loop-free keys) between blocks b != b'."""
        key = "cut_" + fam
        if key not in self._sig:
            N, STRUCT, KNN, NERX = self.cd.keysets()
            keys = STRUCT if fam == "struct" else (KNN if fam == "knn" else np.concatenate([STRUCT, KNN]))
            u = (keys // np.int64(N)).astype(np.int64)
            v = (keys % np.int64(N)).astype(np.int64)
            bu, bv = self.hard[u], self.hard[v]
            m = bu != bv
            W = sp.coo_matrix((np.ones(int(m.sum()), np.float32), (bu[m], bv[m])), shape=(self.npart, self.npart)).tocsr()
            W = W + W.T
            self._sig[key] = W.tocsr()
        return self._sig[key]

    # ------------------------------------------------------------------ evaluation
    def eval_rank(self, rank, tag, quiet=False, splits=("A",)):
        """rank: (nq, >=50) block ids per query (descending, -1 padded) -> ALL + McNemar vs BASE on the
        requested populations.  Exploration reads DEV_A only; DEV_B / full are read by the confirmation script."""
        sel = X.fuse(self.C, rank, rank, "T")
        r, allv, anyv = X.evaluate(self.C, sel, tag)
        out = {"tag": tag}
        for sname, m in (("A", self.A), ("B", self.B), ("full", np.ones(self.nq, bool))):
            if sname not in splits:
                continue
            g, l, p = X.mcnemar(self.base_all[m], allv[m])
            out[sname] = {"ALL": r["ALL_split"][sname if sname != "full" else "ALL"]["ALL"],
                          "ANY": r["ALL_split"][sname if sname != "full" else "ALL"]["ANY"],
                          "gained": g, "lost": l, "p": p,
                          "per_hop": {k: v["ALL"] for k, v in r["ALL_split"][sname if sname != "full" else "ALL"].items() if k.startswith("hop")}}
        out["scope_nodes"] = r["scope_nodes"]
        if not quiet and "A" in out:
            a = out["A"]
            hops = " ".join("%s %.3f" % (k[3:], v) for k, v in sorted(a["per_hop"].items()))
            log("  %-58s DEV_A ALL %.4f (+%d/-%d p=%.1e) ANY %.4f | hops %s | scope %s" % (tag, a["ALL"], a["gained"], a["lost"], a["p"], a["ANY"], hops, out["scope_nodes"]))
        return out, allv


# ---------------------------------------------------------------------- block scoring primitives
def hits_to_blocks(ids, w, mem, npart, how, sizes=None, m=3):
    """aggregate per-query node hits (ids (nq,k) int, w (nq,k) float weights) into (nq, npart) block scores
    through a membership structure mem = (ptr, flat) (node -> blocks; hard or legacy own+out-neighbour).
    how: sum | max | count | topm | mean | sum_norm (sum / size) | sum_sqrt (sum / sqrt size)."""
    ptr, flat = mem
    nq, k = ids.shape
    S = np.zeros((nq, npart), np.float64)
    ok = (ids >= 0).ravel()
    nodes = ids.ravel()[ok].astype(np.int64)
    rows0 = np.repeat(np.arange(nq), k)[ok]
    w0 = w.ravel().astype(np.float64)[ok]
    cnt = (ptr[nodes + 1] - ptr[nodes]).astype(np.int64)
    rows = np.repeat(rows0, cnt)
    wv = np.repeat(w0, cnt)
    # gather member blocks
    starts = ptr[nodes]
    idx = np.repeat(starts - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    b = flat[idx]
    if how in ("sum", "mean", "sum_norm", "sum_sqrt"):
        np.add.at(S, (rows, b), wv)
        if how == "mean":
            Cn = np.zeros((nq, npart), np.float64)
            np.add.at(Cn, (rows, b), 1.0)
            S = np.where(Cn > 0, S / np.maximum(Cn, 1), 0.0)
        elif how == "sum_norm":
            S = S / np.maximum(sizes, 1)[None, :]
        elif how == "sum_sqrt":
            S = S / np.sqrt(np.maximum(sizes, 1))[None, :]
    elif how == "max":
        np.maximum.at(S, (rows, b), wv)
    elif how == "count":
        np.add.at(S, (rows, b), 1.0)
    elif how == "topm":
        key = rows * npart + b
        o = np.lexsort((-wv, key))
        ks, ws = key[o], wv[o]
        st = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        pos = np.arange(len(ks)) - np.repeat(st, np.diff(np.r_[st, len(ks)]))
        keep = pos < m
        np.add.at(S, (ks[keep] // npart, ks[keep] % npart), ws[keep])
    else:
        raise ValueError(how)
    return S


def rank_of(S):
    """(nq, npart) scores -> (nq, npart) block ranking (descending, stable), and the rank-position matrix."""
    order = np.argsort(-S, axis=1, kind="stable")
    return order


def rrf_ranks(rankings, K=K0):
    """canonical partition-level RRF (frozen numerics; rankings[0] = dense supplies the tie-break)."""
    return TA.rrf_partitions([np.asarray(r, np.int32) for r in rankings], rankings[0].shape[1]).astype(np.int64)


def canon_channel_rank(node_lists, mem, npart):
    """the frozen per-channel block ranking: rr(sum of 1/(K0+r)) + rr(max of 1/(K0+r))."""
    return TA.partition_ranking(node_lists, mem, npart).astype(np.int64)


def rank_votes(S):
    """rank-transform a score matrix the frozen way: 1/(K0 + rank)."""
    nq, npart = S.shape
    order = np.argsort(-S, axis=1)
    rank = np.empty((nq, npart), np.int64)
    rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
    return 1.0 / (K0 + rank)


def rankvec(k):
    return 1.0 / (K0 + np.arange(k, dtype=np.float64))


def wj(p, o):
    X.wj(p, o)
