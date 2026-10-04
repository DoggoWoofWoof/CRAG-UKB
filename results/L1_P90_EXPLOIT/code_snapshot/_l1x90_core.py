"""L1 P50 exploitation lane (target: ALL-gold coverage > 0.90 at the frozen P50 budget) -- shared core.

Read-only over the frozen canonical replay caches + the canonical adapter's structural CSR.  Nothing
here touches a CONTRACT_FILE, a canonical manifest, a partition, a production cache or an L1 record.

Populations: the replay-cache dev rows (never TEST).  Every row is assigned once, by a hash of its
query id, to DEV_A (development / exploration) or DEV_B (confirmation).  Numbers on DEV_B are only
looked at for the single preregistered configuration.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset  # noqa: E402

OUT = os.path.join(REPO, "results", "L1_P90_EXPLOIT")
os.makedirs(OUT, exist_ok=True)

K0 = 60          # RRF constant of the frozen contract
P_MAIN = 50      # P50
TOPP = 200       # ranked partitions kept in the cache

CACHES = {
    # canonical Mt-KaHyPar H4_SK partitions (the frozen contract)
    "metaqa": os.path.join(REPO, "data", "l1_canonical", "metaqa", "replay_cache.npz"),
    "squad": os.path.join(REPO, "data", "l1_canonical", "squad", "replay_cache.npz"),
    # musique has no canonical Mt-KaHyPar partition (RESOURCE_INFEASIBLE_LOCAL); PHG+repair arm
    "musique": os.path.join(REPO, "data", "l1_lowmem", "musique", "replay_cache__LOWMEM__PHG_C1_con.npz"),
    # robustness arm: metaqa under PHG+repair
    "metaqa_phg": os.path.join(REPO, "data", "l1_lowmem", "metaqa", "replay_cache__LOWMEM__PHG_REPAIR1_con.npz"),
}
DS_OF = {"metaqa": "metaqa", "squad": "squad", "musique": "musique", "metaqa_phg": "metaqa"}

_T0 = time.time()


def log(*a):
    print("[%7.1fs] %s" % (time.time() - _T0, " ".join(str(x) for x in a)), flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wj(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=1, ensure_ascii=True)


class Cache:
    """one frozen replay cache + the dataset's structural graph, indexed by cache row."""

    def __init__(self, name):
        self.name = name
        self.path = CACHES[name]
        self.ds = DS_OF[name]
        z = np.load(self.path, allow_pickle=True)
        self.meta = json.loads(str(z["meta_json"]))
        self.rows = z["rows"]
        self.base_rank = z["base_rank"]            # (nq, TOPP) int32, -1 padded
        self.gold_part = z["gold_part"]
        self.gold_ptr = z["gold_ptr"]
        self.hard = z["hard"].astype(np.int64)     # node -> block
        self.part_sizes = z["part_sizes"]
        self.hops = z["hops"]
        self.seeds = z["seeds"]                    # (nq, 5) RRF seeds (fused dense+splade top-5 nodes)
        self.ret_rrf = z["ret_rrf"]                # (nq, 200) fused node continuation
        self.ret_dense = z["ret_dense"]
        self.ret_splade = z["ret_splade"]
        self.s_node = z["s_node"]
        self.s_hop = z["s_hop"]
        self.nq = len(self.rows)
        self.npart = int(self.part_sizes.shape[0])
        self.N = int(self.hard.shape[0])
        self.qids = list(self.meta["row_query_ids"])
        assert len(self.qids) == self.nq
        # gold blocks per query (sets) and gold NODE positions (from the adapter, labels only)
        self.gb = [set(int(p) for p in self.gold_part[self.gold_ptr[i]:self.gold_ptr[i + 1]]) for i in range(self.nq)]
        self.cd = CanonicalDataset(self.ds)
        assert self.cd.n_nodes == self.N, (self.cd.n_nodes, self.N)
        self.gold_nodes = [np.asarray(g, np.int64) for g in self.cd.gold([int(r) for r in self.rows])]
        # development / confirmation halves by query-id hash (fixed before any number was read)
        par = np.array([int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in self.qids], np.int8)
        self.split = np.where(par == 0, "A", "B")
        self._csr = {}
        self._blockmat = None

    # ------------------------------------------------------------- graphs
    def csr(self, fam="struct"):
        """row-normalised random-walk transition matrix P = D^-1 A over the undirected, deduped,
        loop-free adjacency (STRUCT = the frozen expansion contract's graph; SK = STRUCT u KNN)."""
        if fam in self._csr:
            return self._csr[fam]
        N, STRUCT, KNN, NERX = self.cd.keysets()
        keys = STRUCT if fam == "struct" else np.concatenate([STRUCT, KNN])
        xadj, adj = self.cd.csr_from_keys(keys, N)
        deg = np.diff(xadj).astype(np.float64)
        inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
        data = np.repeat(inv, np.diff(xadj)).astype(np.float32)
        P = sp.csr_matrix((data, adj.astype(np.int64), xadj), shape=(N, N))
        self._csr[fam] = (P, deg)
        return self._csr[fam]

    def blockmat(self):
        """N x npart indicator (node -> its block)."""
        if self._blockmat is None:
            self._blockmat = sp.csr_matrix((np.ones(self.N, np.float32), (np.arange(self.N), self.hard)),
                                           shape=(self.N, self.npart))
        return self._blockmat

    # ------------------------------------------------------------- metrics
    def cover(self, sel_rows):
        """sel_rows: list of per-query iterables of selected blocks -> per-query ALL / ANY flags."""
        allv = np.zeros(self.nq, np.int8)
        anyv = np.zeros(self.nq, np.int8)
        for i, s in enumerate(sel_rows):
            s = set(int(x) for x in s)
            g = self.gb[i]
            if not g:
                continue
            c = [p in s for p in g]
            allv[i] = all(c)
            anyv[i] = any(c)
        return allv, anyv

    def summary(self, allv, anyv, mask=None):
        m = np.ones(self.nq, bool) if mask is None else mask
        out = {"n": int(m.sum()), "ALL": round(float(allv[m].mean()), 4), "ANY": round(float(anyv[m].mean()), 4)}
        if (self.hops >= 0).any():
            for h in sorted(set(int(x) for x in self.hops[self.hops >= 0])):
                mm = m & (self.hops == h)
                out["hop%d" % h] = {"n": int(mm.sum()), "ALL": round(float(allv[mm].mean()), 4), "ANY": round(float(anyv[mm].mean()), 4)}
        return out


def mcnemar(a, b):
    """exact two-sided McNemar on paired 0/1 vectors: (gained, lost, p)."""
    from scipy.stats import binomtest
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    g = int((b & ~a).sum())
    l = int((a & ~b).sum())
    p = 1.0 if g + l == 0 else float(binomtest(g, g + l, 0.5).pvalue)
    return g, l, p


# ------------------------------------------------------------------ structural block channel
def ppr_block_mass(C, seed_rows, fam="struct", alpha=0.15, iters=10, hub=None, batch=256, seed_weights=None):
    """Personalised PageRank from per-query seed sets over the transition matrix, aggregated to
    blocks.  seed_rows: (nq, s) int array (-1 padded); seed_weights: same shape or None (uniform).
    hub: None | 'sqrt' | 'deg' -> divide the converged node mass by deg^0.5 / deg before block sum.
    Returns (nq, npart) float32 block mass."""
    P, deg = C.csr(fam)
    B = C.blockmat()
    nq = seed_rows.shape[0]
    out = np.zeros((nq, C.npart), np.float32)
    damp = None
    if hub == "sqrt":
        damp = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    elif hub == "deg":
        damp = 1.0 / np.maximum(deg, 1.0)
    for s in range(0, nq, batch):
        e = min(nq, s + batch)
        R = np.zeros((e - s, C.N), np.float32)
        for i in range(s, e):
            sd = seed_rows[i]
            ok = sd >= 0
            if not ok.any():
                continue
            w = np.ones(int(ok.sum()), np.float64) if seed_weights is None else seed_weights[i][ok].astype(np.float64)
            w = w / w.sum()
            np.add.at(R[i - s], sd[ok].astype(np.int64), w.astype(np.float32))
        X = R.copy()
        for _ in range(iters):
            X = (1.0 - alpha) * (P.T @ X.T).T + alpha * R     # x <- (1-a) x P + a r
        if damp is not None:
            X = X * damp.astype(np.float32)[None, :]
        out[s:e] = np.asarray(X @ B)
    return out


def rank_from_scores(S, top=TOPP):
    """descending argsort per row, entries with score 0 marked -1 (unranked)."""
    o = np.argsort(-S, axis=1, kind="stable")[:, :top]
    r = np.full(o.shape, -1, np.int64)
    for i in range(S.shape[0]):
        v = S[i, o[i]]
        r[i] = np.where(v > 0, o[i], -1)
    return r


# ------------------------------------------------------------------ fusion rules (block rankings -> 50 blocks)
def rank_pos(rank_rows, npart):
    """(nq, npart) position of each block in a ranked row (large = unranked)."""
    nq = rank_rows.shape[0]
    pos = np.full((nq, npart), 10 ** 6, np.int64)
    for i in range(nq):
        r = rank_rows[i]
        r = r[r >= 0]
        pos[i, r] = np.arange(len(r))
    return pos


def fuse(C, T_rank, S_rank, rule, B=None):
    """returns list of per-query selected block arrays (P_MAIN)."""
    npart = C.npart
    pt = rank_pos(T_rank, npart)
    ps = rank_pos(S_rank, npart)
    sel = []
    for i in range(C.nq):
        t = T_rank[i][T_rank[i] >= 0]
        s = S_rank[i][S_rank[i] >= 0]
        if rule == "T":
            sel.append(t[:P_MAIN])
            continue
        if rule == "S":
            if len(s) < P_MAIN:                       # fill from text when the walk touches < 50 blocks
                s = np.concatenate([s, t[~np.isin(t, s)]])
            sel.append(s[:P_MAIN])
            continue
        wt = 1.0 / (K0 + pt[i].astype(np.float64))
        ws = 1.0 / (K0 + ps[i].astype(np.float64))
        wt[pt[i] >= 10 ** 6] = 0.0
        ws[ps[i] >= 10 ** 6] = 0.0
        if rule == "RRF":                             # symmetric RRF over all 50 slots
            sc = wt + ws
            o = np.argsort(-sc, kind="stable")
            sel.append(o[:P_MAIN])
        elif rule == "MINRANK":                       # min over channels, tie-break by RRF sum
            mr = np.minimum(pt[i], ps[i]).astype(np.float64)
            key = mr - 1e-3 * (wt + ws)
            o = np.argsort(key, kind="stable")
            sel.append(o[:P_MAIN])
        elif rule == "CORE":                          # protected text core (50-B) + RRF competition for B slots
            core = t[:P_MAIN - B]
            sc = wt + ws
            sc[core] = -1.0
            o = np.argsort(-sc, kind="stable")
            rest = o[: B]
            sel.append(np.concatenate([core, rest]))
        elif rule == "HOLD":                          # text incumbents hold unless a struct challenger outranks them
            inc = list(t[:P_MAIN])
            ch = [b for b in s if b not in set(inc)]
            inc_r = [pt[i, b] for b in inc]
            j = 0
            while j < len(ch) and inc:
                worst = int(np.argmax(inc_r))
                if ps[i, ch[j]] < inc_r[worst]:
                    inc[worst] = ch[j]
                    inc_r[worst] = ps[i, ch[j]]
                    j += 1
                else:
                    break
            sel.append(np.array(inc))
        else:
            raise ValueError(rule)
    return sel


def evaluate(C, sel, name, mask_split=None):
    allv, anyv = C.cover(sel)
    res = {"method": name, "ALL_split": {}, }
    for sname in ("A", "B", "ALL"):
        m = np.ones(C.nq, bool) if sname == "ALL" else (C.split == sname)
        res["ALL_split"][sname] = C.summary(allv, anyv, m)
    res["scope_nodes"] = round(float(np.mean([C.part_sizes[np.asarray(s, np.int64)].sum() for s in sel])), 1)
    return res, allv, anyv
