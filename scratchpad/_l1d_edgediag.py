"""L1 DEVELOPMENT diagnostic (gold-aware; never a candidate, never an arm): which static edges are LOCALIZATION edges?
(user, 2026-09-26, pausing steps 3/4: "Does static graph adjacency actually predict gold co-relevance?" -- per edge family,
missing-gold enrichment, blocks activated per top semantic hit, block precision, STRUCT-only vs KNN-only vs STRUCT+KNN.)

Per query of the development population (loc_population.json): FLAT = the pinned rrf_full (exhaustive dense + SPLADE, as in
steps 1-2; the gold FLAT ranks are asserted == the step-1 record), hits S = FLAT[:200] (rank buckets 1-10 / 11-50 / 51-200),
gold G, missing gold G_miss(M) = {g in G : FLAT rank >= M}, M in (100, 1000, 5000).
Edge families, read-only from the canonical graph (self loops dropped, pairs deduplicated):
    STRUCT_out, STRUCT_in, STRUCT   the structural family by direction (metaqa: the 9 kb.txt relations, head (movie) -> tail;
                                    musique: title_mention, s -> u when the text of s contains the title of u) and undirected
    KNN, NER                        semantic kNN (k = 3) and shares-entity (undirected)
    TITLE                           same title = paragraphs of one source article (derived from nodes.jsonl titles; only if
                                    the dataset has shared titles)
    SK = STRUCT u KNN (the families of the H4_SK hypergraph), SKN = SK u NER, OUT+KNN = STRUCT_out u KNN (directed)
    <relation>_out, <relation>_in   (metaqa) one structural relation, one direction
    OWN                             no edge: the hit's own block only (block level)
    BLOCK:<cell>                    baseline: the other nodes of the hit's block in that frozen partition (per hit, node set)
Sections (every count accumulated over ALL queries and per hop):
    per_hit    for (q, s in S): neighbours N_F(s).  P(gold | u in N_F(s)) and ENRICHMENT = observed / the matched random
               expectation (|N_F(s)| x |G|/N per query; for missing gold, neighbours outside FLAT@M x |G_miss(M)|/(N - M)).
               By hit rank bucket, hit gold status (from a gold hit = pure co-relevance), hit spread bin (|N_F(s)|),
               neighbour degree bin (undirected degree of u in F) and edge coherence quintile (within the family).
    node_set   U = union of N_F(s) over S (top-10 / top-200), outside FLAT@M: size, precision, recall of G_miss(M),
               enrichment, queries whose every missing gold is in U.
    blocks     per cell: mem_F(s) = {P(s)} + {P(u) : u in N_F(s)} (F = STRUCT_out is the served router's membership table);
               B_F(q) = union over S (top-200).  Blocks per hit (and added beyond the own block), per query: blocks, node
               mass, precision = activated blocks holding a missing gold / activated blocks (queries with a missing gold),
               the same over the blocks ADDED beyond OWN (the dilution measure), recall = missing gold nodes whose block is
               activated, random baseline = share of all blocks holding a missing gold.  By hit spread bin and by vote bin
               (number of hits activating the block).
    gold_pairs FLAT-independent co-relevance: share of ordered gold pairs (g1, g2) with g2 in N_F(g1), against the family's
               density (entries / N(N-1)); golds with a gold neighbour; same-block rate per cell against random pairs.
    family_static  label-free statistics (candidate static criteria): edges, degree, COHERENCE = cosine of the endpoints'
               dense vectors (quintile thresholds per family; random-pair baseline), block-internal share per cell, overlap
               with KNN / STRUCT.
Reads only frozen inputs (population, canonical graph, nodes.jsonl titles, dense + SPLADE embeddings, partitions); writes
nothing under data/.

Usage: python scratchpad/_l1d_edgediag.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/edgediag_<dataset>__<tag>.json (write-once) + markdown tables in the log
"""
import collections
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D

log = D.log
HERE, REPO, OUT = D.HERE, D.REPO, D.OUT
NH = D.ACT                                         # 200 hits = FLAT[:200], the activation depth of steps 1-2
MS = (100, 1000, 5000)
RANK_EDGES = np.array([0, 10, 50])
RANK_LABELS = ("1-10", "11-50", "51-200")
DEG_EDGES = np.array([0, 1, 2, 6, 26, 101, 301])
DEG_LABELS = ("0", "1", "2-5", "6-25", "26-100", "101-300", "301+")
VOTE_EDGES = np.array([1, 2, 4, 11])
VOTE_LABELS = ("1", "2-3", "4-10", "11+")
NCOH = 5
COH_LABELS = ("q1 (least coherent)", "q2", "q3", "q4", "q5 (most coherent)")
COH_BATCH = 8192
N_RANDOM_PAIRS = 200000
SEED = 20260926
V1_TAG = "v1"
HIT_FIELDS = (("hits", "hits_with_nbrs", "nbrs", "gold", "exp_gold") + tuple("out@%d" % M for M in MS)
              + tuple("miss@%d" % M for M in MS) + tuple("exp_miss@%d" % M for M in MS) + ("hits_with_gold_nbr",)
              + tuple("hits_with_miss@%d_nbr" % M for M in MS))
EDGE_FIELDS = HIT_FIELDS[2:14]
SET_FIELDS = ("queries", "queries_with_miss", "new_nodes", "found", "missing", "exp_found", "all_found", "found_flat_ext", "all_found_flat_ext")
BLK_FIELDS = (("queries", "blocks", "mass", "added", "added_mass", "any_gold_blocks", "any_gold_added")
              + tuple(f % M for M in MS for f in ("q_miss@%d", "blocks|miss@%d", "added|miss@%d", "miss_blocks@%d",
                                                   "miss_added@%d", "gold_blocks@%d", "missing@%d", "covered@%d",
                                                   "covered_added@%d", "all_covered@%d", "new_mass@%d", "flat_ext_covered@%d",
                                                   "flat_ext_all@%d")))
BSPREAD_FIELDS = ("hits", "blocks", "added") + tuple("added_miss@%d" % M for M in MS) + ("added_any_gold",)
BVOTE_FIELDS = ("blocks", "mass", "any_gold") + tuple("miss@%d" % M for M in MS)
PAIR_FIELDS = ("queries_ge2", "golds", "ordered_pairs", "adjacent", "golds_with_gold_nbr")
BPAIR_FIELDS = ("queries_ge2", "ordered_pairs", "same_block")
assert len(HIT_FIELDS) == 18 and len(EDGE_FIELDS) == 12 and EDGE_FIELDS[0] == "nbrs" and EDGE_FIELDS[-1] == "exp_miss@%d" % MS[-1]


def dbin(x):
    return np.searchsorted(DEG_EDGES, x, side="right") - 1


def ukeys(u, v, N):
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    m = u != v
    return np.unique(np.minimum(u[m], v[m]) * np.int64(N) + np.maximum(u[m], v[m]))


def dkeys(u, v, N):
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    m = u != v
    return np.unique(u[m] * np.int64(N) + v[m])


def csr_directed(keys, N):
    """out-edge CSR of sorted unique directed keys src*N+dst (rows sorted by (src, dst))."""
    s = keys // np.int64(N)
    xadj = np.zeros(N + 1, np.int64)
    xadj[1:] = np.cumsum(np.bincount(s, minlength=N))
    return xadj, (keys % np.int64(N)).astype(np.int32)


def csr_undirected(keys, N):
    """symmetric CSR of sorted unique undirected keys min*N+max: the directed keys of both orientations."""
    u, v = keys // np.int64(N), keys % np.int64(N)
    return csr_directed(np.unique(np.concatenate([u * np.int64(N) + v, v * np.int64(N) + u])), N)


def und_of_directed(keys, N):
    u, v = keys // np.int64(N), keys % np.int64(N)
    return np.unique(np.minimum(u, v) * np.int64(N) + np.maximum(u, v))


def entries(starts, lens):
    """(hit index, CSR position) of every neighbour entry of the rows (starts, lens), grouped by row in row order."""
    lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    hit = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    pos = np.arange(tot, dtype=np.int64) + np.repeat(np.asarray(starts, np.int64) - (np.cumsum(lens) - lens), lens)
    return hit, pos


def pair_cosines(E, keys, N):
    """cosine of the dense vectors of the endpoints of every undirected key (processed shard pair by shard pair, each shard
    read once per pair into memory as float16, products in float32)."""
    out = np.full(len(keys), np.nan, np.float32)
    u, v = keys // np.int64(N), keys % np.int64(N)
    ss = int(E.shard_size)
    su, sv = u // ss, v // ss
    for a in range(E.n_shards):
        A = np.array(E.shard(a))
        for b in range(a, E.n_shards):
            m = np.flatnonzero((su == a) & (sv == b))
            if not len(m):
                continue
            B = A if b == a else np.array(E.shard(b))
            for c0 in range(0, len(m), COH_BATCH):
                mm = m[c0:c0 + COH_BATCH]
                X = A[u[mm] - a * ss].astype(np.float32)
                Y = B[v[mm] - b * ss].astype(np.float32)
                out[mm] = np.einsum("ij,ij->i", X, Y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
                del X, Y
            if b != a:
                del B
        del A
    assert np.isfinite(out).all()
    return out


class Acc(object):
    """sums per stratum: key -> array (n_strata, ...)."""

    def __init__(self, ns):
        self.ns, self.d = ns, collections.OrderedDict()

    def add(self, key, arr, sids):
        arr = np.asarray(arr, np.float64)
        a = self.d.get(key)
        if a is None:
            a = self.d[key] = np.zeros((self.ns,) + arr.shape, np.float64)
        for s in sids:
            a[s] += arr


def build_families(cd, N, parts, log):
    """every edge family as a CSR (xadj, adj int32) + per-entry coherence quintile + per-node degree bin + label-free stats."""
    t0 = time.time()
    s, d, r, ent = cd.family("structural")
    s = np.asarray(s, np.int64)
    d = np.asarray(d, np.int64)
    r = np.asarray(r, np.int64)
    vocab = ent.get("relation_vocabulary") or []
    SD = dkeys(s, d, N)                                             # directed structural pairs
    rel_dir = {}
    if len(vocab) > 1:
        for ri, nm in enumerate(vocab):
            m = r == ri
            rel_dir[nm] = dkeys(s[m], d[m], N)
    del s, d, r
    ks, kd, _, _ = cd.family("knn")
    KNN = ukeys(ks, kd, N)
    ns_, nd_, _, _ = cd.family("ner")
    NER = ukeys(ns_, nd_, N)
    titles = [x.get("title") for x in cd.ds.nodes()]
    assert len(titles) == N
    grp = collections.defaultdict(list)
    for i, t in enumerate(titles):
        if t:
            grp[t].append(i)
    tp = [(a, b) for v in grp.values() if len(v) > 1 for ia, a in enumerate(v) for b in v[ia + 1:]]
    del titles, grp
    TITLE = ukeys([a for a, _ in tp], [b for _, b in tp], N) if tp else np.empty(0, np.int64)
    del tp
    STRUCT = und_of_directed(SD, N)
    fam_keys = collections.OrderedDict()                             # name -> (kind, keys): directed keys or undirected keys
    fam_keys["STRUCT_out"] = ("dir", SD)
    fam_keys["STRUCT_in"] = ("dir", np.unique((SD % np.int64(N)) * np.int64(N) + SD // np.int64(N)))
    fam_keys["STRUCT"] = ("und", STRUCT)
    fam_keys["KNN"] = ("und", KNN)
    fam_keys["NER"] = ("und", NER)
    if len(TITLE):
        fam_keys["TITLE"] = ("und", TITLE)
    fam_keys["SK"] = ("und", np.union1d(STRUCT, KNN))
    fam_keys["SKN"] = ("und", np.union1d(fam_keys["SK"][1], NER))
    ku, kv = KNN // np.int64(N), KNN % np.int64(N)
    fam_keys["OUT+KNN"] = ("dir", np.union1d(SD, np.concatenate([ku * np.int64(N) + kv, kv * np.int64(N) + ku])))
    del ku, kv
    for nm, k in rel_dir.items():
        fam_keys[nm + "_out"] = ("dir", k)
        fam_keys[nm + "_in"] = ("dir", np.unique((k % np.int64(N)) * np.int64(N) + k // np.int64(N)))
    UNION = np.unique(np.concatenate([STRUCT, KNN, NER, TITLE]))
    t1 = time.time()
    COS = pair_cosines(cd.node_embeddings, UNION, N)
    rng = np.random.RandomState(SEED)
    ru, rv = rng.randint(0, N, N_RANDOM_PAIRS), rng.randint(0, N, N_RANDOM_PAIRS)
    RK = ukeys(ru, rv, N)
    RCOS = pair_cosines(cd.node_embeddings, RK, N)
    log("  coherence: %d union pairs + %d random pairs (%.0fs)" % (len(UNION), len(RK), time.time() - t1))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    static = {"random_pair_cosine": {"n": int(len(RK)), "mean": D.q4(RCOS.mean()),
                                     "quantiles": {str(q): D.q4(np.quantile(RCOS, q)) for q in qs}},
              "relation_vocabulary": vocab, "title_groups_shared": int(len(TITLE)), "families": {}}
    del RK, RCOS
    FAM = collections.OrderedDict()
    for nm, (kind, keys) in fam_keys.items():
        if kind == "dir":
            xadj, adj = csr_directed(keys, N)
            und = und_of_directed(keys, N)
        else:
            xadj, adj = csr_undirected(keys, N)
            und = keys
        deg_und = np.bincount(und // np.int64(N), minlength=N) + np.bincount(und % np.int64(N), minlength=N)
        ci = np.searchsorted(UNION, und)
        assert (ci < len(UNION)).all() and (UNION[ci] == und).all()
        cu = COS[ci]
        thr = np.quantile(cu, [0.2, 0.4, 0.6, 0.8]) if len(cu) else np.zeros(4)
        cbin = np.empty(len(adj), np.int8)                          # per CSR entry, chunked
        rows = np.repeat(np.arange(N, dtype=np.int64), np.diff(xadj))
        for c0 in range(0, len(adj), 1 << 21):
            a_, b_ = rows[c0:c0 + (1 << 21)], adj[c0:c0 + (1 << 21)].astype(np.int64)
            kk = np.minimum(a_, b_) * np.int64(N) + np.maximum(a_, b_)
            cbin[c0:c0 + len(kk)] = np.searchsorted(thr, COS[np.searchsorted(UNION, kk)], side="right")
        e = {"kind": kind, "entries": int(len(adj)), "undirected_pairs": int(len(und)),
             "nodes_with_edges": int((deg_und > 0).sum()), "degree_undirected": D.stats(deg_und[deg_und > 0]) if len(und) else None,
             "degree_p90_p99": [float(np.percentile(deg_und[deg_und > 0], 90)), float(np.percentile(deg_und[deg_und > 0], 99))] if len(und) else None,
             "row_length (spread)": D.stats(np.diff(xadj)),
             "coherence": ({"mean": D.q4(cu.mean()), "quantiles": {str(q): D.q4(np.quantile(cu, q)) for q in qs},
                            "quintile_thresholds": [D.q4(x) for x in thr]} if len(cu) else None),
             "overlap_with_KNN": D.q4(np.isin(und, KNN, assume_unique=True).mean()) if len(und) else None,
             "overlap_with_STRUCT": D.q4(np.isin(und, STRUCT, assume_unique=True).mean()) if len(und) else None,
             "block_internal_share": {c: D.q4((P.hard[rows] == P.hard[adj]).mean()) if len(adj) else None for c, P in parts.items()}}
        del rows
        static["families"][nm] = e
        FAM[nm] = {"xadj": xadj, "adj": adj, "cbin": cbin, "dbin": dbin(deg_und).astype(np.int8), "entries": int(len(adj))}
        log("  family %-26s %s entries %9d pairs %9d coherence %s internal %s" % (
            nm, kind, len(adj), len(und), e["coherence"] and e["coherence"]["mean"], e["block_internal_share"]))
    del UNION, COS, fam_keys, SD, STRUCT, KNN, NER, TITLE
    static["random_pair_block_share"] = {c: D.q4(float((P.sizes * (P.sizes - 1)).sum()) / (float(N) * (N - 1))) for c, P in parts.items()}
    static["seconds"] = round(time.time() - t0, 1)
    return FAM, static


class QCtx(object):
    pass


def hit_level(acc, key, lens, hit, nb, pos, F, Q):
    """per-hit and per-entry enrichment counts of one family (F None: no edge-level dimensions)."""
    nh = len(lens)
    isg = Q.gm[nb].astype(np.float64)
    outs = [o[nb].astype(np.float64) for o in Q.out]
    miss = [isg * o for o in outs]
    gold_h = np.bincount(hit, weights=isg, minlength=nh)
    out_h = [np.bincount(hit, weights=o, minlength=nh) for o in outs]
    miss_h = [np.bincount(hit, weights=m, minlength=nh) for m in miss]
    lf = lens.astype(np.float64)
    HF = np.column_stack([np.ones(nh), lf > 0, lf, gold_h, lf * Q.cg] + out_h + miss_h + [out_h[k] * Q.cm[k] for k in range(len(MS))]
                         + [gold_h > 0] + [m > 0 for m in miss_h]).astype(np.float64)
    acc.add(("hit", key, "ALL"), HF.sum(0)[None, :], Q.sids)
    for dim, b, nbins in (("rank", Q.rank_bin, len(RANK_LABELS)), ("gold_hit", Q.gold_hit, 2), ("spread", dbin(lens), len(DEG_LABELS))):
        A = np.zeros((nbins, HF.shape[1]))
        np.add.at(A, b, HF)
        acc.add(("hit", key, dim), A, Q.sids)
    if F is None:
        return
    for dim, b, nbins in (("nbr_degree", F["dbin"][nb], len(DEG_LABELS)), ("coherence", F["cbin"][pos], NCOH)):
        b = b.astype(np.int64)
        cnt = np.bincount(b, minlength=nbins).astype(np.float64)
        gb = np.bincount(b, weights=isg, minlength=nbins)
        ob = [np.bincount(b, weights=o, minlength=nbins) for o in outs]
        mb = [np.bincount(b, weights=m, minlength=nbins) for m in miss]
        acc.add(("edge", key, dim), np.column_stack([cnt, gb, cnt * Q.cg] + ob + mb + [ob[k] * Q.cm[k] for k in range(len(MS))]), Q.sids)


def node_set(acc, key, lens, nb, Q, mark):
    c10 = int(lens[:10].sum())
    for lab, sel in (("top10", nb[:c10]), ("top200", nb)):
        mark[sel] = True
        U = np.flatnonzero(mark)
        mark[U] = False
        R = np.zeros((len(MS), len(SET_FIELDS)))
        for k in range(len(MS)):
            Um = U[Q.out[k][U]]
            found, nm = int(Q.gm[Um].sum()), Q.nmiss[k]
            ff = int(((Q.grank >= MS[k]) & (Q.grank < MS[k] + len(Um))).sum())     # FLAT extended by the same number of nodes
            R[k] = [1, nm > 0, len(Um), found, nm, len(Um) * Q.cm[k], (nm > 0) and found == nm, ff, (nm > 0) and ff == nm]
        acc.add(("set", key, lab), R, Q.sids)


def block_level(acc, key, cell, P, S, hit, nb, lens, Q):
    nh, npart = len(S), P.npart
    own = P.hard[S]
    M2 = np.zeros(nh * npart, bool)
    M2[np.arange(nh, dtype=np.int64) * npart + own] = True
    if len(nb):
        M2[hit * npart + P.hard[nb]] = True
    M2 = M2.reshape(nh, npart)
    nblk = M2.sum(1)
    votes = M2.sum(0)
    act = votes > 0
    ownm = np.zeros(npart, bool)
    ownm[own] = True
    added = act & ~ownm
    GB = Q.gb[cell]
    v = [1, act.sum(), P.sizes[act].sum(), added.sum(), P.sizes[added].sum(), (act & GB["any"]).sum(), (added & GB["any"]).sum()]
    for k in range(len(MS)):
        gbm, hb = GB["miss"][k], GB["miss_nodes_blocks"][k]
        newm = int(P.sizes[act].sum() - GB["flat_cnt"][k][act].sum())    # activated-block nodes outside FLAT@M
        if len(hb):
            cov, cova = act[hb].sum(), added[hb].sum()
            ff = int(((Q.grank >= MS[k]) & (Q.grank < MS[k] + newm)).sum())
            v += [1, act.sum(), added.sum(), (act & gbm).sum(), (added & gbm).sum(), gbm.sum(), len(hb), cov, cova, cov == len(hb),
                  newm, ff, ff == len(hb)]
        else:
            v += [0] * 10 + [newm, 0, 0]
    acc.add(("blk", key, cell), np.array(v, np.float64), Q.sids)
    sb = dbin(lens)
    HB = np.column_stack([np.ones(nh), nblk, nblk - 1] + [np.count_nonzero(M2 & GB["miss"][k][None, :], axis=1) - GB["miss"][k][own]
                                                          for k in range(len(MS))]
                         + [np.count_nonzero(M2 & GB["any"][None, :], axis=1) - GB["any"][own]]).astype(np.float64)
    A = np.zeros((len(DEG_LABELS), len(BSPREAD_FIELDS)))
    np.add.at(A, sb, HB)
    acc.add(("blkspread", key, cell), A, Q.sids)
    ia = np.flatnonzero(act)
    vb = np.searchsorted(VOTE_EDGES, votes[ia], side="right") - 1
    VB = np.column_stack([np.ones(len(ia)), P.sizes[ia], GB["any"][ia]] + [GB["miss"][k][ia] for k in range(len(MS))]).astype(np.float64)
    A = np.zeros((len(VOTE_LABELS), len(BVOTE_FIELDS)))
    np.add.at(A, vb, VB)
    acc.add(("blkvote", key, cell), A, Q.sids)


def gold_pairs(acc, key, F, Q):
    g = Q.g
    ng = len(g)
    if ng < 2:
        return
    xadj, adj = F["xadj"], F["adj"]
    lens = xadj[g + 1] - xadj[g]
    hit, pos = entries(xadj[g], lens)
    isg = Q.gm[adj[pos]].astype(np.float64)
    per = np.bincount(hit, weights=isg, minlength=ng) > 0
    acc.add(("pairs", key), np.array([1, ng, ng * (ng - 1), isg.sum(), per.sum()], np.float64), Q.sids)


def ratio(a, b):
    return None if not b else round(float(a) / float(b), 5)


def summarise(acc, snames, FAM, static, parts, N):
    """derived ratios per stratum from the accumulated sums (the sums are kept too)."""
    out = {"strata": snames, "per_hit": {}, "per_edge": {}, "node_set": {}, "blocks": {}, "gold_pairs": {}, "sums": {}}
    fi = {f: i for i, f in enumerate(HIT_FIELDS)}
    ei = {f: i for i, f in enumerate(EDGE_FIELDS)}
    si = {f: i for i, f in enumerate(SET_FIELDS)}
    bi = {f: i for i, f in enumerate(BLK_FIELDS)}

    def hitrow(x):
        r = {"hits": int(x[fi["hits"]]), "nbrs_per_hit": ratio(x[fi["nbrs"]], x[fi["hits"]]),
             "P_gold": ratio(x[fi["gold"]], x[fi["nbrs"]]), "enrich_gold": ratio(x[fi["gold"]], x[fi["exp_gold"]]),
             "hits_with_gold_nbr": ratio(x[fi["hits_with_gold_nbr"]], x[fi["hits"]])}
        for M in MS:
            r["P_miss@%d" % M] = ratio(x[fi["miss@%d" % M]], x[fi["out@%d" % M]])
            r["enrich_miss@%d" % M] = ratio(x[fi["miss@%d" % M]], x[fi["exp_miss@%d" % M]])
            r["hits_with_miss@%d_nbr" % M] = ratio(x[fi["hits_with_miss@%d_nbr" % M]], x[fi["hits"]])
            r["miss@%d_found_per_hit" % M] = ratio(x[fi["miss@%d" % M]], x[fi["hits"]])
        return r

    def edgerow(x, tot):
        r = {"share_of_entries": ratio(x[ei["nbrs"]], tot), "P_gold": ratio(x[ei["gold"]], x[ei["nbrs"]]),
             "enrich_gold": ratio(x[ei["gold"]], x[ei["exp_gold"]])}
        for M in MS:
            r["P_miss@%d" % M] = ratio(x[ei["miss@%d" % M]], x[ei["out@%d" % M]])
            r["enrich_miss@%d" % M] = ratio(x[ei["miss@%d" % M]], x[ei["exp_miss@%d" % M]])
        return r

    for key, a in acc.d.items():
        sec = key[0]
        out["sums"]["|".join(str(k) for k in key)] = a.tolist()
        for si_, sn in enumerate(snames):
            x = a[si_]
            if sec == "hit":
                _, fam, dim = key
                labs = {"ALL": ["ALL"], "rank": RANK_LABELS, "gold_hit": ("non-gold hit", "gold hit"), "spread": DEG_LABELS}[dim]
                out["per_hit"].setdefault(fam, {}).setdefault(dim, {})[sn] = {lab: hitrow(x[i]) for i, lab in enumerate(labs) if x[i][0] > 0}
            elif sec == "edge":
                _, fam, dim = key
                labs = {"nbr_degree": DEG_LABELS, "coherence": COH_LABELS}[dim]
                tot = x[:, ei["nbrs"]].sum()
                out["per_edge"].setdefault(fam, {}).setdefault(dim, {})[sn] = {lab: edgerow(x[i], tot) for i, lab in enumerate(labs) if x[i][0] > 0}
            elif sec == "set":
                _, fam, lab = key
                e = {}
                for k, M in enumerate(MS):
                    y = x[k]
                    e[str(M)] = {"new_nodes_per_query": ratio(y[si["new_nodes"]], y[si["queries"]]),
                                 "precision": ratio(y[si["found"]], y[si["new_nodes"]]),
                                 "recall_missing_gold_nodes": ratio(y[si["found"]], y[si["missing"]]),
                                 "enrichment": ratio(y[si["found"]], y[si["exp_found"]]),
                                 "queries_all_missing_found": ratio(y[si["all_found"]], y[si["queries_with_miss"]]),
                                 "queries_with_missing": int(y[si["queries_with_miss"]]), "missing_gold_nodes": int(y[si["missing"]]),
                                 "FLAT_ext_recall_same_exposure": ratio(y[si["found_flat_ext"]], y[si["missing"]]),
                                 "FLAT_ext_queries_all_found_same_exposure": ratio(y[si["all_found_flat_ext"]], y[si["queries_with_miss"]])}
                out["node_set"].setdefault(fam, {}).setdefault(lab, {})[sn] = e
            elif sec == "blk":
                _, fam, cell = key
                q = x[bi["queries"]]
                e = {"blocks_per_query": ratio(x[bi["blocks"]], q), "mass_per_query": ratio(x[bi["mass"]], q),
                     "added_blocks_per_query": ratio(x[bi["added"]], q), "added_mass_per_query": ratio(x[bi["added_mass"]], q),
                     "precision_any_gold": ratio(x[bi["any_gold_blocks"]], x[bi["blocks"]]),
                     "added_precision_any_gold": ratio(x[bi["any_gold_added"]], x[bi["added"]])}
                for M in MS:
                    qm = x[bi["q_miss@%d" % M]]
                    rnd = ratio(x[bi["gold_blocks@%d" % M]], qm * parts[cell].npart)
                    pr = ratio(x[bi["miss_blocks@%d" % M]], x[bi["blocks|miss@%d" % M]])
                    pa = ratio(x[bi["miss_added@%d" % M]], x[bi["added|miss@%d" % M]])
                    e[str(M)] = {"queries_with_missing": int(qm), "precision_missing": pr, "random_block_share": rnd,
                                 "enrichment": ratio(pr, rnd) if pr is not None and rnd else None,
                                 "added_precision_missing": pa, "added_enrichment": ratio(pa, rnd) if pa is not None and rnd else None,
                                 "recall_missing_gold_nodes": ratio(x[bi["covered@%d" % M]], x[bi["missing@%d" % M]]),
                                 "recall_added_only": ratio(x[bi["covered_added@%d" % M]], x[bi["missing@%d" % M]]),
                                 "queries_all_missing_covered": ratio(x[bi["all_covered@%d" % M]], qm),
                                 "new_mass_per_query": ratio(x[bi["new_mass@%d" % M]], q),
                                 "FLAT_ext_recall_same_new_mass": ratio(x[bi["flat_ext_covered@%d" % M]], x[bi["missing@%d" % M]]),
                                 "FLAT_ext_all_covered_same_new_mass": ratio(x[bi["flat_ext_all@%d" % M]], qm)}
                sp = acc.d.get(("blkspread", fam, cell))
                if sp is not None:
                    y = sp[si_]
                    tot = y.sum(0)
                    e["blocks_per_hit"] = ratio(tot[1], tot[0])
                    e["added_blocks_per_hit"] = ratio(tot[2], tot[0])
                    e["by_hit_spread"] = {DEG_LABELS[i]: {"hits": int(y[i][0]), "blocks_per_hit": ratio(y[i][1], y[i][0]),
                                                          "added_per_hit": ratio(y[i][2], y[i][0]),
                                                          "added_precision_any_gold": ratio(y[i][6], y[i][2]),
                                                          **{"added_precision_missing@%d" % M: ratio(y[i][3 + k], y[i][2]) for k, M in enumerate(MS)}}
                                          for i in range(len(DEG_LABELS)) if y[i][0] > 0}
                vt = acc.d.get(("blkvote", fam, cell))
                if vt is not None:
                    y = vt[si_]
                    e["by_votes"] = {VOTE_LABELS[i]: {"blocks": int(y[i][0]), "precision_any_gold": ratio(y[i][2], y[i][0]),
                                                      **{"precision_missing@%d" % M: ratio(y[i][3 + k], y[i][0]) for k, M in enumerate(MS)}}
                                     for i in range(len(VOTE_LABELS)) if y[i][0] > 0}
                out["blocks"].setdefault(fam, {}).setdefault(cell, {})[sn] = e
            elif sec == "pairs":
                _, fam = key
                dens = FAM[fam]["entries"] / (float(N) * (N - 1))
                rate = ratio(x[3], x[2])
                out["gold_pairs"].setdefault(fam, {})[sn] = {
                    "queries_ge2_gold": int(x[0]), "adjacent_rate": rate, "density": float("%.3g" % dens),
                    "enrichment": round(rate / dens, 1) if rate is not None and dens else None,
                    "golds_with_gold_nbr": ratio(x[4], x[1])}
            elif sec == "pairs_block":
                _, cell = key
                rb = static["random_pair_block_share"][cell]
                rate = ratio(x[2], x[1])
                out["gold_pairs"].setdefault("SAME_BLOCK:" + cell, {})[sn] = {
                    "queries_ge2_gold": int(x[0]), "adjacent_rate": rate, "density": rb,
                    "enrichment": round(rate / rb, 1) if rate is not None and rb else None}
    return out


def markdown(ds, S, static, snames):
    L = []
    fams_main = [f for f in S["per_hit"] if not f.endswith("_out") and not f.endswith("_in") or f in ("STRUCT_out", "STRUCT_in")]
    fams_rel = [f for f in S["per_hit"] if f not in fams_main]
    h = lambda fam, dim="ALL", sn="ALL", lab="ALL": S["per_hit"].get(fam, {}).get(dim, {}).get(sn, {}).get(lab, {})
    f3 = lambda x: "-" if x is None else ("%.3f" % x)
    f1 = lambda x: "-" if x is None else ("%.1f" % x)
    L.append("\n#### %s -- per hit (S = FLAT top-200): P(gold | u in N_F(s)) and enrichment over random\n" % ds)
    L.append("| family | nbrs/hit | P(g) | enr(g) | enr miss@1000 | enr miss@5000 | gold-hit enr(g) | gold-hit enr miss@5000 | top-10 enr miss@5000 | hits w/ miss@5000 nbr |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for fam in fams_main + fams_rel:
        a, gh, t10 = h(fam), h(fam, "gold_hit", "ALL", "gold hit"), h(fam, "rank", "ALL", "1-10")
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fam, f1(a.get("nbrs_per_hit")), f3(a.get("P_gold")), f1(a.get("enrich_gold")), f1(a.get("enrich_miss@1000")),
            f1(a.get("enrich_miss@5000")), f1(gh.get("enrich_gold")), f1(gh.get("enrich_miss@5000")), f1(t10.get("enrich_miss@5000")),
            f3(a.get("hits_with_miss@5000_nbr"))))
    L.append("\nper hop, enrichment of missing@5000 per hit (all hits):\n")
    L.append("| family | " + " | ".join(snames) + " |")
    L.append("|---|" + "---|" * len(snames))
    for fam in fams_main + fams_rel:
        L.append("| %s | %s |" % (fam, " | ".join(f1(h(fam, "ALL", sn).get("enrich_miss@5000")) for sn in snames)))
    L.append("\n#### %s -- node set U = union of N_F(s) over the top-200 hits, outside FLAT@M\n" % ds)
    L.append("| family | new@1000 | recall@1000 | FLAT-ext recall@1000 (same exposure) | enr@1000 | new@5000 | prec@5000 | recall@5000 | FLAT-ext recall@5000 (same exposure) | enr@5000 | all-missing found@5000 | FLAT-ext all found@5000 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam in list(S["node_set"]):
        e = S["node_set"][fam].get("top200", {}).get("ALL")
        if not e:
            continue
        a, b = e["1000"], e["5000"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fam, f1(a["new_nodes_per_query"]), f3(a["recall_missing_gold_nodes"]), f3(a["FLAT_ext_recall_same_exposure"]), f1(a["enrichment"]),
            f1(b["new_nodes_per_query"]), f3(b["precision"]), f3(b["recall_missing_gold_nodes"]), f3(b["FLAT_ext_recall_same_exposure"]),
            f1(b["enrichment"]), f3(b["queries_all_missing_found"]), f3(b["FLAT_ext_queries_all_found_same_exposure"])))
    for cell in sorted({c for f in S["blocks"] for c in S["blocks"][f]}):
        L.append("\n#### %s / %s -- blocks activated from the top-200 hits: mem_F(s) = own block + blocks of N_F(s)\n" % (ds, cell))
        L.append("| family | blocks/hit | added/hit | blocks/q | mass/q | prec any-gold | prec miss@5000 (enr) | ADDED prec miss@5000 (enr) | recall miss@5000 | recall ADDED only | all covered@5000 | new mass@5000 | FLAT-ext recall@5000 (same new mass) | prec miss@1000 (enr) | ADDED prec miss@1000 (enr) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for fam in S["blocks"]:
            e = S["blocks"][fam].get(cell, {}).get("ALL")
            if not e:
                continue
            b5, b1 = e["5000"], e["1000"]
            L.append("| %s | %s | %s | %s | %s | %s | %s (%s) | %s (%s) | %s | %s | %s | %s | %s | %s (%s) | %s (%s) |" % (
                fam, f1(e.get("blocks_per_hit")), f1(e.get("added_blocks_per_hit")), f1(e["blocks_per_query"]), f1(e["mass_per_query"]),
                f3(e["precision_any_gold"]), f3(b5["precision_missing"]), f1(b5["enrichment"]), f3(b5["added_precision_missing"]),
                f1(b5["added_enrichment"]), f3(b5["recall_missing_gold_nodes"]), f3(b5["recall_added_only"]), f3(b5["queries_all_missing_covered"]),
                f1(b5["new_mass_per_query"]), f3(b5["FLAT_ext_recall_same_new_mass"]),
                f3(b1["precision_missing"]), f1(b1["enrichment"]), f3(b1["added_precision_missing"]), f1(b1["added_enrichment"])))
        L.append("\nrandom block share holding a missing gold: @1000 %s, @5000 %s" % tuple(
            f3(S["blocks"]["OWN"][cell]["ALL"][str(M)]["random_block_share"]) for M in (1000, 5000)))
    L.append("\n#### %s -- gold co-relevance (FLAT-independent): ordered gold pairs (g1, g2) with g2 in N_F(g1)\n" % ds)
    L.append("| family | queries (>=2 gold) | adjacent rate | density | enrichment | golds with a gold nbr | " + " | ".join("rate %s" % sn for sn in snames[1:]) + " |")
    L.append("|---|---|---|---|---|---|" + "---|" * (len(snames) - 1))
    for fam, e in S["gold_pairs"].items():
        a = e.get("ALL")
        if not a:
            continue
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (fam, a["queries_ge2_gold"], f3(a["adjacent_rate"]), a["density"],
                                                         a["enrichment"], f3(a.get("golds_with_gold_nbr")),
                                                         " | ".join(f3(e.get(sn, {}).get("adjacent_rate")) for sn in snames[1:])))
    L.append("\n#### %s -- within-family: per-entry enrichment (missing@5000 / all gold) by edge COHERENCE quintile and by neighbour degree\n" % ds)
    L.append("| family | " + " | ".join(COH_LABELS) + " | quintile thresholds |")
    L.append("|---|" + "---|" * (NCOH + 1))
    for fam, e in S["per_edge"].items():
        c = e.get("coherence", {}).get("ALL", {})
        thr = static["families"][fam]["coherence"]["quintile_thresholds"] if static["families"][fam]["coherence"] else []
        L.append("| %s | %s | %s |" % (fam, " | ".join("%s / %s" % (f1(c.get(l, {}).get("enrich_miss@5000")), f1(c.get(l, {}).get("enrich_gold")))
                                                      for l in COH_LABELS), thr))
    L.append("\n| family | " + " | ".join("deg %s" % l for l in DEG_LABELS[1:]) + " |")
    L.append("|---|" + "---|" * (len(DEG_LABELS) - 1))
    for fam, e in S["per_edge"].items():
        c = e.get("nbr_degree", {}).get("ALL", {})
        L.append("| %s | %s |" % (fam, " | ".join("%s / %s (%s)" % (f1(c.get(l, {}).get("enrich_miss@5000")), f1(c.get(l, {}).get("enrich_gold")),
                                                                    f3(c.get(l, {}).get("share_of_entries"))) for l in DEG_LABELS[1:])))
    L.append("\n#### %s -- label-free family statistics\n" % ds)
    L.append("| family | entries | pairs | nodes w/ edges | degree mean / median / p99 / max | coherence mean | block-internal share | overlap KNN | overlap STRUCT |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for fam, e in static["families"].items():
        dg = e["degree_undirected"] or {}
        L.append("| %s | %d | %d | %d | %s / %s / %s / %s | %s | %s | %s | %s |" % (
            fam, e["entries"], e["undirected_pairs"], e["nodes_with_edges"], dg.get("mean"), dg.get("median"),
            (e["degree_p90_p99"] or [None, None])[1], dg.get("max"), e["coherence"] and e["coherence"]["mean"],
            e["block_internal_share"], e["overlap_with_KNN"], e["overlap_with_STRUCT"]))
    L.append("\nrandom-pair coherence mean %s; random-pair same-block share %s" % (static["random_pair_cosine"]["mean"], static["random_pair_block_share"]))
    return "\n".join(L)


def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else ""
    assert MODE == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    ds, tag = sys.argv[2], sys.argv[3]
    ROWS = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--rows=")), None)
    SMOKE_OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    assert (ROWS is None) == (SMOKE_OUT is None)
    assert SMOKE_OUT is None or not os.path.abspath(SMOKE_OUT).lower().startswith(os.path.abspath(REPO).lower())
    code_sha = D.sha_file(os.path.abspath(__file__))
    lib_sha = D.sha_file(os.path.join(HERE, "_l1d_lib.py"))
    host0 = D.host_state()
    t_all = time.time()
    fp_out = os.path.join(SMOKE_OUT or OUT, "edgediag_%s__%s.json" % (ds, tag))
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr = pop.nq, pop.gptr
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    log("RUN edgediag %s %s: N %d, %d rows, %d gold nodes, cells %s" % (ds, tag, N, nq, pop.ng_tot, cells))
    FAM, static = build_families(cd, N, parts, log)
    log("families built (%.0fs, RSS %.0f MB)" % (static["seconds"], D._rss_mb()))
    hops = sorted(set(int(x) for x in pop.hops))
    snames = ["ALL"] + ["hop%d" % x for x in hops]
    sidx = {x: 1 + i for i, x in enumerate(hops)}
    acc = Acc(len(snames))
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(NH, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(NH, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    Q = QCtx()
    Q.gm = np.zeros(N, bool)
    mark = np.zeros(N, bool)
    Q.rank_bin = np.searchsorted(RANK_EDGES, np.arange(NH), side="right") - 1
    fams = list(FAM)
    LAT = collections.defaultdict(list)
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos_j, fv, frank = D.flat_row(SD[i], SS[i])
            k_ = min(D.N_AGREE, npos_j)
            agd[j] = len(set(od[:D.N_AGREE].tolist()) & set(d200[j, :D.N_AGREE].tolist())) / float(D.N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            g = pop.golds[j]
            POS_FLAT[gptr[j]:gptr[j + 1]] = frank[g]
            S = of[:NH].astype(np.int64)
            Q.g = g
            Q.grank = frank[g]
            Q.gm[g] = True
            Q.out = [frank >= M for M in MS]
            Q.nmiss = [int((frank[g] >= M).sum()) for M in MS]
            Q.cg = len(g) / float(N)
            Q.cm = [Q.nmiss[k] / float(N - M) for k, M in enumerate(MS)]
            Q.gold_hit = Q.gm[S].astype(np.int64)
            Q.sids = (0, sidx[int(pop.hops[j])])
            Q.gb = {}
            for c, P in parts.items():
                gba = np.zeros(P.npart, bool)
                gba[P.hard[g]] = True
                miss_b, mnb = [], []
                for M in MS:
                    gm_ = g[frank[g] >= M]
                    x = np.zeros(P.npart, bool)
                    x[P.hard[gm_]] = True
                    miss_b.append(x)
                    mnb.append(P.hard[gm_])
                Q.gb[c] = {"any": gba, "miss": miss_b, "miss_nodes_blocks": mnb,
                           "flat_cnt": [np.bincount(P.hard[of[:M]], minlength=P.npart) for M in MS]}
            for fam in fams:
                t0 = time.perf_counter()
                F = FAM[fam]
                xadj = F["xadj"]
                lens = xadj[S + 1] - xadj[S]
                hit, pos = entries(xadj[S], lens)
                nb = F["adj"][pos].astype(np.int64)
                hit_level(acc, fam, lens, hit, nb, pos, F, Q)
                node_set(acc, fam, lens, nb, Q, mark)
                for c, P in parts.items():
                    block_level(acc, fam, c, P, S, hit, nb, lens, Q)
                gold_pairs(acc, fam, F, Q)
                LAT[fam].append(time.perf_counter() - t0)
            for c, P in parts.items():
                own = P.hard[S]
                lensb = P.sizes[own]
                hit, pos = entries(P.ptr[own], lensb)
                nb = P.order_nodes[pos].astype(np.int64)
                keep = nb != S[hit]
                hit, nb = hit[keep], nb[keep]
                lens = np.bincount(hit, minlength=NH)
                hit_level(acc, "BLOCK:" + c, lens, hit, nb, None, None, Q)
                node_set(acc, "BLOCK:" + c, lens, nb, Q, mark)
                block_level(acc, "OWN", c, P, S, np.empty(0, np.int64), np.empty(0, np.int64), np.zeros(NH, np.int64), Q)
                cnt = np.bincount(P.hard[g], minlength=P.npart)
                if len(g) >= 2:
                    acc.add(("pairs_block", c), np.array([1, len(g) * (len(g) - 1), float((cnt * (cnt - 1)).sum())]), Q.sids)
            Q.gm[g] = False
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    f1 = os.path.join(OUT, "loc_%s__%s.npz" % (ds, V1_TAG))
    z1 = np.load(f1)
    assert (z1["rows"][:nq] == pop.rows).all()
    ng = int(z1["gptr"][nq])
    assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT gold ranks differ from the v1 record"
    v1check = "FLAT gold ranks == %s (sha %s) on all %d gold nodes" % (D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    S_ = summarise(acc, snames, FAM, static, parts, N)
    md = markdown(ds, S_, static, snames)
    gdir = os.path.join(cd.dir, "graph")
    res = {"dataset": ds, "tag": tag, "mode": "L1_DEVELOPMENT_EDGE_FAMILY_DIAGNOSTIC (gold-aware; never a candidate)",
           "status": "DEVELOPMENT (descriptive)", "definitions": __doc__, "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot,
           "population": pop.record, "cells": {c: {"partition": P.tag, "path": D.rel(P.path), "sha256": D.sha_file(P.path), "npart": P.npart}
                                               for c, P in parts.items()},
           "graph_inputs": {f: {"path": D.rel(os.path.join(gdir, f)), "sha256": D.sha_file(os.path.join(gdir, f))}
                            for f in ("GRAPH_MANIFEST.json", "structural.npz", "knn.npz", "ner.npz")},
           "nodes_jsonl": {"path": D.rel(os.path.join(cd.dir, "nodes.jsonl")), "sha256": D.sha_file(os.path.join(cd.dir, "nodes.jsonl"))},
           "v1_check": v1check, "served_list_agreement": agree, "family_static": static, "results": S_,
           "missing_gold_nodes": {str(M): int((POS_FLAT >= M).sum()) for M in MS},
           "latency_ms_per_query_per_family": {f: D.ms_stats(v) for f, v in LAT.items()},
           "code": {"path": D.rel(os.path.abspath(__file__)), "sha256": code_sha, "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": lib_sha}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": {"NH": NH, "MS": list(MS), "DEG_EDGES": DEG_EDGES.tolist(),
                                                                          "VOTE_EDGES": VOTE_EDGES.tolist(), "NCOH": NCOH, "SEED": SEED,
                                                                          "N_RANDOM_PAIRS": N_RANDOM_PAIRS},
           "platform": D.platform_record(), "host_at_start": host0, "seconds_loop": t_loop, "markdown": md}
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    assert D.sha_file(os.path.abspath(__file__)) == code_sha and D.sha_file(os.path.join(HERE, "_l1d_lib.py")) == lib_sha, "code changed during the run"
    D.G.S.wj(fp_out, res)
    print(md)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))


if __name__ == "__main__":
    main()
