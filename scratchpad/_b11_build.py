"""G2 B1.1 — EXPANDED-UNIVERSE ADMISSION PILOT · per-dataset universe + feature builder.

For one dataset (argv[1] in {metaqa, 2wiki_clean, squad_clean}), build, per VAL query:
  UNIVERSE = original P50  UNION  parameter-free STRUCTURAL directional geometry expansion (M_MAX=256).
Emit a COMPACT feature table (S0/S1/S2, all per-query bounded/relative, NO dataset id / gold / relation-type /
bracket) + labels + recovered/in_p50 flags + per-hop tag, to scratchpad/_b11/{ds}.npz for the LODO stage.

Graph provenance (row-native, native-structural only, synthetic kNN excluded):
  metaqa      -> kb.txt structural (undirected reachability; s_dir uses embedding displacement, not KB orient).
  2wiki/squad -> master_nodes.neighbors (frozen native graph; counts show structural-dominated, not kNN/NER).
Anchors = retrieval-seeded (dense top-SEED_K). No gold, no bracket, no title-match, no test split.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from src.pipeline.standardizer import load_nodes

DS = sys.argv[1] if len(sys.argv) > 1 else "squad_clean"
BASE = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"scratchpad/_b11/{DS}.npz"
K0 = 60; P_MAIN = 50; SEED_K = 5; M_MAX = 256; H = 3; DEG_CAP = 300; FRONTIER_CAP = 1500; PPR_CAP = 4000
EVAL_CAP = int(os.environ.get("B11_CAP", "2000"))   # per-dataset val cap (metaqa balanced across hops)
TOP200 = 200
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)

S0_COLS = ["dense_pct", "splade_pct", "fused_rrf_pct", "retriever_support_frac", "dense_splade_agree"]
S1_COLS = ["min_hop_from_seed_norm", "seed_connectivity_frac", "struct_support_norm", "degree_pct", "ppr_pct"]
S2_COLS = ["s_dir_pct", "dir_exp_rank_pct", "min_exp_hop_norm", "n_struct_anchors_pct",
           "struct_dist_pct", "dir_support_max_pct", "geometry_added_ind"]


# ------------------------------------------------------------------ row space + graph
def doc_rows():
    docs = [n for n in load_nodes(f"data/processed/master_nodes_{DS}.json") if n.metadata.get("type") != "question"]
    id2row = {n.node_id: i for i, n in enumerate(docs)}
    return docs, id2row


def build_adj(docs, id2row):
    """Undirected row-space adjacency (list of neighbor rows). metaqa=kb.txt structural; else master_nodes.neighbors."""
    n = len(docs); adj = [set() for _ in range(n)]
    if DS == "metaqa":
        norm = lambda name: "metaqa_ent_" + name.strip().lower()
        ne = 0
        for line in open("data/original/metaqa/kb.txt", encoding="utf-8"):
            p = line.rstrip("\n").split("|")
            if len(p) != 3:
                continue
            hi = id2row.get(norm(p[0])); ti = id2row.get(norm(p[2]))
            if hi is None or ti is None or hi == ti:
                continue
            adj[hi].add(ti); adj[ti].add(hi); ne += 1
        src = f"kb.txt structural (undirected), edges={ne}"
    else:
        ne = 0
        for i, nd in enumerate(docs):
            for x in nd.neighbors:
                j = id2row.get(x)
                if j is not None and j != i:
                    adj[i].add(j); adj[j].add(i); ne += 1
        src = f"master_nodes.neighbors (undirected), dir_edges={ne}"
    adj = [np.fromiter(s, np.int32) for s in adj]
    deg = np.array([len(a) for a in adj], np.int32)
    return adj, deg, src


# ------------------------------------------------------------------ geometry primitives (Track-A)
def residual(qvec, seed_rows, Xn):
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    if not len(seed_rows):
        return q
    E = Xn[seed_rows]
    U, s, _ = np.linalg.svd(E.T, full_matrices=False); U = U[:, s > 1e-6]
    if U.shape[1] == 0:
        return q
    r = q - U @ (U.T @ q); nn = np.linalg.norm(r)
    return r / nn if nn > 1e-6 else q


def expand_dir(seed_rows, r_q, adj, deg, Xn, M):
    """Parameter-free directional beam: per hop keep top-M neighbors by s_dir = cos(r_q, norm(x_v - x_e)).
    Returns (added_rows in s_dir order, meta) where meta[v]=(first_hop, best_sdir, n_anchor_edges, best_anchor_row)."""
    scope = set(int(x) for x in seed_rows); frontier = list(int(x) for x in seed_rows)
    order = []; vmeta = {}                                  # v -> [first_hop, best_s, anchor_count]
    rq = r_q.astype(np.float32)
    for hop in range(H):
        Vs = []; Ss = []
        for e in frontier:
            if deg[e] > DEG_CAP:
                continue
            nb = adj[e]
            if len(nb) == 0:
                continue
            nb = nb[deg[nb] <= DEG_CAP]
            if len(nb) == 0:
                continue
            delta = Xn[nb] - Xn[e]; nd = np.linalg.norm(delta, axis=1)
            ok = nd > 1e-9
            if not ok.any():
                continue
            Vs.append(nb[ok]); Ss.append((delta[ok] / nd[ok, None]) @ rq)
        if not Vs:
            break
        V = np.concatenate(Vs); S = np.concatenate(Ss).astype(np.float64)
        uv, inv = np.unique(V, return_inverse=True)         # vectorized aggregation over unique targets
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), -1e30); np.maximum.at(best_s, inv, S)
        for k in range(len(uv)):                            # loop now over UNIQUE targets, not all edges
            v = int(uv[k]); m = vmeta.get(v)
            if m is None:
                vmeta[v] = [hop + 1, float(best_s[k]), int(cnt[k])]
            else:
                m[2] += int(cnt[k])
                if best_s[k] > m[1]:
                    m[1] = float(best_s[k])
        new_mask = np.fromiter((int(v) not in scope for v in uv), bool, len(uv))
        if not new_mask.any():
            break
        cv = uv[new_mask]; cs = best_s[new_mask]
        keep = np.argsort(-cs, kind="stable")[:M]
        for idx in keep:
            v = int(cv[idx])
            if v not in scope:
                scope.add(v); order.append((float(cs[idx]), v))
        frontier = [int(cv[i]) for i in keep]
        if len(scope) >= M_MAX * 4:
            break
    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:M_MAX]
    return added, vmeta


def local_ppr(seed_rows, universe_rows, adj, deg, alpha=0.15, iters=20):
    """Bounded personalized PageRank restarted to seeds over the subgraph induced by universe+seeds ONLY
    (no ring -> bounded), hub-pruned (deg<=DEG_CAP), node-capped at PPR_CAP. Edge list -> vectorized power iter."""
    nodes = list(dict.fromkeys([int(x) for x in seed_rows] + [int(x) for x in universe_rows]))[:PPR_CAP]
    idx = {u: k for k, u in enumerate(nodes)}; N = len(idx)
    if N == 0:
        return {}
    src = []; dst = []
    for u in nodes:
        ui = idx[u]
        au = adj[u]
        if len(au) > DEG_CAP:
            continue
        for v in au:
            v = int(v)
            j = idx.get(v)
            if j is not None:
                src.append(ui); dst.append(j)
    s = np.zeros(N)
    for u in seed_rows:
        j = idx.get(int(u))
        if j is not None:
            s[j] = 1.0
    if s.sum() == 0:
        return {}
    s /= s.sum()
    if not src:
        return {u: float(s[idx[u]]) for u in nodes}
    src = np.asarray(src); dst = np.asarray(dst)
    outdeg = np.bincount(src, minlength=N).astype(np.float64); outdeg[outdeg == 0] = 1.0
    r = s.copy()
    for _ in range(iters):
        contrib = (r / outdeg)[src]
        nr = np.zeros(N); np.add.at(nr, dst, contrib)
        r = alpha * s + (1 - alpha) * nr
    return {u: float(r[idx[u]]) for u in nodes}


# ------------------------------------------------------------------ per-query feature construction
def pct(v):
    """within-array percentile, 1=largest; ties share the averaged rank. bounded [0,1]."""
    n = len(v)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(v, kind="stable")
    rank = np.empty(n); rank[order] = np.arange(n)
    # average ties
    vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); starts = csum - counts
    avg = (starts + csum - 1) / 2.0
    rank = avg[inv]
    return rank / (n - 1)


def hop_features(seeds, universe, adj, deg):
    """min hop from any seed, seed-connectivity fraction, #distinct seeds reaching, BFS distance — per universe row."""
    ns = max(len(seeds), 1)
    # per-seed BFS distance up to H
    dist = {}                                  # row -> min hop over seeds
    reach_count = {}                           # row -> #seeds reaching within H
    for si in seeds:
        seen = {int(si): 0}; frontier = [int(si)]
        for h in range(1, H + 1):
            nf = []
            for u in frontier:
                if deg[u] > DEG_CAP:
                    continue
                for v in adj[u]:
                    v = int(v)
                    if v not in seen and deg[v] <= DEG_CAP:
                        seen[v] = h; nf.append(v)
            if len(nf) > FRONTIER_CAP:                     # hub-blowup guard: keep lowest-degree frontier
                nf = list(np.asarray(nf)[np.argsort(deg[np.asarray(nf)])[:FRONTIER_CAP]])
            frontier = nf
            if not frontier:
                break
        for v, hv in seen.items():
            if v == int(si):
                continue
            dist[v] = min(dist.get(v, 99), hv)
            reach_count[v] = reach_count.get(v, 0) + 1
    uni = list(universe)
    minhop = np.zeros(len(uni)); conn = np.zeros(len(uni)); supp = np.zeros(len(uni)); dnorm = np.zeros(len(uni))
    for j, c in enumerate(uni):
        c = int(c)
        if c in dist:
            h = dist[c]; minhop[j] = 1.0 / (1.0 + h); conn[j] = reach_count[c] / ns
            supp[j] = reach_count[c]; dnorm[j] = h
    return minhop, conn, supp, dnorm


def build():
    os.makedirs("scratchpad/_b11", exist_ok=True)
    log(f"=== B1.1 BUILD {DS}  M_MAX={M_MAX} SEED_K={SEED_K} H={H} EVAL_CAP={EVAL_CAP} ===")
    docs, id2row = doc_rows(); ndoc = len(docs)
    adj, deg, gsrc = build_adj(docs, id2row); log(f"graph: {gsrc}  ndoc={ndoc}")
    Q = json.load(open(BASE + "query_ids_all.json"))
    ids = Q["ids"]; golds = Q["golds"]; hops = Q["hops"]; si = Q["split_indices"]
    val_rows = list(si["val"])
    dense_top = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
    splade_top = np.load(BASE + "splade_top200_all.npy", mmap_mode="r")
    nodes = np.load(BASE + "nodes.npy", mmap_mode="r")
    Xn = np.load(BASE + "nodes.npy", mmap_mode="r")   # will normalize per-slice
    qall = np.load(BASE + "queries_all.npy", mmap_mode="r")
    # normalize node embeddings once (fits in RAM: ndoc x 1536 float32)
    Xn = np.asarray(nodes, np.float32); Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
    log(f"val queries={len(val_rows)} nodes={nodes.shape}")

    # dataset-balanced eval sampling (metaqa across hops)
    def hop_of(qr):
        h = hops[qr]
        return int(h) if h is not None else 1
    if DS == "metaqa":
        buckets = {1: [], 2: [], 3: []}
        for qr in val_rows:
            buckets[hop_of(qr)].append(qr)
        per = EVAL_CAP // 3; sel = []
        for h in (1, 2, 3):
            b = buckets[h]; np.random.shuffle(b); sel += b[:per]
        val_rows = sorted(sel)
    else:
        np.random.shuffle(val_rows); val_rows = sorted(val_rows[:EVAL_CAP])
    log(f"eval queries after cap={len(val_rows)}")

    rows_out = []          # feature rows
    grp = []               # per-query candidate counts
    y = []; recov = []; inp50 = []; qhop = []; qids_out = []
    stat = {"n_q": 0, "n_q_with_gold": 0, "recovered_golds": 0, "p50_golds": 0, "added_total": 0,
            "universe_total": 0, "gold_in_universe": 0, "gold_total": 0}
    for qi, qr in enumerate(val_rows):
        g_rows = set()
        for gid in golds[qr]:
            r = id2row.get(gid)
            if r is not None:
                g_rows.add(r)
        if not g_rows:
            continue
        stat["n_q_with_gold"] += 1; stat["gold_total"] += len(g_rows)
        dtop = np.asarray(dense_top[qr], np.int64)          # top-200 dense rows
        stop = np.asarray(splade_top[qr], np.int64)          # top-200 splade rows
        drank = {int(r): k for k, r in enumerate(dtop)}
        srank = {int(r): k for k, r in enumerate(stop)}
        # P50 via RRF over the union of the two top-200 lists
        union = list(dict.fromkeys([int(x) for x in dtop] + [int(x) for x in stop]))
        rrf = np.array([1.0 / (K0 + drank.get(c, TOP200)) + 1.0 / (K0 + srank.get(c, TOP200)) for c in union])
        p50 = [union[k] for k in np.argsort(-rrf, kind="stable")[:P_MAIN]]
        p50set = set(p50)
        # geometry expansion
        seeds = [int(x) for x in dtop[:SEED_K]]
        r_q = residual(np.asarray(qall[qr], np.float64), seeds, Xn)
        added, vmeta = expand_dir(seeds, r_q, adj, deg, Xn, M_MAX)
        added = [v for v in added if v not in p50set]
        universe = p50 + added
        uni_arr = np.array(universe, np.int64)
        n = len(universe)
        stat["added_total"] += len(added); stat["universe_total"] += n
        giu = g_rows & set(universe); stat["gold_in_universe"] += len(giu)
        rec_g = (g_rows & set(added)) - p50set
        stat["recovered_golds"] += len(rec_g); stat["p50_golds"] += len(g_rows & p50set)
        stat["n_q"] += 1

        # ---- S0 retrieval-relative ----
        qn = Xn[uni_arr]                                     # (n,1536) normalized
        qv = np.asarray(qall[qr], np.float32); qv = qv / (np.linalg.norm(qv) + 1e-9)
        dense_score = qn @ qv                                # exact cos
        sp_rank = np.array([srank.get(int(c), TOP200) for c in universe], np.float64)
        dr_rank = np.array([drank.get(int(c), TOP200) for c in universe], np.float64)
        rrf_u = 1.0 / (K0 + dr_rank) + 1.0 / (K0 + sp_rank)
        dense_pct = pct(dense_score); splade_pct = pct(-sp_rank); fused_pct = pct(rrf_u)
        support = ((dr_rank < K0).astype(np.float64) + (sp_rank < K0).astype(np.float64)) / 2.0
        agree = 1.0 - np.abs(dense_pct - splade_pct)
        S0 = np.stack([dense_pct, splade_pct, fused_pct, support, agree], 1)

        # ---- S1 structural ----
        minhop, conn, supp, dbfs = hop_features(seeds, universe, adj, deg)
        degc = deg[uni_arr].astype(np.float64)
        ppr = local_ppr(seeds, universe, adj, deg)
        pprv = np.array([ppr.get(int(c), 0.0) for c in universe])
        S1 = np.stack([minhop, conn, pct(supp), pct(degc), pct(pprv)], 1)

        # ---- S2 directional geometry ----
        sdir = np.full(n, np.nan); exphop = np.zeros(n); nanchor = np.zeros(n); dirmax = np.full(n, np.nan)
        add_order = {v: k for k, v in enumerate(added)}
        for j, c in enumerate(universe):
            c = int(c)
            if c in vmeta:
                fh, best_s, nedge = vmeta[c]
                sdir[j] = best_s; dirmax[j] = best_s; exphop[j] = 1.0 / (1.0 + fh); nanchor[j] = nedge
        # candidates with no directional evidence -> abstain at the LOW end of the percentile
        def pct_abstain(v):
            m = ~np.isnan(v)
            out = np.zeros(len(v))
            if m.sum() > 0:
                out[m] = pct(v[m]) * 0.5 + 0.5     # defined -> upper half
            return out                              # undefined -> 0 (abstain)
        s_dir_pct = pct_abstain(sdir)
        dir_exp_rank = np.array([1.0 - add_order[int(c)] / max(len(added) - 1, 1) if int(c) in add_order else 0.0
                                 for c in universe])
        geo_ind = np.array([1.0 if int(c) in add_order else 0.0 for c in universe])
        S2 = np.stack([s_dir_pct, dir_exp_rank, exphop, pct(nanchor), pct(dbfs),
                       pct_abstain(dirmax), geo_ind], 1)

        feat = np.concatenate([S0, S1, S2], 1).astype(np.float32)
        lab = np.array([1 if int(c) in g_rows else 0 for c in universe], np.int8)
        rflag = np.array([1 if int(c) in rec_g else 0 for c in universe], np.int8)
        pflag = np.array([1 if int(c) in p50set else 0 for c in universe], np.int8)
        rows_out.append(feat); y.append(lab); recov.append(rflag); inp50.append(pflag)
        grp.append(n); qhop.append(hop_of(qr)); qids_out.append(ids[qr])
        if (qi + 1) % 500 == 0:
            log(f"  {qi+1}/{len(val_rows)} q  cum_recovered_golds={stat['recovered_golds']}")

    X = np.concatenate(rows_out).astype(np.float32)
    out = dict(X=X, y=np.concatenate(y), recov=np.concatenate(recov), inp50=np.concatenate(inp50),
               groups=np.array(grp, np.int64), qhop=np.array(qhop, np.int32),
               S0=np.array(S0_COLS), S1=np.array(S1_COLS), S2=np.array(S2_COLS))
    np.savez_compressed(OUT, **out)
    stat["feature_dim"] = X.shape[1]; stat["graph_src"] = gsrc
    json.dump(stat, open(f"scratchpad/_b11/{DS}_stat.json", "w"), indent=1)
    log(f"DONE {DS}: q={stat['n_q']} cands={len(np.concatenate(y))} feat_dim={X.shape[1]}")
    log(f"  STAT {json.dumps(stat)}")
    log("B11_BUILD_DONE")


if __name__ == "__main__":
    build()
