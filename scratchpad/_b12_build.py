"""G2 B1.2 — INVARIANCE REPAIR · per-dataset universe + feature builder.

Same universe as B1.1 (P50 UNION parameter-free structural directional expansion, M_MAX=256) but emits, per
candidate, FIVE feature blocks so the LODO can compare BASE / S0 / old-S1 / new-S1' / old-S2 / new-S2':
  S0    (5)  retrieval-relative      — unchanged (already candidate-local; identity-clean in B1.1).
  S1old (5)  structural (B1.1)        — MIXES candidate-local + QUERY_GRAPH_REGIME (raw degree, raw reach-count
                                        percentile, raw ppr percentile) -> the elevated dataset fingerprint.
  S2old (7)  directional (B1.1)       — mostly candidate-local + n_struct_anchors (raw edge count = QGR).
  S1new (5)  structural INVARIANT     — CANDIDATE_LOCAL only: seed-normalized support, local degree ratio,
                                        ppr share, structure-vs-retrieval rank advantage.
  S2new (6)  directional INVARIANT    — CANDIDATE_LOCAL only: s_dir percentile among THIS query's traversed
                                        edges, directional margin, directional-vs-retrieval advantage.
Also stores DIAGNOSTIC channels (dense_rank, splade_rank; 200 = absent) for the "why can S0 admit recovered
golds?" dense/SPLADE decomposition, and records the FROZEN GLOBAL COMPUTE CONTRACT + per-query edges-visited /
expansion-latency / candidate-count (mean/p50/p95). No dataset-specific runtime branch; dense graphs obey the
same caps as sparse. No dataset id / gold / relation-type / bracket / target stats in any model feature.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from src.pipeline.standardizer import load_nodes

DS = sys.argv[1] if len(sys.argv) > 1 else "squad_clean"
BASE = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"scratchpad/_b12/{DS}.npz"

# ---- FROZEN GLOBAL COMPUTE CONTRACT (identical for every dataset; no runtime branch on DS) ----
K0 = 60; P_MAIN = 50; SEED_K = 5; M_MAX = 256
MAX_HOPS = 3                    # H
MAX_FRONTIER = 1500            # per-seed BFS frontier cap (keep lowest-degree)
DEG_CAP = 300                 # hub prune (a node with deg>cap is not expanded through)
MAX_EDGES_SCORED = 400000     # per-query hard budget across expansion + hop-BFS + ppr edge scans
PPR_CAP = 4000                # ppr node cap
TOP_POOL = 50; TOP200 = 200
EVAL_CAP = int(os.environ.get("B12_CAP", "2000"))
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)

S0_COLS = ["dense_pct", "splade_pct", "fused_rrf_pct", "retriever_support_frac", "dense_splade_agree"]
S1OLD_COLS = ["min_hop_from_seed_norm", "seed_connectivity_frac", "struct_support_pct", "degree_pct", "ppr_pct"]
S2OLD_COLS = ["s_dir_pct", "dir_exp_rank_pct", "min_exp_hop_norm", "n_struct_anchors_pct",
              "struct_dist_pct", "dir_support_max_pct", "geometry_added_ind"]
# ---- new CANDIDATE_LOCAL invariant variants ----
S1NEW_COLS = ["min_hop_from_seed_norm", "seed_support_frac", "ppr_share_pct",
              "degree_local_ratio", "struct_vs_retrieval_adv"]
S2NEW_COLS = ["s_dir_pct_local", "dir_margin", "dir_exp_rank_pct", "min_exp_hop_norm",
              "dir_vs_retrieval_adv", "geometry_added_ind"]

# feature-separation manifest (CANDIDATE_LOCAL vs QUERY_GRAPH_REGIME) — analysis metadata, not a model input
FEATURE_SEPARATION = {
    "dense_pct": "CANDIDATE_LOCAL", "splade_pct": "CANDIDATE_LOCAL", "fused_rrf_pct": "CANDIDATE_LOCAL",
    "retriever_support_frac": "CANDIDATE_LOCAL", "dense_splade_agree": "CANDIDATE_LOCAL",
    "min_hop_from_seed_norm": "CANDIDATE_LOCAL", "seed_connectivity_frac": "CANDIDATE_LOCAL",
    "struct_support_pct": "QUERY_GRAPH_REGIME", "degree_pct": "QUERY_GRAPH_REGIME", "ppr_pct": "QUERY_GRAPH_REGIME",
    "s_dir_pct": "CANDIDATE_LOCAL", "dir_exp_rank_pct": "CANDIDATE_LOCAL", "min_exp_hop_norm": "CANDIDATE_LOCAL",
    "n_struct_anchors_pct": "QUERY_GRAPH_REGIME", "struct_dist_pct": "CANDIDATE_LOCAL",
    "dir_support_max_pct": "CANDIDATE_LOCAL", "geometry_added_ind": "CANDIDATE_LOCAL",
    "seed_support_frac": "CANDIDATE_LOCAL", "ppr_share_pct": "CANDIDATE_LOCAL",
    "degree_local_ratio": "CANDIDATE_LOCAL", "struct_vs_retrieval_adv": "CANDIDATE_LOCAL",
    "s_dir_pct_local": "CANDIDATE_LOCAL", "dir_margin": "CANDIDATE_LOCAL", "dir_vs_retrieval_adv": "CANDIDATE_LOCAL",
}


# ------------------------------------------------------------------ row space + graph
def doc_rows():
    docs = [n for n in load_nodes(f"data/processed/master_nodes_{DS}.json") if n.metadata.get("type") != "question"]
    id2row = {n.node_id: i for i, n in enumerate(docs)}
    return docs, id2row


def build_adj(docs, id2row):
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


def expand_dir(seed_rows, r_q, adj, deg, Xn, M, budget):
    """Parameter-free directional beam. Returns (added_rows in s_dir order, vmeta, edges_scored).
    vmeta[v] = [first_hop, best_s_dir, anchor_edge_count]. Bounded by MAX_HOPS / DEG_CAP / MAX_EDGES_SCORED."""
    scope = set(int(x) for x in seed_rows); frontier = list(int(x) for x in seed_rows)
    order = []; vmeta = {}; escored = 0
    rq = r_q.astype(np.float32)
    for hop in range(MAX_HOPS):
        Vs = []; Ss = []
        for e in frontier:
            if deg[e] > DEG_CAP:
                continue
            nb = adj[e]
            if len(nb) == 0:
                continue
            escored += len(nb)
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
        uv, inv = np.unique(V, return_inverse=True)
        cnt = np.bincount(inv)
        best_s = np.full(len(uv), -1e30); np.maximum.at(best_s, inv, S)
        for k in range(len(uv)):
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
        if len(scope) >= M_MAX * 4 or escored >= budget:
            break
    added = [v for _, v in sorted(order, key=lambda x: -x[0])][:M_MAX]
    return added, vmeta, escored


def local_ppr(seed_rows, universe_rows, adj, deg, alpha=0.15, iters=20):
    nodes = list(dict.fromkeys([int(x) for x in seed_rows] + [int(x) for x in universe_rows]))[:PPR_CAP]
    idx = {u: k for k, u in enumerate(nodes)}; N = len(idx); escored = 0
    if N == 0:
        return {}, 0
    src = []; dst = []
    for u in nodes:
        ui = idx[u]; au = adj[u]
        if len(au) > DEG_CAP:
            continue
        escored += len(au)
        for v in au:
            v = int(v); j = idx.get(v)
            if j is not None:
                src.append(ui); dst.append(j)
    s = np.zeros(N)
    for u in seed_rows:
        j = idx.get(int(u))
        if j is not None:
            s[j] = 1.0
    if s.sum() == 0:
        return {}, escored
    s /= s.sum()
    if not src:
        return {u: float(s[idx[u]]) for u in nodes}, escored
    src = np.asarray(src); dst = np.asarray(dst)
    outdeg = np.bincount(src, minlength=N).astype(np.float64); outdeg[outdeg == 0] = 1.0
    r = s.copy()
    for _ in range(iters):
        contrib = (r / outdeg)[src]
        nr = np.zeros(N); np.add.at(nr, dst, contrib)
        r = alpha * s + (1 - alpha) * nr
    return {u: float(r[idx[u]]) for u in nodes}, escored


def pct(v):
    """within-array percentile, 1=largest; averaged ties; bounded [0,1]."""
    n = len(v)
    if n <= 1:
        return np.ones(n)
    vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); starts = csum - counts
    avg = (starts + csum - 1) / 2.0
    return avg[inv] / (n - 1)


def hop_features(seeds, universe, adj, deg):
    """min hop, seed-reach count, BFS distance per universe row + edges scanned."""
    ns = max(len(seeds), 1); escored = 0
    dist = {}; reach_count = {}
    for si in seeds:
        seen = {int(si): 0}; frontier = [int(si)]
        for h in range(1, MAX_HOPS + 1):
            nf = []
            for u in frontier:
                if deg[u] > DEG_CAP:
                    continue
                escored += len(adj[u])
                for v in adj[u]:
                    v = int(v)
                    if v not in seen and deg[v] <= DEG_CAP:
                        seen[v] = h; nf.append(v)
            if len(nf) > MAX_FRONTIER:
                nf = list(np.asarray(nf)[np.argsort(deg[np.asarray(nf)])[:MAX_FRONTIER]])
            frontier = nf
            if not frontier:
                break
        for v, hv in seen.items():
            if v == int(si):
                continue
            dist[v] = min(dist.get(v, 99), hv)
            reach_count[v] = reach_count.get(v, 0) + 1
    uni = list(universe)
    minhop = np.zeros(len(uni)); supp = np.zeros(len(uni)); dnorm = np.zeros(len(uni))
    for j, c in enumerate(uni):
        c = int(c)
        if c in dist:
            minhop[j] = 1.0 / (1.0 + dist[c]); supp[j] = reach_count[c]; dnorm[j] = dist[c]
    return minhop, supp, dnorm, ns, escored


def build():
    os.makedirs("scratchpad/_b12", exist_ok=True)
    log(f"=== B1.2 BUILD {DS}  M_MAX={M_MAX} SEED_K={SEED_K} HOPS={MAX_HOPS} EDGE_BUDGET={MAX_EDGES_SCORED} CAP={EVAL_CAP} ===")
    docs, id2row = doc_rows(); ndoc = len(docs)
    adj, deg, gsrc = build_adj(docs, id2row); log(f"graph: {gsrc}  ndoc={ndoc}")
    Q = json.load(open(BASE + "query_ids_all.json"))
    ids = Q["ids"]; golds = Q["golds"]; hops = Q["hops"]; si = Q["split_indices"]
    val_rows = list(si["val"])
    dense_top = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
    splade_top = np.load(BASE + "splade_top200_all.npy", mmap_mode="r")
    nodes = np.load(BASE + "nodes.npy", mmap_mode="r")
    qall = np.load(BASE + "queries_all.npy", mmap_mode="r")
    Xn = np.asarray(nodes, np.float32); Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
    log(f"val queries={len(val_rows)} nodes={nodes.shape}")

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

    S0r = []; S1o = []; S2o = []; S1n = []; S2n = []
    dranks = []; sranks = []                       # diagnostic channels (per candidate)
    grp = []; y = []; recov = []; inp50 = []; qhop = []; qids_out = []
    edges_per_q = []; lat_per_q = []; cand_per_q = []
    stat = {"n_q": 0, "recovered_golds": 0, "p50_golds": 0, "added_total": 0, "universe_total": 0,
            "gold_in_universe": 0, "gold_total": 0}
    for qi, qr in enumerate(val_rows):
        g_rows = set()
        for gid in golds[qr]:
            r = id2row.get(gid)
            if r is not None:
                g_rows.add(r)
        if not g_rows:
            continue
        t_q = time.time()
        stat["gold_total"] += len(g_rows)
        dtop = np.asarray(dense_top[qr], np.int64); stop = np.asarray(splade_top[qr], np.int64)
        drank = {int(r): k for k, r in enumerate(dtop)}; srank = {int(r): k for k, r in enumerate(stop)}
        union = list(dict.fromkeys([int(x) for x in dtop] + [int(x) for x in stop]))
        rrf = np.array([1.0 / (K0 + drank.get(c, TOP200)) + 1.0 / (K0 + srank.get(c, TOP200)) for c in union])
        p50 = [union[k] for k in np.argsort(-rrf, kind="stable")[:P_MAIN]]; p50set = set(p50)
        seeds = [int(x) for x in dtop[:SEED_K]]
        r_q = residual(np.asarray(qall[qr], np.float64), seeds, Xn)
        added, vmeta, esc_dir = expand_dir(seeds, r_q, adj, deg, Xn, M_MAX, MAX_EDGES_SCORED)
        added = [v for v in added if v not in p50set]
        universe = p50 + added; uni_arr = np.array(universe, np.int64); n = len(universe)
        stat["added_total"] += len(added); stat["universe_total"] += n
        stat["gold_in_universe"] += len(g_rows & set(universe))
        rec_g = (g_rows & set(added)) - p50set
        stat["recovered_golds"] += len(rec_g); stat["p50_golds"] += len(g_rows & p50set); stat["n_q"] += 1

        # ---- S0 retrieval-relative (unchanged) ----
        qn = Xn[uni_arr]; qv = np.asarray(qall[qr], np.float32); qv = qv / (np.linalg.norm(qv) + 1e-9)
        dense_score = qn @ qv
        sp_rank = np.array([srank.get(int(c), TOP200) for c in universe], np.float64)
        dr_rank = np.array([drank.get(int(c), TOP200) for c in universe], np.float64)
        rrf_u = 1.0 / (K0 + dr_rank) + 1.0 / (K0 + sp_rank)
        dense_pct = pct(dense_score); splade_pct = pct(-sp_rank); fused_pct = pct(rrf_u)
        support = ((dr_rank < K0).astype(np.float64) + (sp_rank < K0).astype(np.float64)) / 2.0
        agree = 1.0 - np.abs(dense_pct - splade_pct)
        S0 = np.stack([dense_pct, splade_pct, fused_pct, support, agree], 1)

        # ---- structural primitives ----
        minhop, supp, dbfs, ns, esc_hop = hop_features(seeds, universe, adj, deg)
        degc = deg[uni_arr].astype(np.float64)
        ppr, esc_ppr = local_ppr(seeds, universe, adj, deg)
        pprv = np.array([ppr.get(int(c), 0.0) for c in universe])
        conn = supp / ns                                   # seed_support_frac / seed_connectivity_frac

        # ---- S1 OLD (mixes QGR: raw degree pct, raw support pct, raw ppr pct) ----
        S1old = np.stack([minhop, conn, pct(supp), pct(degc), pct(pprv)], 1)

        # ---- S1 NEW invariant (CANDIDATE_LOCAL) ----
        med_deg = np.median(degc) if len(degc) else 0.0
        degree_local_ratio = degc / (degc + med_deg + 1.0)                         # local-relative, bounded
        ppr_share = pprv / (pprv.sum() + 1e-12); ppr_share_pct = pct(ppr_share)     # concentration in THIS universe
        struct_score = conn + 0.5 * minhop + 0.5 * ppr_share_pct
        struct_rank_pct = pct(struct_score)
        struct_vs_retrieval_adv = np.clip(struct_rank_pct - fused_pct, -1, 1) * 0.5 + 0.5   # ->[0,1]
        S1new = np.stack([minhop, conn, ppr_share_pct, degree_local_ratio, struct_vs_retrieval_adv], 1)

        # ---- directional primitives ----
        sdir = np.full(n, np.nan); exphop = np.zeros(n); nanchor = np.zeros(n); dirmax = np.full(n, np.nan)
        add_order = {v: k for k, v in enumerate(added)}
        for j, c in enumerate(universe):
            c = int(c)
            if c in vmeta:
                fh, best_s, nedge = vmeta[c]
                sdir[j] = best_s; dirmax[j] = best_s; exphop[j] = 1.0 / (1.0 + fh); nanchor[j] = nedge

        def pct_abstain(v):                                # defined -> upper half [0.5,1]; undefined -> 0
            m = ~np.isnan(v); out = np.zeros(len(v))
            if m.sum() > 0:
                out[m] = pct(v[m]) * 0.5 + 0.5
            return out
        s_dir_pct = pct_abstain(sdir)
        dir_exp_rank = np.array([1.0 - add_order[int(c)] / max(len(added) - 1, 1) if int(c) in add_order else 0.0
                                 for c in universe])
        geo_ind = np.array([1.0 if int(c) in add_order else 0.0 for c in universe])

        # ---- S2 OLD (n_struct_anchors_pct = raw edge count = QGR) ----
        S2old = np.stack([s_dir_pct, dir_exp_rank, exphop, pct(nanchor), pct(dbfs),
                          pct_abstain(dirmax), geo_ind], 1)

        # ---- S2 NEW invariant (CANDIDATE_LOCAL) ----
        m = ~np.isnan(sdir)
        dir_margin = np.zeros(n)
        if m.sum() > 1:
            md = np.median(sdir[m]); sd = sdir[m].std() + 1e-6
            dm = np.clip((sdir[m] - md) / sd, -3, 3) / 6.0 + 0.5      # ->[0,1], 0.5=median traversed
            dir_margin[m] = dm
        dir_vs_retrieval_adv = np.clip(s_dir_pct - dense_pct, -1, 1) * 0.5 + 0.5    # directional beats similarity?
        S2new = np.stack([s_dir_pct, dir_margin, dir_exp_rank, exphop, dir_vs_retrieval_adv, geo_ind], 1)

        S0r.append(S0); S1o.append(S1old); S2o.append(S2old); S1n.append(S1new); S2n.append(S2new)
        dranks.append(dr_rank.astype(np.int16)); sranks.append(sp_rank.astype(np.int16))
        y.append(np.array([1 if int(c) in g_rows else 0 for c in universe], np.int8))
        recov.append(np.array([1 if int(c) in rec_g else 0 for c in universe], np.int8))
        inp50.append(np.array([1 if int(c) in p50set else 0 for c in universe], np.int8))
        grp.append(n); qhop.append(hop_of(qr)); qids_out.append(ids[qr])
        edges_per_q.append(esc_dir + esc_hop + esc_ppr); lat_per_q.append(time.time() - t_q); cand_per_q.append(n)
        if (qi + 1) % 500 == 0:
            log(f"  {qi+1}/{len(val_rows)} q  cum_recovered={stat['recovered_golds']}")

    def q3(a):
        a = np.asarray(a, np.float64)
        return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2)}
    contract = {"MAX_HOPS": MAX_HOPS, "MAX_FRONTIER": MAX_FRONTIER, "MAX_EDGES_SCORED": MAX_EDGES_SCORED,
                "DEG_CAP": DEG_CAP, "PPR_CAP": PPR_CAP, "M_MAX": M_MAX, "TOP_POOL": TOP_POOL, "SEED_K": SEED_K,
                "K0": K0, "P_MAIN": P_MAIN,
                "edges_visited": q3(edges_per_q), "expansion_latency_s": q3(lat_per_q),
                "candidate_count": q3(cand_per_q), "edge_budget_hit_frac": round(
                    float(np.mean(np.asarray(edges_per_q) >= MAX_EDGES_SCORED)), 4)}
    out = dict(
        S0=np.concatenate(S0r).astype(np.float32), S1old=np.concatenate(S1o).astype(np.float32),
        S2old=np.concatenate(S2o).astype(np.float32), S1new=np.concatenate(S1n).astype(np.float32),
        S2new=np.concatenate(S2n).astype(np.float32),
        dense_rank=np.concatenate(dranks), splade_rank=np.concatenate(sranks),
        y=np.concatenate(y), recov=np.concatenate(recov), inp50=np.concatenate(inp50),
        groups=np.array(grp, np.int64), qhop=np.array(qhop, np.int32),
        S0_COLS=np.array(S0_COLS), S1OLD_COLS=np.array(S1OLD_COLS), S2OLD_COLS=np.array(S2OLD_COLS),
        S1NEW_COLS=np.array(S1NEW_COLS), S2NEW_COLS=np.array(S2NEW_COLS))
    np.savez_compressed(OUT, **out)
    stat.update({"graph_src": gsrc, "compute_contract": contract})
    json.dump(stat, open(f"scratchpad/_b12/{DS}_stat.json", "w"), indent=1)
    log(f"DONE {DS}: q={stat['n_q']} cands={len(np.concatenate(y))} recovered={stat['recovered_golds']}")
    log(f"  CONTRACT {json.dumps(contract)}")
    log("B12_BUILD_DONE")


if __name__ == "__main__":
    build()
