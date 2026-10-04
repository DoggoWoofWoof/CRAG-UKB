"""B1.9 STEP 3+4 — coherent label-free ENVIRONMENTS, and whether descriptors track reserve utility.

FEASIBILITY QUESTION FIRST (cheapest decisive check):
    A corpus-level predictor fit from 3 whole-corpus points is worthless. Environments only help if the
    descriptors and the reserve utility BOTH vary WITHIN a corpus. If within-corpus variance is negligible,
    the 70-odd "points" are really 3 clusters and the analysis must stay descriptive.

ENVIRONMENT CONSTRUCTION (label-free, deterministic, coherence-preserving):
    Deterministic graph-Voronoi partition (multi-source BFS) into `NENV` CONNECTED regions. The regions
    partition the SAMPLE, not the graph -- every descriptor is still computed against the FULL adjacency and
    the FULL-corpus semantic neighbourhood, so no edge is cut and retrieval semantics are untouched.
    Uses: graph topology + node embeddings only. No labels, no queries, no dataset id.

QUERY -> ENVIRONMENT assignment: a query's SEEDS are dense_top200[:SEED_K] (exactly _b12_build's definition) --
    label-free and inference-safe. The query joins the environment holding most of its seeds.

U_E(R) uses TRAIN relevance labels, AFTER the fact, as supervision only -- never as a descriptor input.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from src.pipeline.standardizer import load_nodes
from _b14_twochannel import load
from _b17_marginal import utilities, R_VALS
from _b19_descriptors import build_adj, running_topk

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
OUT = "results/GENERALIZATION/_g2_b19b_environments.json"
NENV = 24            # target environments per corpus
PER_ENV_SAMPLE = 300
MIN_Q = 25           # an environment needs this many queries for a usable U_E(R)
SEED_K = 5; K_TOP = 200; NB_CAP = 32; DEG_CAP = 300; CHUNK = 20000
EVAL_CAP = {"metaqa": 2000, "2wiki_clean": 2000, "squad_clean": 1000}   # B12_CAP used per build
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)


def graph_voronoi_partition(adj, deg, n, nenv):
    """Deterministic multi-source BFS (graph Voronoi): exactly `nenv` CONNECTED regions covering the graph.
    Seeds are evenly spaced by node index among structured nodes -- deterministic, label-free, query-free.
    (A greedy one-region-at-a-time BFS carve degenerates: boundary nodes whose neighbours are already claimed
    become singleton regions, which produced 11k regions instead of 24.)"""
    from collections import deque
    cand = np.where(deg > 0)[0]
    if len(cand) == 0:
        return np.zeros(n, np.int32), 1
    k = int(min(nenv, len(cand)))
    seeds = cand[np.linspace(0, len(cand) - 1, k).astype(np.int64)]
    env = np.full(n, -1, np.int32); q = deque()
    for e, s_ in enumerate(seeds):
        env[s_] = e; q.append(int(s_))
    while q:
        u = q.popleft()
        for v in adj[u]:
            if env[v] == -1:
                env[v] = env[u]; q.append(int(v))
    miss = np.where(env == -1)[0]              # isolated / unreachable: deterministic index blocks
    if len(miss):
        env[miss] = (np.arange(len(miss)) * k // len(miss)).astype(np.int32)
    return env, k


def env_descriptors(ds, env, nenv_actual, adj, deg, n, rng):
    """z_E for each environment -- FULL adjacency, FULL-corpus semantic neighbourhood. Label-free."""
    path = f"data/ukb_storage/{ds}/gte_qwen/nodes.npy"
    X = np.load(path, mmap_mode="r")
    samples, owner = [], []
    for e in range(nenv_actual):
        rows = np.where((env == e) & (deg > 0))[0]
        if len(rows) < 20:
            continue
        take = rows if len(rows) <= PER_ENV_SAMPLE else rng.choice(rows, PER_ENV_SAMPLE, replace=False)
        samples.append(np.sort(take)); owner.append(np.full(len(take), e))
    if not samples:
        return {}
    samp = np.concatenate(samples); own = np.concatenate(owner)
    Xs = np.array(X[samp], dtype=np.float32); Xs /= (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    log(f"    streaming top-{K_TOP} for {len(samp)} env-sampled nodes over {n}")
    tv, ti = running_topk(Xs, path, n, X.shape[1], K_TOP, CHUNK, samp)
    rnd_rows = rng.choice(n, size=min(512, n), replace=False)
    Xr = np.array(X[rnd_rows], dtype=np.float32); Xr /= (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)

    per = {}
    for si, u in enumerate(samp):
        e = int(own[si]); nb = adj[u]
        nb_use = nb if len(nb) <= NB_CAP else nb[:NB_CAP]
        nbs = set(int(x) for x in nb_use); m = len(nbs)
        t10, t50 = set(ti[si, :10].tolist()), set(ti[si, :50].tolist())
        jac = []
        for v in nb_use[:8]:
            a_, b_ = set(adj[u].tolist()), set(adj[v].tolist()); un = len(a_ | b_)
            if un:
                jac.append(len(a_ & b_) / un)
        s2 = set()
        for v in nb_use:
            if deg[v] <= DEG_CAP:
                s2.update(adj[v].tolist())
        s2 -= nbs; s2.discard(int(u))
        Xn_ = np.array(X[np.array(sorted(nbs), np.int64)], dtype=np.float32)
        Xn_ /= (np.linalg.norm(Xn_, axis=1, keepdims=True) + 1e-9)
        d = per.setdefault(e, {k: [] for k in ["deg", "ov10", "ov50", "jac", "h2", "ecos", "rcos", "t10cos"]})
        d["deg"].append(len(nb)); d["ov10"].append(len(nbs & t10) / m); d["ov50"].append(len(nbs & t50) / m)
        d["jac"].append(float(np.mean(jac)) if jac else 0.0)
        d["h2"].append(len(s2) / max(len(nb), 1))
        d["ecos"].append(float((Xs[si] @ Xn_.T).mean())); d["rcos"].append(float((Xs[si] @ Xr.T).mean()))
        d["t10cos"].append(float(tv[si, :10].mean()))

    out = {}
    for e, d in per.items():
        ec, rc, tc = np.mean(d["ecos"]), np.mean(d["rcos"]), np.mean(d["t10cos"])
        out[e] = {"n_sampled": len(d["deg"]),
                  "mean_degree": float(np.mean(d["deg"])),
                  "log_mean_degree": float(np.log1p(np.mean(d["deg"]))),
                  "StructSemOverlap@10": float(np.mean(d["ov10"])),
                  "StructSemOverlap@50": float(np.mean(d["ov50"])),
                  "neighbor_jaccard_redundancy": float(np.mean(d["jac"])),
                  "hop2_over_hop1": float(np.mean(d["h2"])),
                  "edge_semantic_lift": float((ec - rc) / max(tc - rc, 1e-9))}
    return out


def query_seed_envs(ds, env):
    """Reproduce _b12_build's exact query list & seeds (dense top-5). Golds used ONLY to reproduce row order."""
    BASE = f"data/ukb_storage/{ds}/gte_qwen/"
    docs = [nd for nd in load_nodes(f"data/processed/master_nodes_{ds}.json") if nd.metadata.get("type") != "question"]
    id2row = {nd.node_id: i for i, nd in enumerate(docs)}
    Q = json.load(open(BASE + "query_ids_all.json"))
    golds = Q["golds"]; hops = Q["hops"]; val_rows = list(Q["split_indices"]["val"])
    np.random.seed(0)                                   # identical RNG replay to _b12_build
    if ds == "metaqa":
        buckets = {1: [], 2: [], 3: []}
        for qr in val_rows:
            h = hops[qr]; buckets[int(h) if h is not None else 1].append(qr)
        per = EVAL_CAP[ds] // 3; sel = []
        for h in (1, 2, 3):
            b = buckets[h]; np.random.shuffle(b); sel += b[:per]
        val_rows = sorted(sel)
    else:
        np.random.shuffle(val_rows); val_rows = sorted(val_rows[:EVAL_CAP[ds]])
    dense_top = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
    qenv = []
    for qr in val_rows:
        if not any(id2row.get(g) is not None for g in golds[qr]):
            continue                                    # same skip as the builder (row-order alignment only)
        seeds = np.asarray(dense_top[qr][:SEED_K], np.int64)
        es = [int(env[s]) for s in seeds if 0 <= s < len(env)]
        qenv.append(int(np.bincount(es).argmax()) if es else -1)
    return np.array(qenv, np.int32)


def main():
    rng = np.random.default_rng(0)
    res = {"CONSTRUCTION": {"method": "deterministic graph-Voronoi (multi-source BFS) regions; descriptors use FULL adjacency and "
                                      "FULL-corpus semantic neighbourhood (regions partition the sample, not "
                                      "the graph -- no edge is cut)",
                            "uses_labels": "NO (U_E(R) uses TRAIN labels afterward as supervision only)",
                            "uses_target_queries": "NO for z_E", "encoder_passes": 0,
                            "query_to_env": "seeds = dense_top200[:5] (label-free, = _b12_build definition)"}}
    ENV = {}
    for ds in DSES:
        log(f"--- {ds}")
        docs = [nd for nd in load_nodes(f"data/processed/master_nodes_{ds}.json")
                if nd.metadata.get("type") != "question"]
        id2row = {nd.node_id: i for i, nd in enumerate(docs)}; n = len(docs)
        adj, deg, src = build_adj(ds, docs, id2row); del docs
        env, ne = graph_voronoi_partition(adj, deg, n, NENV)
        log(f"    {ne} raw regions (target {NENV})")
        zs = env_descriptors(ds, env, ne, adj, deg, n, rng)
        qenv = query_seed_envs(ds, env)
        d = load(ds); UR, _ = utilities(d)
        assert len(qenv) == len(UR), f"query alignment {len(qenv)} vs {len(UR)}"
        rows = []
        for e, z in zs.items():
            m = qenv == e
            if m.sum() < MIN_Q:
                continue
            u = UR[m]
            netR = {str(R_VALS[i]): float(u[:, i].sum()) for i in range(4)}
            bi = int(np.argmax([u[:, i].sum() for i in range(4)]))
            rows.append({**z, "env": int(e), "n_queries": int(m.sum()),
                         "NET_by_R": netR, "best_R": R_VALS[bi],
                         "NET_per_query_at_best_R": float(u[:, bi].sum() / m.sum()),
                         "NET_per_query_at_R16": float(u[:, 3].sum() / m.sum())})
        ENV[ds] = rows
        log(f"    usable environments: {len(rows)} / {len(zs)}  (>= {MIN_Q} queries)")
    res["ENVIRONMENTS"] = ENV

    # ---------------- feasibility: between- vs within-corpus variance
    FEATS = ["log_mean_degree", "neighbor_jaccard_redundancy", "StructSemOverlap@10", "StructSemOverlap@50",
             "edge_semantic_lift", "hop2_over_hop1"]
    feas = {}
    for f in FEATS:
        gm, allv, wv = [], [], []
        for ds in DSES:
            v = np.array([r[f] for r in ENV[ds]])
            if len(v) == 0:
                continue
            gm.append(v.mean()); allv.append(v); wv.append(v.std())
        allc = np.concatenate(allv)
        between = float(np.std(gm)); within = float(np.mean(wv))
        feas[f] = {"between_corpus_std": round(between, 5), "mean_within_corpus_std": round(within, 5),
                   "within_over_between": round(within / max(between, 1e-9), 4),
                   "per_corpus_mean": {ds: round(float(np.mean([r[f] for r in ENV[ds]])), 5) for ds in DSES},
                   "per_corpus_std": {ds: round(float(np.std([r[f] for r in ENV[ds]])), 5) for ds in DSES}}
    res["FEASIBILITY_VARIANCE"] = feas

    # utility variance too
    uy = {ds: np.array([r["NET_per_query_at_R16"] for r in ENV[ds]]) for ds in DSES}
    res["FEASIBILITY_UTILITY"] = {ds: {"n_env": len(uy[ds]), "mean": round(float(uy[ds].mean()), 5) if len(uy[ds]) else None,
                                       "std": round(float(uy[ds].std()), 5) if len(uy[ds]) else None,
                                       "min": round(float(uy[ds].min()), 5) if len(uy[ds]) else None,
                                       "max": round(float(uy[ds].max()), 5) if len(uy[ds]) else None} for ds in DSES}

    # ---------------- STEP 4: correlation / monotonicity, per corpus and leave-one-corpus-out sign stability
    def spear(x, y):
        if len(x) < 6 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return None
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])
    st4 = {}
    for f in FEATS:
        per_ds, signs = {}, []
        for ds in DSES:
            x = np.array([r[f] for r in ENV[ds]]); y = uy[ds]
            s = spear(x, y); per_ds[ds] = None if s is None else round(s, 4)
            if s is not None and abs(s) > 0.15:
                signs.append(int(np.sign(s)))
        xa = np.concatenate([[r[f] for r in ENV[ds]] for ds in DSES])
        ya = np.concatenate([uy[ds] for ds in DSES])
        st4[f] = {"within_corpus_spearman": per_ds, "pooled_spearman": round(spear(xa, ya) or 0.0, 4),
                  "sign_stable_within": bool(len(signs) >= 2 and len(set(signs)) == 1),
                  "n_corpora_with_signal": len(signs)}
    res["STEP4_MONOTONICITY"] = st4

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("\n=== ENVIRONMENT FEASIBILITY: within- vs between-corpus variance ===")
    print(f"  {'descriptor':32s} {'between':>10s} {'within':>10s} {'w/b':>7s}")
    for f in FEATS:
        v = feas[f]
        print(f"  {f:32s} {v['between_corpus_std']:10.5f} {v['mean_within_corpus_std']:10.5f} {v['within_over_between']:7.3f}")
    print("\n=== per-corpus environment utility (NET per query at R=16) ===")
    for ds in DSES:
        u = res["FEASIBILITY_UTILITY"][ds]
        print(f"  {ds:14s} n_env {u['n_env']:3d}  mean {u['mean']}  std {u['std']}  range [{u['min']}, {u['max']}]")
    print("\n=== STEP 4: does the descriptor track reserve utility ACROSS environments? ===")
    print(f"  {'descriptor':32s}" + "".join(f"{d[:9]:>11s}" for d in DSES) + f"{'pooled':>9s} {'stable':>8s}")
    for f in FEATS:
        v = st4[f]
        row = f"  {f:32s}"
        for ds in DSES:
            s = v["within_corpus_spearman"][ds]
            row += f"{'None':>11s}" if s is None else f"{s:11.3f}"
        print(row + f"{v['pooled_spearman']:9.3f} {str(v['sign_stable_within']):>8s}")
    print("\n[b19b] wrote", OUT)


if __name__ == "__main__":
    main()
