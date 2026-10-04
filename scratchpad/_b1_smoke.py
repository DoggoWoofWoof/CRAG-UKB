"""G2 · B1 — universal POOL-ADMISSION LODO smoke (MECHANICS validation, small).

Scope: original P50 candidate universe (NO geometry expansion in this smoke -> RECOVERED_GOLD_ADMISSION is
N/A here; geometry + S2 directional features are deferred to the full run). Learned admission selects TOP_POOL=50
from the ~5000 P50 candidates. Compares, under TRUE 3-way LODO over {2wiki, musique, squad} (val, local):
  BASE_FUSED top50 (reference)  vs  S0-linear  vs  S1-linear.
Validates: feature alignment (identical columns across datasets), NO leakage (target excluded from train;
normalization stats from TRAIN only; no gold/label/dataset-id in features), normalization, candidate budget,
pool metrics. Linear scorer only (logistic regression); MLP deferred per the directive.

Features implement G2_UNIVERSAL_FEATURE_CONTRACT S0 + a bounded S1 (PPR deferred to full run; hop via H<=2 BFS
from retrieval-seed anchors). All features are per-query bounded/relative. No dataset id. No bracket rule.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
from src.pipeline.standardizer import load_nodes

DSES = ["2wiki_clean", "musique_clean", "squad_clean"]
CORP = "data/l2_corpus"; TOP_POOL = 50; K0 = 60; QCAP = 700; SEED_K = 5; H = 2; DEG_CAP = 2000
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)

S0_COLS = ["dense_rank_pct", "splade_rank_pct", "fused_rrf_pct", "retriever_support_frac", "dense_splade_agreement"]
S1_COLS = S0_COLS + ["min_hop_from_seed_norm", "seed_connectivity_frac", "struct_support_count_norm", "degree_pct"]


def build_adj(ds):
    """row-space structural adjacency from master_nodes.neighbors (uniform interface). row == doc file order."""
    p = f"data/processed/master_nodes_{ds}.json"
    nodes = load_nodes(p if os.path.exists(p) else "data/processed/master_nodes.json")
    docs = [n for n in nodes if n.metadata.get("type") != "question"]
    id2row = {n.node_id: i for i, n in enumerate(docs)}
    adj = [None] * len(docs)
    deg = np.zeros(len(docs), np.int32)
    for i, n in enumerate(docs):
        nb = [id2row[x] for x in n.neighbors if x in id2row]
        adj[i] = nb; deg[i] = len(nb)
    return adj, deg, len(docs)


def hop_sets(seeds, adj, deg):
    """per-seed 1-hop and <=2-hop reachable sets (row ids), with a degree cap to avoid hub blowup."""
    one = []; two = []
    for s in seeds:
        h1 = set(v for v in adj[s] if deg[v] <= DEG_CAP)
        h2 = set(h1)
        for v in list(h1):
            if deg[v] <= DEG_CAP:
                h2.update(w for w in adj[v] if deg[w] <= DEG_CAP)
        one.append(h1); two.append(h2)
    return one, two


def features(ds, adj, deg, dense_top, qi_list, arrs):
    off, ci, dsc, spc, lab, qmeta = arrs
    X0 = []; X1 = []; Y = []; groups = []; meta = []
    for qi in qi_list:
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        if n < TOP_POOL + 1:
            continue
        cand = np.asarray(ci[s:e], np.int64)
        d = np.asarray(dsc[s:e], np.float64); sp = np.asarray(spc[s:e], np.float64)
        y = (np.asarray(lab[s:e]) == 1).astype(np.int8)
        # ranks -> percentiles (1 = best). argsort desc, rank position.
        def pct(v):
            r = np.empty(n, np.float64); r[np.argsort(-v, kind="stable")] = np.arange(n)
            return 1.0 - r / max(n - 1, 1)
        dpct = pct(d); sppct = pct(sp)
        drank = (n - 1) * (1 - dpct); srank = (n - 1) * (1 - sppct)
        rrf = 1.0 / (K0 + drank) + 1.0 / (K0 + srank); frrf = pct(rrf)
        support = ((drank < K0).astype(np.float64) + (srank < K0).astype(np.float64)) / 2.0
        agree = 1.0 - np.abs(dpct - sppct)
        f0 = np.stack([dpct, sppct, frrf, support, agree], 1)
        # S1 structural from retrieval-seed anchor
        ra = int(qmeta[qi]["row_all"]); seeds = [int(x) for x in np.asarray(dense_top[ra][:SEED_K])]
        one, two = hop_sets(seeds, adj, deg)
        ns = max(len(seeds), 1)
        onehop = np.zeros(n); twohop = np.zeros(n); minhopnorm = np.zeros(n)
        for j, c in enumerate(cand):
            in1 = sum(1 for hs in one if c in hs); in2 = sum(1 for hs in two if c in hs)
            onehop[j] = in1 / ns; twohop[j] = in2 / ns
            minhopnorm[j] = 1.0 if in1 > 0 else (0.5 if in2 > 0 else 0.0)   # 1/(1+hop): hop0 N/A,1->.5? use 1,.5,0
        degc = deg[cand].astype(np.float64); degpct = pct(degc)
        f1 = np.concatenate([f0, np.stack([minhopnorm, twohop, onehop, degpct], 1)], 1)
        X0.append(f0); X1.append(f1); Y.append(y); groups.append(n)
        meta.append({"qi": qi, "cand": cand, "y": y,
                     "base_top50": set(int(c) for c in cand[np.argsort(-rrf, kind="stable")[:TOP_POOL]]),
                     "n_gold": int(y.sum())})
    return (np.concatenate(X0), np.concatenate(X1), np.concatenate(Y),
            np.asarray(groups, np.int64), meta)


def load_arrays(ds):
    D = f"{CORP}/{ds}/val"
    off = np.load(f"{D}/query_offsets.npy")
    ci = np.load(f"{D}/cand_ids.npy", mmap_mode="r")
    dsc = np.load(f"{D}/dense_score.npy", mmap_mode="r")
    spc = np.load(f"{D}/splade_scope_score.npy", mmap_mode="r")
    lab = np.load(f"{D}/labels.npy", mmap_mode="r")
    qm = json.load(open(f"{D}/query_meta.json"))
    qmeta = qm if isinstance(qm, list) else [qm[k] for k in sorted(qm, key=lambda x: int(x))]
    dense_top = np.load(f"data/ukb_storage/{ds}/gte_qwen/dense_top200_all.npy", mmap_mode="r")
    return off, ci, dsc, spc, lab, qmeta, dense_top


def fit_logreg(X, y, iters=300, lr=0.5, l2=1e-3, wpos=None):
    """tiny logistic regression (numpy), class-balanced. Returns (w,b)."""
    n, d = X.shape; w = np.zeros(d); b = 0.0
    if wpos is None:
        pos = max(y.sum(), 1); wpos = (n - pos) / pos
    sw = np.where(y == 1, wpos, 1.0)
    for _ in range(iters):
        z = X @ w + b; p = 1.0 / (1.0 + np.exp(-z)); g = (p - y) * sw
        gw = X.T @ g / n + l2 * w; gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def admit_metrics(meta, score_by_q):
    """pool metrics @50 for a learned admission (score_by_q: qi -> per-cand score)."""
    ANY = REC_n = REC_d = ALLq = 0; churn = []; net = []; retain_n = retain_d = 0
    nq = 0
    for m in meta:
        qi = m["qi"]; cand = m["cand"]; y = m["y"]; ng = m["n_gold"]
        if ng == 0:
            continue
        nq += 1
        sc = score_by_q[qi]
        top = cand[np.argsort(-sc, kind="stable")[:TOP_POOL]]
        tset = set(int(c) for c in top)
        gold_rows = set(int(c) for c, yy in zip(cand, y) if yy == 1)
        gin = len(gold_rows & tset)
        ANY += int(gin >= 1); REC_n += gin; REC_d += ng; ALLq += int(gin == ng)
        base = m["base_top50"]; base_gin = len(gold_rows & base)
        net.append(gin - base_gin)
        churn.append(len(tset ^ base) / (2 * TOP_POOL))
        retain_n += len(gold_rows & base & tset); retain_d += base_gin
    return {"nq": nq, "ANY@50": round(ANY / max(nq, 1), 4), "GOLD_RECALL@50": round(REC_n / max(REC_d, 1), 4),
            "ALL@50": round(ALLq / max(nq, 1), 4), "NET_GOLD_GAIN@50_mean": round(float(np.mean(net)), 4),
            "POOL_CHURN_mean": round(float(np.mean(churn)), 4),
            "ORIGINAL_GOLD_RETENTION@50": round(retain_n / max(retain_d, 1), 4),
            "RECOVERED_GOLD_ADMISSION@50": "N/A (no geometry expansion in smoke)"}


def base_metrics(meta):
    ANY = REC_n = REC_d = ALLq = 0; nq = 0
    for m in meta:
        if m["n_gold"] == 0:
            continue
        nq += 1; cand = m["cand"]; y = m["y"]; base = m["base_top50"]
        gold_rows = set(int(c) for c, yy in zip(cand, y) if yy == 1); gin = len(gold_rows & base)
        ANY += int(gin >= 1); REC_n += gin; REC_d += m["n_gold"]; ALLq += int(gin == m["n_gold"])
    return {"nq": nq, "ANY@50": round(ANY / max(nq, 1), 4), "GOLD_RECALL@50": round(REC_n / max(REC_d, 1), 4),
            "ALL@50": round(ALLq / max(nq, 1), 4)}


def main():
    log(f"=== B1 POOL-ADMISSION LODO smoke  datasets={DSES}  TOP_POOL={TOP_POOL} QCAP={QCAP} ===")
    DATA = {}
    for ds in DSES:
        arrs = load_arrays(ds); off = arrs[0]; qmeta = arrs[5]; dense_top = arrs[6]
        adj, deg, ndoc = build_adj(ds)
        nq = len(off) - 1
        qi_list = [qi for qi in range(nq) if qmeta[qi].get("N_GOLD_IN_SCOPE", 0) >= 1][:QCAP]
        X0, X1, Y, G, meta = features(ds, adj, deg, dense_top, qi_list,
                                      (off, arrs[1], arrs[2], arrs[3], arrs[4], qmeta))
        DATA[ds] = {"X0": X0, "X1": X1, "Y": Y, "G": G, "meta": meta, "ndoc": ndoc}
        log(f"{ds}: queries={len(meta)} cands={len(Y)} golds={int(Y.sum())} ndoc={ndoc} base={base_metrics(meta)}")

    # feature-alignment assertions
    assert all(DATA[d]["X0"].shape[1] == len(S0_COLS) for d in DSES), "S0 column mismatch"
    assert all(DATA[d]["X1"].shape[1] == len(S1_COLS) for d in DSES), "S1 column mismatch"
    log(f"FEATURE_ALIGN OK: S0={S0_COLS} S1_extra={S1_COLS[len(S0_COLS):]}")

    results = {}
    for target in DSES:
        train_ds = [d for d in DSES if d != target]
        assert target not in train_ds, "LODO leakage: target in train"
        row = {"train": train_ds}
        for tag, key, cols in (("S0", "X0", S0_COLS), ("S1", "X1", S1_COLS)):
            # dataset-balanced pooled train: equal query weight -> subsample each train ds to min cand count
            parts_X = []; parts_Y = []
            counts = [len(DATA[d]["Y"]) for d in train_ds]; cap = min(counts + [200000])  # smoke subsample
            for d in train_ds:
                Xd = DATA[d][key]; Yd = DATA[d]["Y"]; idx = np.random.permutation(len(Yd))[:cap]
                parts_X.append(Xd[idx]); parts_Y.append(Yd[idx])
            Xtr = np.concatenate(parts_X); Ytr = np.concatenate(parts_Y)
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6                      # TRAIN-only normalization
            w, b = fit_logreg((Xtr - mu) / sd, Ytr)
            # eval target once (normalize with TRAIN stats)
            Xt = DATA[target][key]; meta = DATA[target]["meta"]; G = DATA[target]["G"]
            st = np.concatenate([[0], np.cumsum(G)])
            sc_all = ((Xt - mu) / sd) @ w + b
            score_by_q = {meta[i]["qi"]: sc_all[st[i]:st[i + 1]] for i in range(len(meta))}
            row[tag] = admit_metrics(meta, score_by_q)
            row[tag + "_train_norm_mu"] = [round(float(x), 3) for x in mu]
        row["BASE_FUSED"] = base_metrics(DATA[target]["meta"])
        results[target] = row
        log(f"TARGET {target}: BASE={row['BASE_FUSED']}")
        log(f"   S0={row['S0']}")
        log(f"   S1={row['S1']}")

    out = {"phase": "G2 B1 pool-admission LODO smoke (MECHANICS); P50 scope only; linear; TEST untouched",
           "datasets": DSES, "TOP_POOL": TOP_POOL, "QCAP": QCAP, "seed_k": SEED_K, "H": H,
           "S0_COLS": S0_COLS, "S1_COLS": S1_COLS,
           "leakage_checks": {"target_excluded_from_train": True, "normalization_train_only": True,
                              "no_gold_or_label_in_features": True, "no_dataset_id_feature": True},
           "deferred_to_full_run": ["geometry expansion (P50 U <=256)", "RECOVERED_GOLD_ADMISSION",
                                    "S2 directional geometry", "PPR feature", "small MLP", "metaqa/hotpot/webqsp (Modal)"],
           "RESULTS": results, "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b1_smoke.json", "w"), indent=1, default=str)
    log("B1_SMOKE_DONE -> results/GENERALIZATION/_g2_b1_smoke.json")


if __name__ == "__main__":
    main()
