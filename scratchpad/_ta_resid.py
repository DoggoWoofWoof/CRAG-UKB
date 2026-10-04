"""G2 ARCHITECTURE DECISION EXPERIMENT -- L1 COVERAGE ONLY (parameter-free, no L2, no TEST).

COMPETITIVE STRUCTURAL VOTING  vs  ADDITIVE STRUCTURAL CAPACITY.

Arms (BASE P50 is NEVER evicted in any additive arm, so coverage is monotone non-decreasing):

  A  BASE                       canonical Dense+SPLADE -> partition voting -> RRF -> P50 -> hard union
  B  STRUCT_NODE_ADD_M          BASE + top-M structural nodes outside BASE          M  in {8,16,32,64}
  C  STRUCT_PART_ADD_Ps         BASE + top-Ps structurally supported OUT-OF-P50     Ps in {1,2,4,8}
                                partitions (node rank -> 1/(K0+j) -> summed per partition)
  D  PREPARTITION_TRACK_A       read from the already-measured _g2_ta_{ds}.json (NOT re-run)

Matched-scope controls (STEP 4):
  RANDOM_NODE_ADD_n             n random eligible out-of-scope nodes, 5 deterministic seeds
  RETRIEVAL_NODE_ADD_n          n next-best retrieval nodes outside BASE
                                  dense      : EXACT full-corpus cosine continuation (unbounded depth)
                                  splade200  : cached splade_top200 continuation (depth-limited)
                                  rrf200     : canonical node_rrf(dense200, splade200) continuation
  RANDOM_PART_ADD_Ps            Ps random out-of-P50 partitions, 5 deterministic seeds
  RETRIEVAL_PART_ADD_Ps         the next Ps partitions of the canonical fused ranking (positions 50..50+Ps)
  *_matchedPART_Ps              node arms re-run at the per-query node count STRUCT_PART_Ps actually added

ZERO learned parameters. No gold in seeding/discovery/selection/ranking. No dataset-identity branch.
No benchmark entity annotations. No TEST split. No encoder pass.

  python scratchpad/_ta_resid.py <dataset>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
M_BUDGETS = [8, 16, 32, 64]
PS_BUDGETS = [1, 2, 4, 8]
BEAM_MAIN = 64            # traversal beam = the largest node budget under test (frozen for all corpora)
BEAM_DEEP = 256           # frozen Track-A contract M_MAX, depth check on the primary seed source only
RAND_REPS = 5
EVAL_CAP = int(os.environ.get("TA_CAP", "2000"))
EMB = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"results/GENERALIZATION/_g2_resid_{DS}.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
np.random.seed(0)


def mcnemar(cur, base):
    from math import comb
    win = int(((cur == 1) & (base == 0)).sum()); los = int(((cur == 0) & (base == 1)).sum())
    n2 = win + los
    p = 1.0 if n2 == 0 else min(1.0, 2.0 * sum(comb(n2, i) for i in range(min(win, los) + 1)) / (2.0 ** n2))
    return {"gained": win, "lost": los, "net": win - los, "mcnemar_p": round(p, 5),
            "sig_p<0.05": bool(p < 0.05)}


def main():
    log(f"=== L1 ARCHITECTURE DECISION {DS}  M={M_BUDGETS} Ps={PS_BUDGETS} beam={BEAM_MAIN} ===")
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
    ndocs = len(hard)

    j = json.load(open(EMB + "query_ids_all.json"))
    golds, hops, si = j["golds"], j.get("hops", [None] * len(j["ids"])), j["split_indices"]
    rows = si["val"] if si.get("val") else si["all"]
    if DS == "metaqa":            # identical dev protocol to _ta_run.py (seed 0, hop-balanced sample)
        buck = {1: [], 2: [], 3: []}
        for r in rows:
            if hops[r] in buck:
                buck[hops[r]].append(r)
        per = EVAL_CAP // 3; sel = []
        for h in (1, 2, 3):
            b = list(buck[h]); np.random.shuffle(b); sel += b[:per]
        rows = sorted(sel)
    else:
        rows = list(rows); np.random.shuffle(rows); rows = sorted(rows[:EVAL_CAP])

    dense_all = np.load(EMB + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(EMB + "splade_top200_all.npy", mmap_mode="r")
    qall = np.load(EMB + "queries_all.npy", mmap_mode="r")
    nodes = np.load(EMB + "nodes.npy", mmap_mode="r")

    keep, gold_rows = [], []
    for r in rows:
        g = [id2row[x] for x in golds[r] if x in id2row]
        if g:
            keep.append(r); gold_rows.append(g)
    rows = keep; nq = len(rows)
    log(f"dev queries={nq} docs={ndocs} npart={npart}")

    d200 = np.stack([np.asarray(dense_all[i]) for i in rows]).astype(np.int64)
    s200 = np.stack([np.asarray(splade_all[i]) for i in rows]).astype(np.int64)
    dK, sK = d200[:, :TA.K_LOCK], s200[:, :TA.K_LOCK]

    log("normalizing node embeddings ...")
    Xn = np.array(nodes, np.float32)
    for a in range(0, len(Xn), 100000):
        b = min(a + 100000, len(Xn)); Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    Qm = np.stack([np.asarray(qall[i], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)

    # ============================================ STEP 0 : authoritative BASE + parity gate
    t = time.time()
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    t_base = time.time() - t
    base_sel = [set(int(x) for x in base_rank[qi][:TA.P_MAIN]) for qi in range(nq)]
    base_scope_size = np.array([int(part_sizes[sorted(s)].sum()) for s in base_sel], np.int64)

    ind_all_base = np.zeros(nq, np.int8); ind_any_base = np.zeros(nq, np.int8)
    for qi in range(nq):
        c = [int(hard[g]) in base_sel[qi] for g in gold_rows[qi]]
        ind_all_base[qi] = int(all(c)); ind_any_base[qi] = int(any(c))
    BASE_ALL = float(ind_all_base.mean()); BASE_ANY = float(ind_any_base.mean())
    log(f"BASE_ALL_P50={BASE_ALL:.4f} BASE_ANY_P50={BASE_ANY:.4f} "
        f"BASE_SCOPE_NODES={base_scope_size.mean():.1f} ({t_base:.1f}s)")

    parity = {"status": "NO_REFERENCE"}
    rp = f"results/GENERALIZATION/_g2_ta_{DS}.json"
    ref = None
    if os.path.exists(rp):
        ref = json.load(open(rp))["CONFIGS"]["BASE"]
        if ref["n"] != nq:                       # capped smoke run -- not comparable, not a failure
            parity = {"status": "NO_REFERENCE_AT_THIS_CAP", "ref_n": ref["n"], "this_n": nq}
            ref = None
    if ref is not None and parity["status"] == "NO_REFERENCE":
        ok = (abs(ref["ALL@P50"] - round(BASE_ALL, 4)) < 1e-9
              and abs(ref["ANY@P50"] - round(BASE_ANY, 4)) < 1e-9
              and abs(ref["hard_union_nodes_mean"] - round(float(base_scope_size.mean()), 1)) < 1e-6)
        parity = {"status": "EXACT" if ok else "MISMATCH", "ref_ALL": ref["ALL@P50"],
                  "ref_ANY": ref["ANY@P50"], "ref_n": ref["n"], "ref_scope": ref["hard_union_nodes_mean"]}
    log(f"BASE_PARTITION_PARITY = {parity['status']}")
    if parity["status"] == "MISMATCH":
        json.dump({"dataset": DS, "BASE_PARTITION_PARITY": parity, "ABORTED_PER_STEP0": True},
                  open(OUT, "w"), indent=1)
        log("PARITY BROKEN -- stopping for this corpus per STEP 0"); return

    # ============================================ STEP 1 : universal retrieval seeds
    fused200 = [TA.node_rrf(d200[qi], s200[qi]) for qi in range(nq)]
    SEEDS = {"RRF":    lambda qi: fused200[qi][:TA.SEED_K],
             "DENSE":  lambda qi: [int(x) for x in dK[qi][:TA.SEED_K]],
             "SPLADE": lambda qi: [int(x) for x in sK[qi][:TA.SEED_K]]}

    # ============================================ STEP 2 : one parameter-free structural ranking
    def structural_residual(seed_fn, beam, tag):
        """ordered by s_dir; every node already inside the BASE P50 hard union is REMOVED."""
        out = []; esc = 0; nraw = 0; hopc = {1: 0, 2: 0, 3: 0}; t = time.time()
        for qi in range(nq):
            sd = seed_fn(qi)
            r_q = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
            a, vm, e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, beam)
            esc += e; nraw += len(a)
            res = [v for v in a if int(hard[v]) not in base_sel[qi]]
            for v in res:
                h = vm.get(v, [0])[0]
                if h in hopc:
                    hopc[h] += 1
            out.append(res)
        tot = max(sum(len(x) for x in out), 1)
        c = {"tag": tag, "beam": beam, "edges_traversed": int(esc), "sec": round(time.time() - t, 1),
             "structural_nodes_evaluated": int(nraw),
             "raw_expanded_per_query": round(nraw / nq, 1),
             "residual_available_per_query": round(sum(len(x) for x in out) / nq, 1),
             "frac_raw_already_in_BASE_scope": round(1.0 - tot / max(nraw, 1), 4),
             "residual_hop_frac": {str(h): round(hopc[h] / tot, 4) for h in (1, 2, 3)},
             "sec_vs_BASE": round((time.time() - t) / max(t_base, 1e-9), 1)}
        log(f"  [{tag}] avail/q={c['residual_available_per_query']:.1f} raw/q={c['raw_expanded_per_query']:.1f} "
            f"dropped_in_scope={c['frac_raw_already_in_BASE_scope']:.3f} edges={esc:,} {c['sec']}s")
        return out, c

    # ============================================ STEP 3C : parameter-free partition aggregation
    def partition_residual(resid):
        """node rank j -> fixed RRF contribution 1/(K0+j), summed into that node's canonical C
        partition; out-of-P50 partitions ranked by the summed contribution (stable ties).
        No learning, no tuning, no gold. Returns per-query ordered out-of-P50 partition list."""
        out = []
        for qi in range(nq):
            acc = {}
            for jj, v in enumerate(resid[qi]):
                p = int(hard[v])
                if p < 0 or p in base_sel[qi]:
                    continue
                acc[p] = acc.get(p, 0.0) + 1.0 / (TA.K0 + jj)
            out.append([p for p, _ in sorted(acc.items(), key=lambda kv: -kv[1])])
        return out

    # ============================================ STEP 3C control : retrieval partition continuation
    retr_part = [[int(x) for x in base_rank[qi][TA.P_MAIN:TA.P_MAIN + max(PS_BUDGETS)]] for qi in range(nq)]

    # ============================================ STEP 4 : retrieval node continuations
    log("exact full-corpus dense continuation ...")
    t = time.time(); dense_cont = []; TOPC = max(2000, 20 * max(M_BUDGETS)); dpar = 0
    for a in range(0, nq, 200):
        b = min(a + 200, nq)
        S = Qm[a:b] @ Xn.T
        kk = min(TOPC, S.shape[1] - 1)
        part = np.argpartition(-S, kk, axis=1)[:, :kk]
        ordv = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1,
                                                   kind="stable"), 1)
        for k in range(b - a):
            qi = a + k
            dpar += int(np.array_equal(ordv[k][:TA.K_LOCK], d200[qi][:TA.K_LOCK]))
            dense_cont.append([int(v) for v in ordv[k] if int(hard[v]) not in base_sel[qi]])
        del S
    dense_cost = {"tag": "RETRIEVAL_NODE_dense", "sec": round(time.time() - t, 1),
                  "exact_top100_equals_cached_dense_top100": f"{dpar}/{nq}"}
    log(f"  dense continuation {dense_cost['sec']}s  exact-vs-cached top100: {dpar}/{nq}")
    splade_cont = [[int(v) for v in s200[qi] if int(hard[v]) not in base_sel[qi]] for qi in range(nq)]
    rrf_cont = [[int(v) for v in fused200[qi] if int(hard[v]) not in base_sel[qi]] for qi in range(nq)]

    def random_nodes(counts, seed):
        r = np.random.default_rng(seed); out = []
        for qi in range(nq):
            need = int(counts[qi]); got = []
            while len(got) < need:
                for v in r.integers(0, ndocs, size=max(4 * need, 32)):
                    v = int(v)
                    if hard[v] >= 0 and int(hard[v]) not in base_sel[qi]:
                        got.append(v)
                        if len(got) == need:
                            break
            out.append(got)
        return out

    def random_partitions(Ps, seed):
        r = np.random.default_rng(seed); out = []
        allp = np.arange(npart)
        for qi in range(nq):
            elig = allp[~np.isin(allp, list(base_sel[qi]))]
            out.append([int(x) for x in r.choice(elig, size=min(Ps, len(elig)), replace=False)])
        return out

    # ============================================ STEP 6 : coverage metrics
    def score(added_nodes, added_parts, name):
        anyc = allc = 0; newly = 0; worse = 0; grec = 0
        ind_all = np.zeros(nq, np.int8)
        nadd = np.zeros(nq, np.int64); padd = np.zeros(nq, np.int64)
        hop_n = {}; hop_all = {}; hop_any = {}
        for qi in range(nq):
            an = set(added_nodes[qi]) if added_nodes is not None else set()
            ap = set(added_parts[qi]) if added_parts is not None else set()
            ap -= base_sel[qi]
            nadd[qi] = len(an) + (int(part_sizes[sorted(ap)].sum()) if ap else 0)
            padd[qi] = len(ap)
            cov, bcov = [], []
            for g in gold_rows[qi]:
                p = int(hard[g]); b0 = p in base_sel[qi]
                bcov.append(b0); cov.append(b0 or (p in ap) or (g in an))
            grec += sum(1 for a_, b_ in zip(cov, bcov) if a_ and not b_)
            isall = all(cov); isany = any(cov)
            allc += isall; anyc += isany; ind_all[qi] = int(isall)
            newly += int(isall and not ind_all_base[qi])
            worse += int((not isall) and ind_all_base[qi])
            h = hops[rows[qi]]
            if h is not None:
                k = str(h); hop_n[k] = hop_n.get(k, 0) + 1
                hop_all[k] = hop_all.get(k, 0) + int(isall); hop_any[k] = hop_any.get(k, 0) + int(isany)
        sc = base_scope_size + nadd
        d = {"n": nq, "ANY": round(anyc / nq, 4), "ALL": round(allc / nq, 4),
             "dALL_vs_BASE": round(allc / nq - BASE_ALL, 4),
             "queries_newly_ALL_covered": int(newly),
             "newly_recovered_gold_occurrences": int(grec),
             "queries_worsened": int(worse),
             "added_nodes_mean": round(float(nadd.mean()), 2),
             "added_partitions_mean": round(float(padd.mean()), 2),
             "final_scope_nodes_mean": round(float(sc.mean()), 1),
             "scope_growth_pct": round(100.0 * float(nadd.mean()) / float(base_scope_size.mean()), 3),
             "paired_vs_BASE_ALL": mcnemar(ind_all, ind_all_base)}
        if hop_n:
            d["per_hop"] = {h: {"n": hop_n[h], "ANY": round(hop_any[h] / hop_n[h], 4),
                                "ALL": round(hop_all[h] / hop_n[h], 4)} for h in sorted(hop_n)}
        assert worse == 0, f"{name}: additive arm made {worse} queries WORSE -- implementation bug"
        return d

    def score_rand(fn, tag, reps=RAND_REPS):
        acc = [fn(s) for s in range(reps)]
        out = dict(acc[0])
        for k in ("ALL", "ANY", "dALL_vs_BASE", "added_nodes_mean", "final_scope_nodes_mean",
                  "scope_growth_pct", "added_partitions_mean"):
            out[k] = round(float(np.mean([a[k] for a in acc])), 4)
        out["ALL_std"] = round(float(np.std([a["ALL"] for a in acc])), 5)
        out["queries_newly_ALL_covered"] = round(float(np.mean([a["queries_newly_ALL_covered"] for a in acc])), 2)
        out["newly_recovered_gold_occurrences"] = round(
            float(np.mean([a["newly_recovered_gold_occurrences"] for a in acc])), 2)
        out["reps"] = reps; out["per_rep_ALL"] = [a["ALL"] for a in acc]
        if "per_hop" in acc[0]:
            out["per_hop"] = {h: {"n": acc[0]["per_hop"][h]["n"],
                                  "ANY": round(float(np.mean([a["per_hop"][h]["ANY"] for a in acc])), 4),
                                  "ALL": round(float(np.mean([a["per_hop"][h]["ALL"] for a in acc])), 4)}
                              for h in acc[0]["per_hop"]}
        return out

    RES = {"BASE": score(None, None, "BASE")}
    COST = {"BASE": {"sec": round(t_base, 1), "edges_traversed": 0, "structural_nodes_evaluated": 0}}

    # ---- structural residuals
    resid = {}
    for sn, sf in SEEDS.items():
        resid[sn], COST[f"STRUCT_{sn}_beam{BEAM_MAIN}"] = structural_residual(sf, BEAM_MAIN,
                                                                              f"STRUCT_{sn}_beam{BEAM_MAIN}")
    resid_deep, COST[f"STRUCT_RRF_beam{BEAM_DEEP}"] = structural_residual(SEEDS["RRF"], BEAM_DEEP,
                                                                          f"STRUCT_RRF_beam{BEAM_DEEP}")
    presid = {sn: partition_residual(resid[sn]) for sn in SEEDS}
    presid_deep = partition_residual(resid_deep)

    # ---- B : node arms + matched node controls
    for M in M_BUDGETS:
        cnt = np.full(nq, M, np.int64)
        RES[f"RANDOM_NODE_ADD_M{M}"] = score_rand(
            lambda s, M=M, c=cnt: score(random_nodes(c, s), None, f"rand_M{M}_s{s}"), f"rand_M{M}")
        for nm, src in (("dense", dense_cont), ("splade200", splade_cont), ("rrf200", rrf_cont)):
            RES[f"RETRIEVAL_NODE_ADD_{nm}_M{M}"] = score([x[:M] for x in src], None, f"retr_{nm}_M{M}")
        for sn in SEEDS:
            RES[f"STRUCT_NODE_ADD_{sn}_M{M}"] = score([x[:M] for x in resid[sn]], None, f"struct_{sn}_M{M}")
        RES[f"STRUCT_NODE_ADD_RRFdeep_M{M}"] = score([x[:M] for x in resid_deep], None, f"deep_M{M}")

    # ---- C : partition arms + matched partition/node controls
    for Ps in PS_BUDGETS:
        sp = [x[:Ps] for x in presid["RRF"]]
        RES[f"STRUCT_PART_ADD_RRF_Ps{Ps}"] = score(None, sp, f"spart_RRF_Ps{Ps}")
        for sn in ("DENSE", "SPLADE"):
            RES[f"STRUCT_PART_ADD_{sn}_Ps{Ps}"] = score(None, [x[:Ps] for x in presid[sn]], f"spart_{sn}")
        RES[f"STRUCT_PART_ADD_RRFdeep_Ps{Ps}"] = score(None, [x[:Ps] for x in presid_deep], f"spartdeep")
        RES[f"RETRIEVAL_PART_ADD_Ps{Ps}"] = score(None, [x[:Ps] for x in retr_part], f"rpart_Ps{Ps}")
        RES[f"RANDOM_PART_ADD_Ps{Ps}"] = score_rand(
            lambda s, Ps=Ps: score(None, random_partitions(Ps, s), f"randpart_Ps{Ps}_s{s}"), f"randpart{Ps}")
        # node-count-matched controls for the partition arm
        cnt = np.array([int(part_sizes[[p for p in sp[qi] if p not in base_sel[qi]]].sum())
                        if len(sp[qi]) else 0 for qi in range(nq)], np.int64)
        RES[f"RANDOM_NODE_matchedPART_Ps{Ps}"] = score_rand(
            lambda s, c=cnt: score(random_nodes(c, s), None, "randnode_matched"), "randnode_matched")
        RES[f"RETRIEVAL_NODE_dense_matchedPART_Ps{Ps}"] = score(
            [dense_cont[qi][:int(cnt[qi])] for qi in range(nq)], None, "retrnode_matched")
        RES[f"STRUCT_NODE_matchedPART_Ps{Ps}"] = score(
            [resid["RRF"][qi][:int(cnt[qi])] for qi in range(nq)], None, "structnode_matched")

    # ============================================ STEP 7 : structure-specific gain
    STEP7 = {}
    for M in M_BUDGETS:
        s = RES[f"STRUCT_NODE_ADD_RRF_M{M}"]["dALL_vs_BASE"]
        rn = RES[f"RANDOM_NODE_ADD_M{M}"]["dALL_vs_BASE"]
        chans = ("dense", "splade200", "rrf200")
        bc = max(chans, key=lambda n: RES[f"RETRIEVAL_NODE_ADD_{n}_M{M}"]["dALL_vs_BASE"])
        rt = RES[f"RETRIEVAL_NODE_ADD_{bc}_M{M}"]["dALL_vs_BASE"]
        STEP7[f"NODE_M{M}"] = {"dALL_STRUCT": s, "dALL_RANDOM": rn, "dALL_RETRIEVAL_best": rt,
                               "best_retrieval_channel": bc,
                               "STRUCT_NODE_GAIN_OVER_RANDOM": round(s - rn, 4),
                               "STRUCT_NODE_GAIN_OVER_RETRIEVAL": round(s - rt, 4)}
    for Ps in PS_BUDGETS:
        s = RES[f"STRUCT_PART_ADD_RRF_Ps{Ps}"]["dALL_vs_BASE"]
        rn = RES[f"RANDOM_PART_ADD_Ps{Ps}"]["dALL_vs_BASE"]
        rt = RES[f"RETRIEVAL_PART_ADD_Ps{Ps}"]["dALL_vs_BASE"]
        STEP7[f"PART_Ps{Ps}"] = {"dALL_STRUCT": s, "dALL_RANDOM": rn, "dALL_RETRIEVAL": rt,
                                 "STRUCT_PART_GAIN_OVER_RANDOM": round(s - rn, 4),
                                 "STRUCT_PART_GAIN_OVER_RETRIEVAL": round(s - rt, 4),
                                 "vs_node_at_matched_nodes": round(
                                     s - RES[f"STRUCT_NODE_matchedPART_Ps{Ps}"]["dALL_vs_BASE"], 4)}

    # ============================================ D : pre-partition reference (NOT re-run)
    PREP = {}
    if os.path.exists(rp):
        C = json.load(open(rp))["CONFIGS"]
        for k, v in C.items():
            if k == "BASE":
                continue
            PREP[k] = {"ALL@P50": v["ALL@P50"], "ANY@P50": v["ANY@P50"],
                       "net": v["paired_vs_BASE_ALL"]["net"], "p": v["paired_vs_BASE_ALL"]["mcnemar_p"],
                       "scope": v["hard_union_nodes_mean"],
                       "per_hop": v.get("per_hop")}
        gated = [k for k in PREP if not k.startswith("A1b")]
        PREP["_BEST_PARITY_GATED"] = max(gated, key=lambda k: PREP[k]["ALL@P50"])
        PREP["_BEST_ANY"] = max((k for k in PREP if not k.startswith("_")),
                                key=lambda k: PREP[k]["ALL@P50"])

    out = {"dataset": DS, "n_dev_queries": nq, "n_docs": int(ndocs), "npart": int(npart),
           "M_BUDGETS": M_BUDGETS, "PS_BUDGETS": PS_BUDGETS,
           "CONTRACT": {"K0": TA.K0, "K": TA.K_LOCK, "P_MAIN": TA.P_MAIN, "SEED_K": TA.SEED_K,
                        "BEAM_MAIN": BEAM_MAIN, "BEAM_DEEP": BEAM_DEEP, "MAX_HOPS": TA.MAX_HOPS,
                        "DEG_CAP": TA.DEG_CAP, "MAX_EDGES_SCORED": TA.MAX_EDGES_SCORED,
                        "RAND_REPS": RAND_REPS, "learned_parameters": 0, "uses_gold_in_scope": False,
                        "uses_dataset_id": False, "uses_entity_annotations": False,
                        "additive_only_BASE_never_evicted": True,
                        "partition_aggregation": "node rank j -> 1/(K0+j), summed into hard[node]; "
                                                 "out-of-P50 partitions ranked by the sum (stable ties)"},
           "BASE_PARTITION_PARITY": parity,
           "BASE_ANY_P50": round(BASE_ANY, 4), "BASE_ALL_P50": round(BASE_ALL, 4),
           "BASE_SCOPE_NODES": round(float(base_scope_size.mean()), 1),
           "mean_partition_size": round(float(part_sizes.mean()), 1),
           "RESULTS": RES, "COST": COST, "STEP7_structure_specific_gain": STEP7,
           "PREPARTITION_REFERENCE": PREP}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    # ------------------------------------------------------------------ console
    print(f"\n=== {DS}  n={nq}  BASE_ALL_P50={BASE_ALL:.4f}  BASE_ANY={BASE_ANY:.4f}  "
          f"scope={base_scope_size.mean():.0f}  parity={parity['status']} ===")
    print("  --- NODE arms (added nodes = M) ---")
    print(f"  {'arm':36s} " + " ".join(f"{'M'+str(M):>18s}" for M in M_BUDGETS))
    for f in ["RANDOM_NODE_ADD", "RETRIEVAL_NODE_ADD_dense", "RETRIEVAL_NODE_ADD_splade200",
              "RETRIEVAL_NODE_ADD_rrf200", "STRUCT_NODE_ADD_RRF", "STRUCT_NODE_ADD_DENSE",
              "STRUCT_NODE_ADD_SPLADE", "STRUCT_NODE_ADD_RRFdeep"]:
        cells = [f"{RES[f'{f}_M{M}']['ALL']:.4f}({RES[f'{f}_M{M}']['dALL_vs_BASE']:+.4f})"
                 for M in M_BUDGETS]
        print(f"  {f:36s} " + " ".join(f"{c:>18s}" for c in cells))
    print("  --- PARTITION arms (added nodes vary) ---")
    print(f"  {'arm':36s} " + " ".join(f"{'Ps'+str(P):>18s}" for P in PS_BUDGETS))
    for f in ["RANDOM_PART_ADD", "RETRIEVAL_PART_ADD", "STRUCT_PART_ADD_RRF", "STRUCT_PART_ADD_RRFdeep",
              "RANDOM_NODE_matchedPART", "RETRIEVAL_NODE_dense_matchedPART", "STRUCT_NODE_matchedPART"]:
        cells = [f"{RES[f'{f}_Ps{P}']['ALL']:.4f}({RES[f'{f}_Ps{P}']['dALL_vs_BASE']:+.4f})"
                 for P in PS_BUDGETS]
        print(f"  {f:36s} " + " ".join(f"{c:>18s}" for c in cells))
    print("  added nodes: " + "  ".join(
        f"Ps{P}={RES[f'STRUCT_PART_ADD_RRF_Ps{P}']['added_nodes_mean']:.0f}" for P in PS_BUDGETS))
    print("  --- STEP 7 structure-specific gain ---")
    for k, v in STEP7.items():
        g = [(a, b) for a, b in v.items() if a.startswith("STRUCT")]
        print(f"    {k:12s} " + "  ".join(f"{a.split('GAIN_OVER_')[-1]} {b:+.4f}" for a, b in g))
    if "per_hop" in RES["BASE"]:
        print("  --- MetaQA per-hop ALL ---")
        keyset = [("BASE", "BASE")] + \
                 [(f"RANDOM_NODE_ADD_M32", "RANDOM_NODE +32"),
                  (f"RETRIEVAL_NODE_ADD_dense_M32", "RETRIEVAL_NODE +32"),
                  (f"STRUCT_NODE_ADD_RRF_M32", "STRUCT_NODE +32"),
                  (f"RANDOM_PART_ADD_Ps4", "RANDOM_PART +4"),
                  (f"RETRIEVAL_PART_ADD_Ps4", "RETRIEVAL_PART +4"),
                  (f"STRUCT_PART_ADD_RRF_Ps4", "STRUCT_PART +4")]
        print(f"    {'arm':24s} {'hop1':>8s} {'hop2':>8s} {'hop3':>8s}")
        for k, lab in keyset:
            ph = RES[k]["per_hop"]
            print(f"    {lab:24s} " + " ".join(f"{ph[h]['ALL']:8.4f}" for h in ("1", "2", "3")))
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
