"""G2 Track-A CORRECTED runner: M=0 parity gate + BASE / A1 / A2 partition-level co-scoping.

  python scratchpad/_ta_run.py <dataset> [M]

Emits results/GENERALIZATION/_g2_ta_{ds}.json. Dev (val) population only. TEST never touched.
No gold enters scope construction; gold is used only to score the resulting partitions.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
M_LIST = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else str(TA.M_MAX)).split(",")]
EVAL_CAP = int(os.environ.get("TA_CAP", "2000"))
BASE = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"results/GENERALIZATION/_g2_ta_{DS}.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
np.random.seed(0)


def main():
    log(f"=== TRACK-A PRE-PARTITION {DS}  K={TA.K_LOCK} P={TA.P_MAIN} SEED_K={TA.SEED_K} M={M_LIST} ===")
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)

    j = json.load(open(BASE + "query_ids_all.json"))
    ids, golds, hops, si = j["ids"], j["golds"], j.get("hops", [None] * len(j["ids"])), j["split_indices"]
    rows = si["val"] if si.get("val") else si["all"]
    # deterministic dev sample; metaqa balanced per hop (matches the frozen Track-A/B1.x protocol)
    if DS == "metaqa":
        buck = {1: [], 2: [], 3: []}
        for r in rows:
            h = hops[r]
            if h in buck:
                buck[h].append(r)
        per = EVAL_CAP // 3; sel = []
        for h in (1, 2, 3):
            b = list(buck[h]); np.random.shuffle(b); sel += b[:per]
        rows = sorted(sel)
    else:
        rows = list(rows); np.random.shuffle(rows); rows = sorted(rows[:EVAL_CAP])

    dense_all = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(BASE + "splade_top200_all.npy", mmap_mode="r")
    qall = np.load(BASE + "queries_all.npy", mmap_mode="r")
    nodes = np.load(BASE + "nodes.npy", mmap_mode="r")

    # keep only queries with >=1 in-corpus gold (scope-construction never sees gold)
    keep, gold_rows = [], []
    for r in rows:
        g = [id2row[x] for x in golds[r] if x in id2row]
        if g:
            keep.append(r); gold_rows.append(g)
    rows = keep
    nq = len(rows)
    log(f"dev queries={nq} (of {len(si.get('val') or si['all'])} val) npart={npart}")

    dK = np.stack([np.asarray(dense_all[i][:TA.K_LOCK]) for i in rows]).astype(np.int64)
    sK = np.stack([np.asarray(splade_all[i][:TA.K_LOCK]) for i in rows]).astype(np.int64)

    # ---------------------------------------------------------------- normalized embeddings
    log("normalizing node embeddings ...")
    Xn = np.array(nodes, np.float32)
    for a in range(0, len(Xn), 100000):
        b = min(a + 100000, len(Xn))
        Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)

    # ---------------------------------------------------------------- BASE (canonical)
    log("BASE: canonical partition voting ...")
    t = time.time()
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    base_rank = TA.rrf_partitions([PR_d, PR_s], npart)
    t_base = time.time() - t
    log(f"BASE done {t_base:.1f}s")

    # ---------------------------------------------------------------- structural expansion (once per config)
    base_sel = [set(int(x) for x in base_rank[qi][:TA.P_MAIN]) for qi in range(nq)]

    def run_expansion(seed_fn, M, tag):
        """seed_fn(qi) -> seed rows. Returns per-query added rows (s_dir order) + counters.

        MECHANISM DIAGNOSTIC: the canonical mem_idx already votes each retrieved node into its own
        partition AND its 1-hop neighbours' partitions. So a hop-1 structural discovery carries
        evidence the base router has already counted. We measure how much of the expansion is
        hop-1 (redundant with mem_idx) and how much lands in partitions BASE already selected."""
        added, esc, nstruct = [], 0, 0
        hopcnt = {1: 0, 2: 0, 3: 0}; in_base_part = 0; novel_part = 0
        t = time.time()
        for qi in range(nq):
            if M <= 0:
                added.append([]); continue
            sd = seed_fn(qi)
            r_q = TA.residual(np.asarray(qall[rows[qi]], np.float64), sd, Xn)
            a, vm, e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, M)
            added.append(a); esc += e; nstruct += len(a)
            for v in a:
                h = vm.get(v, [0])[0]
                if h in hopcnt:
                    hopcnt[h] += 1
                if int(hard[v]) in base_sel[qi]:
                    in_base_part += 1
                else:
                    novel_part += 1
        tot = max(nstruct, 1)
        return added, {"edges_scanned": int(esc), "structural_nodes": int(nstruct),
                       "sec": round(time.time() - t, 1), "tag": tag,
                       "hop_frac": {str(h): round(hopcnt[h] / tot, 4) for h in (1, 2, 3)},
                       "frac_in_partition_BASE_already_selected": round(in_base_part / tot, 4),
                       "frac_in_novel_partition": round(novel_part / tot, 4)}

    # ---------------------------------------------------------------- A1 / A2 over the M sweep
    fused_nodes = [TA.node_rrf(dK[qi], sK[qi]) for qi in range(nq)]
    dpresent = [set(int(x) for x in dK[qi]) for qi in range(nq)]
    spresent = [set(int(x) for x in sK[qi]) for qi in range(nq)]
    configs = {"BASE": (base_rank, {"sec": round(t_base, 1)})}
    for M in M_LIST:
        log(f"--- M={M} : A1 (RRF before structure) ---")
        a1_added, a1_cost = run_expansion(lambda qi: fused_nodes[qi][:TA.SEED_K], M, f"A1_M{M}")
        t = time.time()
        if M > 0 and any(len(a) for a in a1_added):
            PR_g = TA.partition_ranking(a1_added, mem, npart)
            mask_g = np.array([1.0 if len(a) else 0.0 for a in a1_added])
            a1_rank = TA.rrf_partitions([PR_d, PR_s, PR_g], npart, masks=[None, None, mask_g])
        else:
            a1_rank = TA.rrf_partitions([PR_d, PR_s], npart)
        a1_cost["vote_sec"] = round(time.time() - t, 1)
        configs[f"A1a_M{M}"] = (a1_rank, a1_cost)

        # A1b -- the LITERAL directive reading: node-level RRF fuses dense+splade FIRST, structural
        # nodes append to that single fused stream, and ONE partition vote runs over it. Free: reuses
        # a1_added (same seeds). NOTE: A1b at M=0 is NOT expected to equal BASE -- moving RRF ahead of
        # voting is itself an architectural change -- so its M=0 row is the control that prices it.
        fpresent = [set(fused_nodes[qi][:TA.K_LOCK]) for qi in range(nq)]
        flist = [list(fused_nodes[qi][:TA.K_LOCK]) +
                 [v for v in a1_added[qi] if v not in fpresent[qi]] for qi in range(nq)]
        t = time.time()
        a1b_rank = TA.partition_ranking(flist, mem, npart)
        configs[f"A1b_M{M}"] = (a1b_rank, {"tag": f"A1b_M{M}", "vote_sec": round(time.time() - t, 1),
                                           "note": "single fused stream, one partition vote",
                                           "structural_nodes": a1_cost["structural_nodes"]})

        log(f"--- M={M} : A2 (structure before RRF) ---")
        a2d_added, a2d_cost = run_expansion(lambda qi: [int(x) for x in dK[qi][:TA.SEED_K]], M, f"A2d_M{M}")
        a2s_added, a2s_cost = run_expansion(lambda qi: [int(x) for x in sK[qi][:TA.SEED_K]], M, f"A2s_M{M}")
        t = time.time()
        dlist = [list(dK[qi]) + [v for v in a2d_added[qi] if v not in dpresent[qi]] for qi in range(nq)]
        slist = [list(sK[qi]) + [v for v in a2s_added[qi] if v not in spresent[qi]] for qi in range(nq)]
        PR_d2 = TA.partition_ranking(dlist, mem, npart)
        PR_s2 = TA.partition_ranking(slist, mem, npart)
        a2_rank = TA.rrf_partitions([PR_d2, PR_s2], npart)
        configs[f"A2_M{M}"] = (a2_rank, {"dense": a2d_cost, "splade": a2s_cost,
                                         "vote_sec": round(time.time() - t, 1)})
        # TASK-4 dense-vs-SPLADE expansion ablation. Free: reuses PR_d/PR_s (BASE) and PR_d2/PR_s2 (A2).
        # Which retrieval channel should seed/carry structural propagation? At M=0 both collapse to BASE,
        # so they also widen the parity gate.
        configs[f"A2dOnly_M{M}"] = (TA.rrf_partitions([PR_d2, PR_s], npart),
                                    {"note": "structural expansion on the DENSE channel only"})
        configs[f"A2sOnly_M{M}"] = (TA.rrf_partitions([PR_d, PR_s2], npart),
                                    {"note": "structural expansion on the SPLADE channel only"})

    # ---------------------------------------------------------------- scoring
    def score(rank, name):
        anyc = allc = 0; scopes = []; churn = []; adm = 0; evi = 0
        ind_all = np.zeros(nq, np.int8); ind_any = np.zeros(nq, np.int8)
        hop_any = {}; hop_all = {}; hop_n = {}
        gp_new_total = 0
        for qi in range(nq):
            sel = set(int(x) for x in rank[qi][:TA.P_MAIN])
            bsel = set(int(x) for x in base_rank[qi][:TA.P_MAIN])
            g = gold_rows[qi]
            gparts = set(int(hard[x]) for x in g)
            ret = sum(1 for x in g if int(hard[x]) in sel)
            isany = ret > 0; isall = ret == len(g)
            anyc += isany; allc += isall
            ind_all[qi] = int(isall); ind_any[qi] = int(isany)
            scopes.append(int(part_sizes[list(sel)].sum()) if sel else 0)
            churn.append(len(sel ^ bsel) / (2 * TA.P_MAIN))
            adm += len(gparts & (sel - bsel))          # gold-bearing partitions newly admitted
            evi += len(gparts & (bsel - sel))          # gold-bearing partitions evicted
            gp_new_total += len(sel - bsel)
            h = hops[rows[qi]]
            if h is not None:
                hop_n[str(h)] = hop_n.get(str(h), 0) + 1
                hop_any[str(h)] = hop_any.get(str(h), 0) + int(isany)
                hop_all[str(h)] = hop_all.get(str(h), 0) + int(isall)
        d = {"n": nq, "ANY@P50": round(anyc / nq, 4), "ALL@P50": round(allc / nq, 4),
             "hard_union_nodes_mean": round(float(np.mean(scopes)), 1),
             "hard_union_nodes_median": round(float(np.median(scopes)), 1),
             "partition_churn_mean": round(float(np.mean(churn)), 4),
             "partitions_changed_mean": round(gp_new_total / nq, 2),
             "gold_bearing_partitions_admitted": int(adm),
             "gold_bearing_partitions_evicted": int(evi)}
        d["_ind_all"] = ind_all; d["_ind_any"] = ind_any
        if hop_n:
            d["per_hop"] = {h: {"n": hop_n[h], "ANY": round(hop_any[h] / hop_n[h], 4),
                                "ALL": round(hop_all[h] / hop_n[h], 4)} for h in sorted(hop_n)}
        return d

    res = {"dataset": DS, "M_SWEEP": M_LIST, "n_dev_queries": nq, "npart": npart,
           "CONTRACT": {"K0": TA.K0, "K": TA.K_LOCK, "P_MAIN": TA.P_MAIN, "SEED_K": TA.SEED_K,
                        "M_SWEEP": M_LIST, "MAX_HOPS": TA.MAX_HOPS, "DEG_CAP": TA.DEG_CAP,
                        "MAX_EDGES_SCORED": TA.MAX_EDGES_SCORED,
                        "learned_parameters": 0, "uses_gold_in_scope": False,
                        "uses_dataset_id": False,
                        "adjacency": "master_nodes.neighbors (uniform, no dataset branch)"},
           "CONFIGS": {k: score(v[0], k) for k, v in configs.items()},
           "COST": {k: v[1] for k, v in configs.items()}}

    # paired test vs BASE (McNemar exact on the ALL@P50 indicator; no gold in scope construction)
    from math import comb
    CF = res["CONFIGS"]; b_all = CF["BASE"]["_ind_all"]; b_any = CF["BASE"]["_ind_any"]
    for k, d in CF.items():
        for nm, base_ind in (("ALL", b_all), ("ANY", b_any)):
            cur = d["_ind_" + nm.lower()]
            win = int(((cur == 1) & (base_ind == 0)).sum()); los = int(((cur == 0) & (base_ind == 1)).sum())
            n2 = win + los
            pv = 1.0 if n2 == 0 else min(1.0, 2.0 * sum(comb(n2, i) for i in range(min(win, los) + 1)) / (2.0 ** n2))
            d[f"paired_vs_BASE_{nm}"] = {"gained": win, "lost": los, "net": win - los,
                                         "mcnemar_p": round(pv, 5), "sig_p<0.05": bool(pv < 0.05)}
    for d in CF.values():
        d.pop("_ind_all", None); d.pop("_ind_any", None)

    # ---------------------------------------------------------------- M=0 PARITY GATE
    log("M=0 parity gate ...")
    a1_0 = TA.rrf_partitions([PR_d, PR_s], npart)
    d0 = [list(dK[qi]) for qi in range(nq)]; s0 = [list(sK[qi]) for qi in range(nq)]
    a2_0 = TA.rrf_partitions([TA.partition_ranking(d0, mem, npart),
                              TA.partition_ranking(s0, mem, npart)], npart)
    p1 = bool(np.array_equal(a1_0[:, :TA.P_MAIN], base_rank[:, :TA.P_MAIN]))
    p2 = bool(np.array_equal(a2_0[:, :TA.P_MAIN], base_rank[:, :TA.P_MAIN]))
    gate = {"A1a_vs_BASE": "EXACT" if p1 else "FAIL", "A2_vs_BASE": "EXACT" if p2 else "FAIL"}
    # every config actually built at M=0 must reproduce BASE's top-P exactly (A1b is exempt by
    # construction -- it moves RRF to the node level, which changes the object even with no expansion)
    for k, v in configs.items():
        if k.endswith("_M0") and not k.startswith("A1b"):
            gate[k] = "EXACT" if np.array_equal(v[0][:, :TA.P_MAIN], base_rank[:, :TA.P_MAIN]) else "FAIL"
    gate["gate"] = "EXACT" if all(x == "EXACT" for x in gate.values()) else "FAIL"
    res["M0_PARTITION_PARITY"] = gate
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    C = res["CONFIGS"]
    print("")
    print(f"=== {DS}  n={nq}  M_SWEEP={M_LIST} ===")
    print(f"  M0_PARTITION_PARITY = {res['M0_PARTITION_PARITY']['gate']}")
    print(f"  {'cfg':10s} {'ANY@P50':>8s} {'ALL@P50':>8s} {'scope':>8s} {'churn':>7s} {'gpADM':>6s} {'gpEVI':>6s}")
    for k in C:
        d = C[k]
        print(f"  {k:10s} {d['ANY@P50']:8.4f} {d['ALL@P50']:8.4f} {d['hard_union_nodes_mean']:8.0f} "
              f"{d['partition_churn_mean']:7.4f} {d['gold_bearing_partitions_admitted']:6d} "
              f"{d['gold_bearing_partitions_evicted']:6d}"
              f"   ALLnet {d['paired_vs_BASE_ALL']['net']:+4d} p={d['paired_vs_BASE_ALL']['mcnemar_p']:.3f}")
    if "per_hop" in C["BASE"]:
        print("  per-hop ALL@P50:")
        for h in sorted(C["BASE"]["per_hop"]):
            print(f"    hop{h} (n={C['BASE']['per_hop'][h]['n']}): " +
                  "  ".join(f"{k} {C[k]['per_hop'][h]['ALL']:.4f}" for k in C))
    print("  expansion diagnostics (fraction of structural nodes):")
    for k, v in res["COST"].items():
        if "hop_frac" in v:
            print(f"    {k:10s} hop1 {v['hop_frac']['1']:.3f} hop2 {v['hop_frac']['2']:.3f} "
                  f"hop3 {v['hop_frac']['3']:.3f} | in_BASE_partition {v['frac_in_partition_BASE_already_selected']:.3f}")
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
