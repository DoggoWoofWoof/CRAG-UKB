"""G2 FINAL L1 EXPERIMENT -- COMBINED PARAMETER-FREE RESIDUAL (L1 coverage only).

    BASE_P50  +  retrieval residual  +  structural residual

Arms (BASE P50 is NEVER evicted -> coverage monotone non-decreasing, queries_worsened must be 0):

  BASE            canonical Dense+SPLADE -> partition voting -> RRF -> P50 -> hard union
  RET32 / RET64   next 32 / 64 canonical retrieval-continuation nodes outside BASE P50
                  CANON channel = node_rrf(dense200, splade200): the SAME fusion rule as BASE.
                  dense200 / splade200 single-channel variants also reported.
  STRUCT32/64     top 32 / 64 parameter-free structural residual nodes outside BASE P50,
                  universal RRF seeds, beam 64, s_dir ordering (identical to _ta_resid.py)
  COMBINED_32_32  BASE u RET32 u STRUCT32, deduplicated (NOT forced to 64 unique)
  RANDOM_matchedCOMBINED   per-query random node count matched exactly to COMBINED unique adds
  RANDOM32/64     reference

ZERO learned parameters. No gold anywhere in seeding/discovery/selection. No dataset-identity branch.
No benchmark entity annotations. No TEST split. No encoder pass. No L2.

Large-corpus path (webqsp / hotpotqa_clean): node matrix held fp16-normalised, all arithmetic fp32
(EXHAUSTIVE_SHARDED_FP16_IP, the already-locked storage method) because a fp32 copy does not fit in
RAM. The 4 small corpora keep the exact fp32 path so their BASE stays bit-identical.

  python scratchpad/_ta_comb.py <dataset>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
BEAM_MAIN = 64
BUDGETS = [32, 64]
RAND_REPS = 5
EVAL_CAP = int(os.environ.get("TA_CAP", "2000"))
MIN_VAL = 1000            # eval-sampling rule (NOT part of the method): if |val|<1000 extend with train
EMB = f"data/ukb_storage/{DS}/gte_qwen/"
OUT = f"results/GENERALIZATION/_g2_comb_{DS}.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
np.random.seed(0)


class F32View:
    """fp16 storage, fp32 compute. Row access returns float32 so residual()/expand_dir() are
    numerically fp32 exactly as on the small corpora."""

    def __init__(self, A):
        self.A = A

    def __getitem__(self, k):
        return np.asarray(self.A[k], np.float32)

    def __len__(self):
        return len(self.A)


def mcnemar(cur, base):
    from math import comb
    win = int(((cur == 1) & (base == 0)).sum()); los = int(((cur == 0) & (base == 1)).sum())
    n2 = win + los
    p = 1.0 if n2 == 0 else min(1.0, 2.0 * sum(comb(n2, i) for i in range(min(win, los) + 1)) / (2.0 ** n2))
    return {"gained": win, "lost": los, "net": win - los, "mcnemar_p": round(p, 5),
            "sig_p<0.05": bool(p < 0.05)}


def main():
    log(f"=== COMBINED RESIDUAL {DS}  budgets={BUDGETS} beam={BEAM_MAIN} ===")
    hard, mem, npart, (adjp, adji), deg, id2row = TA.load_topology(DS, log)
    part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
    ndocs = len(hard)
    BIG = ndocs * 1536 * 4 > 1.2e9

    j = json.load(open(EMB + "query_ids_all.json"))
    golds, hops, si = j["golds"], j.get("hops", [None] * len(j["ids"])), j["split_indices"]
    rows = list(si["val"]) if si.get("val") else list(si["all"])
    sample_rule = "val"
    if len(rows) < MIN_VAL and si.get("train"):
        rows = rows + list(si["train"]); sample_rule = "val+train (val<1000; TEST never touched)"
    val_set = set(si.get("val", []))
    if DS == "metaqa":                      # identical dev protocol to _ta_resid.py
        buck = {1: [], 2: [], 3: []}
        for r in rows:
            if hops[r] in buck:
                buck[hops[r]].append(r)
        per = EVAL_CAP // 3; sel = []
        for h in (1, 2, 3):
            b = list(buck[h]); np.random.shuffle(b); sel += b[:per]
        rows = sorted(sel); sample_rule = "val, hop-balanced (identical to _ta_resid.py)"
    else:
        np.random.shuffle(rows); rows = sorted(rows[:EVAL_CAP])

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
    in_val = np.array([1 if r in val_set else 0 for r in rows], np.int8)
    log(f"dev queries={nq} ({sample_rule}) docs={ndocs} npart={npart} BIG={BIG} "
        f"golds/q={np.mean([len(g) for g in gold_rows]):.2f}")

    d200 = np.stack([np.asarray(dense_all[i]) for i in rows]).astype(np.int64)
    s200 = np.stack([np.asarray(splade_all[i]) for i in rows]).astype(np.int64)
    dK, sK = d200[:, :TA.K_LOCK], s200[:, :TA.K_LOCK]

    log("normalizing node embeddings (" + ("fp16 store / fp32 compute" if BIG else "fp32 exact") + ") ...")
    if BIG:
        X16 = np.empty((ndocs, nodes.shape[1]), np.float16)
        for a in range(0, ndocs, 50000):
            b = min(a + 50000, ndocs)
            blk = np.array(nodes[a:b], np.float32)   # np.asarray on a same-dtype mmap returns a READ-ONLY view
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
            X16[a:b] = blk.astype(np.float16)
            del blk
        Xn = F32View(X16)
    else:
        Xn = np.array(nodes, np.float32)
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    Qm = np.stack([np.asarray(qall[i], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)

    # ---------------------------------------------------------------- BASE + parity
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
    rp = f"results/GENERALIZATION/_g2_resid_{DS}.json"
    if os.path.exists(rp):
        R = json.load(open(rp))
        ok = (abs(R["BASE_ALL_P50"] - round(BASE_ALL, 4)) < 1e-9
              and abs(R["BASE_ANY_P50"] - round(BASE_ANY, 4)) < 1e-9
              and abs(R["BASE_SCOPE_NODES"] - round(float(base_scope_size.mean()), 1)) < 1e-6
              and R["n_dev_queries"] == nq)
        parity = {"status": "EXACT" if ok else "MISMATCH", "vs": rp,
                  "ref_ALL": R["BASE_ALL_P50"], "ref_ANY": R["BASE_ANY_P50"],
                  "ref_scope": R["BASE_SCOPE_NODES"], "ref_n": R["n_dev_queries"]}
    log(f"BASE_PARITY_VS_RESID_EXPERIMENT = {parity['status']}")
    if parity["status"] == "MISMATCH":
        json.dump({"dataset": DS, "BASE_PARITY": parity, "ABORTED": True}, open(OUT, "w"), indent=1)
        log("PARITY BROKEN -- stopping")
        return

    # ---------------------------------------------------------------- retrieval residual (cached, ~free)
    t = time.time()
    fused200 = [TA.node_rrf(d200[qi], s200[qi]) for qi in range(nq)]
    RET = {"rrf200": [[int(v) for v in fused200[qi] if int(hard[v]) not in base_sel[qi]] for qi in range(nq)],
           "dense200": [[int(v) for v in d200[qi] if int(hard[v]) not in base_sel[qi]] for qi in range(nq)],
           "splade200": [[int(v) for v in s200[qi] if int(hard[v]) not in base_sel[qi]] for qi in range(nq)]}
    t_ret = time.time() - t
    CANON_RET = "rrf200"
    ret_avail = {k: round(float(np.mean([len(x) for x in v])), 1) for k, v in RET.items()}
    ret_short64 = {k: int(sum(1 for x in v if len(x) < 64)) for k, v in RET.items()}
    log(f"retrieval residual built in {t_ret:.2f}s  avail/q={ret_avail}  "
        f"queries with <64 available: {ret_short64}")

    # ---------------------------------------------------------------- structural residual (beam 64)
    t = time.time(); esc = 0; nraw = 0
    struct = []
    for qi in range(nq):
        sd = fused200[qi][:TA.SEED_K]
        r_q = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        a, vm, e = TA.expand_dir(sd, r_q, adjp, adji, deg, Xn, BEAM_MAIN)
        esc += e; nraw += len(a)
        struct.append([v for v in a if int(hard[v]) not in base_sel[qi]])
        if (qi + 1) % 500 == 0:
            log(f"   struct {qi+1}/{nq} edges={esc:,}")
    t_struct = time.time() - t
    struct_avail = float(np.mean([len(x) for x in struct]))
    log(f"  [STRUCT_RRF_beam64] avail/q={struct_avail:.1f} raw/q={nraw/nq:.1f} "
        f"edges={esc:,} {t_struct:.1f}s ({t_struct/max(t_base,1e-9):.1f}x BASE)")

    # ---------------------------------------------------------------- scoring
    def score(added_nodes, name):
        anyc = allc = newly = worse = grec = 0
        ind_all = np.zeros(nq, np.int8)
        nadd = np.zeros(nq, np.int64)
        hop_n, hop_all = {}, {}
        for qi in range(nq):
            an = set(added_nodes[qi]) if added_nodes is not None else set()
            nadd[qi] = len(an)
            cov, bcov = [], []
            for g in gold_rows[qi]:
                b0 = int(hard[g]) in base_sel[qi]
                bcov.append(b0); cov.append(b0 or (g in an))
            grec += sum(1 for a_, b_ in zip(cov, bcov) if a_ and not b_)
            isall = all(cov); anyc += any(cov); allc += isall; ind_all[qi] = int(isall)
            newly += int(isall and not ind_all_base[qi]); worse += int((not isall) and ind_all_base[qi])
            h = hops[rows[qi]]
            if h is not None:
                k = str(h); hop_n[k] = hop_n.get(k, 0) + 1; hop_all[k] = hop_all.get(k, 0) + int(isall)
        d = {"n": nq, "ANY": round(anyc / nq, 4), "ALL": round(allc / nq, 4),
             "dALL_vs_BASE": round(allc / nq - BASE_ALL, 4),
             "queries_newly_ALL_covered": int(newly),
             "newly_recovered_gold_occurrences": int(grec),
             "queries_worsened": int(worse),
             "unique_added_nodes_mean": round(float(nadd.mean()), 2),
             "final_scope_nodes_mean": round(float((base_scope_size + nadd).mean()), 1),
             "scope_growth_pct": round(100.0 * float(nadd.mean()) / float(base_scope_size.mean()), 3),
             "paired_vs_BASE_ALL": mcnemar(ind_all, ind_all_base),
             "_ind": ind_all}
        if in_val.sum() and in_val.sum() != nq:
            d["ALL_val_only"] = round(float(ind_all[in_val == 1].mean()), 4)
            d["n_val_only"] = int(in_val.sum())
        if hop_n:
            d["per_hop"] = {h: {"n": hop_n[h], "ALL": round(hop_all[h] / hop_n[h], 4)} for h in sorted(hop_n)}
        assert worse == 0, f"{name}: additive arm made {worse} queries WORSE -- implementation bug"
        return d

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

    def score_rand(counts, tag):
        acc = [score(random_nodes(counts, s), f"{tag}_s{s}") for s in range(RAND_REPS)]
        out = {}
        for k in acc[0]:
            if k == "_ind":
                continue
            out[k] = (round(float(np.mean([a[k] for a in acc])), 4)
                      if isinstance(acc[0][k], (int, float)) else acc[0][k])
        out["ALL_std"] = round(float(np.std([a["ALL"] for a in acc])), 5)
        out["reps"] = RAND_REPS; out["per_rep_ALL"] = [a["ALL"] for a in acc]
        out["_ind"] = acc[0]["_ind"]
        return out

    RES = {"BASE": score(None, "BASE")}
    for M in BUDGETS:
        for ch, src in RET.items():
            RES[f"RET{M}_{ch}"] = score([x[:M] for x in src], f"RET{M}_{ch}")
        RES[f"RET{M}"] = RES[f"RET{M}_{CANON_RET}"]
        RES[f"STRUCT{M}"] = score([x[:M] for x in struct], f"STRUCT{M}")
        RES[f"RANDOM{M}"] = score_rand(np.full(nq, M, np.int64), f"RANDOM{M}")

    comb = [list(dict.fromkeys(RET[CANON_RET][qi][:32] + struct[qi][:32])) for qi in range(nq)]
    RES["COMBINED_32_32"] = score(comb, "COMBINED_32_32")
    ucnt = np.array([len(c) for c in comb], np.int64)
    RES["RANDOM_matchedCOMBINED"] = score_rand(ucnt, "RANDOM_matchedCOMBINED")
    for ch in RET:
        if ch != CANON_RET:
            RES[f"COMBINED_32_32_{ch}"] = score(
                [list(dict.fromkeys(RET[ch][qi][:32] + struct[qi][:32])) for qi in range(nq)], f"comb_{ch}")

    # ---------------------------------------------------------------- overlap + marginals
    ov = np.array([len(set(RET[CANON_RET][qi][:32]) & set(struct[qi][:32])) for qi in range(nq)], np.float64)
    r_only = np.array([len(set(RET[CANON_RET][qi][:32]) - set(struct[qi][:32])) for qi in range(nq)], np.float64)
    s_only = np.array([len(set(struct[qi][:32]) - set(RET[CANON_RET][qi][:32])) for qi in range(nq)], np.float64)
    OVER = {"channel": CANON_RET,
            "mean_RET32_INTERSECT_STRUCT32": round(float(ov.mean()), 3),
            "overlap_fraction_of_32": round(float(ov.mean()) / 32.0, 4),
            "retrieval_only_mean": round(float(r_only.mean()), 2),
            "structural_only_mean": round(float(s_only.mean()), 2),
            "unique_added_mean": round(float(ucnt.mean()), 2),
            "queries_with_any_overlap": int((ov > 0).sum()),
            "RET32_supplied_mean": round(float(np.mean([len(x[:32]) for x in RET[CANON_RET]])), 2),
            "STRUCT32_supplied_mean": round(float(np.mean([len(x[:32]) for x in struct])), 2)}

    C, R32, S32 = RES["COMBINED_32_32"], RES["RET32"], RES["STRUCT32"]
    MARG = {"STRUCT_MARGINAL_AFTER_RET": round(C["ALL"] - R32["ALL"], 4),
            "RET_MARGINAL_AFTER_STRUCT": round(C["ALL"] - S32["ALL"], 4),
            "paired_COMBINED_vs_RET32": mcnemar(C["_ind"], R32["_ind"]),
            "paired_COMBINED_vs_STRUCT32": mcnemar(C["_ind"], S32["_ind"]),
            "paired_COMBINED_vs_RET64": mcnemar(C["_ind"], RES["RET64"]["_ind"]),
            "paired_COMBINED_vs_STRUCT64": mcnemar(C["_ind"], RES["STRUCT64"]["_ind"]),
            "paired_COMBINED_vs_RANDOMmatched": mcnemar(C["_ind"], RES["RANDOM_matchedCOMBINED"]["_ind"]),
            "COMBINED_minus_RET64": round(C["ALL"] - RES["RET64"]["ALL"], 4),
            "COMBINED_minus_STRUCT64": round(C["ALL"] - RES["STRUCT64"]["ALL"], 4),
            "additivity_gap": round(C["dALL_vs_BASE"]
                                    - (R32["dALL_vs_BASE"] + S32["dALL_vs_BASE"]), 4)}

    COST = {"BASE_sec": round(t_base, 1),
            "RET_build_sec": round(t_ret, 2),
            "STRUCT_traversal_sec": round(t_struct, 1),
            "STRUCT_edges_traversed": int(esc),
            "STRUCT_nodes_evaluated": int(nraw),
            "STRUCT_x_BASE": round(t_struct / max(t_base, 1e-9), 1),
            "RET_x_BASE": round(t_ret / max(t_base, 1e-9), 3),
            "COMBINED_incremental_sec_above_STRUCT32": round(t_ret, 2),
            "COMBINED_incremental_pct_above_STRUCT32": round(100.0 * t_ret / max(t_struct, 1e-9), 2)}

    for k in RES:
        RES[k] = {a: b for a, b in RES[k].items() if a != "_ind"}
    out = {"dataset": DS, "n_dev_queries": nq, "sample_rule": sample_rule,
           "n_docs": int(ndocs), "npart": int(npart), "big_corpus_fp16_path": bool(BIG),
           "CONTRACT": {"K0": TA.K0, "K": TA.K_LOCK, "P_MAIN": TA.P_MAIN, "SEED_K": TA.SEED_K,
                        "BEAM": BEAM_MAIN, "MAX_HOPS": TA.MAX_HOPS, "DEG_CAP": TA.DEG_CAP,
                        "canonical_retrieval_channel": CANON_RET, "learned_parameters": 0,
                        "uses_gold_in_scope": False, "uses_dataset_id": False,
                        "uses_entity_annotations": False, "additive_only_BASE_never_evicted": True},
           "BASE_PARITY": parity, "BASE_ANY_P50": round(BASE_ANY, 4), "BASE_ALL_P50": round(BASE_ALL, 4),
           "BASE_SCOPE_NODES": round(float(base_scope_size.mean()), 1),
           "mean_partition_size": round(float(part_sizes.mean()), 1),
           "retrieval_available_per_query": ret_avail,
           "queries_with_fewer_than_64_retrieval_candidates": ret_short64,
           "structural_available_per_query": round(struct_avail, 1),
           "OVERLAP": OVER, "MARGINAL": MARG, "RESULTS": RES, "COST": COST}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    # ---------------------------------------------------------------- console
    print(f"\n=== {DS}  n={nq}  BASE_ALL={BASE_ALL:.4f}  BASE_ANY={BASE_ANY:.4f}  "
          f"scope={base_scope_size.mean():.0f}  parity={parity['status']} ===")
    print(f"  {'arm':26s} {'ALL':>8s} {'dALL':>9s} {'newQ':>6s} {'gold':>6s} "
          f"{'uniq':>7s} {'scope%':>7s} {'ANY':>8s} {'wors':>5s}")
    order = ["BASE", "RET32", "STRUCT32", "RET64", "STRUCT64", "COMBINED_32_32",
             "RANDOM_matchedCOMBINED", "RANDOM32", "RANDOM64",
             "RET32_dense200", "RET32_splade200", "COMBINED_32_32_dense200"]
    for k in order:
        v = RES[k]
        print(f"  {k:26s} {v['ALL']:8.4f} {v['dALL_vs_BASE']:+9.4f} "
              f"{v['queries_newly_ALL_covered']:6} {v['newly_recovered_gold_occurrences']:6} "
              f"{v['unique_added_nodes_mean']:7.2f} {v['scope_growth_pct']:7.3f} {v['ANY']:8.4f} "
              f"{v['queries_worsened']:5}")
    print(f"  OVERLAP  |RET32 & STRUCT32|={OVER['mean_RET32_INTERSECT_STRUCT32']:.3f} "
          f"({OVER['overlap_fraction_of_32']:.1%} of 32)  ret_only={OVER['retrieval_only_mean']:.2f} "
          f"struct_only={OVER['structural_only_mean']:.2f}  unique={OVER['unique_added_mean']:.2f}")
    print(f"  MARGINAL struct_after_ret={MARG['STRUCT_MARGINAL_AFTER_RET']:+.4f} "
          f"(p={MARG['paired_COMBINED_vs_RET32']['mcnemar_p']}) "
          f"ret_after_struct={MARG['RET_MARGINAL_AFTER_STRUCT']:+.4f} "
          f"(p={MARG['paired_COMBINED_vs_STRUCT32']['mcnemar_p']})")
    print(f"  COMBINED-RET64={MARG['COMBINED_minus_RET64']:+.4f} "
          f"(net {MARG['paired_COMBINED_vs_RET64']['net']:+}, p={MARG['paired_COMBINED_vs_RET64']['mcnemar_p']})  "
          f"COMBINED-STRUCT64={MARG['COMBINED_minus_STRUCT64']:+.4f} "
          f"(net {MARG['paired_COMBINED_vs_STRUCT64']['net']:+}, p={MARG['paired_COMBINED_vs_STRUCT64']['mcnemar_p']})")
    print(f"  COST base={COST['BASE_sec']}s ret={COST['RET_build_sec']}s struct={COST['STRUCT_traversal_sec']}s "
          f"({COST['STRUCT_x_BASE']}x BASE)  combined_increment_over_STRUCT32="
          f"{COST['COMBINED_incremental_pct_above_STRUCT32']}%")
    if "per_hop" in RES["BASE"]:
        print(f"    {'arm':26s} {'hop1':>8s} {'hop2':>8s} {'hop3':>8s}")
        for k in ["BASE", "RET32", "STRUCT32", "RET64", "STRUCT64", "COMBINED_32_32"]:
            ph = RES[k]["per_hop"]
            print(f"    {k:26s} " + " ".join(f"{ph[h]['ALL']:8.4f}" for h in ("1", "2", "3")))
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
