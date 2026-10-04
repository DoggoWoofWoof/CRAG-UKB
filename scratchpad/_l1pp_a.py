"""ROUND A -- STEP 0 swap audit, STEP 1 channel redefinition + parity, STEP 2 partition graph.

Verification only.  No selector search is reopened.

  python scratchpad/_l1pp_a.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1kb_core as KB
import _l1kb_router as JR

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
NDUMP = 3


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def audit_swaps(ds, z, meta, C, GRP, goldp, buckets):
    """STEP 0: full provenance dump of the frozen B6 swap for representative queries."""
    ctxs = KB.contexts(z, meta, C, 6)
    hops = z["hops"]
    out = []
    for name, qsel in buckets.items():
        for qi in qsel[:NDUMP]:
            c = ctxs[qi]
            X, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], 6)
            fs = c["prot_set"] | set(X)
            cands = c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
            rec = {
                "bucket": name, "qi": int(qi), "hop": int(hops[qi]),
                "BASE_top50_in_base_rank_order": [int(x) for x in c["prot"]] + [int(x) for x in c["bnd"]],
                "protected_core_44": [int(x) for x in c["prot"]],
                "boundary_6": [int(x) for x in c["bnd"]],
                "n_challengers": len(set(c["chal"])),
                "challengers": [int(x) for x in dict.fromkeys(c["chal"])][:24],
                "channel_ranks": {str(int(p)): {
                    "canonical": int(c["cpos"][p]) if p in c["cpos"] else None,
                    "structural": int(c["spos"][p]) if p in c["spos"] else None,
                    "retrieval": int(c["rpos"][p]) if p in c["rpos"] else None,
                    "rrf_score": round(
                        (1.0 / (PP.K0 + c["cpos"][p]) if p in c["cpos"] else 0.0)
                        + (1.0 / (PP.K0 + c["spos"][p]) if p in c["spos"] else 0.0)
                        + (1.0 / (PP.K0 + c["rpos"][p]) if p in c["rpos"] else 0.0), 8),
                    "incumbent": bool(p in c["bnd"])} for p in cands[:24]},
                "swapped_in": sorted(int(p) for p in X if p not in c["bnd"]),
                "swapped_out": sorted(int(p) for p in c["bnd"] if p not in X),
                "final_top50_size": len(fs),
                "gold_partitions_EVAL_ONLY": sorted(int(g) for g in goldp[qi]),
                "gold_covered_BASE": bool(goldp[qi] <= c["base50"]),
                "gold_covered_FINAL": bool(goldp[qi] <= fs),
            }
            assert rec["final_top50_size"] == PP.P
            out.append(rec)
    return out


def verify(ds, z, meta, C, ch):
    """the eight explicit STEP-0 verification gates."""
    V = {}
    nq = meta["n_dev_queries"]
    # 1. base_rank provenance
    V["base_rank_provenance"] = {
        "claim": "base_rank == rrf_partitions([PR_dense, PR_splade])",
        "status": ch["BASE_RANK_PARITY"]}
    # 2. ret_rrf provenance
    bad = 0
    for qi in range(min(nq, 200)):
        f = TA.node_rrf(z["ret_dense"][qi], z["ret_splade"][qi])[:TA.TOP200]
        c = [int(x) for x in z["ret_rrf"][qi] if int(x) >= 0]
        if f[:len(c)] != c:
            bad += 1
    V["ret_rrf_provenance"] = {
        "claim": "ret_rrf == node_rrf(dense200, splade200)",
        "status": "EXACT" if bad == 0 else "MISMATCH", "checked": min(nq, 200), "bad": bad}
    # 3. measured correlation / redundancy between the two lexical granularities
    ctxs = KB.contexts(z, meta, C, 6)
    tot = inc = 0
    rho = []
    for qi, c in enumerate(ctxs):
        rr = [p for p in c["rpos"] if p not in c["base50"]]
        tot += len(rr); inc += sum(1 for p in rr if p in c["cpos"])
        both = [p for p in rr if p in c["cpos"]]
        if len(both) >= 5:
            rho.append(spearman(np.array([c["cpos"][p] for p in both], float),
                                np.array([c["rpos"][p] for p in both], float)))
    V["channel_redundancy"] = {
        "frac_retrieval_challengers_already_canonically_ranked": round(inc / max(1, tot), 4),
        "mean_spearman_canonical_vs_retrieval_on_shared": round(float(np.mean(rho)), 4)
        if rho else None, "n_queries_with_ge5_shared": len(rho)}
    # 4. absent-channel semantics
    miss = 0
    for c in ctxs[:200]:
        for p in c["chal"]:
            if p not in c["cpos"] and p not in c["spos"] and p not in c["rpos"]:
                miss += 1
    V["absent_channel_semantics"] = {
        "claim": "a channel with no evidence for a candidate contributes exactly 0.0",
        "status": "VERIFIED_BY_CONSTRUCTION", "candidates_with_no_channel_at_all": miss}
    # 5. symmetric scoring: incumbents and challengers scored by the identical expression
    asym = 0
    for qi, c in enumerate(ctxs[:300]):
        cands = c["bnd"] + [p for p in dict.fromkeys(c["chal"]) if p not in c["bnd"]]
        sc = []
        for p in cands:                      # recomputed with NO knowledge of incumbency
            s = (1.0 / (PP.K0 + c["cpos"][p]) if p in c["cpos"] else 0.0) \
                + (1.0 / (PP.K0 + c["spos"][p]) if p in c["spos"] else 0.0) \
                + (1.0 / (PP.K0 + c["rpos"][p]) if p in c["rpos"] else 0.0)
            sc.append((-s, c["cpos"].get(p, 10 ** 6), p))
        sc.sort()
        X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], 6)
        if [p for _, _, p in sc[:6]] != X:
            asym += 1
    V["symmetric_scoring"] = {
        "claim": "incumbent-blind rescoring reproduces the selection", "mismatches": asym,
        "checked": min(300, len(ctxs)), "status": "EXACT" if asym == 0 else "MISMATCH"}
    # 6. deterministic ties: permuting the challenger list must not change the output
    rng = np.random.RandomState(0); dif = 0
    for qi, c in enumerate(ctxs[:300]):
        X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], 6)
        ch2 = list(dict.fromkeys(c["chal"])); rng.shuffle(ch2)
        X2, _ = KB.f6_select(c["bnd"], ch2, c["spos"], c["rpos"], c["cpos"], 6)
        if sorted(X) != sorted(X2):
            dif += 1
    V["deterministic_ties"] = {"claim": "output invariant to challenger input order",
                               "differences": dif, "checked": min(300, len(ctxs)),
                               "status": "EXACT" if dif == 0 else "MISMATCH"}
    # 7. exact P=50
    bad50 = 0
    for qi, c in enumerate(ctxs):
        X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], 6)
        if len(c["prot_set"] | set(X)) != PP.P:
            bad50 += 1
    V["exact_P50"] = {"violations": bad50, "checked": len(ctxs),
                      "status": "EXACT" if bad50 == 0 else "MISMATCH"}
    # 8. no gold in ranking
    V["no_gold_in_ranking"] = {
        "claim": "the selector signature is sel(context, qi, extra); the context carries only "
                 "prot/bnd/chal/cpos/spos/rpos/sagg/ragg/base50 and gold is passed ONLY to the "
                 "scorer after selection",
        "status": "VERIFIED_BY_CONSTRUCTION"}
    return V


def main():
    OUT = {"PPR_CONTRACT": {"alpha": PP.PPR_ALPHA, "iters": PP.PPR_ITERS,
                            "audit": "l3_methods._ppr / l3_solvers select alpha per run from a "
                                     "small grid BY GOLD RECALL (ppr_best_alpha) and "
                                     "LEVEL3_README marks it exploratory, so no canonical alpha "
                                     "could be inherited; 0.85 is the only non-gold-selected "
                                     "alpha in the codebase (_qppr_ball default) and the "
                                     "standard PageRank damping. iters=20 and P=D^-1 A are "
                                     "inherited unchanged."},
           "STEP0_AUDIT": {}, "STEP1_CHANNELS": {}, "STEP2_PARTITION_GRAPH": {}}
    dumps = {}
    for ds in PP.DSETS:
        z, meta = PP.load(ds)
        nq = meta["n_dev_queries"]; hops = z["hops"]
        goldp = PP.goldparts(z, meta)
        topo = TA.load_topology(ds, log=lambda *a: None)
        C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
        ch = PP.channels(ds, z, meta, topo)
        assert ch["BASE_RANK_PARITY"] == "EXACT", f"{ds}: base_rank provenance broken"

        # STEP 0 -- representative queries
        if ds in ("metaqa", "webqsp", "hotpotqa_clean"):
            if ds == "metaqa":
                bk = {"metaqa_hop2": [i for i in range(nq) if hops[i] == 2],
                      "metaqa_hop3": [i for i in range(nq) if hops[i] == 3]}
            else:
                bk = {ds: list(range(nq))}
            dumps[ds] = audit_swaps(ds, z, meta, C, GRP, goldp, bk)

        OUT["STEP0_AUDIT"][ds] = verify(ds, z, meta, C, ch)

        # STEP 1 -- channel separation statistics
        npart = ch["npart"]
        pd_, ps_ = PP.rank_pos(ch["PR_d"], npart), PP.rank_pos(ch["PR_s"], npart)
        pb = PP.rank_pos(z["base_rank"][:, :npart] if z["base_rank"].shape[1] >= npart
                         else ch["base_rank_replay"], npart)
        r_ds = float(np.mean([spearman(pd_[qi], ps_[qi]) for qi in range(min(nq, 300))]))
        OUT["STEP1_CHANNELS"][ds] = {
            "npart": int(npart), "BASE_RANK_PARITY": ch["BASE_RANK_PARITY"],
            "mean_spearman_DENSE_vs_SPLADE_partition_rank": round(r_ds, 4),
            "mean_spearman_DENSE_vs_BASE": round(float(np.mean(
                [spearman(pd_[qi], pb[qi]) for qi in range(min(nq, 300))])), 4),
            "mean_spearman_SPLADE_vs_BASE": round(float(np.mean(
                [spearman(ps_[qi], pb[qi]) for qi in range(min(nq, 300))])), 4)}

        # STEP 2 -- partition graph
        t = time.time()
        G = PP.partition_graph(ds, topo, log=log)
        deg_out = np.bincount(G["edge_src"], minlength=int(G["npart"]))
        w = G["edge_w"]
        OUT["STEP2_PARTITION_GRAPH"][ds] = {
            "npart": int(G["npart"]), "n_nodes": int(G["n_nodes"]),
            "inter_partition_edges": int(len(w)),
            "inter_partition_edge_mass": int(w.sum()),
            "internal_self_mass": int(G["internal_mass"].sum()),
            "frac_mass_internal": round(float(G["internal_mass"].sum())
                                        / float(G["internal_mass"].sum() + w.sum()), 4),
            "density": round(float(len(w)) / (int(G["npart"]) ** 2), 6),
            "outdeg_mean": round(float(deg_out.mean()), 2),
            "outdeg_median": int(np.median(deg_out)),
            "outdeg_max": int(deg_out.max()),
            "isolated_partitions": int((deg_out == 0).sum()),
            "edge_w_mean": round(float(w.mean()), 2), "edge_w_max": int(w.max()),
            "cache_bytes": os.path.getsize(f"{PP.PPD}/graph/pgraph_{ds}.npz"),
            "build_sec": round(time.time() - t, 1)}
        g = OUT["STEP2_PARTITION_GRAPH"][ds]
        log("%-15s npart %5d  inter-edges %8d (density %.4f)  outdeg mean %7.1f max %5d  "
            "isolated %3d  internal-mass %.3f" % (
                ds, g["npart"], g["inter_partition_edges"], g["density"], g["outdeg_mean"],
                g["outdeg_max"], g["isolated_partitions"], g["frac_mass_internal"]))
        v = OUT["STEP0_AUDIT"][ds]
        log("%-15s gates: base %s  ret %s  sym %s  ties %s  P50 %s | redundancy %.4f rho %s"
            % (ds, v["base_rank_provenance"]["status"], v["ret_rrf_provenance"]["status"],
               v["symmetric_scoring"]["status"], v["deterministic_ties"]["status"],
               v["exact_P50"]["status"],
               v["channel_redundancy"]["frac_retrieval_challengers_already_canonically_ranked"],
               v["channel_redundancy"]["mean_spearman_canonical_vs_retrieval_on_shared"]))
    json.dump(OUT, open(f"{PP.PPD}/diag/round_a.json", "w"), indent=1)
    json.dump(dumps, open(f"{PP.PPD}/diag/swap_dump.json", "w"), indent=1)
    log("wrote round_a.json + swap_dump.json")


if __name__ == "__main__":
    main()
