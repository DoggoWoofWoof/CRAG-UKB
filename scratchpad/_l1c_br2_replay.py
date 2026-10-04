"""BOUNDED_VERTEXCUT_R2 step 4 (twelfth ruling; PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json written first): replay the bounded-overlap
substrate (every node in <= 2 blocks, cap C = 100, k = 2 k_f) as a static routing BASE on DEV_A of metaqa / squad / musique, gated
against each dataset's canonical hard BASE with the section-19 / 20 routing, serving, exposure and matched-budget contracts imported
unchanged.  Post-hoc structural-mechanism test: DEV_A only, no DEV_B, no TEST, nothing promoted, served L1 unchanged.

Cells: R2_P50 (protocol cell, exposure flag), R2_MATCHED_LE (the pre-registered SIGNAL cell: equal unique-node budget), R2_MATCHED_GE
(informational), R2_HOME_ONLY_VOTE (diagnostic), OWNERS_ONLY_P50 / OWNERS_ONLY_MATCHED_LE (the R = 1 control: the same owner partition
without alternates), HARD_OWN_BLOCK_VOTE (diagnostic), the GATE / contrast BASE; on metaqa the section-19 V1 cells are recomputed,
asserted equal to the frozen record and compared with R2 (informational).

    python -u scratchpad/_l1c_br2_replay.py <metaqa|squad|musique>   -> results/L1_COVPART/br2_replay_A_<ds>.json (+ .log via the shell)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core as G  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_vcut_replay as R  # noqa: E402  (section-19 functions; its main() is never called)
import _l1c_br2_lib as L  # noqa: E402

X = S.X
K0, K_LOCK, P_MAIN = S.K0, S.K_LOCK, S.P_MAIN
OUT = V.OUT
PDIR = V.PDIR
log = S.log
sha_file, pin = L.sha_file, L.pin
ALPHA = 0.01
GATE = {"metaqa": ("HARD_MTK_BASE", "metaqa", "metaqa_phg"), "squad": ("HARD_MTK_BASE", "squad", "squad_phg"), "musique": ("HARD_PHG_BASE", "musique", None)}
K_V1 = 1004
T0 = time.time()


def label(cmp):
    if cmp["lost"] > cmp["gained"] and cmp["p"] < ALPHA:
        return "LOSS"
    if cmp["gained"] > cmp["lost"] and cmp["p"] < ALPHA:
        return "GAIN"
    return "NEUTRAL"


def exposure_hard(D, sel_rows):
    return np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in sel_rows])


def main(ds):
    gate_label, gate_cache, contrast_cache = GATE[ds]
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
    for mod, pn in pre["code"]["new_modules"].items():
        if mod in ("_l1c_br2_lib.py", "_l1c_br2_replay.py"):
            assert sha_file(os.path.join(HERE, mod)) == pn["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    for nm in [gate_cache] + ([contrast_cache] if contrast_cache else []):
        assert sha_file(X.CACHES[nm]) == pre["caches_read_only"][nm]["sha256"], "cache %s changed" % nm
    fp_out = os.path.join(OUT, "br2_replay_A_%s.json" % ds)
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out

    D = G.Data(gate_cache, dense_fp32=False)
    D2 = G.Data(contrast_cache, dense_fp32=False) if contrast_cache else None
    if D2 is not None:
        assert (D.rows == D2.rows).all() and D.C.qids == D2.C.qids, "the two served caches must hold the same rows"
    mA = D.C.split == "A"
    masks = R.hop_masks(D, mA)
    log("%s: gate %s (%s) contrast %s | nq %d DEV_A %d hops %s" % (ds, gate_label, gate_cache, contrast_cache, D.nq, int(mA.sum()), {kk: int(vv.sum()) for kk, vv in masks.items()}))

    # ---- the bounded-overlap substrate (read only; asserted against its record)
    Gr = V.Graph(ds)
    assert Gr.N == D.N and Gr.keys_sha == pre["graphs_frozen"][ds]["struct_keys_sha256"] and Gr.C == pre["constants"]["C"]
    r2_p, home, alt, k = L.load_r2(ds, Gr)
    r2_rec = json.load(open(r2_p[:-4] + ".json", encoding="utf-8"))
    assert r2_rec["output"]["sha256"] == sha_file(r2_p) and r2_rec["representation"]["k"] == k == L.k_of(Gr) and r2_rec["preregistration"]["sha256"] == sha_file(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"))
    N, C = Gr.N, int(Gr.C)
    deg0 = Gr.deg == 0
    hub = Gr.hub
    t = time.time()
    mem_all, Y, blocks_csc, sizes = L.membership(home, alt, N, k)
    lam = np.diff(mem_all[0])
    assert lam.max() <= L.R_MAX and lam.min() == 1 and sizes.max() <= C and (alt[deg0] == -1).all() and (home >= 0).all()
    mem_vc = R.vote_table(mem_all, home, hub)
    mem_home = R.home_table(home)
    none = np.full(N, -1, np.int64)
    _, _, blocks_own, sizes_own = L.membership(home, none, N, k)
    ptr, flat = mem_all
    rep = {"k": k, "C": C, "R": L.R_MAX, "RF": round(float(lam.sum()) / N, 4), "share_lambda_2": round(float((lam == 2).mean()), 4),
           "block_size": {"mean": round(float(sizes.mean()), 2), "max": int(sizes.max()), "min": int(sizes.min()), "at_C": int((sizes == C).sum())},
           "owner_block_size": {"mean": round(float(sizes_own.mean()), 2), "max": int(sizes_own.max()), "min": int(sizes_own.min())},
           "hubs": int(hub.sum()), "degree_0": int(deg0.sum()), "hubs_with_alternate": int((hub & (alt >= 0)).sum()),
           "vote_table_entries": {"mem_vc": int(len(mem_vc[1])), "mem_all": int(len(flat)), "mem_home": int(len(mem_home[1]))},
           "containment_1hop": r2_rec["containment"]["R2"]["containment_1hop"], "containment_1hop_owners_only": r2_rec["containment"]["owners_only_R1"]["containment_1hop"],
           "seconds": round(time.time() - t, 1)}
    log("representation: %s" % json.dumps(rep))

    # ---- hub / degree-0 share of the served hits and of the gold (descriptive)
    di, si = D.d_ids[:, :K_LOCK], D.s_ids[:, :K_LOCK]
    assert di.min() >= 0 and si.min() >= 0
    hub_gold = np.array([float(hub[g].mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    q_with_hub_gold = np.array([bool(hub[g].any()) for g in D.C.gold_nodes])
    d0_gold = np.array([float(deg0[g].mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    q_with_d0_gold = np.array([bool(deg0[g].any()) for g in D.C.gold_nodes])
    alt_gold = np.array([float((alt[g] >= 0).mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    shares = {"served_hits_hub_share": {"dense_top%d" % K_LOCK: round(float(hub[di].mean()), 4), "splade_top%d" % K_LOCK: round(float(hub[si].mean()), 4)},
              "served_hits_degree_0_share": {"dense_top%d" % K_LOCK: round(float(deg0[di].mean()), 4), "splade_top%d" % K_LOCK: round(float(deg0[si].mean()), 4)},
              "served_hits_with_alternate_share": {"dense_top%d" % K_LOCK: round(float((alt[di] >= 0).mean()), 4), "splade_top%d" % K_LOCK: round(float((alt[si] >= 0).mean()), 4)},
              "gold_hub_share": {"gold_nodes_hub_fraction_A_mean": round(float(hub_gold[mA].mean()), 4), "queries_A_with_hub_gold": int((q_with_hub_gold & mA).sum())},
              "gold_degree_0_share": {"gold_nodes_degree_0_fraction_A_mean": round(float(d0_gold[mA].mean()), 4), "queries_A_with_degree_0_gold": int((q_with_d0_gold & mA).sum())},
              "gold_with_alternate_share": {"gold_nodes_with_alternate_fraction_A_mean": round(float(alt_gold[mA].mean()), 4)}}

    # ---- baselines (served) + the node-level check
    base = D.base_all.astype(np.int8)
    a_chk, _ = R.served(D, D.base_sel, D.mem_hard)
    assert (a_chk == base).all(), "node-level ALL != served block-level ALL (gate)"
    exp = exposure_hard(D, D.base_sel)
    if D2 is not None:
        base2 = D2.base_all.astype(np.int8)
        a_chk2, _ = R.served(D2, D2.base_sel, D2.mem_hard)
        assert (a_chk2 == base2).all(), "node-level ALL != served block-level ALL (contrast)"
        exp2 = exposure_hard(D2, D2.base_sel)
        log("baselines: %s ALL(A) %.4f exposure %.1f | contrast %s ALL(A) %.4f exposure %.1f" % (gate_label, base[mA].mean(), exp[mA].mean(), contrast_cache, base2[mA].mean(), exp2[mA].mean()))
    else:
        log("baselines: %s ALL(A) %.4f exposure %.1f" % (gate_label, base[mA].mean(), exp[mA].mean()))

    # ---- routing (frozen numerics): R2 vote table, home-only table, the served hard own-block table
    t = time.time()
    R_vc = R.fused_rank(D, mem_vc, k)
    R_home = R.fused_rank(D, mem_home, k)
    R_own = R.fused_rank(D, D.mem_hard, D.npart)
    log("rankings computed (%.0fs)" % (time.time() - t))
    sel50 = [R_vc[i, :P_MAIN] for i in range(D.nq)]
    all50, any50 = R.served(D, sel50, mem_all)
    u50, nom50 = R.unique_exposure(sel50, blocks_csc, N)
    L_le, U_le, L_ge, U_ge = R.matched_prefixes(R_vc, blocks_csc, N, exp)
    sel_le = [R_vc[i, :L_le[i]] for i in range(D.nq)]
    sel_ge = [R_vc[i, :L_ge[i]] for i in range(D.nq)]
    all_le, any_le = R.served(D, sel_le, mem_all)
    all_ge, any_ge = R.served(D, sel_ge, mem_all)
    assert (U_le <= exp).all() and (L_ge >= L_le).all()
    # home-only vote, R2 serving (diagnostic)
    sel_home = [R_home[i, :P_MAIN] for i in range(D.nq)]
    all_home, any_home = R.served(D, sel_home, mem_all)
    u_home, _ = R.unique_exposure(sel_home, blocks_csc, N)
    # OWNERS-ONLY control (R = 1: home vote, home serving, owner blocks) at P50 and at the matched budget
    all_o50, any_o50 = R.served(D, sel_home, mem_home)
    u_o50, _ = R.unique_exposure(sel_home, blocks_own, N)
    Lo_le, Uo_le, Lo_ge, Uo_ge = R.matched_prefixes(R_home, blocks_own, N, exp)
    sel_o_le = [R_home[i, :Lo_le[i]] for i in range(D.nq)]
    all_o_le, any_o_le = R.served(D, sel_o_le, mem_home)
    assert (Uo_le <= exp).all()
    # the served hard partition voted through its own-block table (diagnostic)
    sel_own = [R_own[i, :P_MAIN] for i in range(D.nq)]
    all_own, any_own = R.served(D, sel_own, D.mem_hard)
    exp_own = exposure_hard(D, sel_own)

    # ---- reach decomposition for the R2_P50 cell
    ev = R.evidence(D, mem_vc, k)
    pos = G.positions(R_vc)
    reached = np.zeros(D.nq, bool)
    worst = np.full(D.nq, -1, np.int64)
    n_gold = np.zeros(D.nq, np.int64)
    for i, g in enumerate(D.C.gold_nodes):
        if len(g) == 0:
            continue
        n_gold[i] = len(g)
        okv, best = [], []
        for x in g:
            mb = flat[ptr[x]:ptr[x + 1]]
            okv.append(bool(ev[i, mb].any()))
            best.append(int(pos[i, mb].min()))
        reached[i] = all(okv)
        worst[i] = max(best)
    fail = (all50 == 0) & mA
    dec = {"failures_A": int(fail.sum()), "UNREACHED": int((fail & ~reached).sum()), "REACHED_WEAK": int((fail & reached).sum()),
           "reached_rate_A": round(float(reached[mA].mean()), 4),
           "worst_gold_position_among_failures": R.stat(worst[fail]) if fail.any() else None,
           "failures_with_worst_le_100": int((fail & (worst < 100)).sum()), "failures_with_worst_le_200": int((fail & (worst < 200)).sum()),
           "failures_with_hub_gold": int((fail & q_with_hub_gold).sum()), "failures_with_degree_0_gold": int((fail & q_with_d0_gold).sum()),
           "gold_nodes_per_query_A": R.stat(n_gold[mA])}
    hard_fail = (base == 0) & mA
    both = {"fail_both": int((fail & hard_fail).sum()), "fail_R2_only": int((fail & ~hard_fail).sum()), "fail_hard_only": int((~fail & hard_fail).sum())}
    fail_le = (all_le == 0) & mA
    both_le = {"fail_both": int((fail_le & hard_fail).sum()), "fail_R2_only": int((fail_le & ~hard_fail).sum()), "fail_hard_only": int((~fail_le & hard_fail).sum())}

    # ---- statistics (DEV_A only)
    vs = "vs_" + gate_label
    cells = {
        "R2_P50": {"ALL": R.rates(all50, masks), "ANY": R.rates(any50, masks), vs: R.compare(base, all50, masks),
                   "exposure_unique_A": R.stat(u50, mA), "exposure_nominal_A": R.stat(nom50, mA), "blocks": P_MAIN,
                   "queries_A_with_unique_le_hard": int(((u50 <= exp) & mA).sum()), "unique_over_hard_ratio_A": round(float(u50[mA].mean() / exp[mA].mean()), 4)},
        "R2_MATCHED_LE (signal)": {"ALL": R.rates(all_le, masks), "ANY": R.rates(any_le, masks), vs: R.compare(base, all_le, masks),
                                   "blocks_used_A": R.stat(L_le, mA), "exposure_unique_A": R.stat(U_le, mA),
                                   "rule": "largest fused prefix with unique exposure <= the %s P50 exposure of the same query" % gate_label},
        "R2_MATCHED_GE (informational)": {"ALL": R.rates(all_ge, masks), "ANY": R.rates(any_ge, masks), vs: R.compare(base, all_ge, masks),
                                          "blocks_used_A": R.stat(L_ge, mA), "exposure_unique_A": R.stat(U_ge, mA), "overshoot_nodes_A": R.stat(U_ge - exp, mA)},
        "R2_HOME_ONLY_VOTE (diagnostic)": {"ALL": R.rates(all_home, masks), "ANY": R.rates(any_home, masks), vs: R.compare(base, all_home, masks),
                                           "vs_R2_P50": R.compare(all50, all_home, masks), "exposure_unique_A": R.stat(u_home, mA)},
        "OWNERS_ONLY_P50 (R = 1 control)": {"ALL": R.rates(all_o50, masks), "ANY": R.rates(any_o50, masks), vs: R.compare(base, all_o50, masks),
                                            "exposure_unique_A": R.stat(u_o50, mA), "blocks": P_MAIN, "note": "the same owner partition (k = 2 k_f, degree-0 placed) without alternates: home vote, home serving"},
        "OWNERS_ONLY_MATCHED_LE (R = 1 control)": {"ALL": R.rates(all_o_le, masks), "ANY": R.rates(any_o_le, masks), vs: R.compare(base, all_o_le, masks),
                                                   "blocks_used_A": R.stat(Lo_le, mA), "exposure_unique_A": R.stat(Uo_le, mA)},
        "HARD_OWN_BLOCK_VOTE (diagnostic)": {"ALL": R.rates(all_own, masks), "ANY": R.rates(any_own, masks), vs: R.compare(base, all_own, masks),
                                             "exposure_A": R.stat(exp_own, mA), "partition": gate_cache},
        gate_label: {"ALL": R.rates(base, masks), "exposure_A": R.stat(exp, mA), "npart": int(D.npart), "cache": gate_cache, "role": "GATE"},
    }
    cells["R2_MATCHED_LE (signal)"]["vs_OWNERS_ONLY_MATCHED_LE (alternates' value at equal exposure; informational)"] = R.compare(all_o_le, all_le, masks)
    cells["R2_P50"]["vs_OWNERS_ONLY_P50 (informational)"] = R.compare(all_o50, all50, masks)
    if D2 is not None:
        cells["R2_P50"]["vs_HARD_PHG_BASE (contrast)"] = R.compare(base2, all50, masks)
        cells["R2_MATCHED_LE (signal)"]["vs_HARD_PHG_BASE (contrast)"] = R.compare(base2, all_le, masks)
        cells["HARD_PHG_BASE (contrast)"] = {"ALL": R.rates(base2, masks), "exposure_A": R.stat(exp2, mA), "npart": int(D2.npart), "cache": contrast_cache, "role": "contrast only"}

    # ---- metaqa: the section-19 V1 cells recomputed (asserted equal to the frozen record) and compared with R2 (informational)
    v1 = None
    if ds == "metaqa":
        t = time.time()
        rec19 = json.load(open(os.path.join(OUT, "vcut_replay_A_metaqa.json"), encoding="utf-8"))
        assert sha_file(os.path.join(OUT, "vcut_replay_A_metaqa.json")) == pre["records_read_only"]["vcut_replay_A_metaqa.json"]["sha256"]
        cap_p = os.path.join(X.REPO, pre["v1_substrate_metaqa"]["path"])
        assert sha_file(cap_p) == pre["v1_substrate_metaqa"]["sha256"]
        z = np.load(cap_p).astype(np.int64)
        assert len(z) == Gr.E and z.min() >= 0 and z.max() < K_V1
        mem1, lam1, home1, ties1, sizes1, blocks1 = R.memberships(Gr, z, K_V1)
        mem_vc1 = R.vote_table(mem1, home1, hub)
        R_v1 = R.fused_rank(D, mem_vc1, K_V1)
        sel1 = [R_v1[i, :P_MAIN] for i in range(D.nq)]
        all1, any1 = R.served(D, sel1, mem1)
        u1, _ = R.unique_exposure(sel1, blocks1, N)
        L1_le, U1_le, _, _ = R.matched_prefixes(R_v1, blocks1, N, exp)
        sel1_le = [R_v1[i, :L1_le[i]] for i in range(D.nq)]
        all1_le, _ = R.served(D, sel1_le, mem1)
        p50_19 = rec19["cells"]["VCUT_P50"]
        le_19 = rec19["cells"]["VCUT_MATCHED_LE"]
        got = {"VCUT_P50_ALL_all": round(float(all1[mA].mean()), 4), "VCUT_P50_unique_mean": round(float(u1[mA].mean()), 2),
               "VCUT_MATCHED_LE_ALL_all": round(float(all1_le[mA].mean()), 4), "VCUT_MATCHED_LE_blocks_mean": round(float(L1_le[mA].mean()), 2)}
        want = {"VCUT_P50_ALL_all": p50_19["ALL"]["all"], "VCUT_P50_unique_mean": p50_19["exposure_unique_A"]["mean"],
                "VCUT_MATCHED_LE_ALL_all": le_19["ALL"]["all"], "VCUT_MATCHED_LE_blocks_mean": le_19["blocks_used_A"]["mean"]}
        assert got == want, (got, want)
        v1 = {"reproduced_section_19": got, "RF_V1": round(float(lam1.sum()) / N, 4), "k_V1": K_V1, "seconds": round(time.time() - t, 1),
              "V1_P50": {"ALL": R.rates(all1, masks), "exposure_unique_A": R.stat(u1, mA)},
              "V1_MATCHED_LE": {"ALL": R.rates(all1_le, masks), "blocks_used_A": R.stat(L1_le, mA), "exposure_unique_A": R.stat(U1_le, mA)},
              "R2_P50_vs_V1_P50 (informational)": R.compare(all1, all50, masks),
              "R2_MATCHED_LE_vs_V1_MATCHED_LE (informational)": R.compare(all1_le, all_le, masks),
              "note": "R = 2 bounded (RF %.3f) against the unbounded V1 (RF %.3f, k = 1004, CAP100); not a decision cell" % (rep["RF"], float(lam1.sum()) / N)}
        log("V1 reproduced: %s | R2 vs V1: P50 +%d/-%d p=%.3g, matched +%d/-%d p=%.3g" % (
            json.dumps(got), v1["R2_P50_vs_V1_P50 (informational)"]["all"]["gained"], v1["R2_P50_vs_V1_P50 (informational)"]["all"]["lost"], v1["R2_P50_vs_V1_P50 (informational)"]["all"]["p"],
            v1["R2_MATCHED_LE_vs_V1_MATCHED_LE (informational)"]["all"]["gained"], v1["R2_MATCHED_LE_vs_V1_MATCHED_LE (informational)"]["all"]["lost"], v1["R2_MATCHED_LE_vs_V1_MATCHED_LE (informational)"]["all"]["p"]))

    # ---- verdict (the pre-registered rule)
    p50 = cells["R2_P50"][vs]["all"]
    le = cells["R2_MATCHED_LE (signal)"][vs]["all"]
    mean_u = cells["R2_P50"]["exposure_unique_A"]["mean"]
    mean_h = cells[gate_label]["exposure_A"]["mean"]
    primary = label(p50)
    flag = "EXPOSURE_LE_HARD" if mean_u <= mean_h else "EXPOSURE_ABOVE_HARD"
    matched = label(le)
    efficient = primary != "LOSS" and flag == "EXPOSURE_LE_HARD"
    alt_value = label(cells["R2_MATCHED_LE (signal)"]["vs_OWNERS_ONLY_MATCHED_LE (alternates' value at equal exposure; informational)"]["all"])
    ver = {"VERDICT": "%s_%s|MATCHED_%s" % (primary, flag, matched), "primary_label_R2_P50": primary, "exposure_flag": flag, "matched_label_R2_MATCHED_LE": matched,
           "EFFICIENT": efficient, "alpha": ALPHA, "gate": gate_label, "R2_P50_vs_gate": p50, "R2_MATCHED_LE_vs_gate": le,
           "R2_mean_unique_exposure_P50": mean_u, "hard_mean_exposure": mean_h, "alternates_value_at_matched_exposure (informational)": alt_value,
           "status": "post-hoc structural-mechanism result on DEV_A; not a confirmation; not promotable; served L1 unchanged"}
    if ds == "metaqa":
        ver["RETAINED"] = matched == "GAIN"
        ver["role"] = "metaqa: RETAINED iff R2_MATCHED_LE vs HARD_MTK_BASE is GAIN; EFFICIENT iff R2_P50 != LOSS and exposure LE_HARD"
    else:
        ver["SAFE"] = matched != "LOSS"
        ver["role"] = "text: SAFE iff R2_MATCHED_LE vs gate != LOSS; EFFICIENT iff R2_P50 != LOSS and exposure LE_HARD"
    res = {"RECORD": "BOUNDED_VERTEXCUT_R2_REPLAY", "dataset": ds, "cache": gate_cache, "contrast_cache": contrast_cache, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": pin(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")), "substrate": pin(r2_p), "substrate_record": pin(r2_p[:-4] + ".json"),
           "code": pin(os.path.abspath(__file__)), "lib": pin(os.path.join(HERE, "_l1c_br2_lib.py")),
           "n_DEV_A": int(mA.sum()), "hop_counts_A": {kk: int(vv.sum()) for kk, vv in masks.items()},
           "representation": rep, "shares": shares, "cells": cells, "reach_decomposition_R2_P50": dec,
           "failure_overlap_with_gate_A": {"R2_P50": both, "R2_MATCHED_LE": both_le}, "v1_metaqa": v1, "verdict": ver,
           "constants": {"K0": K0, "K_LOCK": K_LOCK, "P_MAIN": P_MAIN, "R": L.R_MAX, "C": C, "k": k}, "seconds": round(time.time() - T0, 1)}
    S.wj(fp_out, res)
    log("%s R2_P50 ALL(A) %.4f (gate %.4f; +%d/-%d p=%.3g) unique %.1f vs hard %.1f | MATCHED_LE ALL %.4f (+%d/-%d p=%.3g; %.1f blocks, %.1f nodes) | "
        "GE %.4f | HOME_ONLY %.4f | OWNERS_ONLY P50 %.4f (%.1f nodes) MATCHED %.4f (%.1f blocks) | OWN_BLOCK %.4f -> %s (alternates at matched: %s) (%.0fs) -> %s" % (
            ds, cells["R2_P50"]["ALL"]["all"], cells[gate_label]["ALL"]["all"], p50["gained"], p50["lost"], p50["p"], mean_u, mean_h,
            cells["R2_MATCHED_LE (signal)"]["ALL"]["all"], le["gained"], le["lost"], le["p"], cells["R2_MATCHED_LE (signal)"]["blocks_used_A"]["mean"],
            cells["R2_MATCHED_LE (signal)"]["exposure_unique_A"]["mean"], cells["R2_MATCHED_GE (informational)"]["ALL"]["all"], cells["R2_HOME_ONLY_VOTE (diagnostic)"]["ALL"]["all"],
            cells["OWNERS_ONLY_P50 (R = 1 control)"]["ALL"]["all"], cells["OWNERS_ONLY_P50 (R = 1 control)"]["exposure_unique_A"]["mean"],
            cells["OWNERS_ONLY_MATCHED_LE (R = 1 control)"]["ALL"]["all"], cells["OWNERS_ONLY_MATCHED_LE (R = 1 control)"]["blocks_used_A"]["mean"],
            cells["HARD_OWN_BLOCK_VOTE (diagnostic)"]["ALL"]["all"], ver["VERDICT"], alt_value, time.time() - T0, os.path.relpath(fp_out, X.REPO)))
    for nm in masks:
        if nm == "all":
            continue
        c = cells["R2_P50"][vs][nm]
        d = cells["R2_MATCHED_LE (signal)"][vs][nm]
        log("  %s n %d: P50 %.4f vs hard %.4f (+%d/-%d p=%.3g) | MATCHED_LE %.4f (+%d/-%d p=%.3g)" % (nm, c["n"], c["cell_ALL"], c["base_ALL"], c["gained"], c["lost"], c["p"],
                                                                                                      d["cell_ALL"], d["gained"], d["lost"], d["p"]))
    log("  reach: %s" % json.dumps(dec))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
