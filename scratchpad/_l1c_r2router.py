"""R2_LEGACY_ROUTER (thirteenth ruling, 2026-09-16; PREREGISTRATION_R2_LEGACY_ROUTER.json written first): the frozen bounded-overlap
substrate of section 22 (every node in <= 2 blocks, cap C = 100, k = 2 k_f; memberships read from the pinned parts/<ds>__BR2_k<k>__R2.npz,
NOT rebuilt) served through the CANONICAL frozen router -- each top-100 hit votes for its own blocks AND for the blocks of its directed
STRUCT out-neighbours (the legacy rule the served hard BASE uses; _l1s_core.Data.legacy_mem), applied to the R2 vote table of section 22
(hub -> H(v) only, non-hub -> M(v)); same P50, same RRF, same serving (M(g) meets the selection), same exposure and matched-budget
contracts, imported unchanged.  Partitioning and routing are separate mechanisms: section 22 removed the router with the substrate;
this arm puts the router back on the bounded substrate.  Post-hoc DEV_A mechanism test: no DEV_B, no TEST, nothing promoted, served L1
unchanged, no composition with BALANCED_H2_PATCH1 / H2_PATCH1 / SAFE.

Cells: R2ROUTER_P50 (PRIMARY: ALL-gold at 50 fused blocks, unique-node exposure, McNemar vs the GATE), R2ROUTER_MATCHED_LE (SECONDARY:
equal unique-node budget), R2ROUTER_MATCHED_GE (informational), R2_MEMBERSHIP_VOTE_P50 (the section-22 cell recomputed and asserted
equal to its record: the router's value on the same substrate), OWNERS_LEGACY_P50 / OWNERS_LEGACY_MATCHED_LE (R = 1 control: the same
owner partition without alternates under the same router), the GATE / contrast BASE; diagnostics: hub-gold reach, non-hub reach, votes
per hit and blocks with evidence for the gate table, the section-22 table, the router table and the control (on metaqa asserted equal
to the seen br2_why_A_metaqa.json numbers), the UNREACHED / REACHED_WEAK decomposition of the primary cell's failures, failure overlap.

    python -u scratchpad/_l1c_r2router.py <metaqa|squad|musique>   -> results/L1_COVPART/r2router_A_<ds>.json (+ .log via the shell)
    python -u scratchpad/_l1c_r2router.py metaqa --dry             -> no record, no ranking, no coverage: builds the tables, asserts that
                                                                      the spread rule on the hard own-block table reproduces the served
                                                                      legacy table exactly, and that the four reach tables reproduce the
                                                                      seen diagnostic numbers (run before the pre-registration)
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
import _l1c_br2_why as W  # noqa: E402  (spread_table + reach: the exact functions whose reach numbers justified this arm; main() never called)

X = S.X
K0, K_LOCK, P_MAIN = S.K0, S.K_LOCK, S.P_MAIN
OUT = V.OUT
PDIR = V.PDIR
log = S.log
sha_file, pin = L.sha_file, L.pin
ALPHA = 0.01
GATE = {"metaqa": ("HARD_MTK_BASE", "metaqa", "metaqa_phg"), "squad": ("HARD_MTK_BASE", "squad", "squad_phg"), "musique": ("HARD_PHG_BASE", "musique", None)}
PRE = os.path.join(OUT, "PREREGISTRATION_R2_LEGACY_ROUTER.json")
NEW_MODULES = ("_l1c_r2router.py", "_l1c_r2router_summary.py")
PRIMARY = "R2ROUTER_P50 (primary)"
SECONDARY = "R2ROUTER_MATCHED_LE (secondary)"
T0 = time.time()

spread_table = W.spread_table
reach = W.reach


def label(cmp):
    if cmp["lost"] > cmp["gained"] and cmp["p"] < ALPHA:
        return "LOSS"
    if cmp["gained"] > cmp["lost"] and cmp["p"] < ALPHA:
        return "GAIN"
    return "NEUTRAL"


def exposure_hard(D, sel_rows):
    return np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in sel_rows])


def same_table(a, b):
    return len(a[0]) == len(b[0]) and len(a[1]) == len(b[1]) and (a[0] == b[0]).all() and (a[1] == b[1]).all()


def main(ds, dry=False):
    gate_label, gate_cache, contrast_cache = GATE[ds]
    pre = None
    if not dry:
        pre = json.load(open(PRE, encoding="utf-8"))
        for mod in NEW_MODULES:
            assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
        for mod, pn in pre["code"]["imports_unchanged"].items():
            assert sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
        for nm in [gate_cache] + ([contrast_cache] if contrast_cache else []):
            assert sha_file(X.CACHES[nm]) == pre["caches_read_only"][nm]["sha256"], "cache %s changed" % nm
        for nm, pn in pre["records_read_only"].items():
            assert sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "record %s changed" % nm
        fp_out = os.path.join(OUT, "r2router_A_%s.json" % ds)
        assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    else:
        assert ds == "metaqa", "the dry run is metaqa only (reach numbers already seen there); no text-set number before the pre-registration"
        fp_out = None

    D = G.Data(gate_cache, dense_fp32=False)
    D2 = G.Data(contrast_cache, dense_fp32=False) if (contrast_cache and not dry) else None
    if D2 is not None:
        assert (D.rows == D2.rows).all() and D.C.qids == D2.C.qids, "the two served caches must hold the same rows"
    mA = D.C.split == "A"
    masks = R.hop_masks(D, mA)
    log("%s: gate %s (%s) contrast %s | nq %d DEV_A %d hops %s%s" % (ds, gate_label, gate_cache, contrast_cache, D.nq, int(mA.sum()), {kk: int(vv.sum()) for kk, vv in masks.items()}, " | DRY" if dry else ""))

    # ---- the frozen bounded-overlap substrate (read only; asserted against its record and the pre-registration)
    Gr = V.Graph(ds)
    assert Gr.N == D.N and Gr.C == 100
    r2_p, home, alt, k = L.load_r2(ds, Gr)
    r2_rec = json.load(open(r2_p[:-4] + ".json", encoding="utf-8"))
    assert r2_rec["output"]["sha256"] == sha_file(r2_p) and r2_rec["representation"]["k"] == k == L.k_of(Gr)
    if pre is not None:
        assert Gr.keys_sha == pre["graphs_frozen"][ds]["struct_keys_sha256"] and sha_file(r2_p) == pre["substrates_frozen"][ds]["sha256"]
    N, C = Gr.N, int(Gr.C)
    deg0 = Gr.deg == 0
    hub = Gr.hub
    t = time.time()
    mem_all, Y, blocks_csc, sizes = L.membership(home, alt, N, k)
    lam = np.diff(mem_all[0])
    assert lam.max() <= L.R_MAX and lam.min() == 1 and sizes.max() <= C and (alt[deg0] == -1).all() and (home >= 0).all()
    mem_vc = R.vote_table(mem_all, home, hub)      # section 22's vote table: hub -> H(v), non-hub -> M(v)
    mem_home = R.home_table(home)
    none = np.full(N, -1, np.int64)
    _, _, blocks_own, sizes_own = L.membership(home, none, N, k)

    # ---- the canonical router: own votes + the votes of the directed STRUCT out-neighbours (legacy rule) on the multi-membership tables
    xo, ao = D.cd.struct_csr(directed=True)
    D.cd._csr.clear()
    sp_hard = spread_table(D.mem_hard, xo, ao, N, D.npart)
    assert same_table(sp_hard, D.mem), "the spread rule on the hard own-block table must reproduce the served legacy vote table exactly"
    sp_r2 = spread_table(mem_vc, xo, ao, N, k)      # R2_LEGACY_ROUTER's vote table
    sp_own = spread_table(mem_home, xo, ao, N, k)   # the R = 1 control's vote table
    log("tables: legacy(hard) reproduced (%d entries) | router %d entries (%.2f per node) | control %d entries | membership %d (RF %.4f) (%.0fs)" % (
        len(D.mem[1]), len(sp_r2[1]), len(sp_r2[1]) / N, len(sp_own[1]), len(mem_all[1]), lam.sum() / N, time.time() - t))

    # ---- reach diagnostics (evidence through the vote table; gold served through the membership table)
    t = time.time()
    reach_tables = {}
    reach_all = {}
    for name, (vote, memb, npart) in (("HARD_LEGACY (gate table)", (D.mem, D.mem_hard, D.npart)),
                                      ("R2_MEMBERSHIP_VOTE (section 22 table)", (mem_vc, mem_all, k)),
                                      ("R2_LEGACY_ROUTER", (sp_r2, mem_all, k)),
                                      ("OWNERS_LEGACY (R = 1 control)", (sp_own, mem_home, k))):
        reach_tables[name], reach_all[name] = reach(D, vote, memb, npart, mA, masks, hub)
    seen = None
    if ds == "metaqa":
        why = json.load(open(os.path.join(OUT, "br2_why_A_metaqa.json"), encoding="utf-8"))["tables"]
        seen = {"HARD_LEGACY (gate table)": "HARD_LEGACY (served BASE: own block + directed out-neighbour blocks; k 432)",
                "R2_MEMBERSHIP_VOTE (section 22 table)": "R2_k864 (this ruling: home + one alternate; votes hub -> home only; served through both blocks)",
                "R2_LEGACY_ROUTER": "R2_k864 + legacy out-neighbour spread (REACH ONLY; not a cell; the natural follow-up, not evaluated)",
                "OWNERS_LEGACY (R = 1 control)": "OWNERS_ONLY_k864 + legacy out-neighbour spread (REACH ONLY; not a cell)"}
        for a, b in seen.items():
            assert reach_tables[a] == why[b], ("reach diagnostic not reproduced", a)
        log("reach reproduced against br2_why_A_metaqa.json for the four tables")
    for name, r in reach_tables.items():
        log("  reach %-40s votes/hit %s reached_all %s hub %s non-hub %s evidence-blocks %s" % (name, r["votes_per_hit_mean (distinct blocks)"], r["reached_all_gold_rate"],
                                                                                              r["gold_node_reached_rate_A_by_hubness"]["hub"], r["gold_node_reached_rate_A_by_hubness"]["non_hub"], r["blocks_with_evidence_per_query_mean"]))
    log("reach diagnostics (%.0fs)" % (time.time() - t))
    if dry:
        log("DRY OK: tables built, legacy table reproduced, reach reproduced; no ranking, no coverage, no record (%.0fs)" % (time.time() - T0))
        return None

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

    # ---- routing (frozen numerics)
    t = time.time()
    R_sp = R.fused_rank(D, sp_r2, k)          # R2_LEGACY_ROUTER
    R_vc = R.fused_rank(D, mem_vc, k)         # section 22 (membership-only vote), for the router's value
    R_ownsp = R.fused_rank(D, sp_own, k)      # R = 1 control under the same router
    log("rankings computed (%.0fs)" % (time.time() - t))
    sel50 = [R_sp[i, :P_MAIN] for i in range(D.nq)]
    all50, any50 = R.served(D, sel50, mem_all)
    u50, nom50 = R.unique_exposure(sel50, blocks_csc, N)
    L_le, U_le, L_ge, U_ge = R.matched_prefixes(R_sp, blocks_csc, N, exp)
    sel_le = [R_sp[i, :L_le[i]] for i in range(D.nq)]
    sel_ge = [R_sp[i, :L_ge[i]] for i in range(D.nq)]
    all_le, any_le = R.served(D, sel_le, mem_all)
    all_ge, any_ge = R.served(D, sel_ge, mem_all)
    assert (U_le <= exp).all() and (L_ge >= L_le).all()
    # section 22 reproduced (membership-only vote, P50)
    sel_vc = [R_vc[i, :P_MAIN] for i in range(D.nq)]
    all_vc, any_vc = R.served(D, sel_vc, mem_all)
    u_vc, _ = R.unique_exposure(sel_vc, blocks_csc, N)
    rec22 = json.load(open(os.path.join(OUT, "br2_replay_A_%s.json" % ds), encoding="utf-8"))
    got22 = {"R2_P50_ALL_all": round(float(all_vc[mA].mean()), 4), "R2_P50_unique_mean": round(float(u_vc[mA].mean()), 2)}
    want22 = {"R2_P50_ALL_all": rec22["cells"]["R2_P50"]["ALL"]["all"], "R2_P50_unique_mean": rec22["cells"]["R2_P50"]["exposure_unique_A"]["mean"]}
    assert got22 == want22, (got22, want22)
    # R = 1 control under the router: owner blocks, home serving
    sel_o50 = [R_ownsp[i, :P_MAIN] for i in range(D.nq)]
    all_o50, any_o50 = R.served(D, sel_o50, mem_home)
    u_o50, _ = R.unique_exposure(sel_o50, blocks_own, N)
    Lo_le, Uo_le, _, _ = R.matched_prefixes(R_ownsp, blocks_own, N, exp)
    sel_o_le = [R_ownsp[i, :Lo_le[i]] for i in range(D.nq)]
    all_o_le, any_o_le = R.served(D, sel_o_le, mem_home)
    assert (Uo_le <= exp).all()

    # ---- reach decomposition of the primary cell's failures
    ev = R.evidence(D, sp_r2, k)
    pos = G.positions(R_sp)
    ptr, flat = mem_all
    reached = np.zeros(D.nq, bool)
    worst = np.full(D.nq, -1, np.int64)
    n_gold = np.zeros(D.nq, np.int64)
    q_with_hub_gold = np.array([bool(hub[g].any()) for g in D.C.gold_nodes])
    q_with_d0_gold = np.array([bool(deg0[g].any()) for g in D.C.gold_nodes])
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
    assert (reached == reach_all["R2_LEGACY_ROUTER"]).all()
    fail = (all50 == 0) & mA
    dec = {"failures_A": int(fail.sum()), "UNREACHED": int((fail & ~reached).sum()), "REACHED_WEAK": int((fail & reached).sum()),
           "reached_rate_A": round(float(reached[mA].mean()), 4),
           "worst_gold_position_among_failures": R.stat(worst[fail]) if fail.any() else None,
           "failures_with_worst_le_100": int((fail & (worst < 100)).sum()), "failures_with_worst_le_200": int((fail & (worst < 200)).sum()),
           "failures_with_hub_gold": int((fail & q_with_hub_gold).sum()), "failures_with_degree_0_gold": int((fail & q_with_d0_gold).sum()),
           "gold_nodes_per_query_A": R.stat(n_gold[mA])}
    hard_fail = (base == 0) & mA
    both = {"fail_both": int((fail & hard_fail).sum()), "fail_router_only": int((fail & ~hard_fail).sum()), "fail_hard_only": int((~fail & hard_fail).sum())}
    fail_le = (all_le == 0) & mA
    both_le = {"fail_both": int((fail_le & hard_fail).sum()), "fail_router_only": int((fail_le & ~hard_fail).sum()), "fail_hard_only": int((~fail_le & hard_fail).sum())}

    # ---- statistics (DEV_A only)
    vs = "vs_" + gate_label
    cells = {
        PRIMARY: {"ALL": R.rates(all50, masks), "ANY": R.rates(any50, masks), vs: R.compare(base, all50, masks),
                  "exposure_unique_A": R.stat(u50, mA), "exposure_nominal_A": R.stat(nom50, mA), "blocks": P_MAIN,
                  "queries_A_with_unique_lt_hard": int(((u50 < exp) & mA).sum()), "unique_over_hard_ratio_A": round(float(u50[mA].mean() / exp[mA].mean()), 4),
                  "vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)": R.compare(all_vc, all50, masks),
                  "vs_OWNERS_LEGACY_P50 (the alternates' value under the router; informational)": R.compare(all_o50, all50, masks)},
        SECONDARY: {"ALL": R.rates(all_le, masks), "ANY": R.rates(any_le, masks), vs: R.compare(base, all_le, masks),
                    "blocks_used_A": R.stat(L_le, mA), "exposure_unique_A": R.stat(U_le, mA),
                    "rule": "largest fused prefix with unique exposure <= the %s P50 exposure of the same query" % gate_label,
                    "vs_OWNERS_LEGACY_MATCHED_LE (the alternates' value under the router at equal exposure; informational)": R.compare(all_o_le, all_le, masks)},
        "R2ROUTER_MATCHED_GE (informational)": {"ALL": R.rates(all_ge, masks), "ANY": R.rates(any_ge, masks), vs: R.compare(base, all_ge, masks),
                                                "blocks_used_A": R.stat(L_ge, mA), "exposure_unique_A": R.stat(U_ge, mA), "overshoot_nodes_A": R.stat(U_ge - exp, mA)},
        "R2_MEMBERSHIP_VOTE_P50 (section 22 reproduced)": {"ALL": R.rates(all_vc, masks), "ANY": R.rates(any_vc, masks), vs: R.compare(base, all_vc, masks),
                                                          "exposure_unique_A": R.stat(u_vc, mA), "reproduced": got22, "record": "br2_replay_A_%s.json / R2_P50" % ds},
        "OWNERS_LEGACY_P50 (R = 1 control)": {"ALL": R.rates(all_o50, masks), "ANY": R.rates(any_o50, masks), vs: R.compare(base, all_o50, masks),
                                              "exposure_unique_A": R.stat(u_o50, mA), "blocks": P_MAIN,
                                              "note": "the same owner partition (k = 2 k_f, degree-0 placed) without alternates, voted through the same legacy spread; home serving; owner blocks for exposure"},
        "OWNERS_LEGACY_MATCHED_LE (R = 1 control)": {"ALL": R.rates(all_o_le, masks), "ANY": R.rates(any_o_le, masks), vs: R.compare(base, all_o_le, masks),
                                                     "blocks_used_A": R.stat(Lo_le, mA), "exposure_unique_A": R.stat(Uo_le, mA)},
        gate_label: {"ALL": R.rates(base, masks), "exposure_A": R.stat(exp, mA), "npart": int(D.npart), "cache": gate_cache, "role": "GATE"},
    }
    if D2 is not None:
        cells[PRIMARY]["vs_HARD_PHG_BASE (contrast)"] = R.compare(base2, all50, masks)
        cells[SECONDARY]["vs_HARD_PHG_BASE (contrast)"] = R.compare(base2, all_le, masks)
        cells["HARD_PHG_BASE (contrast)"] = {"ALL": R.rates(base2, masks), "exposure_A": R.stat(exp2, mA), "npart": int(D2.npart), "cache": contrast_cache, "role": "contrast only"}

    # ---- the ladder (read-only numbers from the frozen records; no recompute of V1)
    ladder = {"HARD_OWN_BLOCK_VOTE": rec22["cells"]["HARD_OWN_BLOCK_VOTE (diagnostic)"]["ALL"]["all"], "OWNERS_ONLY_P50 (section 22)": rec22["cells"]["OWNERS_ONLY_P50 (R = 1 control)"]["ALL"]["all"],
              "R2_P50 (section 22)": rec22["cells"]["R2_P50"]["ALL"]["all"], "R2_MATCHED_LE (section 22)": rec22["cells"]["R2_MATCHED_LE (signal)"]["ALL"]["all"],
              gate_label + " (legacy spread)": rec22["cells"][gate_label]["ALL"]["all"],
              "OWNERS_LEGACY_P50 (this record)": cells["OWNERS_LEGACY_P50 (R = 1 control)"]["ALL"]["all"], "OWNERS_LEGACY_MATCHED_LE (this record)": cells["OWNERS_LEGACY_MATCHED_LE (R = 1 control)"]["ALL"]["all"],
              "R2ROUTER_P50 (this record)": cells[PRIMARY]["ALL"]["all"], "R2ROUTER_MATCHED_LE (this record)": cells[SECONDARY]["ALL"]["all"]}
    if ds == "metaqa":
        ladder["V1_P50 (section 19)"] = rec22["v1_metaqa"]["V1_P50"]["ALL"]["all"]
        ladder["V1_MATCHED_LE (section 19)"] = rec22["v1_metaqa"]["V1_MATCHED_LE"]["ALL"]["all"]

    # ---- verdict (the pre-registered rule)
    p50 = cells[PRIMARY][vs]["all"]
    le = cells[SECONDARY][vs]["all"]
    mean_u = cells[PRIMARY]["exposure_unique_A"]["mean"]
    mean_h = cells[gate_label]["exposure_A"]["mean"]
    primary = label(p50)
    matched = label(le)
    flag = "EXPOSURE_BELOW_HARD" if mean_u < mean_h else "EXPOSURE_NOT_BELOW_HARD"
    ver = {"VERDICT": "%s_%s|MATCHED_%s" % (primary, flag, matched), "primary_label_R2ROUTER_P50": primary, "exposure_flag": flag, "matched_label_R2ROUTER_MATCHED_LE": matched,
           "alpha": ALPHA, "gate": gate_label, "R2ROUTER_P50_vs_gate": p50, "R2ROUTER_MATCHED_LE_vs_gate": le,
           "router_mean_unique_exposure_P50": mean_u, "hard_mean_exposure": mean_h, "unique_over_hard_ratio": round(mean_u / mean_h, 4),
           "router_value_vs_section22_R2_P50 (informational)": label(cells[PRIMARY]["vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)"]["all"]),
           "alternates_value_under_router_P50 (informational)": label(cells[PRIMARY]["vs_OWNERS_LEGACY_P50 (the alternates' value under the router; informational)"]["all"]),
           "alternates_value_under_router_matched (informational)": label(cells[SECONDARY]["vs_OWNERS_LEGACY_MATCHED_LE (the alternates' value under the router at equal exposure; informational)"]["all"]),
           "status": "post-hoc mechanism result on DEV_A; not a confirmation; not promotable; served L1 unchanged; no composition"}
    if ds == "metaqa":
        ver["GAIN"] = primary == "GAIN"
        ver["EFFICIENT"] = primary != "LOSS" and mean_u < mean_h
        ver["MATCHED_GAIN"] = matched == "GAIN"
        ver["role"] = "metaqa: GAIN iff R2ROUTER_P50 vs HARD_MTK_BASE = GAIN; EFFICIENT iff R2ROUTER_P50 != LOSS and mean unique exposure < the hard mean P50 exposure; MATCHED_GAIN informational"
    else:
        ver["SAFE"] = primary != "LOSS" and matched != "LOSS"
        ver["role"] = "text: SAFE iff R2ROUTER_P50 vs gate != LOSS AND R2ROUTER_MATCHED_LE vs gate != LOSS"
    res = {"RECORD": "R2_LEGACY_ROUTER_REPLAY", "dataset": ds, "cache": gate_cache, "contrast_cache": contrast_cache, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": pin(PRE), "substrate": pin(r2_p), "substrate_record": pin(r2_p[:-4] + ".json"),
           "code": pin(os.path.abspath(__file__)), "spread_and_reach_functions": pin(os.path.join(HERE, "_l1c_br2_why.py")), "lib": pin(os.path.join(HERE, "_l1c_br2_lib.py")),
           "n_DEV_A": int(mA.sum()), "hop_counts_A": {kk: int(vv.sum()) for kk, vv in masks.items()},
           "representation": {"k": k, "C": C, "R": L.R_MAX, "RF": round(float(lam.sum()) / N, 4), "hubs": int(hub.sum()), "degree_0": int(deg0.sum()),
                              "vote_table_entries": {"legacy_hard (gate)": int(len(D.mem[1])), "R2_MEMBERSHIP_VOTE (section 22)": int(len(mem_vc[1])), "R2_LEGACY_ROUTER": int(len(sp_r2[1])), "OWNERS_LEGACY": int(len(sp_own[1]))},
                              "membership_entries": int(len(flat)), "block_size": {"mean": round(float(sizes.mean()), 2), "max": int(sizes.max()), "min": int(sizes.min())},
                              "router": "spread_table(mem_vc, xo, ao, N, k): own vote set (hub -> H(v), non-hub -> M(v)) + the vote sets of the directed STRUCT out-neighbours, deduped; identical rule to _l1s_core.Data.legacy_mem (asserted on the hard partition)"},
           "cells": cells, "reach_tables": reach_tables, "reach_reproduced_on_metaqa": seen is not None, "reach_decomposition_R2ROUTER_P50": dec,
           "failure_overlap_with_gate_A": {"R2ROUTER_P50": both, "R2ROUTER_MATCHED_LE": both_le}, "ladder_ALL_DEV_A": ladder, "verdict": ver,
           "constants": {"K0": K0, "K_LOCK": K_LOCK, "P_MAIN": P_MAIN, "R": L.R_MAX, "C": C, "k": k, "alpha": ALPHA}, "seconds": round(time.time() - T0, 1)}
    S.wj(fp_out, res)
    log("%s R2ROUTER_P50 ALL(A) %.4f (gate %.4f; +%d/-%d p=%.3g) unique %.1f vs hard %.1f (ratio %.3f) | MATCHED_LE %.4f (+%d/-%d p=%.3g; %.1f blocks) | GE %.4f | "
        "section-22 R2_P50 %.4f (router value +%d/-%d p=%.3g) | OWNERS_LEGACY P50 %.4f MATCHED %.4f -> %s (%.0fs) -> %s" % (
            ds, cells[PRIMARY]["ALL"]["all"], cells[gate_label]["ALL"]["all"], p50["gained"], p50["lost"], p50["p"], mean_u, mean_h, mean_u / mean_h,
            cells[SECONDARY]["ALL"]["all"], le["gained"], le["lost"], le["p"], cells[SECONDARY]["blocks_used_A"]["mean"], cells["R2ROUTER_MATCHED_GE (informational)"]["ALL"]["all"],
            cells["R2_MEMBERSHIP_VOTE_P50 (section 22 reproduced)"]["ALL"]["all"],
            cells[PRIMARY]["vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)"]["all"]["gained"],
            cells[PRIMARY]["vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)"]["all"]["lost"],
            cells[PRIMARY]["vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)"]["all"]["p"],
            cells["OWNERS_LEGACY_P50 (R = 1 control)"]["ALL"]["all"], cells["OWNERS_LEGACY_MATCHED_LE (R = 1 control)"]["ALL"]["all"], ver["VERDICT"], time.time() - T0, os.path.relpath(fp_out, X.REPO)))
    for nm in masks:
        if nm == "all":
            continue
        c = cells[PRIMARY][vs][nm]
        d = cells[SECONDARY][vs][nm]
        log("  %s n %d: P50 %.4f vs hard %.4f (+%d/-%d p=%.3g) | MATCHED_LE %.4f (+%d/-%d p=%.3g)" % (nm, c["n"], c["cell_ALL"], c["base_ALL"], c["gained"], c["lost"], c["p"],
                                                                                                      d["cell_ALL"], d["gained"], d["lost"], d["p"]))
    log("  reach decomposition: %s | overlap %s" % (json.dumps(dec), json.dumps(both)))
    return res


if __name__ == "__main__":
    main(sys.argv[1], dry="--dry" in sys.argv[2:])
