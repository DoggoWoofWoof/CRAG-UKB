"""PHASE 15 -- SAFE/F6 boundary-router backward-elimination ablation (Part B).

Branch-level decomposition (F6=STRUCT+RET, F7=STRUCT-only, F8=RET-only, via
_l1ps_router.evaluate's use_s/use_r flags) is ALREADY established by Phase 3
(_l1au_p3.py) and is NOT repeated here. This phase decomposes only the INTERNAL
components of the load-bearing branches, by backward elimination: remove one
component; if the removal causes no significant+practical loss on ANY of the 6
corpora (same McNemar+noise-floor gate as every other phase: HH.mcnemar + HH.FLOOR),
keep it removed and test the next component against that now-simplified baseline;
otherwise put it back and move on. No B sweep, no M sweep (only a single ablation
point per component -- full removal, i.e. M_struct=0 / M_ret=0 / B=50 -- never a
grid search over intermediate values), no new scorer family.

Substrate: the FINAL H4_SK partition (scratchpad/_l1ep/parts/{ds}__H4FAM_SK.npy),
fed through _l1ep_c.rebuild exactly as the shipped _l1ov_eval.selected_blocks()
does, so base_rank/gold_part/part_sizes reflect H4_SK -- not the stale variant_C
partition _l1ps_cache.py's raw cache_{ds}.npz was originally built under (verified
by reading _ta_prepartition.load_topology: it reads variant_C/partition_map.json).
The raw per-node traversal (s_node/s_hop/s_sdir/s_cnt) and retrieval-continuation
(ret_rrf) arrays in that cache are graph/embedding-derived, not partition-choice-
derived, so they carry over unchanged under any hard/npart override -- exactly
_l1ep_c.rebuild's documented contract.

Component -> mechanism -> single-point removal test:
  STRUCT traversal        s_hop-filtered aggregation, hop_cap=1 (drop hop-2/3 evidence)
  STRUCT s_dir            drop the 4th (best-s_dir, max-collapsed) signal from S4's RRF
  STRUCT max collapse     SAME test as s_dir -- s_dir only ever reaches S4 as that one
                          max-collapsed signal (struct_aggregate_full's a[4]); there is no
                          separate way to remove "the collapse" while keeping "the signal"
  STRUCT M_struct=64      M_struct=0 (STRUCT branch contributes zero challengers)
  STRUCT S4               agg="S1" (BEST_NODE_RANK only) in place of S4's 4-signal RRF
  RET M_ret=32            M_ret=0 (RET branch contributes zero challengers)
  RET continuation agg    NO alternate ordering exists for RET in the shipped code (unlike
                          STRUCT's S1-S4, ret_aggregate_full/rfull always orders by first-
                          occurrence rank) -- its only lever is M_ret, tested above
  Boundary top44          B=50 (protection removed entirely, all 50 slots open to competition)
  Boundary B=6            NOT retested -- already the survivor of the round1-4 grid search
                          (results/.../runs/final_B6_S4_F6_Ms64_Mr32.json); retesting the
                          exact value would be the B sweep this phase is told not to run
  Boundary F6 competition sel_base (bnd kept as-is, zero competition) vs sel_f6 (shipped)

  python scratchpad/_l1au_p15_safe_ablate.py run <ds>
  python scratchpad/_l1au_p15_safe_ablate.py run_all
  python scratchpad/_l1au_p15_safe_ablate.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1kb_core as KB
import _l1ep_c as EC
import _l1hu_hard as HH

ROOT = RT.ROOT
DS = RT.DSETS
K0, P = RT.K0, RT.P
EPARTS = "scratchpad/_l1ep/parts"
AOUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)

SHIP = dict(M_struct=64, M_ret=32, agg="S4", hop_cap=None, B=6, competition="F6")
STEP_ORDER = ["TRAVERSAL_HOP1", "SDIR_DROP", "MSTRUCT_ZERO", "AGG_S1", "MRET_ZERO",
             "F6_COMPETITION_OFF", "TOP44_OFF_B50"]


def order_s4_nosdir(agg):
    ps = list(agg)
    rk = {}
    for key in (lambda p: agg[p][0], lambda p: -agg[p][1], lambda p: agg[p][3]):
        for r, p in enumerate(sorted(ps, key=lambda p: (key(p), p))):
            rk[p] = rk.get(p, 0.0) + 1.0 / (K0 + r)
    return [p for p in sorted(ps, key=lambda p: (-rk[p], p))]


def build_struct_agg(z, hard, M_struct, nq, hop_cap=None):
    aggs = []
    for qi in range(nq):
        agg = {}
        sn = z["s_node"][qi]; sh = z["s_hop"][qi]; sd = z["s_sdir"][qi]; sc = z["s_cnt"][qi]
        for jj in range(min(len(sn), M_struct)):
            v = int(sn[jj])
            if v < 0:
                break
            if hop_cap is not None and int(sh[jj]) > hop_cap:
                continue
            p = int(hard[v])
            if p < 0:
                continue
            a = agg.get(p)
            if a is None:
                agg[p] = [jj, 1, 1.0 / (K0 + jj), int(sh[jj]), float(sd[jj]), int(sc[jj])]
            else:
                a[1] += 1; a[2] += 1.0 / (K0 + jj)
                a[3] = min(a[3], int(sh[jj])); a[4] = max(a[4], float(sd[jj])); a[5] += int(sc[jj])
        aggs.append(agg)
    return aggs


def order_agg(aggs, mode):
    if mode == "S4_NOSDIR":
        return [order_s4_nosdir(a) for a in aggs]
    return [RT.order_struct(a, mode) for a in aggs]


def struct_sfull(z, hard, nq, M_struct, hop_cap, agg_mode):
    if M_struct == 0:
        return [[] for _ in range(nq)]
    aggs = build_struct_agg(z, hard, M_struct, nq, hop_cap=hop_cap)
    return order_agg(aggs, agg_mode)


def ret_rfull(z, hard, nq, M_ret):
    if M_ret == 0:
        return [[] for _ in range(nq)]
    out = []
    for qi in range(nq):
        agg = RT.ret_aggregate_full(z, qi, hard, M_ret)
        out.append([p for p, _ in sorted(agg.items(), key=lambda kv: (kv[1][0], kv[0]))])
    return out


def _prep(ds, log=log):
    hard = np.load("%s/%s__H4FAM_SK.npy" % (EPARTS, ds))
    hard = np.asarray(hard, np.int64)
    npart = int(hard.max()) + 1
    z0 = np.load("%s/runs/cache_%s.npz" % (ROOT, ds), allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}
    z.update(EC.rebuild(ds, hard, npart, z, meta, log))
    nq = meta["n_dev_queries"]
    base50 = [set(int(x) for x in z["base_rank"][qi][:P]) for qi in range(nq)]
    goldp = KB.goldparts(z, meta)
    ind_base = np.array([int(goldp[qi] <= base50[qi]) for qi in range(nq)], np.int8)
    cpos = [{int(p): r for r, p in enumerate(z["base_rank"][qi])} for qi in range(nq)]
    hops = np.asarray(z["hops"]) if "hops" in z.keys() else -np.ones(nq, np.int64)
    return dict(ds=ds, hard=hard, npart=npart, z=z, meta=meta, nq=nq, base50=base50,
               goldp=goldp, ind_base=ind_base, cpos=cpos, hops=hops)


def eval_variant(E, sfull, rfull, M_struct, M_ret, agg_key, B, sel=KB.sel_f6):
    nq = E["nq"]
    C = {"base50": E["base50"], "cpos": E["cpos"],
        "sfull": {(M_struct, agg_key): sfull}, "rfull": {M_ret: rfull},
        "saggf": {M_struct: [{}] * nq}, "raggf": {M_ret: [{}] * nq}}
    cfg = dict(M_struct=M_struct, M_ret=M_ret, agg=agg_key, B=B)
    ctxs = KB.contexts(E["z"], E["meta"], C, B, cfg=cfg)
    r = KB.run_selector(ctxs, sel, E["goldp"], E["ind_base"])
    return r["ind"], round(r["ALL"], 4)


def cell(E, cfg):
    """cfg: {M_struct, M_ret, agg, hop_cap, B, competition}. agg in {S1,S2,S3,S4,S4_NOSDIR}.
    competition="OFF" replays sel_base (boundary kept as-is, no competition) -- respected here
    so a cur dict with competition already turned OFF stays OFF if cell() is called on it again."""
    sfull = struct_sfull(E["z"], E["hard"], E["nq"], cfg["M_struct"], cfg["hop_cap"], cfg["agg"])
    rfull = ret_rfull(E["z"], E["hard"], E["nq"], cfg["M_ret"])
    sel = KB.sel_base if cfg.get("competition") == "OFF" else KB.sel_f6
    ind, allv = eval_variant(E, sfull, rfull, cfg["M_struct"], cfg["M_ret"], cfg["agg"], cfg["B"], sel=sel)
    return ind, allv


def run(ds, log=log):
    E = _prep(ds, log)
    floor = HH.FLOOR.get(ds, 0.0)
    log("%s: nq=%d, base ALL_P50=%.4f" % (ds, E["nq"], float(E["ind_base"].mean())))

    cur = dict(SHIP)
    ind_ship, all_ship = cell(E, cur)
    ind_cur, all_cur = ind_ship, all_ship
    log("  SHIPPED (H4_SK, M_struct=64,M_ret=32,agg=S4,hop_cap=None,B=6,F6): ALL=%.4f" % all_cur)

    steps = {}

    def try_step(name, candidate_cfg, sel=KB.sel_f6):
        nonlocal cur, ind_cur, all_cur
        if sel is KB.sel_f6:
            if candidate_cfg.get("M_struct", cur["M_struct"]) == 0 and cur["M_struct"] == 0 \
               and candidate_cfg.get("agg", cur["agg"]) != cur["agg"]:
                steps[name] = {"skipped": True, "reason": "STRUCT branch already eliminated"}
                return
            ind_new, all_new = cell(E, candidate_cfg)
        else:
            sfull = struct_sfull(E["z"], E["hard"], E["nq"], cur["M_struct"], cur["hop_cap"], cur["agg"])
            rfull = ret_rfull(E["z"], E["hard"], E["nq"], cur["M_ret"])
            ind_new, all_new = eval_variant(E, sfull, rfull, cur["M_struct"], cur["M_ret"],
                                            cur["agg"], candidate_cfg["B"], sel=sel)
        g_, l_, p_ = HH.mcnemar(ind_cur, ind_new)
        delta = round(all_new - all_cur, 4)
        sig = bool(p_ < 0.05 and abs(delta) > floor)
        steps[name] = {"ALL_before": all_cur, "ALL_after": all_new, "delta": delta,
                       "gained": g_, "lost": l_, "p": p_, "floor": floor, "sig_regression": sig and delta < 0,
                       "removed": (not (sig and delta < 0))}
        if not (sig and delta < 0):
            cur.update({k: v for k, v in candidate_cfg.items() if k in cur})
            ind_cur, all_cur = ind_new, all_new
            steps[name]["new_current_best"] = dict(cur)
        return steps[name]

    try_step("TRAVERSAL_HOP1", dict(cur, hop_cap=1))
    try_step("SDIR_DROP", dict(cur, agg="S4_NOSDIR"))
    try_step("MSTRUCT_ZERO", dict(cur, M_struct=0))
    if cur["M_struct"] == 0:
        steps["AGG_S1"] = {"skipped": True, "reason": "STRUCT branch already eliminated by MSTRUCT_ZERO"}
    else:
        try_step("AGG_S1", dict(cur, agg="S1"))
    try_step("MRET_ZERO", dict(cur, M_ret=0))
    try_step("TOP44_OFF_B50", dict(cur, B=50))
    try_step("F6_COMPETITION_OFF", dict(cur, competition="OFF"), sel=KB.sel_base)

    per_hop = None
    if (E["hops"] >= 0).any():
        per_hop = {"BASE": KB.per_hop_table(E["ind_base"], E["hops"]),
                  "SHIPPED": KB.per_hop_table(ind_ship, E["hops"]),
                  "FINAL_MINIMAL": KB.per_hop_table(ind_cur, E["hops"])}

    out = {"ds": ds, "nq": E["nq"], "floor": floor, "BASE_ALL_P50": round(float(E["ind_base"].mean()), 4),
          "SHIPPED_ALL": round(float(all_ship), 4),
          "STEPS": steps, "FINAL_MINIMAL_CFG": cur, "FINAL_MINIMAL_ALL": all_cur,
          "per_hop": per_hop}
    os.makedirs("%s/safe_ablate" % AOUT, exist_ok=True)
    fp = "%s/safe_ablate/P15_%s.json" % (AOUT, ds)
    json.dump(out, open(fp, "w"), indent=1, default=str)
    log("wrote %s" % fp)
    return out


def run_all():
    for ds in DS:
        run(ds)


def report():
    rows = {}
    for ds in DS:
        fp = "%s/safe_ablate/P15_%s.json" % (AOUT, ds)
        if os.path.exists(fp):
            rows[ds] = json.load(open(fp))
    if not rows:
        print("no P15 results yet"); return {}

    flags = {}
    for step in STEP_ORDER:
        any_sig_reg = False
        detail = {}
        for ds, r in rows.items():
            s = r["STEPS"].get(step)
            if s is None or s.get("skipped"):
                detail[ds] = "skipped"
                continue
            detail[ds] = {"delta": s["delta"], "p": s["p"], "sig_regression": s["sig_regression"]}
            if s["sig_regression"]:
                any_sig_reg = True
        flags[step] = {"REQUIRED": any_sig_reg, "per_corpus": detail}

    NAME_MAP = {
        "TRAVERSAL_HOP1": "TRAVERSAL_REQUIRED (hop2/3 evidence)",
        "SDIR_DROP": "SDIR_REQUIRED",
        "MSTRUCT_ZERO": "MSTRUCT_REQUIRED",
        "AGG_S1": "S4_REQUIRED (vs S1)",
        "MRET_ZERO": "MRET_REQUIRED",
        "F6_COMPETITION_OFF": "F6_COMPETITION_REQUIRED",
        "TOP44_OFF_B50": "TOP44_PROTECTION_REQUIRED",
    }
    verdict = {NAME_MAP[k]: v["REQUIRED"] for k, v in flags.items()}
    verdict["MAX_COLLAPSE_REQUIRED"] = verdict["SDIR_REQUIRED"]
    verdict["_MAX_COLLAPSE_NOTE"] = ("identical test to SDIR_REQUIRED -- s_dir only reaches S4 "
                                     "as the max-collapsed 4th signal, no separate seam exists")
    verdict["B6_REQUIRED"] = "NOT_RETESTED_see_final_B6_S4_F6_Ms64_Mr32.json"
    verdict["_B6_NOTE"] = ("B was already swept in rounds 1-4 and B=6 is the on-disk survivor "
                           "(results/.../runs/final_B6_S4_F6_Ms64_Mr32.json); Part B was told "
                           "not to re-run a B sweep, so this flag is answered by citation")

    minimal_router = {ds: r["FINAL_MINIMAL_CFG"] for ds, r in rows.items()}
    all_same = len(set(json.dumps(v, sort_keys=True) for v in minimal_router.values())) == 1

    print("== SAFE/F6 router backward-elimination verdict (H4_SK substrate) ==")
    for k, v in verdict.items():
        if k.startswith("_"):
            continue
        print("  %-28s %s" % (k, v))
    print("\nMINIMAL_SAFE_ROUTER per corpus:")
    for ds, cfg in minimal_router.items():
        print("  %-16s %s  (ALL=%.4f, base=%.4f)" %
             (ds, cfg, rows[ds]["FINAL_MINIMAL_ALL"], rows[ds]["BASE_ALL_P50"]))
    print("\nuniversal (all 6 corpora agree on one minimal config): %s" % all_same)

    out = {"flags": verdict, "per_step": flags, "minimal_router_per_corpus": minimal_router,
          "universal_single_config": all_same,
          "per_corpus_hop_breakdown": {ds: r.get("per_hop") for ds, r in rows.items() if r.get("per_hop")}}
    json.dump(out, open("%s/PHASE15_SAFE_ABLATION_REPORT.json" % AOUT, "w"), indent=1)
    print("\nwrote %s/PHASE15_SAFE_ABLATION_REPORT.json" % AOUT)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
