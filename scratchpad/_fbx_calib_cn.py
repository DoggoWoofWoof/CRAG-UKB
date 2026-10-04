"""FBX_SCALE stage 3a rung 3b (addendum 3) -- score the R3b maps under the UNCHANGED addendum-1 gate, on the restricted grid K in {100, 250, 500, 1000}.

A sibling of _fbx_calib.py EVAL (kept unchanged so the records it wrote keep the code hash they were written with): the per-row arithmetic is the same, the methods are PHG (the
reference), LP_S (rung 3, re-scored as the control) and the R3b maps CN_S (graph S, the gate) and CN_SK (S u KNN, diagnostic).  The full-population regression gate of _fbx_calib.py
(PHG_k25928 vs the l3w record) is not re-run here (K 25,928 is not in this grid); in its place EVERY PHG and LP_S cell (SV and BP arrays, own and matched, all rows, all four K)
must equal the v1 EVAL npz bit for bit -- the v1 EVAL passed the regression, so equality certifies the same bundle, the same arithmetic and the same reference.

  python scratchpad/_fbx_calib_cn.py EVAL <tag> <bundle_tag> <cn_maps_tag>
"""
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _fbx_calib as C
import _fbx_part as P

log = D.log
RT, KS = C.RT, C.KS
KGRID = (100, 250, 500, 1000)
GATE_K = C.GATE_K
METHODS = ("PHG", "LP_S", "CN_S", "CN_SK")
V1_TAG = "v1"
ADDENDUM3 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_3.json")


def load_maps(cn_tag):
    """{(method, K): int32 block table}, each checked against its record (PHG: its RUN.json; LP_S: CALIB_MAPS lp_v1; CN_*: the cn maps record)."""
    lrec = C.jl(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % C.LP_MAPS_TAG))
    crec = C.jl(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % cn_tag))
    maps, meta = {}, {}
    for K in KGRID:
        rj = os.path.join(C.PARTS, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K)
        R = C.jl(rj)
        p = os.path.join(D.REPO, R["output"]["file"])
        assert D.sha_file(p) == R["output"]["sha256"], "PHG map K %d differs from its RUN.json" % K
        assert R["STATUS"] == "OK" and R["balance"]["every_block_used"]
        maps[("PHG", K)] = np.load(p).astype(np.int32)
        meta[("PHG", K)] = {"sha256": R["output"]["sha256"], "run_json_sha256": D.sha_file(rj), "file": R["output"]["file"]}
        for m in METHODS[1:]:
            p = C.map_path(m, K)
            src = lrec if m.startswith("LP_") else crec
            assert D.sha_file(p) == src["maps"]["%s__k%d" % (m, K)]["sha256"], "map %s K %d differs from its record" % (m, K)
            maps[(m, K)] = np.load(p)
            meta[(m, K)] = {"sha256": src["maps"]["%s__k%d" % (m, K)]["sha256"]}
    return maps, meta, crec, lrec


def run_eval(tag, btag, cn_tag):
    out_j = os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.json" % tag)
    out_z = out_j.replace(".json", ".npz")
    assert not os.path.exists(out_j), "write-once: %s exists" % out_j
    assert os.path.exists(ADDENDUM3), "addendum 3 must exist before R3b is scored"
    bdir = os.path.join(C.WORK, "bundle_%s" % btag)
    bmeta = C.jl(os.path.join(bdir, "meta.json"))
    assert bmeta["ev_sha256"] == D.sha_file(os.path.join(bdir, "ev.npz")), "the bundle's ev.npz differs from its meta"
    with np.load(os.path.join(bdir, "ev.npz")) as zev:
        ev = {k: zev[k] for k in zev.files}
    ford = np.load(os.path.join(bdir, "ford.npy"), mmap_mode="r")
    N = bmeta["N"]
    nq = bmeta["n_rows"]
    assert nq == 786
    gptr, gpos, rows = ev["gptr"], ev["gpos"], ev["rows"]
    assert D.sha_file(C.POP_JSON).startswith(C.POP_SHA_PREFIX) and [int(r) for r in C.jl(C.POP_JSON)["rows"][:nq]] == [int(r) for r in rows[:nq]]
    golds = [ev["gold"][gptr[j]:gptr[j + 1]] for j in range(nq)]
    maps, mmeta, crec, lrec = load_maps(cn_tag)
    tabs = {K: KS.mult_tables(K)[0][C.ALPHA] for K in KGRID}
    wseed = 1.0 / np.arange(1.0, C.ACT + 1.0)
    MB = np.array(C.M_CURVE)
    NB = len(MB)
    ngo = np.diff(gptr)[:nq]
    keys = [(m, K, mode) for m in METHODS for K in KGRID for mode in ("own", "matched")]
    SV = {k: np.zeros((nq, NB), np.int16) for k in keys}
    BP = {k: np.zeros(nq, np.int64) for k in keys}
    CM = {k: np.zeros(nq, np.int64) for k in keys}
    LO = {k: np.zeros(nq, np.int64) for k in keys}
    NOEV = {m: np.zeros(nq, bool) for m in METHODS}
    B0 = {m: np.zeros(nq, np.int64) for m in METHODS}
    H100 = {m: maps[(m, C.K_REF)].astype(np.int64) for m in METHODS}
    t_ = time.time()
    for j in range(nq):
        g = golds[j]
        fo = np.asarray(ford[j])
        oq = ev["oq"][ev["oq_ptr"][j]:ev["oq_ptr"][j + 1]].astype(np.int64)
        s0, s1 = ev["e_ptr"][j], ev["e_ptr"][j + 1]
        u = ev["eu"][s0:s1].astype(np.int64)
        hit = ev["eh"][s0:s1].astype(np.int64)
        x = ev["ex"][s0:s1].astype(np.float64)
        gp = gpos[gptr[j]:gptr[j + 1]]
        nh = int(ev["nh"][j])
        for m in METHODS:                        # B100(q) on the SAME partitioner's K 100 map
            Cm = np.bincount(hit * C.K_REF + H100[m][u], weights=x, minlength=nh * C.K_REF).reshape(nh, C.K_REF)
            sdiv = wseed[:nh] @ (Cm > 0)
            kn_, npz_ = RT.knee(sdiv)
            NOEV[m][j] = not npz_
            B0[m][j] = kn_ if npz_ else C.K_REF
        for K in KGRID:
            b_ref = None
            for m in METHODS:
                hard = maps[(m, K)]
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro = np.lexsort((np.arange(K), fe))
                pr = RT.inv(ro)
                prh = pr[hard]
                lo = int(pr[hard[g]].max()) + 1
                b_own = K if NOEV[m][j] else int(tabs[K][B0[m][j]])
                if m == "PHG":
                    b_ref = b_own
                for mode, b in (("own", b_own), ("matched", b_ref)):
                    if mode == "matched" and (m == "PHG" or b == b_own):
                        k1 = (m, K, "own")
                        for arr in (SV, BP, CM, LO):
                            arr[(m, K, mode)][j] = arr[k1][j]
                        continue
                    contn = prh < b
                    cm = int(contn.sum())
                    cs = np.cumsum(contn[fo])
                    cg = contn[g]
                    mp = cs[gp] - 1
                    n_ = np.minimum(MB, cm)
                    sv = np.array([(cg & (mp < n_[i])).sum() for i in range(NB)])
                    k1 = (m, K, mode)
                    SV[k1][j], BP[k1][j], CM[k1][j], LO[k1][j] = sv, b, cm, lo
        if j % 50 == 0 or j == nq - 1:
            log("  EVAL rows %d / %d (%.0fs, RSS %.0f MB)" % (j + 1, nq, time.time() - t_, D._rss_mb()))
    sec_loop = round(time.time() - t_, 1)
    # ---- the equality check against the v1 EVAL: PHG and LP_S cells must be identical -------------------------------------------------
    zv1 = np.load(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % V1_TAG))
    v1_sha = D.sha_file(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % V1_TAG))
    chk = {}
    for m in ("PHG", "LP_S"):
        for K in KGRID:
            for mode in ("own", "matched"):
                a1 = SV[(m, K, mode)] == zv1["SV__%s__k%d__%s" % (m, K, mode)]
                a2 = BP[(m, K, mode)] == zv1["BP__%s__k%d__%s" % (m, K, mode)]
                assert a1.all() and a2.all(), "control cell %s K %d %s differs from the v1 EVAL" % (m, K, mode)
                chk["%s|K%d|%s" % (m, K, mode)] = "equal (SV %s, BP %d rows)" % (a1.shape, len(a2))
    log("control check: PHG and LP_S cells (2 methods x %d K x own/matched) are bit-identical to the v1 EVAL" % len(KGRID))
    # ---- tables (the same shape as the v1 EVAL) ---------------------------------------------------------------------------------------
    from scipy.stats import binomtest
    allc = lambda sv: [int((sv[:, i] == ngo).sum()) for i in range(NB)]
    res = {}
    for K in KGRID:
        ref_all = SV[("PHG", K, "own")] == ngo[:, None]
        for m in METHODS:
            for mode in ("own", "matched"):
                k1 = (m, K, mode)
                sv = SV[k1]
                allm = sv == ngo[:, None]
                e = {"ALL": allc(sv), "ANY": [int((sv[:, i] > 0).sum()) for i in range(NB)], "gold_served": [int(sv[:, i].sum()) for i in range(NB)],
                     "B_P": {"mean": round(float(BP[k1].mean()), 1), "median": float(np.median(BP[k1])), "min": int(BP[k1].min()), "max": int(BP[k1].max())},
                     "B_P_over_K_mean": round(float((BP[k1] / float(K)).mean()), 4), "rows_with_B_P_eq_K": int((BP[k1] == K).sum()),
                     "contacted_mass": {"mean": round(float(CM[k1].mean()), 0), "median": float(np.median(CM[k1]))}, "contacted_mass_over_N_mean": round(float((CM[k1] / float(N)).mean()), 4),
                     "lost_to_reach (B_P < LO)": int((BP[k1] < LO[k1]).sum())}
                if m != "PHG" or mode != "own":
                    gain = [int((allm[:, i] & ~ref_all[:, i]).sum()) for i in range(NB)]
                    lost = [int((~allm[:, i] & ref_all[:, i]).sum()) for i in range(NB)]
                    e["vs_PHG_gained"], e["vs_PHG_lost"] = gain, lost
                    e["vs_PHG_p_sign_descriptive"] = [(1.0 if g_ + l_ == 0 else float("%.3g" % binomtest(g_, g_ + l_, 0.5).pvalue)) for g_, l_ in zip(gain, lost)]
                res["%s|K%d|%s" % (m, K, mode)] = e

    def method_verdict(names):
        loss_max, fail = 0, False
        cells = {}
        for K in GATE_K:
            for nm in names:
                ph = res["PHG|K%d|own" % K]["ALL"]
                mm = res["%s|K%d|matched" % (nm, K)]["ALL"]
                ls = [p - q for p, q in zip(ph, mm)]
                cells["%s K%d" % (nm, K)] = ls
                loss_max = max(loss_max, max(ls))
                if max(ls) >= C.LOSS_FAIL:
                    fail = True
                if res["%s|K%d|own" % (nm, K)]["rows_with_B_P_eq_K"] > nq / 2.0:
                    fail = True
        v = "FAILED" if fail else ("RECALL_PRESERVING" if loss_max <= C.LOSS_PASS else "BORDERLINE")
        return {"verdict": v, "max_loss_rows_vs_PHG_matched_over_gate_cells": int(loss_max), "loss_rows_by_cell_and_B_N": cells}
    verdicts = {"R3 LP on S (control; equals the v1 verdict)": method_verdict(["LP_S"]), "R3b CN on S (deployable; the gate)": method_verdict(["CN_S"]),
                "R3b CN on S u KNN (diagnostic)": method_verdict(["CN_SK"])}
    for k_, v_ in verdicts.items():
        log("%-46s %s (max loss %d rows)" % (k_, v_["verdict"], v_["max_loss_rows_vs_PHG_matched_over_gate_cells"]))
    for K in KGRID:
        for m in METHODS:
            log("K %5d %-6s matched ALL %s | own ALL %s  B_P/K %.3f" % (K, m, res["%s|K%d|matched" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["B_P_over_K_mean"]))
    rec = {"stage": "FBX_SCALE 3a rung 3b: WebQSP calibration of the R3b closed-neighbourhood LP", "tag": tag, "status": "DEVELOPMENT (calibration of a systems substrate)", "definitions": __doc__,
           "N": N, "n_rows": nq, "M_curve": list(C.M_CURVE), "K_grid": list(KGRID), "gate_K": list(GATE_K), "loss_pass_rows": C.LOSS_PASS, "loss_fail_rows": C.LOSS_FAIL,
           "control_check_vs_v1_eval": {"eval_npz_sha256": v1_sha, "cells": chk}, "results": res, "verdicts": verdicts,
           "map_costs_and_structure": {k: {kk: vv for kk, vv in v.items() if kk in ("method", "seconds", "K", "sweeps_run", "moved_per_sweep", "km1_before", "km1_after", "km1_bookkeeping_consistent",
                                                                                       "pool_bytes", "c_peak_rss_kb", "sha256", "stats", "init_stats")} for k, v in crec["maps"].items()},
           "seconds_loop": sec_loop,
           "inputs": {"bundle_meta": {"path": D.rel(os.path.join(bdir, "meta.json")), "sha256": D.sha_file(os.path.join(bdir, "meta.json"))},
                      "cn_maps_record": {"path": D.rel(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % cn_tag)), "sha256": D.sha_file(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % cn_tag))},
                      "lp_maps_record": {"path": D.rel(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % C.LP_MAPS_TAG)), "sha256": D.sha_file(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % C.LP_MAPS_TAG))},
                      "phg_reference": {"%s" % K: mmeta[("PHG", K)] for K in KGRID}, "population": {"path": D.rel(C.POP_JSON), "sha256": D.sha_file(C.POP_JSON)},
                      "addendum": {D.rel(C.ADDENDUM): D.sha_file(C.ADDENDUM), D.rel(C.ADDENDUM2): D.sha_file(C.ADDENDUM2), D.rel(ADDENDUM3): D.sha_file(ADDENDUM3)}},
           "code": {"scratchpad/_fbx_calib_cn.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_calib.py": D.sha_file(os.path.join(D.HERE, "_fbx_calib.py")),
                    "scratchpad/_fbx_part.py": D.sha_file(os.path.join(D.HERE, "_fbx_part.py"))},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "platform": D.platform_record(), "process_peak_rss_mb": D.peak_rss_mb(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    arrays = {"rows": rows[:nq], "gptr": gptr[:nq + 1], "ngold": ngo}
    for (m, K, mode) in keys:
        arrays["SV__%s__k%d__%s" % (m, K, mode)] = SV[(m, K, mode)]
        arrays["BP__%s__k%d__%s" % (m, K, mode)] = BP[(m, K, mode)]
    np.savez_compressed(out_z, **arrays)
    rec["npz"] = {"path": os.path.basename(out_z), "sha256": D.sha_file(out_z)}
    C.wj(out_j, rec)
    log("EVAL -> %s %s" % (out_j, D.sha_file(out_j)[:12]))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "EVAL":
        run_eval(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
