"""FBX_SCALE stage 4A (addendum 4) -- score the rung-4 maps under the UNCHANGED addendum-1 gate, K in {100, 250, 500, 1000}.

A sibling of _fbx_calib_cn.py (which stays unchanged): the per-row arithmetic is identical; what changes is the method list and one extra reference.
Methods:  PHG    = the seven WebQSP PHG maps (H4_SK, NP4)               -- the gate's reference
          PHG_S  = the frozen PHG on the STRUCT-only hypergraph (H4_S)  -- control: the ceiling of a graph-S partition (no candidate)
          ML_S   = coarsen (twins + agglomeration) -> vertex-weighted PHG on the quotient -> projection -> R3b closed-neighbourhood LP   -- the candidate (graph S)
          ML_S_proj = the projection before the LP refinement            -- diagnostic
Modes per (method, K): own (B_P from the method's own K100 map), matched (B_P = PHG's own B_P at that K: the gate), matchedS (B_P = PHG_S's own B_P: same-graph reference).
The gate is exactly addendum 1's: K in {100, 250, 500}, every B_N, rows below PHG at matched B_P: <= 7 PASS, >= 24 FAIL (or B_P = K on > half the rows), else BORDERLINE.

  python scratchpad/_ml_eval.py CONTROL <tag> <bundle_tag>                       # PHG and PHG_S only; the PHG cells must equal the v1 EVAL bit for bit
  python scratchpad/_ml_eval.py EVAL    <tag> <bundle_tag> <control_tag>         # all four methods; PHG and PHG_S cells must equal the control EVAL bit for bit
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _fbx_calib as C

log = D.log
RT, KS = C.RT, C.KS
KGRID = (100, 250, 500, 1000)
GATE_K = C.GATE_K
MODES = ("own", "matched", "matchedS")
PARTS_S = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "parts_S")
ML = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "ml")
ADDENDUM3 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_3.json")
ADDENDUM4 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_4.json")
V1_TAG = "v1"


def load_maps(methods):
    maps, meta = {}, {}
    for K in KGRID:
        R = C.jl(os.path.join(C.PARTS, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K))
        p = os.path.join(D.REPO, R["output"]["file"])
        assert D.sha_file(p) == R["output"]["sha256"] and R["STATUS"] == "OK" and R["balance"]["every_block_used"], "PHG map K %d" % K
        maps[("PHG", K)] = np.load(p).astype(np.int32)
        meta[("PHG", K)] = {"sha256": R["output"]["sha256"], "file": R["output"]["file"]}
        if "PHG_S" in methods:
            R = C.jl(os.path.join(PARTS_S, "webqsp__H4_S_k%d__PHG_con.RUN.json" % K))
            p = os.path.join(D.REPO, R["output"]["file"])
            assert D.sha_file(p) == R["output"]["sha256"] and R["STATUS"] == "OK" and R["balance"]["every_block_used"], "PHG_S map K %d" % K
            maps[("PHG_S", K)] = np.load(p).astype(np.int32)
            meta[("PHG_S", K)] = {"sha256": R["output"]["sha256"], "file": R["output"]["file"], "run_json_sha256": D.sha_file(os.path.join(PARTS_S, "webqsp__H4_S_k%d__PHG_con.RUN.json" % K))}
        if "ML_S" in methods:
            rj = os.path.join(ML, "webqsp__ML_S_k%d.RUN.json" % K)
            if not os.path.exists(rj):
                # a cell that failed its own validity gate has no map: allowed only outside the gate cells (K = 1000 is a diagnostic), and its FAILED record is pinned
                fj = os.path.join(ML, "webqsp__ML_S_k%d.FAILED.json" % K)
                Rf = C.jl(fj)
                assert K not in GATE_K and Rf["STATUS"] == "PARTITION_INVALID", "ML_S K %d has no map and is not an allowed invalid diagnostic cell" % K
                meta[("ML_S", K)] = {"STATUS": "PARTITION_INVALID", "file": D.rel(fj), "sha256": D.sha_file(fj)}
                continue
            R = C.jl(rj)
            assert R["STATUS"] == "OK"
            for m, key in (("ML_S", "output"), ("ML_S_proj", "projection_map")):
                p = os.path.join(D.REPO, R[key]["file"])
                assert D.sha_file(p) == R[key]["sha256"], "%s map K %d differs from its RUN.json" % (m, K)
                maps[(m, K)] = np.load(p).astype(np.int32)
                meta[(m, K)] = {"sha256": R[key]["sha256"], "file": R[key]["file"], "run_json_sha256": D.sha_file(rj)}
    return maps, meta


def run_eval(mode_name, tag, btag, ctag):
    methods = ("PHG", "PHG_S") if mode_name == "CONTROL" else ("PHG", "PHG_S", "ML_S", "ML_S_proj")
    out_j = os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.json" % tag)
    out_z = out_j.replace(".json", ".npz")
    assert not os.path.exists(out_j), "write-once: %s exists" % out_j
    if mode_name == "EVAL":
        assert os.path.exists(ADDENDUM4), "addendum 4 must exist before the rung-4 maps are scored"
    bdir = os.path.join(C.WORK, "bundle_%s" % btag)
    bmeta = C.jl(os.path.join(bdir, "meta.json"))
    assert bmeta["ev_sha256"] == D.sha_file(os.path.join(bdir, "ev.npz"))
    with np.load(os.path.join(bdir, "ev.npz")) as zev:
        ev = {k: zev[k] for k in zev.files}
    ford = np.load(os.path.join(bdir, "ford.npy"), mmap_mode="r")
    N, nq = bmeta["N"], bmeta["n_rows"]
    assert nq == 786
    gptr, gpos, rows = ev["gptr"], ev["gpos"], ev["rows"]
    assert D.sha_file(C.POP_JSON).startswith(C.POP_SHA_PREFIX) and [int(r) for r in C.jl(C.POP_JSON)["rows"][:nq]] == [int(r) for r in rows[:nq]]
    golds = [ev["gold"][gptr[j]:gptr[j + 1]] for j in range(nq)]
    maps, mmeta = load_maps(methods)
    tabs = {K: KS.mult_tables(K)[0][C.ALPHA] for K in KGRID}
    wseed = 1.0 / np.arange(1.0, C.ACT + 1.0)
    MB = np.array(C.M_CURVE)
    NB = len(MB)
    ngo = np.diff(gptr)[:nq]
    keys = [(m, K, mode) for m in methods for K in KGRID for mode in MODES if (m, K) in maps]
    SV = {k: np.zeros((nq, NB), np.int16) for k in keys}
    BP = {k: np.zeros(nq, np.int64) for k in keys}
    CMs = {k: np.zeros(nq, np.int64) for k in keys}
    LO = {k: np.zeros(nq, np.int64) for k in keys}
    NOEV = {m: np.zeros(nq, bool) for m in methods}
    B0 = {m: np.zeros(nq, np.int64) for m in methods}
    H100 = {m: maps[(m, C.K_REF)].astype(np.int64) for m in methods}
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
        for m in methods:
            Cm = np.bincount(hit * C.K_REF + H100[m][u], weights=x, minlength=nh * C.K_REF).reshape(nh, C.K_REF)
            sdiv = wseed[:nh] @ (Cm > 0)
            kn_, npz_ = RT.knee(sdiv)
            NOEV[m][j] = not npz_
            B0[m][j] = kn_ if npz_ else C.K_REF
        for K in KGRID:
            b_own_of = {m: (K if NOEV[m][j] else int(tabs[K][B0[m][j]])) for m in methods}
            b_ref, b_refS = b_own_of["PHG"], b_own_of["PHG_S"]
            for m in methods:
                if (m, K) not in maps:
                    continue
                hard = maps[(m, K)]
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro = np.lexsort((np.arange(K), fe))
                pr = RT.inv(ro)
                prh = pr[hard]
                lo = int(pr[hard[g]].max()) + 1
                b_own = b_own_of[m]
                for mode, b in (("own", b_own), ("matched", b_ref), ("matchedS", b_refS)):
                    k1 = (m, K, mode)
                    if mode != "own" and b == b_own:
                        for arr in (SV, BP, CMs, LO):
                            arr[k1][j] = arr[(m, K, "own")][j]
                        continue
                    contn = prh < b
                    cm = int(contn.sum())
                    cs = np.cumsum(contn[fo])
                    cg = contn[g]
                    mp = cs[gp] - 1
                    n_ = np.minimum(MB, cm)
                    sv = np.array([(cg & (mp < n_[i])).sum() for i in range(NB)])
                    SV[k1][j], BP[k1][j], CMs[k1][j], LO[k1][j] = sv, b, cm, lo
        if j % 50 == 0 or j == nq - 1:
            log("  EVAL rows %d / %d (%.0fs, RSS %.0f MB)" % (j + 1, nq, time.time() - t_, D._rss_mb()))
    sec_loop = round(time.time() - t_, 1)
    # ---- bit-equality of the cells that an earlier record already fixed ----------------------------------------------------------------
    chk = {}
    if mode_name == "CONTROL":
        zprev, prev_sha, prev_methods = np.load(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % V1_TAG)), D.sha_file(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % V1_TAG)), ("PHG",)
        prev_modes = ("own", "matched")
    else:
        zprev, prev_sha, prev_methods = np.load(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % ctag)), D.sha_file(os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % ctag)), ("PHG", "PHG_S")
        prev_modes = MODES
    for m in prev_methods:
        for K in KGRID:
            for mode in prev_modes:
                a1 = SV[(m, K, mode)] == zprev["SV__%s__k%d__%s" % (m, K, mode)]
                a2 = BP[(m, K, mode)] == zprev["BP__%s__k%d__%s" % (m, K, mode)]
                assert a1.all() and a2.all(), "cell %s K %d %s differs from the earlier EVAL" % (m, K, mode)
                chk["%s|K%d|%s" % (m, K, mode)] = "equal"
    log("regression: %d cells bit-identical to %s" % (len(chk), prev_sha[:12]))
    from scipy.stats import binomtest
    allc = lambda sv: [int((sv[:, i] == ngo).sum()) for i in range(NB)]
    res = {}
    for K in KGRID:
        ref_all = SV[("PHG", K, "own")] == ngo[:, None]
        refS_all = SV[("PHG_S", K, "own")] == ngo[:, None]
        for m in methods:
            if (m, K) not in maps:
                continue
            for mode in MODES:
                k1 = (m, K, mode)
                sv = SV[k1]
                allm = sv == ngo[:, None]
                e = {"ALL": allc(sv), "ANY": [int((sv[:, i] > 0).sum()) for i in range(NB)], "gold_served": [int(sv[:, i].sum()) for i in range(NB)],
                     "B_P": {"mean": round(float(BP[k1].mean()), 1), "median": float(np.median(BP[k1])), "min": int(BP[k1].min()), "max": int(BP[k1].max())},
                     "B_P_over_K_mean": round(float((BP[k1] / float(K)).mean()), 4), "rows_with_B_P_eq_K": int((BP[k1] == K).sum()),
                     "contacted_mass": {"mean": round(float(CMs[k1].mean()), 0), "median": float(np.median(CMs[k1]))}, "contacted_mass_over_N_mean": round(float((CMs[k1] / float(N)).mean()), 4),
                     "lost_to_reach (B_P < LO)": int((BP[k1] < LO[k1]).sum())}
                for rn, ra in (("PHG", ref_all), ("PHG_S", refS_all)):
                    gain = [int((allm[:, i] & ~ra[:, i]).sum()) for i in range(NB)]
                    lost = [int((~allm[:, i] & ra[:, i]).sum()) for i in range(NB)]
                    e["vs_%s_gained" % rn], e["vs_%s_lost" % rn] = gain, lost
                    e["vs_%s_p_sign_descriptive" % rn] = [(1.0 if g_ + l_ == 0 else float("%.3g" % binomtest(g_, g_ + l_, 0.5).pvalue)) for g_, l_ in zip(gain, lost)]
                res["%s|K%d|%s" % (m, K, mode)] = e

    def method_verdict(nm, ref, mode):
        loss_max, fail, cells = 0, False, {}
        for K in GATE_K:
            ph = res["%s|K%d|own" % (ref, K)]["ALL"]
            mm = res["%s|K%d|%s" % (nm, K, mode)]["ALL"]
            ls = [p - q for p, q in zip(ph, mm)]
            cells["%s K%d" % (nm, K)] = ls
            loss_max = max(loss_max, max(ls))
            if max(ls) >= C.LOSS_FAIL:
                fail = True
            if res["%s|K%d|own" % (nm, K)]["rows_with_B_P_eq_K"] > nq / 2.0:
                fail = True
        v = "FAILED" if fail else ("RECALL_PRESERVING" if loss_max <= C.LOSS_PASS else "BORDERLINE")
        return {"verdict": v, "max_loss_rows_vs_%s_%s_over_gate_cells" % (ref, mode): int(loss_max), "loss_rows_by_cell_and_B_N": cells}
    verdicts = {"PHG_S control vs PHG (S-arm ceiling)": method_verdict("PHG_S", "PHG", "matched")}
    if mode_name == "EVAL":
        verdicts.update({"ML_S vs PHG (THE GATE)": method_verdict("ML_S", "PHG", "matched"), "ML_S_proj vs PHG (diagnostic)": method_verdict("ML_S_proj", "PHG", "matched"),
                         "ML_S vs PHG_S (same graph, diagnostic)": method_verdict("ML_S", "PHG_S", "matchedS")})
    for k_, v_ in verdicts.items():
        log("%-44s %s (%s)" % (k_, v_["verdict"], [v for kk, v in v_.items() if kk.startswith("max_loss")][0]))
    for K in KGRID:
        for m in methods:
            if (m, K) not in maps:
                log("K %5d %-9s NO MAP (%s)" % (K, m, mmeta[("ML_S", K)]["STATUS"]))
                continue
            log("K %5d %-9s matched ALL %s | own ALL %s  B_P/K %.3f" % (K, m, res["%s|K%d|matched" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["B_P_over_K_mean"]))
    rec = {"stage": "FBX_SCALE 4A: WebQSP rung-4 (hypergraph-aware multilevel) " + mode_name, "tag": tag, "status": "DEVELOPMENT (calibration of a systems substrate)", "definitions": __doc__,
           "N": N, "n_rows": nq, "M_curve": list(C.M_CURVE), "K_grid": list(KGRID), "gate_K": list(GATE_K), "loss_pass_rows": C.LOSS_PASS, "loss_fail_rows": C.LOSS_FAIL, "methods": list(methods),
           "regression_vs_earlier_eval": {"eval_npz_sha256": prev_sha, "cells": chk}, "results": res, "verdicts": verdicts, "seconds_loop": sec_loop,
           "inputs": {"bundle_meta": {"path": D.rel(os.path.join(bdir, "meta.json")), "sha256": D.sha_file(os.path.join(bdir, "meta.json"))},
                      "maps": {"%s|%s" % (m, K): mmeta[(m, K)] for (m, K) in mmeta}, "population": {"path": D.rel(C.POP_JSON), "sha256": D.sha_file(C.POP_JSON)},
                      "addendum": {D.rel(C.ADDENDUM): D.sha_file(C.ADDENDUM), D.rel(C.ADDENDUM2): D.sha_file(C.ADDENDUM2), D.rel(ADDENDUM3): D.sha_file(ADDENDUM3)}},
           "code": {"scratchpad/_ml_eval.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_calib.py": D.sha_file(os.path.join(D.HERE, "_fbx_calib.py"))},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "platform": D.platform_record(), "process_peak_rss_mb": D.peak_rss_mb(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if mode_name == "EVAL":
        rec["inputs"]["addendum"][D.rel(ADDENDUM4)] = D.sha_file(ADDENDUM4)
    arrays = {"rows": rows[:nq], "gptr": gptr[:nq + 1], "ngold": ngo}
    for (m, K, mode) in keys:
        arrays["SV__%s__k%d__%s" % (m, K, mode)] = SV[(m, K, mode)]
        arrays["BP__%s__k%d__%s" % (m, K, mode)] = BP[(m, K, mode)]
    np.savez_compressed(out_z, **arrays)
    rec["npz"] = {"path": os.path.basename(out_z), "sha256": D.sha_file(out_z)}
    C.wj(out_j, rec)
    log("%s -> %s %s" % (mode_name, out_j, D.sha_file(out_j)[:12]))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "CONTROL":
        run_eval("CONTROL", a[1], a[2], None)
    elif len(a) == 4 and a[0] == "EVAL":
        run_eval("EVAL", a[1], a[2], a[3])
    else:
        raise SystemExit(__doc__)
