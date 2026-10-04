"""FBX_SCALE stage 4C (addendum 5) -- score the SK-arm candidates against the frozen PHG_SK under the addendum-1 gate, K in {100, 250, 500}.

The per-row arithmetic is _ml_eval.py's (itself _fbx_calib_cn.py's), unchanged.  What changes is the comparison: EVERY candidate here is built on the frozen H4_SK hypergraph (STRUCT u KNN) and
is scored against the frozen PHG_SK maps, so algorithm and graph are never mixed (no S-only candidate, no PHG_S).
Methods:  PHG  = the seven WebQSP PHG maps (H4_SK, NP4), K in {100, 250, 500} used here   -- the reference, M0
          <name> = a candidate: <method>_W<Wmax>_L<level> (clusters -> quotient -> vertex-weighted PHG -> projection -> shared refinement)  or  M0R (the frozen PHG_SK map put through the
                   same shared refinement, no clustering: separates the refinement from the coarsening)
Modes per (method, K): own (B_P from the method's own K100 map), matched (B_P = PHG's own B_P at that K: THE GATE).
The gate is addendum 1's: rows below PHG at matched B_P over every B_N: <= 7 PASS, >= 24 FAIL (or B_P = K on > half the rows), else BORDERLINE.  A candidate is scored at the gate cells it has a
map for; a cell is attempted only when the largest cluster weight is <= 10 % of the block size (Wmax <= 0.10 N / K), otherwise SKIPPED_BY_RULE (declared in addendum 5).

  python scratchpad/_ml2_eval.py EVAL <tag> <bundle_tag> <control_tag> <name>[,<name>...]     # control_tag = an earlier EVAL whose PHG cells must be reproduced bit for bit
"""
import json
import os
import re
import sys
import time

import numpy as np

import _l1d_lib as D
import _fbx_calib as C

log = D.log
RT, KS = C.RT, C.KS
KGRID = (100, 250, 500)
GATE_K = C.GATE_K
MODES = ("own", "matched")
ML2 = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "ml2")
ADDENDUM5 = os.path.join(C.OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json")
BLOCK_RULE = 0.10


def attempted(name, K, N):
    """the declared attempt rule: a cluster candidate is run at K only if its cap Wmax <= 10 % of the block N / K; M0R has no clusters."""
    m = re.match(r"^M[1-4]_W(\d+)_L\d+$", name)
    if not m:
        return True
    return int(m.group(1)) <= BLOCK_RULE * N / float(K)


def load_maps(names, N):
    maps, meta = {}, {}
    for K in KGRID:
        R = C.jl(os.path.join(C.PARTS, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K))
        p = os.path.join(D.REPO, R["output"]["file"])
        assert D.sha_file(p) == R["output"]["sha256"] and R["STATUS"] == "OK" and R["balance"]["every_block_used"], "PHG map K %d" % K
        maps[("PHG", K)] = np.load(p).astype(np.int32)
        meta[("PHG", K)] = {"STATUS": "OK", "sha256": R["output"]["sha256"], "file": R["output"]["file"]}
        for nm in names:
            base = os.path.join(ML2, "webqsp__%s_k%d" % (nm, K))
            if not attempted(nm, K, N):
                meta[(nm, K)] = {"STATUS": "SKIPPED_BY_RULE"}
                continue
            if os.path.exists(base + ".RUN.json"):
                Rc = C.jl(base + ".RUN.json")
                assert Rc["STATUS"] == "OK"
                p = os.path.join(D.REPO, Rc["output"]["file"])
                assert D.sha_file(p) == Rc["output"]["sha256"], "%s K %d map differs from its RUN.json" % (nm, K)
                maps[(nm, K)] = np.load(p).astype(np.int32)
                meta[(nm, K)] = {"STATUS": "OK", "sha256": Rc["output"]["sha256"], "file": Rc["output"]["file"], "run_json_sha256": D.sha_file(base + ".RUN.json")}
            else:
                fj = base + ".FAILED.json"
                Rf = C.jl(fj)                       # a missing record is an error: every attempted cell has a RUN or a FAILED record
                assert Rf["STATUS"] == "PARTITION_INVALID"
                meta[(nm, K)] = {"STATUS": "PARTITION_INVALID", "file": D.rel(fj), "sha256": D.sha_file(fj)}
    return maps, meta


def run_eval(tag, btag, ctag, names):
    out_j = os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.json" % tag)
    out_z = out_j.replace(".json", ".npz")
    assert not os.path.exists(out_j), "write-once: %s exists" % out_j
    assert os.path.exists(ADDENDUM5), "addendum 5 must exist before the SK-arm candidates are scored"
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
    maps, mmeta = load_maps(names, N)
    methods = ["PHG"] + [nm for nm in names if (nm, C.K_REF) in maps]
    dropped = [nm for nm in names if nm not in methods]
    if dropped:
        log("no K100 map (PARTITION_INVALID) -> not scored: %s" % dropped)
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
            b_ref = b_own_of["PHG"]
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
                for mode, b in (("own", b_own), ("matched", b_ref)):
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
    # ---- the reference cells an earlier record already fixed must be reproduced bit for bit ---------------------------------------------
    zprev_p = os.path.join(C.OUTR, "CALIB_EVAL_WEBQSP__%s.npz" % ctag)
    zprev, prev_sha = np.load(zprev_p), D.sha_file(zprev_p)
    chk = {}
    for K in KGRID:
        for mode in MODES:
            a1 = SV[("PHG", K, mode)] == zprev["SV__PHG__k%d__%s" % (K, mode)]
            a2 = BP[("PHG", K, mode)] == zprev["BP__PHG__k%d__%s" % (K, mode)]
            assert a1.all() and a2.all(), "PHG cell K %d %s differs from the earlier EVAL" % (K, mode)
            chk["PHG|K%d|%s" % (K, mode)] = "equal"
    log("regression: %d PHG cells bit-identical to %s" % (len(chk), prev_sha[:12]))
    from scipy.stats import binomtest
    allc = lambda sv: [int((sv[:, i] == ngo).sum()) for i in range(NB)]
    res = {}
    for K in KGRID:
        ref_all = SV[("PHG", K, "own")] == ngo[:, None]
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
                gain = [int((allm[:, i] & ~ref_all[:, i]).sum()) for i in range(NB)]
                lost = [int((~allm[:, i] & ref_all[:, i]).sum()) for i in range(NB)]
                e["vs_PHG_gained"], e["vs_PHG_lost"] = gain, lost
                e["vs_PHG_p_sign_descriptive"] = [(1.0 if g_ + l_ == 0 else float("%.3g" % binomtest(g_, g_ + l_, 0.5).pvalue)) for g_, l_ in zip(gain, lost)]
                res["%s|K%d|%s" % (m, K, mode)] = e

    def verdict(nm):
        loss_max, fail, cells, present = 0, False, {}, []
        for K in GATE_K:
            if (nm, K) not in maps:
                continue
            present.append(K)
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
        return {"verdict": v, "gate_cells_scored": present, "gate_cells_not_scored": {str(K): mmeta[(nm, K)]["STATUS"] for K in GATE_K if (nm, K) not in maps},
                "max_loss_rows_vs_PHG_matched_over_scored_gate_cells": int(loss_max), "loss_rows_by_cell_and_B_N": cells}
    verdicts = {nm: verdict(nm) for nm in methods if nm != "PHG"}
    for nm, v_ in verdicts.items():
        log("%-16s %-18s max loss %3d rows; scored K %s; not scored %s" % (nm, v_["verdict"], v_["max_loss_rows_vs_PHG_matched_over_scored_gate_cells"], v_["gate_cells_scored"], v_["gate_cells_not_scored"]))
    for K in KGRID:
        for m in methods:
            if (m, K) in maps:
                log("K %4d %-16s matched ALL %s | own ALL %s  B_P/K %.3f" % (K, m, res["%s|K%d|matched" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["B_P_over_K_mean"]))
    rec = {"stage": "FBX_SCALE 4C: WebQSP SK-arm multilevel lab (SK candidates vs PHG_SK)", "tag": tag, "status": "DEVELOPMENT (calibration of a systems substrate)", "definitions": __doc__,
           "N": N, "n_rows": nq, "M_curve": list(C.M_CURVE), "K_grid": list(KGRID), "gate_K": list(GATE_K), "loss_pass_rows": C.LOSS_PASS, "loss_fail_rows": C.LOSS_FAIL, "block_rule": BLOCK_RULE,
           "candidates_declared": list(names), "candidates_scored": [m for m in methods if m != "PHG"], "candidates_without_K100_map": dropped,
           "regression_vs_earlier_eval": {"eval_npz_sha256": prev_sha, "cells": chk}, "results": res, "verdicts": verdicts, "seconds_loop": sec_loop,
           "inputs": {"bundle_meta": {"path": D.rel(os.path.join(bdir, "meta.json")), "sha256": D.sha_file(os.path.join(bdir, "meta.json"))},
                      "maps": {"%s|%s" % (m, K): mmeta[(m, K)] for (m, K) in mmeta}, "population": {"path": D.rel(C.POP_JSON), "sha256": D.sha_file(C.POP_JSON)},
                      "addendum": {D.rel(C.ADDENDUM): D.sha_file(C.ADDENDUM), D.rel(C.ADDENDUM2): D.sha_file(C.ADDENDUM2), D.rel(ADDENDUM5): D.sha_file(ADDENDUM5)}},
           "code": {"scratchpad/_ml2_eval.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_calib.py": D.sha_file(os.path.join(D.HERE, "_fbx_calib.py"))},
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
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "EVAL":
        run_eval(a[1], a[2], a[3], [x for x in a[4].split(",") if x])
    else:
        raise SystemExit(__doc__)
