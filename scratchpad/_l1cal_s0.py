"""STEP 0 -- state the capacity claim with its candidate universe attached, then close PPR.

The previous phase reported "CAP_B6 = 21/677 on MetaQA" and, separately, that adding PPR reach
drives REACH to 0 while CAP_B6 rises to 215.  Those two numbers are about DIFFERENT universes and
must never be quoted without one.  This emits all three classes explicitly:

    A  CAP_B6_CURRENT_POOL        boundary + S4/ret challengers  (what the router actually sees)
    B  CAP_B6_PPR_EXPANDED_POOL   the same, plus PPR-reached partitions
    C  CAP_P50                    more than 50 gold partitions -- impossible under any rule

and then runs the DIAGNOSTIC the directive asks for: the protected-core swap with PPR admitted as a
fourth voting channel, at B in {6, 8, 12}, on all six corpora.  A PPR-reached partition with no
S4 and no node-retrieval evidence would otherwise score ~0 and could never win a slot, so without
giving PPR a vote the "capacity" question cannot even be posed.

    capacity-limited   ->  ALL climbs with B
    ranking-limited    ->  ALL stays flat or negative at every B

DIAGNOSTIC ONLY.  Nothing here is promotable and no PPR variant continues past this file.

  python scratchpad/_l1cal_s0.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1pp_core as PP
import _l1kb_core as KB

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0, P = CC.K0, CC.P
M_PPR = 64                      # same depth as the frozen M_struct, so the pools are comparable


def ppr_lists(ds, nq):
    """top-M_PPR partitions per query from the cached global partition PPR of the previous phase."""
    f = f"{PP.PPD}/ppr/mass_{ds}.npz"
    if not os.path.exists(f):
        return None
    m = np.load(f)["mass_global"]
    order = np.argsort(-m, axis=1, kind="stable")[:, :M_PPR]
    out = []
    for qi in range(nq):
        out.append([int(p) for p in order[qi] if m[qi, int(p)] > 0])
    return out


def classify(ctxs, goldp, nq, Bv, extra_pool=None):
    """CAP_P50 / POOL(unreachable) / CAP_B / RANKING-solvable, against a named universe."""
    o = {"CAP_P50": 0, "POOL_UNREACHABLE": 0, f"CAP_B{Bv}": 0, "SOLVABLE": 0}
    for qi, c in enumerate(ctxs):
        g = goldp[qi]
        if len(g) > P:
            o["CAP_P50"] += 1; continue
        uni = set(c["bnd"]) | set(c["chal"])
        if extra_pool is not None:
            uni |= {p for p in extra_pool[qi] if p not in c["base50"]}
        miss = g - c["prot_set"]
        if not (miss <= uni):
            o["POOL_UNREACHABLE"] += 1; continue
        if len(miss) > Bv:
            o[f"CAP_B{Bv}"] += 1; continue
        o["SOLVABLE"] += 1
    return o


def swap_with_ppr(c, K, Bv, plist, ppos):
    """frozen F6 arithmetic with PPR admitted as a fourth co-equal channel."""
    cands = c["bnd"] + [p for p in dict.fromkeys(list(c["chal"]) + plist) if p not in c["bnd"]]
    sc = []
    for p in cands:
        s = 1.0 / (K0 + c["cpos"][p]) if p in c["cpos"] else 0.0
        if p in c["spos"]:
            s += 1.0 / (K0 + c["spos"][p])
        if p in c["rpos"]:
            s += 1.0 / (K0 + c["rpos"][p])
        if p in ppos:
            s += 1.0 / (K0 + ppos[p])
        sc.append((-s, c["cpos"].get(p, 10 ** 6), p))
    sc.sort()
    return [p for _, _, p in sc[:Bv]]


def main():
    OUT = {}
    for ds in CC.DSETS:
        row = {"CLASSES": {}, "PPR_DIAGNOSTIC": {}}
        S = CC.substrate(ds, 6)
        nq, goldp = S["nq"], S["goldp"]
        pl = ppr_lists(ds, nq)
        base_all = float(S["ind_base"].mean())
        row["BASE_ALL"] = round(base_all, 4)
        for Bv in (6, 8, 12):
            Sb = CC.substrate(ds, Bv)
            cur = classify(Sb["ctxs"], goldp, nq, Bv, None)
            exp = classify(Sb["ctxs"], goldp, nq, Bv, pl) if pl else None
            row["CLASSES"][f"B{Bv}"] = {"CAP_B_CURRENT_POOL": cur[f"CAP_B{Bv}"],
                                        "POOL_UNREACHABLE_CURRENT": cur["POOL_UNREACHABLE"],
                                        "CAP_P50": cur["CAP_P50"],
                                        "SOLVABLE_CURRENT": cur["SOLVABLE"]}
            if exp:
                row["CLASSES"][f"B{Bv}"].update(
                    {"CAP_B_PPR_EXPANDED_POOL": exp[f"CAP_B{Bv}"],
                     "POOL_UNREACHABLE_PPR_EXPANDED": exp["POOL_UNREACHABLE"],
                     "SOLVABLE_PPR_EXPANDED": exp["SOLVABLE"]})
            # --- the diagnostic run: does more boundary capacity let PPR pay off?
            f6 = np.zeros(nq, np.int8); pp = np.zeros(nq, np.int8)
            for qi, c in enumerate(Sb["ctxs"]):
                X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], Bv)
                f6[qi] = int(goldp[qi] <= (c["prot_set"] | set(X)))
                if pl:
                    # challengers must live OUTSIDE base50, exactly as the frozen pool does
                    pq = [p for p in pl[qi] if p not in c["base50"]]
                    ppos = {p: r for r, p in enumerate(pq)}
                    Y = swap_with_ppr(c, Sb["KS"][qi], Bv, pq, ppos)
                    assert len(c["prot_set"] | set(Y)) == P
                    pp[qi] = int(goldp[qi] <= (c["prot_set"] | set(Y)))
            e = {"F6_ALL": round(float(f6.mean()), 4),
                 "F6_dBASE": round(float(f6.mean() - base_all), 4)}
            if pl:
                e.update({"F6_PLUS_PPR_ALL": round(float(pp.mean()), 4),
                          "PPR_dBASE": round(float(pp.mean() - base_all), 4),
                          "PPR_dF6": round(float(pp.mean() - f6.mean()), 4),
                          "vs_F6": PP.mcnemar(pp, f6)})
            row["PPR_DIAGNOSTIC"][f"B{Bv}"] = e
            log("%-15s B%-2d  CAP_B cur %4d / ppr-exp %4d   CAP_P50 %3d | F6 %+.4f  F6+PPR %+.4f "
                "(dF6 %+.4f)" % (ds[:14], Bv, cur[f"CAP_B{Bv}"],
                                 exp[f"CAP_B{Bv}"] if exp else -1, cur["CAP_P50"],
                                 e["F6_dBASE"], e.get("PPR_dBASE", float("nan")),
                                 e.get("PPR_dF6", float("nan"))))
        OUT[ds] = row
        os.makedirs(f"{CC.CALD}/diag", exist_ok=True)
        json.dump(OUT, open(f"{CC.CALD}/diag/step0_capacity.json", "w"), indent=1)
    log("wrote step0_capacity.json")


if __name__ == "__main__":
    main()
