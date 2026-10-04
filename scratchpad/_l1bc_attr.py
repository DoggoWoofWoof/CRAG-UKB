"""L1 BACKWARD CAUSAL PHASE -- STEP 11 (per-query causal attribution) + STEP 12 (latency).

STEP 11 answers "why did THIS query change", never "the aggregate went up".  For every query whose
exact-P50 outcome differs from SAFE the record names the partition that entered or left, the
evidence that put it there, and -- for a loss -- the SAFE required partition that was displaced and
what displaced it.

  GAIN mechanisms (a gained query can carry more than one; the primary is the first that holds)
    ASSIGNMENT_DIVERSIFICATION      the introduced partition won atoms of a FAMILY no SAFE pick won
    INDEPENDENT_EVIDENCE_COVERAGE   it won at least one atom outright in the assignment
    DOMINATED_INCUMBENT_REMOVED     the SAFE pick it replaced was Pareto-dominated in the pool
    NOVELTY                         it covered an evidence atom the SAFE final set covered not at all
    ORDER_ONLY                      none of the above -- the tie-break alone moved it

  LOSS mechanisms
    DISPLACED_REQUIRED_WON_NO_ATOM  the lost required partition won zero atoms, so the assignment
                                    could not see it at all
    DISPLACED_BY_LARGER_MASS        it did win atoms, but the partition taking its slot won more
    DISPLACED_BY_ORDER              it won no less mass; the tie-break moved it out

STEP 12 times the selection stage ALONE against the frozen L1 work it sits on top of (the residual
and the bounded traversal are unchanged by every method here, so they are the denominator).

  python scratchpad/_l1bc_attr.py <ds> [method]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1ss_core as SS
import _l1sr_eval as EV
import _l1pp_core as PP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1bc_methods as ME
import _l1tp_core as TP

P, B, MISS = BC.P, BC.B, BC.MISS
RCOL = BC.RCOL
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
GAINM = ["ASSIGNMENT_DIVERSIFICATION", "INDEPENDENT_EVIDENCE_COVERAGE",
         "DOMINATED_INCUMBENT_REMOVED", "NOVELTY", "ORDER_ONLY"]
LOSSM = ["DISPLACED_REQUIRED_WON_NO_ATOM", "DISPLACED_BY_LARGER_MASS", "DISPLACED_BY_ORDER"]


def fam_of(aid, fam):
    nch, nsd = int(fam[0]), int(fam[1])
    return "CH" if aid < nch else ("SEED" if aid < nch + nsd else "SRC")


def attribute(ds, method="G4_ASSIGNMENT", log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    gm = {k: 0 for k in GAINM}
    lm = {k: 0 for k in LOSSM}
    rows, hs, hm = [], np.zeros(nq, np.int8), np.zeros(nq, np.int8)
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        X, cands, sc = LG.safe_pick(c)
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        pool = [int(p) for p in cands if int(p) in ixp]
        REQ = set(int(p) for p in goldp[qi])
        s, tie = ME.scores(r, ixp, prot, pool, safe_ord, method)
        mtop = [pool[j] for j in np.lexsort((tie, -s))][:B]
        stop = [int(p) for p in X]
        hs[qi] = int(REQ <= (pset | set(stop)))
        hm[qi] = int(REQ <= (pset | set(mtop)))
        if hs[qi] == hm[qi]:
            continue
        # ---- evidence for the attribution
        n_atom = r["n_atom"]
        Wd = np.zeros((max(n_atom, 1), r["n"]))
        if n_atom:
            Wd[r["A_aid"], r["A_pix"]] = r["A_w"]
        allp = prot + pool
        allix = np.array([ixp[p] for p in allp], np.int64)
        Wa = Wd[:, allix]
        win = np.asarray(np.argmax(Wa, 1))
        best = Wa[np.arange(len(Wa)), win]
        won = {}
        for e in range(len(Wa)):
            if best[e] > 0:
                won.setdefault(allp[win[e]], []).append((e, float(best[e])))
        pix = np.array([ixp[p] for p in pool], np.int64)
        M = np.stack([r[k][pix].astype(np.float64) for k in RCOL], 1)
        nd, _ = LG.pareto_front(M)
        ndm = {int(p): bool(nd[i]) for i, p in enumerate(pool)}
        selix = np.array([ixp[p] for p in (pset | set(stop)) if p in ixp], np.int64)
        cov = Wd[:, selix].max(1) if len(selix) else np.zeros(len(Wd))
        famof = lambda p: set(fam_of(e, r["fam"]) for e, _ in won.get(p, []))
        mass = lambda p: sum(w for _, w in won.get(p, []))
        safe_fams = set().union(*[famof(p) for p in stop]) if stop else set()

        entered = [p for p in mtop if p not in stop]
        leftout = [p for p in stop if p not in mtop]
        if hm[qi] > hs[qi]:
            key = [p for p in entered if p in REQ] or entered
            p0 = key[0]
            if famof(p0) - safe_fams:
                g = "ASSIGNMENT_DIVERSIFICATION"
            elif won.get(p0):
                g = "INDEPENDENT_EVIDENCE_COVERAGE"
            elif any(not ndm.get(q, True) for q in leftout):
                g = "DOMINATED_INCUMBENT_REMOVED"
            elif p0 in ixp and not np.allclose(np.maximum(cov, Wd[:, ixp[p0]]), cov):
                g = "NOVELTY"
            else:
                g = "ORDER_ONLY"
            gm[g] += 1
            rows.append({"q": qi, "hop": int(hops[qi]), "kind": "GAIN", "mech": g,
                         "entered": [int(x) for x in key[:3]],
                         "entered_required": [int(x) for x in entered if x in REQ],
                         "won_families": sorted(famof(p0)), "won_mass": round(mass(p0), 5),
                         "won_atoms": len(won.get(p0, [])),
                         "safe_pick_families": sorted(safe_fams),
                         "displaced": [int(x) for x in leftout],
                         "safe_order_index": safe_ord.get(p0, -1)})
        else:
            key = [p for p in leftout if p in REQ] or leftout
            p0 = key[0]
            takers = sorted(entered, key=lambda q: -mass(q))
            if not won.get(p0):
                l = "DISPLACED_REQUIRED_WON_NO_ATOM"
            elif takers and mass(takers[0]) > mass(p0):
                l = "DISPLACED_BY_LARGER_MASS"
            else:
                l = "DISPLACED_BY_ORDER"
            lm[l] += 1
            rows.append({"q": qi, "hop": int(hops[qi]), "kind": "LOSS", "mech": l,
                         "lost_required": [int(x) for x in key[:3]],
                         "lost_won_mass": round(mass(p0), 5), "lost_won_atoms": len(won.get(p0, [])),
                         "lost_on_front": bool(ndm.get(p0, False)),
                         "displaced_by": [int(x) for x in takers[:3]],
                         "taker_families": sorted(famof(takers[0])) if takers else [],
                         "taker_mass": round(mass(takers[0]), 5) if takers else 0.0,
                         "safe_order_index": safe_ord.get(p0, -1)})
        if (qi + 1) % 500 == 0:
            log(f"   {ds} attr {qi+1}/{nq}")
    st = PP.mcnemar(hm, hs)
    return {"ds": ds, "method": method, "nq": nq, "safe": round(float(hs.mean()), 4),
            "meth": round(float(hm.mean()), 4), "mcnemar": {k: (int(v) if isinstance(v, (int, np.integer))
            else (round(float(v), 4) if isinstance(v, float) else bool(v))) for k, v in st.items()},
            "gain_mech": gm, "loss_mech": lm, "n_rows": len(rows), "rows": rows}


def latency(ds, n=120, log=log):
    """STEP 12 -- the frozen L1 work is the denominator; only the selection stage is new."""
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    z = S["z"]
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    n = min(n, int(T["nq"][0]))
    t_res = t_exp = t_safe = 0.0
    t_m = {m: 0.0 for m in ME.METHODS}
    for qi in range(n):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        t = time.perf_counter()
        rq = TA.residual(C["Qm"][qi].astype(np.float64), sd, C["Xn"])
        t_res += time.perf_counter() - t
        t = time.perf_counter()
        SS.expand_feat(sd, rq, C["adjp"], C["adji"], C["deg"], C["Xn"], C["W"], want_future=False)
        t_exp += time.perf_counter() - t
        c = S["ctxs"][qi]
        t = time.perf_counter()
        X, cands, sc = LG.safe_pick(c)
        t_safe += time.perf_counter() - t
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        pool = [int(p) for p in cands if int(p) in ixp]
        for m in ME.METHODS:
            t = time.perf_counter()
            s, tie = ME.scores(r, ixp, prot, pool, safe_ord, m)
            np.lexsort((tie, -s))
            t_m[m] += time.perf_counter() - t
    f = 1000.0 / n
    base = (t_res + t_exp + t_safe) * f
    out = {"ds": ds, "n": n, "L1_RESIDUAL_MS": round(t_res * f, 3),
           "L1_TRAVERSAL_MS": round(t_exp * f, 3), "L1_SAFE_SELECTOR_MS": round(t_safe * f, 3),
           "L1_FROZEN_TOTAL_MS": round(base, 3)}
    for m in ME.METHODS:
        ov = t_m[m] * f
        out[m + "_MS"] = round(ov, 3)
        out[m + "_PCT_OF_L1"] = round(100.0 * ov / (base + ov), 2)
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
    meth = sys.argv[2] if len(sys.argv) > 2 else "G4_ASSIGNMENT"
    A = attribute(ds, meth)
    with open(f"{BC.BCD}/diag/attr_{ds}_{meth}.json", "w") as fh:
        json.dump(A, fh, indent=1)
    L = latency(ds)
    with open(f"{BC.BCD}/diag/lat_{ds}.json", "w") as fh:
        json.dump(L, fh, indent=1)
    log(ds, meth, json.dumps({k: A[k] for k in ("safe", "meth", "mcnemar", "gain_mech",
                                                "loss_mech", "n_rows")}, indent=1))
    log(ds, "LATENCY", json.dumps(L, indent=1))
