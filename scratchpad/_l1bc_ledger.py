"""L1 BACKWARD CAUSAL PHASE -- STEP 1 (backward loss ledger) + STEP 2 (oracle ladder).

Nothing here changes the frozen output.  Gold is used for EVALUATION ONLY.

REQUIRED(q)      = goldp[qi], the partitions the query actually needs (exact-P50 is
                   "REQUIRED subset of the final 50", which is what EV.evaluate scores).
SELECTED_SAFE(q) = prot(44) | f6_select(...) -- the frozen selection, reproduced from the ctx and
                   gated against the known frozen accuracy.
MISSING(q)       = REQUIRED - SELECTED_SAFE.

Every missing required partition gets ONE primary label, assigned by walking the backward chain in
the directive order and stopping at the FIRST stage that prevents recovery:

  P50_CAPACITY               more required partitions than the contract has slots
  UNREACHABLE                the bounded machinery never touched it at all
  NOT_IN_CANDIDATE_UNIVERSE  touched, but never offered to the selector
  B_CAPACITY                 offered, but more than B=6 of them were needed at once
  POINTWISE_RANKING          offered and affordable, yet NO single evidence column ranks it top-B
                             among the candidates -- no individual score would have found it
  REDUNDANCY                 a chosen challenger contributes zero new evidence while this one does
  SET_SELECTION              Pareto non-dominated among the candidates and still not chosen
  FINAL_FUSION               Pareto-dominated, but at least one single column did have it top-B --
                             the equal-RRF fusion is what lost it

The secondary co-occurrence matrix is reported too, so the cost of the ordering is visible.

ORACLE LADDER (STEP 2).  prot(44) is the frozen protected core in every rung.
  O0 SAFE actual
  O1 perfect selection over the frozen candidate universe bnd|chal, B=6
  O2 same universe, B=50 (may evict prot)          -- the "is B blocking it" probe
  O3 perfect selection over everything VISITED, B=6
  O4 everything visited, B=50
  O5 |REQUIRED| <= 50
The mutually exclusive chain is O0 <= O1 <= O3 <= O4 <= O5; O2 sits off-chain between O1 and O4.

  python scratchpad/_l1bc_ledger.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1kb_core as KB
import _l1bc_core as BC

P, B, K0, MISS = BC.P, BC.B, BC.K0, BC.MISS
RCOL = BC.RCOL
LABELS = ["P50_CAPACITY", "UNREACHABLE", "NOT_IN_CANDIDATE_UNIVERSE", "B_CAPACITY",
          "POINTWISE_RANKING", "REDUNDANCY", "SET_SELECTION", "FINAL_FUSION"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def safe_pick(c):
    """the frozen F6 selection, rebuilt from the ctx alone.  X, ordered candidate pool."""
    b50, bnd, cpos, rpos, spos = c["base50"], c["bnd"], c["cpos"], c["rpos"], c["spos"]
    SF = [p for p, _ in sorted(spos.items(), key=lambda kv: kv[1])]
    chal = [p for p in SF if p not in b50] + [p for p in c["_RF"] if p not in b50]
    X, sc = KB.f6_select(bnd, chal, spos, rpos, cpos, B)
    cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
    return list(X), cands, sc


def pareto_front(M):
    """M[i, d] smaller-is-better.  returns bool non-dominated mask and the domination count."""
    n = len(M)
    nd = np.ones(n, bool)
    dom = np.zeros(n, np.int32)
    for j in range(n):
        le = (M <= M[j]).all(1)
        lt = (M < M[j]).any(1)
        d = le & lt
        d[j] = False
        dom[j] = int(d.sum())
        if d.any():
            nd[j] = False
    return nd, dom


def run(ds, log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)

    O = {k: np.zeros(nq, np.int8) for k in ["O0", "O1", "O2", "O3", "O4", "O5"]}
    cnt = {L: np.zeros(nq, np.int32) for L in LABELS}
    co = {L: np.zeros(len(LABELS), np.int64) for L in LABELS}
    nmiss = np.zeros(nq, np.int32)
    nreq = np.zeros(nq, np.int32)
    ncand = np.zeros(nq, np.int32)

    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        pid = r["pid"]
        ixp = {int(p): i for i, p in enumerate(pid)}
        REQ = set(int(p) for p in goldp[qi])
        prot = set(c["prot"])
        X, cands, _ = safe_pick(c)
        sel = prot | set(X)
        assert len(sel) == P, f"{len(sel)} != {P}"
        candset = set(int(p) for p in cands)
        reach = set(int(p) for p, f in zip(pid, r["in_reach"]) if f)
        nreq[qi] = len(REQ)

        # ---------------- STEP 2 oracle ladder
        out = REQ - prot
        O["O0"][qi] = int(REQ <= sel)
        O["O1"][qi] = int(len(out) <= B and out <= candset)
        O["O2"][qi] = int(len(REQ) <= P and REQ <= (prot | candset))
        O["O3"][qi] = int(len(out) <= B and out <= reach)
        O["O4"][qi] = int(len(REQ) <= P and REQ <= (prot | reach))
        O["O5"][qi] = int(len(REQ) <= P)

        miss = sorted(REQ - sel)
        nmiss[qi] = len(miss)
        ncand[qi] = len(cands)
        if not miss:
            continue

        # ---------------- STEP 1 ledger.  columns, coverage matrix, Pareto, all over `cands`.
        cix = np.array([ixp[p] for p in cands], np.int64)
        M = np.stack([r[k][cix].astype(np.float64) for k in RCOL], 1)
        nd, _ = pareto_front(M)
        ndm = {int(p): bool(nd[i]) for i, p in enumerate(cands)}
        topB = {int(p): False for p in cands}
        for d in range(M.shape[1]):
            ordr = np.lexsort((cix, M[:, d]))
            for t in ordr[:B]:
                if M[t, d] < MISS:
                    topB[int(cands[t])] = True

        n_atom, n = r["n_atom"], r["n"]
        Wd = np.zeros((max(n_atom, 1), n))
        if n_atom:
            Wd[r["A_aid"], r["A_pix"]] = r["A_w"]
        selix = np.array([ixp[p] for p in sel if p in ixp], np.int64)
        cov = Wd[:, selix].max(1) if len(selix) else np.zeros(len(Wd))
        # a chosen challenger is REDUNDANT iff dropping it leaves the covered evidence unchanged
        waste = False
        for x in X:
            oth = np.array([ixp[p] for p in sel if p != x and p in ixp], np.int64)
            if len(oth) and np.allclose(Wd[:, oth].max(1), cov):
                waste = True
                break

        # P50_CAPACITY: the excess beyond the contract, charged to the worst by canonical rank
        cap = set()
        if len(REQ) > P:
            byc = sorted(miss, key=lambda p: (-(r["r_canon"][ixp[p]] if p in ixp else MISS), p))
            cap = set(byc[:len(REQ) - P])
        rest = [p for p in miss if p not in cap]
        unre = set(p for p in rest if p not in reach)
        rest = [p for p in rest if p not in unre]
        nocu = set(p for p in rest if p not in candset)
        rest = [p for p in rest if p not in nocu]
        # B_CAPACITY: more simultaneously-needed candidates than the swap allowance
        bcap = set()
        if len(rest) > B:
            best = lambda p: min(float(r[k][ixp[p]]) for k in RCOL)
            bcap = set(sorted(rest, key=lambda p: (-best(p), p))[:len(rest) - B])
        rest = [p for p in rest if p not in bcap]

        for p in miss:
            if p in cap:
                L = "P50_CAPACITY"
            elif p in unre:
                L = "UNREACHABLE"
            elif p in nocu:
                L = "NOT_IN_CANDIDATE_UNIVERSE"
            elif p in bcap:
                L = "B_CAPACITY"
            elif not topB.get(p, False):
                L = "POINTWISE_RANKING"
            elif waste and (p in ixp) and Wd[:, ixp[p]].max(initial=0.0) > 0 and \
                    not np.allclose(np.maximum(cov, Wd[:, ixp[p]]), cov):
                L = "REDUNDANCY"
            elif ndm.get(p, False):
                L = "SET_SELECTION"
            else:
                L = "FINAL_FUSION"
            cnt[L][qi] += 1
            sec = [p in cap, p in unre, p in nocu, p in bcap, not topB.get(p, False),
                   bool(waste and p in ixp and not np.allclose(
                       np.maximum(cov, Wd[:, ixp[p]]), cov)), ndm.get(p, False),
                   bool(topB.get(p, False) and not ndm.get(p, False))]
            co[L] += np.asarray(sec, np.int64)

        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}")

    return dict(ds=ds, nq=nq, hops=hops, O=O, cnt=cnt, co=co, nmiss=nmiss, nreq=nreq, ncand=ncand)


def masks(ds, hops, nq):
    if ds == "metaqa":
        h = np.asarray(hops)
        return [("hop1", h == 1), ("hop2", h == 2), ("hop3", h == 3), ("ALL", np.ones(nq, bool))]
    return [("ALL", np.ones(nq, bool))]


def report(R):
    ds, nq = R["ds"], R["nq"]
    MS = masks(ds, R["hops"], nq)
    out = {"ds": ds, "nq": nq, "oracle": {}, "ledger": {}, "chain": {}, "co": {}}
    for nm, m in MS:
        out["oracle"][nm] = {k: round(float(R["O"][k][m].mean()), 4) for k in
                             ["O0", "O1", "O2", "O3", "O4", "O5"]}
        o = out["oracle"][nm]
        out["chain"][nm] = {
            "SELECTION_O1_O0": round(o["O1"] - o["O0"], 4),
            "CANDIDATE_UNIVERSE_O3_O1": round(o["O3"] - o["O1"], 4),
            "B_CAPACITY_O4_O3": round(o["O4"] - o["O3"], 4),
            "REACH_O5_O4": round(o["O5"] - o["O4"], 4),
            "P50_CAPACITY_1_O5": round(1.0 - o["O5"], 4),
            "B50_IN_UNIVERSE_O2_O1": round(o["O2"] - o["O1"], 4),
            "TOTAL_1_O0": round(1.0 - o["O0"], 4)}
        tot = float(R["nmiss"][m].sum())
        out["ledger"][nm] = {"missing_total": int(tot), "missing_per_q": round(tot / max(m.sum(), 1), 3),
                             "required_per_q": round(float(R["nreq"][m].mean()), 3),
                             "candidates_per_q": round(float(R["ncand"][m].mean()), 1)}
        for L in LABELS:
            v = int(R["cnt"][L][m].sum())
            out["ledger"][nm][L] = v
            out["ledger"][nm][L + "_pct"] = round(100.0 * v / tot, 1) if tot else 0.0
    out["co"] = {L: {LABELS[i]: int(v) for i, v in enumerate(R["co"][L]) if v} for L in LABELS
                 if R["cnt"][L].sum()}
    return out


if __name__ == "__main__":
    dss = sys.argv[1:] or ["metaqa"]
    os.makedirs(f"{BC.BCD}/diag", exist_ok=True)
    for ds in dss:
        R = run(ds)
        rep = report(R)
        with open(f"{BC.BCD}/diag/ledger_{ds}.json", "w") as fh:
            json.dump(rep, fh, indent=1)
        log(ds, json.dumps(rep["oracle"], indent=1))
        log(ds, json.dumps(rep["ledger"], indent=1))
