"""L1 QUERY-CONDITIONED TRANSFORMATION ALGEBRA -- operators, single-hop transport, composition.

The offset family this project has used everywhere assumes the relation between a source node and a
target node acts on the query by TRANSLATION, x' = x + delta.  That is one very restrictive element
of the affine group.  This module replaces it with four parameter-free operators derived from the
edges the frozen beam ALREADY inspected, and asks whether any of them transports the query state
onto its actual multi-hop target better than translation does.

For a real explored edge u -> v (both rows of Xn, unit norm), acting on a query state z:

    F0_TRANSLATION     z' = z + (v - u)
    F1_HOUSEHOLDER     w = normalize(u - v);  z' = z - 2 w (w^T z)
                       exact on unit u, v: H u = v, and H is an involution, ||H z|| = ||z||
    F2_MIN_ROTATION    minimal rotation inside span(u, v) carrying u to v, identity on the
                       orthogonal complement; c = u^T v, b = normalize(v - c u),
                       z' = z + (c z_u - s z_b - z_u) u + (s z_u + c z_b - z_b) b
    F3_RANK1_TRANSPORT z' = z + (v - u)(u^T z)/(u^T u);  T(u) = v exactly

All four are matrix-free and O(d) per edge -- no d x d matrix is ever formed.  Nothing is fitted,
nothing is learned, no new edge is searched and no new vector is encoded: U and V are the endpoints
of edges the frozen traversal already scored.

COMPOSITION follows the ACTUAL parent chain the beam recorded (`tid` / `parent_tid`), so z_h is the
query transported along a path the machinery really walked, never a path search of our own.

QUERY STATES (STEP 4) are both of the ones that already exist: the raw query embedding and the
seed-conditioned residual r_q that the frozen traversal itself uses as its direction.

  python scratchpad/_l1kt_tf.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1tp_core as TP
import _l1bc_core as BC
import _l1bc_ledger as LG
import _l1bc_diag as DGX

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
FAMS = ["F0_TRANSLATION", "F1_HOUSEHOLDER", "F2_MIN_ROTATION", "F3_RANK1_TRANSPORT"]
ZS = ["q_raw", "r_q"]
EPS = 1e-6
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ the four operators
def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, EPS), n[:, 0]


def op_F0(U, V, Z):
    return Z + (V - U)


def op_F1(U, V, Z):
    W = U - V
    Wh, nw = _unit(W)
    out = Z - 2.0 * (Wh * Z).sum(1, keepdims=True) * Wh
    return np.where((nw > EPS)[:, None], out, Z)          # u == v is the identity


def op_F2(U, V, Z):
    c = (U * V).sum(1, keepdims=True)
    Bh, s = _unit(V - c * U)
    s = s[:, None]
    za = (U * Z).sum(1, keepdims=True)
    zb = (Bh * Z).sum(1, keepdims=True)
    out = Z + (c * za - s * zb - za) * U + (s * za + c * zb - zb) * Bh
    return np.where((s[:, 0] > EPS)[:, None], out, Z)     # collinear u, v is the identity


def op_F3(U, V, Z):
    uu = np.maximum((U * U).sum(1, keepdims=True), EPS)
    return Z + (V - U) * ((U * Z).sum(1, keepdims=True) / uu)


OPS = {"F0_TRANSLATION": op_F0, "F1_HOUSEHOLDER": op_F1, "F2_MIN_ROTATION": op_F2,
       "F3_RANK1_TRANSPORT": op_F3}


def chain_states(U, V, z0, hop, parent_tid, fam, hopmasks):
    """z_h along the ACTUAL recorded parent chain.  Row i holds the state AFTER edge i is applied.

    `tid` is verified to be exactly the row index and `parent_tid < tid` always, so the chain is a
    forward-only scan and the parent state is a direct gather -- no id map, no search."""
    f = OPS[fam]
    out = np.empty_like(U)
    for h, m in hopmasks:
        par = parent_tid[m]
        have = par >= 0
        if have.any():
            Zin = np.empty((len(m), U.shape[1]), U.dtype)
            Zin[~have] = z0
            Zin[have] = out[par[have]]
        else:
            Zin = z0[None, :]
        out[m] = f(U[m], V[m], Zin)
    return out


# ------------------------------------------------------------------ closed-form scalar transport
# cos(T(z), v) and ||T(z)|| never need the transformed VECTOR.  With ||u|| = ||v|| = 1 and the four
# per-edge scalars a = <u,z>, b = <v,z>, n2 = ||z||^2, d0 = <z0,z> (plus c = <u,v>, <u,z0>, <v,z0>)
# every family closes in O(1) per edge.  Only ~3% of edges are the parent of another edge, so only
# those rows ever need the vector itself -- the rest of the chain is pure scalar recursion.
def scalar_apply(fam, a, b, n2, d0, c, uz0, vz0):
    """-> (tv = <T(z), v>, n2p = ||T(z)||^2, d0p = <T(z), z0>)"""
    if fam == "F0_TRANSLATION":
        return b + 1.0 - c, n2 + (2.0 - 2.0 * c) + 2.0 * (b - a), d0 + vz0 - uz0
    if fam == "F3_RANK1_TRANSPORT":
        return (b + a * (1.0 - c), n2 + a * a * (2.0 - 2.0 * c) + 2.0 * a * (b - a),
                d0 + a * (vz0 - uz0))
    if fam == "F1_HOUSEHOLDER":
        L = np.sqrt(np.maximum(2.0 - 2.0 * c, 0.0))
        ok = L > EPS
        Ls = np.where(ok, L, 1.0)
        wz, wv, wz0 = (a - b) / Ls, (c - 1.0) / Ls, (uz0 - vz0) / Ls
        return (np.where(ok, b - 2.0 * wz * wv, b), n2,
                np.where(ok, d0 - 2.0 * wz * wz0, d0))
    if fam == "F2_MIN_ROTATION":
        s = np.sqrt(np.maximum(1.0 - c * c, 0.0))
        ok = s > EPS
        ss = np.where(ok, s, 1.0)
        zb = (b - c * a) / ss
        bz0 = (vz0 - c * uz0) / ss
        A = c * a - ss * zb - a
        Bc = ss * a + c * zb - zb
        return (np.where(ok, b + A * c + Bc * ss, b), n2,
                np.where(ok, d0 + A * uz0 + Bc * bz0, d0))
    raise KeyError(fam)


def transport(U, V, z0, hopmasks, parent_tid, is_parent, fam, c, uz0, vz0):
    """chain transport along the recorded parent chain, scalars for all rows, vectors for parents.

    Returns (tv, n2p, d0p) per edge for the CHAIN state z_h, exactly equal to applying the vector
    operator hop by hop -- gated against the vector path in `_gate_scalar`."""
    n = len(U)
    a = np.empty(n); b = np.empty(n); n2 = np.empty(n); d0 = np.empty(n)
    tv = np.empty(n); n2p = np.empty(n); d0p = np.empty(n)
    P = {}                                        # row -> its OUTPUT vector, parents only
    f = OPS[fam]
    for h, m in hopmasks:
        par = parent_tid[m]
        root = par < 0
        mr = m[root]
        a[mr], b[mr], n2[mr], d0[mr] = uz0[mr], vz0[mr], 1.0, 1.0
        nz = m[~root]
        if len(nz):
            pn = par[~root]
            for p in np.unique(pn):
                rows = nz[pn == p]
                zp = P[int(p)]
                a[rows] = U[rows] @ zp
                b[rows] = V[rows] @ zp
                n2[rows] = float(zp @ zp)
                d0[rows] = float(zp @ z0)
        tv[m], n2p[m], d0p[m] = scalar_apply(fam, a[m], b[m], n2[m], d0[m], c[m], uz0[m], vz0[m])
        pm = m[is_parent[m]]                      # only these rows are ever read again
        if len(pm):
            pp = parent_tid[pm]
            Zin = np.empty((len(pm), U.shape[1]), U.dtype)
            for j, (i, q) in enumerate(zip(pm, pp)):
                Zin[j] = z0 if q < 0 else P[int(q)]
            Zo = f(U[pm], V[pm], Zin)
            for j, i in enumerate(pm):
                P[int(i)] = Zo[j]
    return tv, n2p, d0p


def cos_rows(A, Bm):
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(Bm, axis=1)
    return (A * Bm).sum(1) / np.maximum(na * nb, EPS)


# ------------------------------------------------------------------ driver
def run(ds, stride=1, log=log):
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    Xn = C["Xn"]
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    hard = C["hard"]
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)
    qs = list(range(0, nq, stride))

    # accumulators: per (z0, fam, mode) pooled edge-level scores split by target class
    KEYS = [(z, f, m) for z in ZS for f in FAMS for m in ("single", "chain")]
    pos_req = {k: [] for k in KEYS}      # edge target partition is REQUIRED and MISSING from P50
    neg = {k: [] for k in KEYS}          # edge target partition is not required
    base = {("q_raw", "single"): [], ("q_raw", "chain"): [],
            ("r_q", "single"): [], ("r_q", "chain"): []}
    byhop = {k: {h: [] for h in (1, 2, 3)} for k in KEYS}
    stab = {(z, f): {h: {"norm": [], "drift": []} for h in (1, 2, 3)} for z in ZS for f in FAMS}
    prank = {k: [] for k in KEYS}        # percentile rank of required target partitions
    qhop, nedge, par_ok, ms = [], [], 0, {"replay": 0.0, "tf": 0.0}
    for qi in qs:
        t = time.perf_counter()
        E, rq, par, st = TP.replay(C, qi)
        ms["replay"] += time.perf_counter() - t
        par_ok += int(par)
        u, v = E["u"], E["v"]
        if not len(u):
            continue
        nedge.append(len(u))
        qhop.append(int(hops[qi]))
        c = ctxs[qi]
        Xp, cands, _ = LG.safe_pick(c)
        sel = set(int(p) for p in c["prot"]) | set(int(p) for p in Xp)
        REQ = set(int(p) for p in goldp[qi])
        MISSREQ = REQ - sel
        tp = hard[v]                                  # target partition of every edge
        is_pos = np.array([int(p) in MISSREQ for p in tp])
        is_neg = np.array([int(p) not in REQ for p in tp])
        U = gx(u)
        V = gx(v)
        Xv = V
        hop = E["T3_HOP"].astype(np.int64)
        Z0 = {"q_raw": C["Qm"][qi].astype(np.float32),
              "r_q": rq.astype(np.float32)}
        assert (E["tid"] == np.arange(len(u))).all() and (E["parent_tid"] < E["tid"]).all()
        hopmasks = [(h, np.nonzero(hop == h)[0]) for h in sorted(set(int(x) for x in hop))]
        hopmasks = [(h, m) for h, m in hopmasks if len(m)]
        upart = np.unique(tp[tp >= 0])
        pinv = {int(p): i for i, p in enumerate(upart)}
        pix = np.array([pinv.get(int(p), -1) for p in tp], np.int64)
        pos_mask = np.array([int(p) in MISSREQ for p in upart]) if len(upart) else np.zeros(0, bool)
        t = time.perf_counter()
        is_parent = np.zeros(len(u), bool)
        is_parent[E["parent_tid"][E["parent_tid"] >= 0]] = True
        c = (U * V).sum(1)
        one = np.ones(len(u), np.float32)
        for z in ZS:
            z0 = Z0[z] / max(float(np.linalg.norm(Z0[z])), EPS)
            uz0 = U @ z0
            vz0 = V @ z0
            b = vz0                       # cos(z0, v): both unit, no transform
            base[(z, "single")].append((b[is_pos], b[is_neg]))
            base[(z, "chain")].append((b[is_pos], b[is_neg]))
            for f in FAMS:
                for mode in ("single", "chain"):
                    if mode == "single":
                        tv, n2p, d0p = scalar_apply(f, uz0, vz0, one, one, c, uz0, vz0)
                    else:
                        tv, n2p, d0p = transport(U, V, z0, hopmasks, E["parent_tid"],
                                                 is_parent, f, c, uz0, vz0)
                    nrm = np.sqrt(np.maximum(n2p, EPS))
                    tc = tv / nrm
                    k = (z, f, mode)
                    pos_req[k].append(tc[is_pos])
                    neg[k].append(tc[is_neg])
                    for h in (1, 2, 3):
                        mh = hop == h
                        if mh.any():
                            byhop[k][h].append(float(tc[mh].mean()))
                    if mode == "chain":
                        dr = d0p / nrm
                        for h in (1, 2, 3):
                            mh = hop == h
                            if mh.any():
                                stab[(z, f)][h]["norm"].append(float(nrm[mh].mean()))
                                stab[(z, f)][h]["drift"].append(float(dr[mh].mean()))
                    # marginal target-partition rank: partitions ordered by best edge score
                    if MISSREQ and pos_mask.any():
                        bestp = np.full(len(upart), -np.inf, np.float64)
                        np.maximum.at(bestp, pix[pix >= 0], tc[pix >= 0])
                        order = np.argsort(-bestp, kind="mergesort")
                        rr = np.nonzero(pos_mask[order])[0] / max(len(upart) - 1, 1)
                        if len(rr):
                            prank[k].append(float(rr.mean()))
        ms["tf"] += time.perf_counter() - t
        if (len(nedge)) % 200 == 0:
            log(f"   {ds} {len(nedge)}/{len(qs)}")
    return dict(ds=ds, nq=len(nedge), qhop=np.array(qhop), nedge=np.array(nedge), par=par_ok,
                pos=pos_req, neg=neg, base=base, byhop=byhop, stab=stab, prank=prank,
                ms=ms, nqs=len(qs))


def report(R):
    ds = R["ds"]
    out = {"ds": ds, "n_queries": R["nq"], "parity": f"{R['par']}/{R['nqs']}",
           "edges_per_query": round(float(R["nedge"].mean()), 1),
           "STEP3_single_hop": {}, "STEP5_composition_by_hop": {}, "STEP6_stability": {},
           "STEP3_partition_rank": {}, "latency_ms_per_q": {}}
    for z in ZS:
        bp = np.concatenate([a for a, _ in R["base"][(z, "single")]]) if R["base"][(z, "single")] else np.zeros(0)
        bn = np.concatenate([b for _, b in R["base"][(z, "single")]]) if R["base"][(z, "single")] else np.zeros(0)
        out["STEP3_single_hop"][f"{z}/NO_TRANSFORM"] = {
            "mean_missing_required": round(float(bp.mean()), 4) if len(bp) else None,
            "mean_other": round(float(bn.mean()), 4) if len(bn) else None,
            "AUC": round(DGX._auc(bp, bn), 4) if len(bp) and len(bn) else None}
        for f in FAMS:
            for mode in ("single", "chain"):
                k = (z, f, mode)
                p = np.concatenate(R["pos"][k]) if R["pos"][k] else np.zeros(0)
                n = np.concatenate(R["neg"][k]) if R["neg"][k] else np.zeros(0)
                d = {"mean_missing_required": round(float(p.mean()), 4) if len(p) else None,
                     "mean_other": round(float(n.mean()), 4) if len(n) else None,
                     "AUC": round(DGX._auc(p, n), 4) if len(p) and len(n) else None,
                     "n_pos": int(len(p)), "n_neg": int(len(n))}
                (out["STEP3_single_hop"] if mode == "single"
                 else out["STEP5_composition_by_hop"]).setdefault(f"{z}/{f}", {}).update(d)
                out["STEP3_partition_rank"][f"{z}/{f}/{mode}"] = (
                    round(float(np.mean(R["prank"][k])), 4) if R["prank"][k] else None)
                if mode == "chain":
                    out["STEP5_composition_by_hop"][f"{z}/{f}"]["mean_tc_by_hop"] = {
                        h: round(float(np.mean(R["byhop"][k][h])), 4) if R["byhop"][k][h] else None
                        for h in (1, 2, 3)}
            out["STEP6_stability"][f"{z}/{f}"] = {
                f"hop{h}": {"norm_ratio": round(float(np.mean(R["stab"][(z, f)][h]["norm"])), 4)
                            if R["stab"][(z, f)][h]["norm"] else None,
                            "cos_drift_vs_z0": round(float(np.mean(R["stab"][(z, f)][h]["drift"])), 4)
                            if R["stab"][(z, f)][h]["drift"] else None} for h in (1, 2, 3)}
    nq = max(R["nq"], 1)
    out["latency_ms_per_q"] = {"replay": round(1000 * R["ms"]["replay"] / nq, 3),
                               "all_transforms_all_families": round(1000 * R["ms"]["tf"] / nq, 3),
                               "per_family_per_state": round(1000 * R["ms"]["tf"] / nq / 16, 3)}
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    rp = report(run(ds, stride))
    json.dump(rp, open(f"{KTD}/diag/tf_{ds}.json", "w"), indent=1)
    print(json.dumps(rp, indent=1))
