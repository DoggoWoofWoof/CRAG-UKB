"""STEP 1 / STEP 2 / STEP 6 gates for the transformation algebra -- persisted, not ad hoc.

Checks, on REAL replayed edges (nothing synthetic):
  G1  every operator maps its own source endpoint onto its own target endpoint, T(u) = v
  G2  the two orthogonal families are exact isometries, ||T z|| = ||z||
  G3  F2_MIN_ROTATION is the identity on span(u, v)^perp, F1_HOUSEHOLDER on (u - v)^perp
  G4  F1 is an involution
  G5  the composed chain equals sequential manual application of the same operators
  G6  the O(1) closed-form scalar transport equals the O(d) vector path

  python scratchpad/_l1kt_gates.py [ds] [n_queries]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1tp_core as TP
import _l1kt_tf as TF

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
OPS = {"F0_TRANSLATION": TF.op_F0, "F1_HOUSEHOLDER": TF.op_F1,
       "F2_MIN_ROTATION": TF.op_F2, "F3_RANK1_TRANSPORT": TF.op_F3}


def main(ds="musique_clean", nqq=6):
    C = TP.ctx(ds)
    Xn = C["Xn"]
    gx = lambda ix: np.asarray(Xn[ix], np.float32)
    U, V, EE = [], [], []
    for qi in range(0, 2000, 331):
        E, rq, par, st = TP.replay(C, qi)
        if len(E["u"]):
            U.append(gx(E["u"])); V.append(gx(E["v"])); EE.append((E, rq))
        if len(U) >= nqq:
            break
    U = np.concatenate(U); V = np.concatenate(V)
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(U.shape).astype(np.float32)
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    out = {"ds": ds, "n_edges": int(len(U)), "G1_max_abs_T_u_minus_v": {},
           "G2_norm_ratio_mean_std": {}, "G3_fixes_complement": {},
           "G4_F1_involution_max_err": None, "G5_chain_equals_manual_max_err": {},
           "G6_scalar_vs_vector_max_err": {}}
    for f, op in OPS.items():
        out["G1_max_abs_T_u_minus_v"][f] = float(np.abs(op(U, V, U) - V).max())
        r = np.linalg.norm(op(U, V, Zn), axis=1)
        out["G2_norm_ratio_mean_std"][f] = [round(float(r.mean()), 6), round(float(r.std()), 6)]
    # G3: components of Zn orthogonal to the operator plane must be untouched
    c = (U * V).sum(1, keepdims=True)
    Bh = V - c * U
    Bh = Bh / np.maximum(np.linalg.norm(Bh, axis=1, keepdims=True), TF.EPS)
    Zp = Zn - (U * Zn).sum(1, keepdims=True) * U - (Bh * Zn).sum(1, keepdims=True) * Bh
    out["G3_fixes_complement"]["F2_MIN_ROTATION"] = float(np.abs(TF.op_F2(U, V, Zp) - Zp).max())
    W = U - V
    W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), TF.EPS)
    Zw = Zn - (W * Zn).sum(1, keepdims=True) * W
    out["G3_fixes_complement"]["F1_HOUSEHOLDER"] = float(np.abs(TF.op_F1(U, V, Zw) - Zw).max())
    out["G4_F1_involution_max_err"] = float(
        np.abs(TF.op_F1(U, V, TF.op_F1(U, V, Zn)) - Zn).max())
    # G5 / G6 on one real query, along its actual parent chain
    E, rq = EE[0]
    u, v = E["u"], E["v"]
    Uq, Vq = gx(u), gx(v)
    hop = E["T3_HOP"].astype(np.int64)
    hm = [(h, np.nonzero(hop == h)[0]) for h in sorted(set(int(x) for x in hop))]
    hm = [(h, m) for h, m in hm if len(m)]
    isp = np.zeros(len(u), bool)
    isp[E["parent_tid"][E["parent_tid"] >= 0]] = True
    cuv = (Uq * Vq).sum(1)
    z0 = C["Qm"][0].astype(np.float32)
    z0 = z0 / np.linalg.norm(z0)
    uz0, vz0 = Uq @ z0, Vq @ z0
    for f in OPS:
        Zc = TF.chain_states(Uq, Vq, z0, hop, E["parent_tid"], f, hm)
        man = np.zeros_like(Zc)
        for i in range(len(u)):                     # manual sequential application
            zz = z0.copy()
            path, j = [], int(E["parent_tid"][i])
            while j >= 0:
                path.append(j)
                j = int(E["parent_tid"][j])
            for j in reversed(path):
                zz = OPS[f](Uq[j:j + 1], Vq[j:j + 1], zz[None, :])[0]
            man[i] = OPS[f](Uq[i:i + 1], Vq[i:i + 1], zz[None, :])[0]
        out["G5_chain_equals_manual_max_err"][f] = float(np.abs(Zc - man).max())
        tv, n2, d0 = TF.transport(Uq, Vq, z0, hm, E["parent_tid"], isp, f, cuv, uz0, vz0)
        out["G6_scalar_vs_vector_max_err"][f] = {
            "cos_target": float(np.abs(tv - (Zc * Vq).sum(1)).max()),
            "norm_sq": float(np.abs(n2 - (Zc * Zc).sum(1)).max()),
            "drift": float(np.abs(d0 - Zc @ z0).max())}
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "musique_clean"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    os.makedirs(f"{KTD}/diag", exist_ok=True)
    o = main(ds, n)
    json.dump(o, open(f"{KTD}/diag/gates.json", "w"), indent=1)
    print(json.dumps(o, indent=1))
