"""L1_GEOM factorial on DEV_A (one cache = one corpus x one partition):
  P   BASE = Dense(q) + SPLADE(q) block channels, frozen RRF, P50            (P0 Mt-KaHyPar cache / P1 PHG cache)
  F   the same two channels under each parameter-free fusion                 (F1a..F1e)
  G   one more retrieval point from the query geometry (mu, 2q-mu, q-mu, q_hat, shifted seeds, seeds) through the
      same dense index -> one more block channel, fused with the two BASE channels (frozen RRF and each F)
Ceilings and evidence diagnostics tell where the headroom is (fusion vs new evidence) before any arm is judged.
Outputs results/L1_GEOM/ladder_A_<cache>.json and the DEV_A ALL vectors (rows of DEV_A only) for cross-partition tests."""
import os
import sys
import time

import numpy as np

import _l1g_core as G

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
t0 = time.time()
D = G.Data(name, dense_fp32=False)
C = D.C
npart, nq = D.npart, D.nq
m = D.A
gm = G.gold_mask(D)
res = {"cache": name, "partition": G.PARTITION_OF[name], "n_DEV_A": int(m.sum()), "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}, "diag": {}}
vecs = {}
G.log("%s [%s] DEV_A n=%d blocks %d BASE %.4f" % (name, G.PARTITION_OF[name], int(m.sum()), npart, res["BASE_A"]["ALL"]))


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    vecs[tag] = allv[m].astype(np.int8)
    return allv


rd, rs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
evd, evs = G.evidence(D, D.d_ids), G.evidence(D, D.s_ids)
base = G.F0([rd, rs], npart)
assert (base[:, :G.P_MAIN] == D.base_rank[:, :G.P_MAIN]).all(), "frozen RRF not reproduced"
run("P BASE (frozen RRF)", base)
res["diag"]["ceilings"] = G.ceilings(D, [rd, rs], [evd, evs], gm, m)
G.log("  ceilings DEV_A: %s" % res["diag"]["ceilings"])
# agreement diagnostics: does BASE fail where the channels disagree?
A_set, A_rr = G.agreement_set([rd, rs]), G.agreement_rr([rd, rs])
base_all = D.base_all.astype(bool)
qs = np.quantile(A_set[m], [0.25, 0.5, 0.75])
bins = np.digitize(A_set, qs)
res["diag"]["agreement"] = {"A_set_mean": round(float(A_set[m].mean()), 3), "A_rr_mean": round(float(A_rr[m].mean()), 3),
                            "A_set_quartile_edges": [round(float(x), 3) for x in qs],
                            "BASE_ALL_by_A_set_quartile": [round(float(base_all[m & (bins == b)].mean()), 3) for b in range(4)],
                            "corr_Aset_Arr": round(float(np.corrcoef(A_set[m], A_rr[m])[0, 1]), 3)}
G.log("  agreement: %s" % res["diag"]["agreement"])
# ---- F ladder (two channels)
for fname in G.FUSIONS:
    if fname.startswith("F0"):
        continue
    run("F %s" % fname, G.fuse(fname, [rd, rs], npart, ev=[evd, evs]))
# ---- G ladder
pts, gdiag = G.geometry(D)
res["diag"]["geometry"] = gdiag
G.log("  geometry: %s" % gdiag)
ev_ref = evd | evs
chan = {}
for gname, P in pts.items():
    ids, sc = G.retrieve(D, P)
    rg, evg = G.block_channel(D, ids), G.evidence(D, ids)
    chan[gname] = (rg, evg)
    ne = G.new_evidence(D, ids, D.d_ids, gm, ev_ref, m)
    res["diag"].setdefault("new_evidence", {})[gname] = ne
    G.log("  %-38s new hits %.2f | q with new gold-node hit %.3f | q gaining an unreached gold block %.3f | reach %.3f -> %.3f" % (
        gname, ne["frac_new_hits"], ne["q_with_new_gold_node_hit"], ne["q_gaining_unreached_gold_block"], res["diag"]["ceilings"]["reach_any_evidence"], ne["reach_after"]))
    run("G %s | channel alone" % gname, rg)
    for fname in G.FUSIONS:
        run("G %s | + BASE | %s" % (gname, fname), G.fuse(fname, [rd, rs, rg], npart, ev=[evd, evs, evg]))
# the three related points together: q, mu, 2q - mu
r1, e1 = chan["G1 centroid mu"]
r2, e2 = chan["G2 extrapolated 2q - mu"]
for fname in G.FUSIONS:
    run("G12 mu + (2q - mu) | + BASE | %s" % fname, G.fuse(fname, [rd, rs, r1, r2], npart, ev=[evd, evs, e1, e2]))
r4, e4 = chan["G4 projected q_hat = q + P(q - mu)"]
for fname in ["F0 frozen RRF"]:
    run("G124 mu + (2q - mu) + q_hat | + BASE | %s" % fname, G.fuse(fname, [rd, rs, r1, r2, r4], npart, ev=[evd, evs, e1, e2, e4]))
G.S.wj(os.path.join(G.OUT, "ladder_A_%s.json" % name), res)
np.savez_compressed(os.path.join(G.OUT, "vecs_A_%s.npz" % name), rows=D.rows[m], qids=np.array(C.qids, dtype=object)[m].astype(str),
                    **{k: v for k, v in vecs.items()})
G.log("done %.0fs" % (time.time() - t0))
