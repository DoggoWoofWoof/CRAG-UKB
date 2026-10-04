"""Lever 1 (partition quality) on DEV_A: paired comparison of the SAME queries under the Mt-KaHyPar H4_SK partition
and the validated Zoltan-PHG partition of the same corpus (same served hits, same frozen numerics, same P50).
Uses the DEV_A ALL vectors written by _l1g_ladder.py (rows aligned by cache row id) plus partition geometry stats."""
import io
import json
import os

import numpy as np

import _l1g_core as G
import _l1x90_core as X

out = {}
for ds, a, b in (("metaqa", "metaqa", "metaqa_phg"), ("squad", "squad", "squad_phg")):
    va = np.load(os.path.join(G.OUT, "vecs_A_%s.npz" % a), allow_pickle=True)
    vb = np.load(os.path.join(G.OUT, "vecs_A_%s.npz" % b), allow_pickle=True)
    assert (va["rows"] == vb["rows"]).all(), "DEV_A rows differ between the caches"
    e = {"n": int(len(va["rows"])), "arms": {}}
    for arm in ("P BASE (frozen RRF)", "F F1e agreement blend (reciprocal-rank agreement)", "G12 mu + (2q - mu) | + BASE | F0 frozen RRF",
                "G G3 residual direction q - mu | + BASE | F0 frozen RRF"):
        xa, xb = va[arm].astype(bool), vb[arm].astype(bool)
        g, l, p = X.mcnemar(xa, xb)
        e["arms"][arm] = {"MTKAHYPAR": round(float(xa.mean()), 4), "PHG": round(float(xb.mean()), 4), "gained_by_PHG": g, "lost_by_PHG": l, "p": p}
        G.log("%-7s %-62s MtK %.4f -> PHG %.4f (+%d/-%d p=%.1e)" % (ds, arm[:62], xa.mean(), xb.mean(), g, l, p))
    # partition geometry on the whole corpus (query independent) + gold-block statistics on the cache rows
    for tag, name in (("MTKAHYPAR", a), ("PHG", b)):
        C = X.Cache(name)
        ngb = np.array([len(s) for s in C.gb])
        m = C.split == "A"
        xo, ao = C.cd.struct_csr(directed=True)
        hard = C.hard.astype(np.int64)
        cross = float((hard[np.repeat(np.arange(C.N), np.diff(xo))] != hard[np.asarray(ao, np.int64)]).mean())
        C.cd._csr.clear()
        e[tag] = {"blocks": int(C.npart), "size_min": int(C.part_sizes.min()), "size_max": int(C.part_sizes.max()),
                  "gold_blocks_per_query_DEV_A": round(float(ngb[m].mean()), 3), "feasible_P50_DEV_A": round(float((ngb[m] <= G.P_MAIN).mean()), 4),
                  "directed_struct_edges_cut_fraction": round(cross, 4)}
        G.log("  %-9s blocks %d sizes [%d,%d] gold-blocks/q %.3f feasible %.4f cut-edge fraction %.4f" % (
            tag, C.npart, C.part_sizes.min(), C.part_sizes.max(), ngb[m].mean(), (ngb[m] <= G.P_MAIN).mean(), cross))
        del C
    out[ds] = e
G.S.wj(os.path.join(G.OUT, "partition_effect_A.json"), out)
