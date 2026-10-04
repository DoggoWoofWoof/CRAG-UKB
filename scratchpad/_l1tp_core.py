"""L1 PARAMETER-FREE STRUCTURAL-OFFSET TRIPLET PHASE -- substrate.

STEP 1 + STEP 2.  The frozen bounded traversal already computes, for every ACTUAL directed edge
u->v it inspects, the quantity

    S = cosine( r_q , normalize( emb(v) - emb(u) ) )

and then throws it away: `np.maximum.at(best_s, inv, S)` collapses every edge arriving at a node v
into one number attached to v.  That collapse is exactly what this phase stops doing.  Nothing new
is searched, nothing new is encoded, no parameter is introduced -- `want_edges` only exports what the
frozen loop already held in registers, and the frozen parity gate is asserted on every query.

Triplet T = (u, delta_uv, v), delta_uv = normalize(emb(v) - emb(u)).  delta is never materialised as
a vector: only its inner product with the static query residual is ever used, and only to score the
REAL transition u->v (STEP 7's rule).  No u + delta -> FAISS.

Signals, measured individually (STEP 3):
  T0_OFFSET   cosine(r_q, delta_uv)                  -- the relational offset of the real transition
  T1_SOURCE   inference-safe confidence of u          -- its own arrival score (1.0 for a seed)
  T2_TARGET   cosine(q_hat, emb(v))                   -- existing query-target similarity
  T3_HOP      structural hop of the transition        -- provenance control
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1bm_core as BM
import _l1bm_run as RUN
import _l1ss_core as SS

TPD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_TRIPLET"
K0 = EV.K0
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]


def ctx(ds):
    """Everything the replay needs, loaded once."""
    S = EV.substrate(ds)
    gr, _ = DG.gold_rows_of(ds, S["z"])
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, S["z"])
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    return {"S": S, "z": S["z"], "nq": S["nq"], "hard": S["hard"].astype(np.int64),
            "hops": S["hops"], "gold": [np.asarray(g, np.int64) for g in gr],
            "adjp": adjp, "adji": adji, "deg": deg, "Xn": Xn, "Qm": Qm,
            "W": SS.workspace(ndocs), "ndocs": ndocs,
            "fz": {k: np.asarray(S["z"][k]) for k in ("s_node", "s_hop", "s_sdir", "s_cnt")},
            "npart": int(S["hard"].max()) + 1}


def replay(C, qi):
    """One query, frozen-exact, with the per-edge triplet table exported alongside."""
    z = C["z"]
    sd = [int(s) for s in z["seeds"][qi] if s >= 0]
    rq = TA.residual(C["Qm"][qi].astype(np.float64), sd, C["Xn"])
    added, vmeta, st, FE = SS.expand_feat(sd, rq, C["adjp"], C["adji"], C["deg"], C["Xn"], C["W"],
                                          want_future=False, want_edges=True)
    sn, sh, sdd, sc = BM.arrays_of(added, vmeta)
    ok = (int(len(sn) == len(C["fz"]["s_node"][qi])) and (sn == C["fz"]["s_node"][qi]).all()
          and (sh == C["fz"]["s_hop"][qi]).all() and (sdd == C["fz"]["s_sdir"][qi]).all()
          and (sc == C["fz"]["s_cnt"][qi]).all())
    return FE["EDGES"], rq, bool(ok), st
