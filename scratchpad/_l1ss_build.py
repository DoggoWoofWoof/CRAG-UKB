"""STEP 1 -- the exact read-stage dataset, with a parity gate in front of it.

For every query, the node set that the `M_struct = 64` read is applied to is built on the exact frozen
replay, and every inference-safe quantity already available at that moment is recorded per node.
Gold labels are attached afterwards, for evaluation only; no feature sees them.

The parity gate compares the four frozen per-query arrays this run produces against the cached frozen
substrate.  If that is not exact on every query, nothing downstream means anything.

  python scratchpad/_l1ss_build.py ds
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

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
os.makedirs(f"{SS.SSD}/data", exist_ok=True)
os.makedirs(f"{SS.SSD}/diag", exist_ok=True)
MISS = 10 ** 6            # rank given to a partition absent from a prior, as the frozen F6 does


def build(ds="metaqa", want_future=True, beam_rank=None):
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]; hard = S["hard"]; hops = S["hops"]
    gr, _ = DG.gold_rows_of(ds, z)
    Gs = [set(int(x) for x in g) for g in gr]
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    fz = {k: np.asarray(z[k]) for k in ("s_node", "s_hop", "s_sdir", "s_cnt")}

    cols = {k: [] for k in SS.FEATS}
    node, qidx, gold, part = [], [], [], []
    qptr = [0]
    ngold = np.zeros(nq, np.int32)
    par_ok = 0
    st_tot = {"edges": 0, "look_edges": 0, "scope": 0}
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        added, vmeta, st, FE = SS.expand_feat(sd, rq, adjp, adji, deg, Xn, W,
                                              want_future=want_future, beam_rank=beam_rank)
        sn, sh, sdd, sc = BM.arrays_of(added, vmeta)
        par_ok += int((sn == fz["s_node"][qi]).all() and (sh == fz["s_hop"][qi]).all()
                      and (sdd == fz["s_sdir"][qi]).all() and (sc == fz["s_cnt"][qi]).all())
        for k in ("edges", "look_edges", "scope"):
            st_tot[k] += st[k]

        v = FE["node"]
        qh = Qm[qi].astype(np.float64)
        qh = qh / (np.linalg.norm(qh) + 1e-9)
        c = S["ctxs"][qi]
        pv = hard[v].astype(np.int64)
        FE["QSIM"] = np.asarray(Xn[v], np.float64) @ qh
        FE["RSIM"] = np.asarray(Xn[v], np.float64) @ rq
        FE["PARTITION_CANONICAL_PRIOR"] = np.array([c["cpos"].get(int(p), MISS) for p in pv], float)
        FE["PARTITION_RETRIEVAL_PRIOR"] = np.array([c["rpos"].get(int(p), MISS) for p in pv], float)
        for k in SS.FEATS:
            cols[k].append(FE[k])
        node.append(v); part.append(pv)
        qidx.append(np.full(len(v), qi, np.int32))
        gold.append(np.fromiter((int(int(x) in Gs[qi]) for x in v), np.int8, len(v)))
        qptr.append(qptr[-1] + len(v))
        ngold[qi] = len(Gs[qi])
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}  parity {par_ok}/{qi+1}  {time.time()-T0:.0f}s")

    D = {k: np.concatenate(cols[k]).astype(np.float64) for k in SS.FEATS}
    D["node"] = np.concatenate(node).astype(np.int64)
    D["part"] = np.concatenate(part).astype(np.int64)
    D["qidx"] = np.concatenate(qidx)
    D["gold"] = np.concatenate(gold)
    D["qptr"] = np.asarray(qptr, np.int64)
    D["ngold"] = ngold
    D["hops"] = np.asarray(hops) if hops is not None else np.full(nq, -1, np.int64)
    tag = ds if beam_rank is None else f"{ds}__BEAM{beam_rank}"
    np.savez_compressed(f"{SS.SSD}/data/{tag}.npz", **D)

    rep = {"ds": tag, "beam_rank": beam_rank, "nq": nq, "rows": int(len(D["gold"])),
           "rows_per_query": round(len(D["gold"]) / nq, 1),
           "PARITY": f"{par_ok}/{nq}", "BASELINE_PARITY": "EXACT" if par_ok == nq else "BROKEN",
           "gold_nodes_total": int(ngold.sum()),
           "gold_in_candidate_set": int(D["gold"].sum()),
           "gold_in_candidate_frac": round(float(D["gold"].sum() / max(1, ngold.sum())), 4),
           "edges_per_q": round(st_tot["edges"] / nq, 1),
           "lookahead_edges_per_q": round(st_tot["look_edges"] / nq, 1),
           "scope_per_q": round(st_tot["scope"] / nq, 1),
           "future_defined_frac": round(float((D["FUTURE_MAX"] > SS.NEG / 2).mean()), 4)}
    json.dump(rep, open(f"{SS.SSD}/diag/step1_{tag}.json", "w"), indent=1)
    log(json.dumps(rep))
    if beam_rank is None:
        assert par_ok == nq, f"PARITY BROKEN {par_ok}/{nq} -- refusing to trust the dataset"
    return rep


if __name__ == "__main__":
    build(*(sys.argv[1:] or ["metaqa"]))
