"""Beam-policy runner + STEP 1 parity gate.

Every policy is replayed on the SAME frozen inputs the cache builder used (RRF seeds from the cache,
`TA.residual` on the same normalised query/node embeddings, same CSR topology).  The output of each
policy is written back into the four frozen per-query arrays, so downstream everything reads the
identical code path the frozen substrate does (`EV.sf_from_cache` -> S4 -> F6 -> exact P50).

  python scratchpad/_l1bm_run.py step1 [ds]          parity gate only
  python scratchpad/_l1bm_run.py build ds pol,pol,... build + cache those policies
"""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1sr_seq as SQ  # noqa: F401  (kept for the small-corpus loader)
import _l1bm_core as BM

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
os.makedirs(f"{BM.BMD}/cache", exist_ok=True)
os.makedirs(f"{BM.BMD}/diag", exist_ok=True)


def topology(ds):
    _, _, _, (adjp, adji), deg, _ = TA.load_topology(ds, lambda *a: None)
    return np.asarray(adjp), np.asarray(adji), np.asarray(deg)


def run_policy(ds, policy, S=None, emb=None, topo=None, log=log):
    """returns the four frozen arrays + internal-work stats for one policy."""
    S = S or EV.substrate(ds)
    z = S["z"]; nq = S["nq"]
    adjp, adji, deg = topo or topology(ds)
    Xn, Qm = emb if emb else load_emb_frozen(ds, z)[:2]
    sn = np.full((nq, BM.SMAX), -1, np.int32); sh = np.zeros((nq, BM.SMAX), np.int8)
    sd = np.zeros((nq, BM.SMAX), np.float32); sc = np.zeros((nq, BM.SMAX), np.int32)
    agg = {"edges": 0, "look_edges": 0, "cand_total": 0, "scope": 0, "n": nq,
           "frontier": np.zeros(TA.MAX_HOPS), "cands": np.zeros(TA.MAX_HOPS),
           "kept": np.zeros(TA.MAX_HOPS), "hops_seen": np.zeros(TA.MAX_HOPS)}
    t = time.time()
    for qi in range(nq):
        seeds = [int(s) for s in z["seeds"][qi] if s >= 0]
        r_q = TA.residual(Qm[qi].astype(np.float64), seeds, Xn)
        a, vm, st = BM.expand_beam(seeds, r_q, adjp, adji, deg, Xn, BM.BEAM, policy)
        A = BM.arrays_of(a, vm)
        sn[qi], sh[qi], sd[qi], sc[qi] = A
        for k in ("edges", "look_edges", "cand_total", "scope"):
            agg[k] += st[k]
        for h in range(len(st["cands"])):
            agg["frontier"][h] += st["frontier"][h]; agg["cands"][h] += st["cands"][h]
            agg["kept"][h] += st["kept"][h]; agg["hops_seen"][h] += 1
        if (qi + 1) % 500 == 0:
            log(f"   {policy} {ds} {qi+1}/{nq}  {time.time()-t:.0f}s")
    agg["latency_ms"] = round(1000.0 * (time.time() - t) / nq, 3)
    for k in ("frontier", "cands", "kept"):
        agg[k] = [round(float(agg[k][h] / max(1, agg["hops_seen"][h])), 1)
                  for h in range(TA.MAX_HOPS)]
    agg["hops_seen"] = [int(x) for x in agg["hops_seen"]]
    for k in ("edges", "look_edges", "cand_total", "scope"):
        agg[k + "_per_q"] = round(agg[k] / nq, 1)
    return {"s_node": sn, "s_hop": sh, "s_sdir": sd, "s_cnt": sc, "stats": agg}


def build(ds, policies, S=None, emb=None, topo=None, cache=True, log=log):
    S = S or EV.substrate(ds)
    out = {}
    for pol in policies:
        fp = f"{BM.BMD}/cache/{ds}__{pol}.npz"
        if cache and os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            out[pol] = {k: z[k] for k in ("s_node", "s_hop", "s_sdir", "s_cnt")}
            out[pol]["stats"] = json.loads(str(z["stats"]))
            log(f"[cached] {ds} {pol}")
            continue
        emb = emb or load_emb_frozen(ds, S["z"])[:2]; topo = topo or topology(ds)
        r = run_policy(ds, pol, S, emb, topo, log)
        if cache:
            np.savez_compressed(fp, stats=json.dumps(r["stats"]),
                                **{k: r[k] for k in ("s_node", "s_hop", "s_sdir", "s_cnt")})
        out[pol] = r
        st = r["stats"]
        log(f"[built] {ds} {pol}  edges/q {st['edges_per_q']:.0f} (+look {st['look_edges_per_q']:.0f})"
            f"  cand/hop {st['cands']}  kept/hop {st['kept']}  {st['latency_ms']:.2f} ms/q")
    return out, S, emb, topo


def sf_arrays(R, qi, hard, aggname="P0_S4", depth=EV.M_STRUCT):
    return EV.sf_from_cache({"s_node": R["s_node"], "s_hop": R["s_hop"],
                             "s_sdir": R["s_sdir"], "s_cnt": R["s_cnt"]}, qi, hard, aggname, depth)


# ------------------------------------------------------------------------------ STEP 1
def step1(ds="metaqa"):
    S = EV.substrate(ds); z = S["z"]; nq = S["nq"]; hard = S["hard"]
    R = run_policy(ds, "M0_BASELINE", S)
    ok_node = int((R["s_node"] == z["s_node"]).all(1).sum())
    ok_all = int((( R["s_node"] == z["s_node"]) & (R["s_hop"] == z["s_hop"])
                  & (R["s_sdir"] == z["s_sdir"]) & (R["s_cnt"] == z["s_cnt"])).all(1).sum())
    log(f"STEP1 array parity: s_node {ok_node}/{nq}   all four fields {ok_all}/{nq}")

    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    MINE = [sf_arrays(R, qi, hard) for qi in range(nq)]
    ok_sf = sum(int(list(a) == list(b)) for a, b in zip(SAFE, MINE))
    i_safe, _ = EV.evaluate(S, SAFE, "A_F6", (EV.P,))
    i_mine, _ = EV.evaluate(S, MINE, "A_F6", (EV.P,))
    hsafe = EV.by_hop(i_safe[EV.P], S["hops"]); hmine = EV.by_hop(i_mine[EV.P], S["hops"])
    log(f"STEP1 S4 ranking parity {ok_sf}/{nq}")
    log(f"STEP1 SAFE  {hsafe}")
    log(f"STEP1 MINE  {hmine}")
    exact = (ok_all == nq and ok_sf == nq and hsafe == hmine)
    log(f"BASELINE_PARITY = {'EXACT' if exact else 'MISMATCH'}")
    out = {"ds": ds, "nq": nq, "s_node_parity": ok_node, "all_field_parity": ok_all,
           "s4_ranking_parity": ok_sf, "SAFE": hsafe, "REPLAY": hmine,
           "BASELINE_PARITY": "EXACT" if exact else "MISMATCH", "stats": R["stats"]}
    json.dump(out, open(f"{BM.BMD}/diag/step1_{ds}.json", "w"), indent=1)
    if not exact:
        bad = np.where(~(R["s_node"] == z["s_node"]).all(1))[0][:10]
        for qi in bad:
            m = R["s_node"][qi] != z["s_node"][qi]
            log(f"   q{qi}: first diff at slot {int(np.where(m)[0][0])}  "
                f"mine {R['s_node'][qi][m][:5]} frozen {z['s_node'][qi][m][:5]}")
    return out


# --------------------------------------------------------------- big-corpus embeddings
class F32View:
    """exactly the fp16-backed view `_l1ps_cache.py` used to BUILD the frozen substrate."""
    def __init__(self, A):
        self.A = A

    def __getitem__(self, k):
        return np.asarray(self.A[k], np.float32)

    def __len__(self):
        return len(self.A)


def load_emb_frozen(ds, z):
    """node/query embeddings in the SAME precision path the frozen cache builder chose for `ds`.

    The builder switches to fp16 node storage above ~195k docs, so replaying a big corpus in fp32
    would NOT be a replay -- it would be a different (more precise) beam.  Mirroring the switch is
    what makes the STEP 1 parity gate meaningful on webqsp/hotpot as well as on metaqa."""
    E = f"data/ukb_storage/{ds}/gte_qwen/"
    nodes = np.load(E + "nodes.npy", mmap_mode="r")
    qall = np.load(E + "queries_all.npy", mmap_mode="r")
    ndocs, dim = nodes.shape
    big = ndocs * 1536 * 4 > 1.2e9
    if big:
        X16 = np.empty((ndocs, dim), np.float16)
        for a in range(0, ndocs, 50000):
            b = min(a + 50000, ndocs)
            blk = np.array(nodes[a:b], np.float32)
            blk /= (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-9)
            X16[a:b] = blk.astype(np.float16); del blk
        Xn = F32View(X16)
    else:
        Xn = np.array(nodes, np.float32)
        for a in range(0, len(Xn), 100000):
            b = min(a + 100000, len(Xn))
            Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    rows = np.asarray(z["rows"])
    Qm = np.stack([np.asarray(qall[int(i)], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)
    return Xn, Qm, big


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step1"
    if cmd == "step1":
        step1(*(sys.argv[2:] or ["metaqa"]))
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3].split(","))
