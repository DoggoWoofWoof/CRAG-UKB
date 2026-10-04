"""STEPS 3-4 -- true sequential residual reasoning, with path provenance preserved.

STEP 2 established that the frozen expansion is STATIC: one residual r0, computed once against the
seeds, scores hop1, hop2 and hop3 alike (`rq` is assigned before the hop loop and never updated), a
node keeps only the max over incoming edges (`np.maximum.at`), and no parent pointer is ever stored.
STEP 1b established that 89.5% of the gold nodes it misses are inside the 3-hop DEG_CAP closure --
reachable, and dropped by the beam.  So sequential scoring is justified.

    r_0                     = residual(q, seeds, Xn)                 -- unchanged
    directional_h(delta)    = cos(r_{h-1}, delta)                    -- per PATH, not per query
    r_h                     = r_{h-1} - proj_delta(r_{h-1}), renormalised

Each beam path carries its own current node, residual, path, per-hop directional scores and path
confidence.  No learning, no dataset identity, no gold at inference.

SCORERS (STEP 4), all reading the same beam:
    N0  the frozen static score (replayed, parity-checked against the cached s_node)
    N1  sequential residual, terminal directional score
    N2  sequential residual, path minimum
    N3  sequential residual, path geometric mean

Frozen constants kept identical: SEED_K=5, MAX_HOPS=3, DEG_CAP=300, beam=64, SMAX=256, and the
MAX_EDGES_SCORED budget.  One beam slot per node (the frozen beam also deduplicates by node), so
compute is comparable; edges scored is reported.
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP

SRD = f"{PP.ROOT}/L1_STRUCT_RECOVERY"
BEAM = 64
SMAX = 256
EPS = 1e-6
SCORERS = ["N0", "N1", "N2", "N3"]


def load_emb(ds, z):
    """node and query embeddings in the same normalised form the frozen cache builder used."""
    E = f"data/ukb_storage/{ds}/gte_qwen/"
    nodes = np.load(E + "nodes.npy", mmap_mode="r")
    qall = np.load(E + "queries_all.npy", mmap_mode="r")
    rows = np.asarray(z["rows"])
    Xn = np.array(nodes, np.float32)
    for a in range(0, len(Xn), 100000):
        b = min(a + 100000, len(Xn))
        Xn[a:b] /= (np.linalg.norm(Xn[a:b], axis=1, keepdims=True) + 1e-9)
    Qm = np.stack([np.asarray(qall[int(i)], np.float32) for i in rows])
    Qm /= (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)
    return Xn, Qm


def expand_seq(seeds, r0, adjp, adji, deg, Xn, beam=BEAM, hops=TA.MAX_HOPS, cap=TA.DEG_CAP,
               budget=TA.MAX_EDGES_SCORED, update=True, crit="term"):
    """path-level beam with a per-path residual.  Returns node -> provenance record.

    rec[v] = {term, minv, geo, hop, sup, seeds}
        term  terminal directional score of the best path reaching v
        minv  minimum directional score along that path (weakest link)
        geo   geometric mean of the directional scores along that path
        hop   length of that path
        sup   number of distinct beam paths that reached v
        seeds set of distinct retrieval seeds those paths started from
    """
    sd = [int(s) for s in seeds if s >= 0 and deg[int(s)] <= cap]
    if not sd:
        return {}, 0
    P_node = np.array(sd, np.int64)
    P_res = np.repeat(r0[None, :].astype(np.float32), len(sd), axis=0)
    P_min = np.ones(len(sd), np.float32)
    P_log = np.zeros(len(sd), np.float64)
    P_len = np.zeros(len(sd), np.int32)
    P_seed = np.arange(len(sd), dtype=np.int32)
    scope = set(sd)
    rec = {}
    escored = 0
    for h in range(hops):
        PAR, NB = [], []
        for i, e in enumerate(P_node):
            lo, hi = adjp[e], adjp[e + 1]
            if hi <= lo:
                continue
            nb = adji[lo:hi]
            escored += len(nb)
            nb = nb[deg[nb] <= cap]
            if not len(nb):
                continue
            PAR.append(np.full(len(nb), i, np.int32)); NB.append(nb.astype(np.int64))
        if not PAR:
            break
        PAR = np.concatenate(PAR); NB = np.concatenate(NB)
        D = Xn[NB] - Xn[P_node[PAR]]
        nd = np.linalg.norm(D, axis=1)
        ok = nd > 1e-9
        if not ok.any():
            break
        PAR, NB, D, nd = PAR[ok], NB[ok], D[ok], nd[ok]
        D /= nd[:, None]
        S = np.einsum("ij,ij->i", D, P_res[PAR]).astype(np.float32)
        fresh = np.fromiter((int(v) not in scope for v in NB), bool, len(NB))
        if not fresh.any():
            break
        PAR, NB, D, S = PAR[fresh], NB[fresh], D[fresh], S[fresh]
        nmin = np.minimum(P_min[PAR], S)
        nlog = P_log[PAR] + np.log(np.maximum(S, EPS))
        nlen = P_len[PAR] + 1
        # the beam is pruned by the path confidence under test, not always the terminal score
        K = S if crit == "term" else nmin if crit == "minv" else             np.exp(nlog / np.maximum(1, nlen)).astype(np.float32)
        # one beam slot per node: keep the best path to each distinct node, then the global top-beam
        o = np.argsort(-K, kind="stable")
        _, first = np.unique(NB[o], return_index=True)
        o = o[np.sort(first)]
        o = o[np.argsort(-K[o], kind="stable")[:beam]]
        PAR, NB, D, S, nmin, nlog, nlen = (PAR[o], NB[o], D[o], S[o], nmin[o], nlog[o], nlen[o])
        for k in range(len(NB)):
            v = int(NB[k]); scope.add(v)
            r = rec.get(v)
            g = float(np.exp(nlog[k] / max(1, nlen[k])))
            if r is None:
                rec[v] = {"term": float(S[k]), "minv": float(nmin[k]), "geo": g,
                          "hop": int(nlen[k]), "sup": 1, "seeds": {int(P_seed[PAR[k]])}}
            else:
                r["sup"] += 1; r["seeds"].add(int(P_seed[PAR[k]]))
                if S[k] > r["term"]:
                    r.update(term=float(S[k]), minv=float(nmin[k]), geo=g, hop=int(nlen[k]))
        # residual update: r_h = r_{h-1} - proj_delta(r_{h-1}), renormalised
        # update=False is the CONTROL: it must reproduce the frozen static beam exactly.
        R = P_res[PAR] - (S[:, None] * D if update else 0.0)
        nn = np.linalg.norm(R, axis=1)
        bad = nn <= 1e-6
        R[bad] = P_res[PAR][bad]
        nn[bad] = np.linalg.norm(R[bad], axis=1) + 1e-9
        P_res = (R / nn[:, None]).astype(np.float32)
        P_node, P_min, P_log, P_len = NB, nmin, nlog, nlen
        P_seed = P_seed[PAR]
        if escored >= budget:
            break
    return rec, escored


def order_of(rec, scorer):
    """node ordering under one scorer.  Ties broken by node id, so every order is deterministic."""
    key = {"N1": "term", "N2": "minv", "N3": "geo"}[scorer]
    return sorted(rec, key=lambda v: (-rec[v][key], v))[:SMAX]


def build(ds, log=lambda *a: None, cache=True):
    """per query: the frozen static list (N0, parity-checked) and the sequential records."""
    os.makedirs(f"{SRD}/cache", exist_ok=True)
    fp = f"{SRD}/cache/seq_{ds}.pkl"
    if cache and os.path.exists(fp):
        with open(fp, "rb") as fh:
            return pickle.load(fh)
    z, meta = PP.load(ds)
    z = {k: z[k] for k in z.files}
    nq = meta["n_dev_queries"]
    hard, _, npart, (adjp, adji), deg, _ = TA.load_topology(ds, lambda *a: None)
    adjp = np.asarray(adjp); adji = np.asarray(adji); deg = np.asarray(deg)
    Xn, Qm = load_emb(ds, z)
    t0 = time.time()
    OUT = {"nq": nq, "recs": [], "n0": [], "esc_seq": 0, "esc_static": 0, "parity": 0}
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        r0 = TA.residual(Qm[qi].astype(np.float64), sd, Xn).astype(np.float32)
        a, vm, e0 = TA.expand_dir(sd, r0, adjp, adji, deg, Xn, BEAM)
        OUT["esc_static"] += e0
        n0 = [int(v) for v in a][:SMAX]
        OUT["n0"].append(n0)
        cached = [int(v) for v in z["s_node"][qi] if v >= 0]
        OUT["parity"] += int(n0[:len(cached)] == cached)
        rec, e1 = expand_seq(sd, r0, adjp, adji, deg, Xn, BEAM)
        OUT["esc_seq"] += e1
        OUT["recs"].append({v: {**r, "seeds": sorted(r["seeds"])} for v, r in rec.items()})
        if (qi + 1) % 250 == 0:
            log(f"   seq {ds} {qi+1}/{nq}  {time.time()-t0:.0f}s  "
                f"parity {OUT['parity']}/{qi+1}  mean|rec| "
                f"{np.mean([len(r) for r in OUT['recs']]):.0f}")
    OUT["mean_rec"] = float(np.mean([len(r) for r in OUT["recs"]]))
    OUT["secs"] = round(time.time() - t0, 1)
    if cache:
        with open(fp, "wb") as fh:
            pickle.dump(OUT, fh, protocol=5)
    log(f"[seq] {ds} parity {OUT['parity']}/{nq}  mean|rec| {OUT['mean_rec']:.1f}  "
        f"edges static {OUT['esc_static']:,} seq {OUT['esc_seq']:,}  {OUT['secs']}s")
    return OUT


if __name__ == "__main__":
    T0 = time.time()
    lg = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
    for d in (sys.argv[1:] or ["metaqa"]):
        build(d, lg)
