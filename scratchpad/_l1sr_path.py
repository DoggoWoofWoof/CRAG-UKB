"""STEP 3/4 mechanism -- is the residual update removing signal or adding it?

The four scorers can be compared end-to-end, but that only says which wins.  This measures WHY, on
the actual gold paths, with no beam and no ranking in the way:

    BFS from the frozen seeds (MAX_HOPS=3, DEG_CAP=300) with parent pointers -> for each reachable
    gold node, one shortest path seed -> ... -> gold.  Walk it and record, at every hop h:

        s_static[h] = cos(r0,      delta_h)     what the frozen scorer sees
        s_seq[h]    = cos(r_{h-1}, delta_h)     what the sequential scorer sees
        rank_static / rank_seq                  the gold edge's rank among ALL edges leaving that
                                                node, under each residual (1.0 = best)

The ranks are the decisive quantity: an absolute cosine can shift for scale reasons, but the rank of
the gold edge among its competitors is exactly what the beam prunes on.

  python scratchpad/_l1sr_path.py [ds] [n_per_hop]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1sr_seq as SQ
import _l1sr_diag as D1

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def bfs_paths(adjp, adji, deg, seeds, targets, hops=TA.MAX_HOPS, cap=TA.DEG_CAP):
    """shortest seed-rooted paths to each target inside the frozen closure."""
    par = {}
    seen = set()
    fr = []
    for s in seeds:
        s = int(s)
        if deg[s] <= cap and s not in seen:
            seen.add(s); par[s] = None; fr.append(s)
    want = set(int(t) for t in targets)
    out = {}
    for _ in range(hops):
        nf = []
        for e in fr:
            nb = adji[adjp[e]:adjp[e + 1]]
            nb = nb[deg[nb] <= cap]
            for v in nb:
                v = int(v)
                if v in seen:
                    continue
                seen.add(v); par[v] = e; nf.append(v)
                if v in want:
                    p = [v]
                    while par[p[-1]] is not None:
                        p.append(par[p[-1]])
                    out[v] = p[::-1]
        fr = nf
        if not fr:
            break
    return out


def walk(path, r0, qv, adjp, adji, deg, Xn):
    """per-hop directional score and edge rank under the static and the sequential residual."""
    rs = r0.copy()
    rq = r0.copy()
    rows = []
    for h in range(len(path) - 1):
        e, v = path[h], path[h + 1]
        nb = adji[adjp[e]:adjp[e + 1]]
        nb = nb[deg[nb] <= deg.max() + 1]
        D = Xn[nb] - Xn[e]
        nd = np.linalg.norm(D, axis=1)
        ok = nd > 1e-9
        nb, D, nd = nb[ok], D[ok], nd[ok]
        if not len(nb):
            break
        D = D / nd[:, None]
        w = np.where(nb == v)[0]
        if not len(w):
            break
        i = int(w[0])
        a = D @ rq                      # static: the same r0 at every hop
        b = D @ rs                      # sequential: the running residual
        # CONTROLS, so the two directional ranks are interpretable at all:
        #   qsim  rank the same neighbours by plain cosine to the query
        #   rsim  rank them by cosine to the residual (position, not displacement)
        c = Xn[nb] @ qv
        e2 = Xn[nb] @ rq
        pr = lambda x: float((x > x[i]).sum()) / max(1, len(x) - 1)
        rows.append({
            "hop": h + 1, "n_edges": int(len(nb)),
            "s_static": float(a[i]), "s_seq": float(b[i]),
            "rank_static": pr(a), "rank_seq": pr(b), "rank_qsim": pr(c), "rank_rsim": pr(e2),
            "mean_static": float(a.mean()), "mean_seq": float(b.mean())})
        d = D[i]
        s = float(b[i])
        r = rs - s * d
        n = np.linalg.norm(r)
        rs = (r / n).astype(np.float32) if n > 1e-6 else rs
    return rows


def main(ds="metaqa", nper=250):
    nper = int(nper)
    z, meta = PP.load(ds)
    z = {k: z[k] for k in z.files}
    _, _, _, (adjp, adji), deg, _ = TA.load_topology(ds, lambda *a: None)
    adjp = np.asarray(adjp); adji = np.asarray(adji); deg = np.asarray(deg)
    Xn, Qm = SQ.load_emb(ds, z)
    gr, hard = D1.gold_rows_of(ds, z)
    hops = np.asarray(z["hops"]) if "hops" in z else None
    if hops is not None and hops.min() < 0:
        hops = None                      # only metaqa carries a real hop annotation
    ALL = []
    for hq in ([1, 2, 3] if hops is not None else [0]):
        idx = np.where(hops == hq)[0][:nper] if hops is not None else range(nper)
        for qi in idx:
            sd = [int(s) for s in z["seeds"][qi] if s >= 0]
            r0 = TA.residual(Qm[qi].astype(np.float64), sd, Xn).astype(np.float32)
            paths = bfs_paths(adjp, adji, deg, sd, [int(g) for g in gr[qi]])
            for v, p in paths.items():
                for row in walk(p, r0, Qm[qi], adjp, adji, deg, Xn):
                    row["q_hop"] = int(hq); row["path_len"] = len(p) - 1
                    ALL.append(row)
        log(f"  hop{hq}: {len(ALL)} gold-path edges so far")

    out = {"ds": ds, "n_edges": len(ALL), "BY_PATH_POSITION": {}, "BY_QUERY_HOP": {}}
    for h in sorted({r["hop"] for r in ALL}):
        R = [r for r in ALL if r["hop"] == h]
        out["BY_PATH_POSITION"][f"pos{h}"] = {
            "n": len(R),
            "s_static": round(float(np.mean([r["s_static"] for r in R])), 4),
            "s_seq": round(float(np.mean([r["s_seq"] for r in R])), 4),
            **{f"pct_rank_{k}": round(float(np.mean([r[f"rank_{k}"] for r in R])), 4)
               for k in ("static", "seq", "qsim", "rsim")},
            "seq_better_frac": round(float(np.mean([r["rank_seq"] < r["rank_static"]
                                                    for r in R])), 4),
            "mean_edges": round(float(np.mean([r["n_edges"] for r in R])), 1)}
    for hq in sorted({r["q_hop"] for r in ALL}):
        R = [r for r in ALL if r["q_hop"] == hq and r["hop"] >= 2]
        if not R:
            continue
        out["BY_QUERY_HOP"][f"hop{hq}_pos2plus"] = {
            "n": len(R),
            **{f"pct_rank_{k}": round(float(np.mean([r[f"rank_{k}"] for r in R])), 4)
               for k in ("static", "seq", "qsim", "rsim")},
            "seq_better_frac": round(float(np.mean([r["rank_seq"] < r["rank_static"]
                                                    for r in R])), 4)}
    os.makedirs(f"{SQ.SRD}/diag", exist_ok=True)
    fp = f"{SQ.SRD}/diag/step3_mechanism_{ds}.json"
    json.dump(out, open(fp, "w"), indent=1)
    log(f"{ds}: {len(ALL)} gold-path edges")
    log("  pctrank of the GOLD edge among its competitors (0=best, 0.5=chance)")
    log("  position |    n  | edges | STATIC | SEQ    | qsim   | rsim   | seq better")
    for k, v in out["BY_PATH_POSITION"].items():
        log(f"  {k:8s} | {v['n']:5d} | {v['mean_edges']:5.0f} | {v['pct_rank_static']:6.4f} | "
            f"{v['pct_rank_seq']:6.4f} | {v['pct_rank_qsim']:6.4f} | {v['pct_rank_rsim']:6.4f} | "
            f"{v['seq_better_frac']:.1%}")
    for k, v in out["BY_QUERY_HOP"].items():
        log(f"  {k:18s} n={v['n']:5d}  static {v['pct_rank_static']:.4f}  "
            f"seq {v['pct_rank_seq']:.4f}  qsim {v['pct_rank_qsim']:.4f}  "
            f"rsim {v['pct_rank_rsim']:.4f}")
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
