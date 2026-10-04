"""FBX_SCALE stage 3b -- routing fan-out, contacted mass, bytes touched and routing latency of a built partitioner (systems proxy; no tuning).

The frozen policy is kept exactly: ES shard ranking by first occurrence in the router order, B_P(q, K) = K when the localisation has no evidence
else ceil(KNEE_SDIV(q; the SAME partitioner's K 100 map) * (K / 100)^0.75) capped at K (the KSCALE alpha 0.75 table), contacted shards = the first
B_P in the ES order, contacted mass = the sum of their sizes.  The router evidence of the 786 REPORT-44 split-A rows (router order oq and the
localisation entries of the frozen IR_L1, read from the stage-3a bundle) is mapped from WebQSP node ids to Freebase positions through
data/final_canonical/freebase/bridge/webqsp_positions.npy; an id that does not resolve (37.6% of WebQSP nodes) is dropped.  This is a SYSTEMS PROXY: the
evidence is the WebQSP subgraph's own, not IR_L1 run on Freebase (no dense / SPLADE vectors exist for 302M nodes; addendum, stage 2b).  What is measured:

  B_P, B_P / K, shards that hold router evidence, contacted mass (nodes) and its share of N, bytes touched (dense fp16 payload only: 3,072 B / node),
  the number of rows whose resolved gold nodes all sit inside the contacted shards (ALL-gold shard reach) and any of them (ANY), the shard-reach
  requirement LO (ES positions needed to cover every resolved gold), and the per-row routing latency (ES order + B_P + contacted mass), two passes over the rows: the maps' pages are already in the OS page cache (the block-size count read them),
  so this is a WARM-cache latency, not a cold-disk one.

  python scratchpad/_fbx_scale_route.py <bundle_tag> <tag> <bridge: freebase|identity> <record1.json> [record2.json ...]
    each record = FBX_BUILD__<dataset>__<method>__<tag>.json (its maps must include K 100); 'identity' = WebQSP maps with no bridge (the regression against stage 3a's EVAL).
Writes results/FREEBASE_SCALE/FBX_ROUTE__<bridge>__<tag>.json (write-once) and its npz.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

import _l1d_kscale as KS
import _l1d_route as RT

ROOT = os.environ.get("FBX_ROOT") or os.getcwd()
OUTR = os.path.join(ROOT, "results", "FREEBASE_SCALE")
ACT = 200
ALPHA = "0.75"
K_REF = 100
DENSE_BYTES = 1536 * 2
CHUNK = 20_000_000


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


T0 = time.time()


def block_sizes(hard, K):
    sz = np.zeros(K, np.int64)
    for s in range(0, len(hard), CHUNK):
        sz += np.bincount(np.asarray(hard[s:s + CHUNK]), minlength=K)
    return sz


def q(v):
    v = np.asarray(v, np.float64)
    return {"mean": round(float(v.mean()), 4), "p50": round(float(np.percentile(v, 50)), 4), "p95": round(float(np.percentile(v, 95)), 4),
            "p99": round(float(np.percentile(v, 99)), 4), "max": round(float(v.max()), 4)}


def main():
    btag, tag, bridge_mode = sys.argv[1], sys.argv[2], sys.argv[3]
    recs = [json.load(io.open(p, encoding="utf-8")) for p in sys.argv[4:]]
    out_j = os.path.join(OUTR, "FBX_ROUTE__%s__%s.json" % (bridge_mode, tag))
    out_z = out_j[:-5] + ".npz"
    assert not os.path.exists(out_j), "write-once: %s exists" % out_j
    bdir = os.path.join(ROOT, "work", "FBX_CALIB", "bundle_%s" % btag)
    bmeta = json.load(io.open(os.path.join(bdir, "meta.json"), encoding="utf-8"))
    assert bmeta["ev_sha256"] == sha_file(os.path.join(bdir, "ev.npz"))
    with np.load(os.path.join(bdir, "ev.npz")) as z:
        ev = {k: z[k] for k in z.files if k not in ("frank_g",)}
    nq = len(ev["nh"])
    gptr = ev["gptr"]
    if bridge_mode == "freebase":
        bridge = np.load(os.path.join(ROOT, "data", "final_canonical", "freebase", "bridge", "webqsp_positions.npy"))
        assert bridge.dtype == np.int32 and len(bridge) == bmeta["N"]
        bridge_sha = sha_file(os.path.join(ROOT, "data", "final_canonical", "freebase", "bridge", "webqsp_positions.npy"))
    else:
        bridge, bridge_sha = None, None
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    res = {"stage": "FBX_SCALE 3b routing / fan-out proxy", "tag": tag, "bridge": bridge_mode, "bridge_sha256": bridge_sha, "n_rows": nq,
           "bundle": {"tag": btag, "ev_sha256": bmeta["ev_sha256"], "population": bmeta["population"]}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "methods": {}}
    arrays = {}
    # ---- the evidence of every row, mapped once
    ORQ, EU, EH, EX, GO = [], [], [], [], []
    res_oq, res_e, res_g, ngold = np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq, np.int64)
    for j in range(nq):
        oq = ev["oq"][ev["oq_ptr"][j]:ev["oq_ptr"][j + 1]].astype(np.int64)
        s0, s1 = ev["e_ptr"][j], ev["e_ptr"][j + 1]
        u, hit, x = ev["eu"][s0:s1].astype(np.int64), ev["eh"][s0:s1].astype(np.int64), ev["ex"][s0:s1].astype(np.float64)
        g = ev["gold"][gptr[j]:gptr[j + 1]].astype(np.int64)
        if bridge is not None:
            oq, um, gm = bridge[oq].astype(np.int64), bridge[u].astype(np.int64), bridge[g].astype(np.int64)
            res_oq[j], res_e[j] = float((oq >= 0).mean()) if len(oq) else 0.0, float((um >= 0).mean()) if len(um) else 0.0
            keep = um >= 0
            u, hit, x = um[keep], hit[keep], x[keep]
            oq = oq[oq >= 0]
            g = gm[gm >= 0]
        else:
            res_oq[j] = res_e[j] = 1.0
        res_g[j] = len(g) / float(gptr[j + 1] - gptr[j])
        ngold[j] = len(g)
        ORQ.append(oq), EU.append(u), EH.append(hit), EX.append(x), GO.append(g)
    res["evidence_resolved_fraction_mean"] = {"router_order": round(float(res_oq.mean()), 4), "localisation_entries": round(float(res_e.mean()), 4), "gold_nodes": round(float(res_g.mean()), 4)}
    res["rows_with_a_resolved_gold"] = int((ngold > 0).sum())
    res["rows_with_all_gold_resolved"] = int((res_g == 1.0).sum())
    nh_ = ev["nh"]
    for rec in recs:
        meth, ds = rec["method"], rec["dataset"]
        Ks = sorted(int(k) for k in rec["maps"])
        assert K_REF in Ks, "the record has no K 100 map (KNEE_SDIV needs it)"
        N = rec["N"]
        maps = {K: np.load(os.path.join(ROOT, rec["maps"][str(K)]["map_file"]), mmap_mode="r") for K in Ks}
        sizes = {}
        for K in Ks:
            t0 = time.time()
            sizes[K] = block_sizes(maps[K], K)
            assert int(sizes[K].sum()) == N
        log("%s: sizes of %d maps (%.1fs)" % (meth, len(Ks), time.time() - t0))
        h100 = maps[K_REF]
        tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in Ks if K > K_REF}
        out = {"K": {}, "map_records": {"record_method": meth, "dataset": ds, "map_sha256": {str(K): rec["maps"][str(K)]["map_sha256"] for K in Ks}}}
        B0, NOEV = np.zeros(nq, np.int64), np.zeros(nq, bool)
        t0 = time.time()
        for j in range(nq):
            if len(EU[j]) == 0:
                NOEV[j], B0[j] = True, K_REF
                continue
            Cm = np.bincount(EH[j] * K_REF + np.asarray(h100[EU[j]], np.int64), weights=EX[j], minlength=int(nh_[j]) * K_REF).reshape(int(nh_[j]), K_REF)
            kn_, npz_ = RT.knee(wseed[:int(nh_[j])] @ (Cm > 0))
            NOEV[j], B0[j] = (not npz_), (kn_ if npz_ else K_REF)
        log("%s: KNEE_SDIV on the K 100 map (%.1fs); rows with no evidence: %d" % (meth, time.time() - t0, int(NOEV.sum())))
        for K in Ks:
            hard, sz = maps[K], sizes[K]
            tab = tabs.get(K)
            bP = np.zeros(nq, np.int64)
            cm = np.zeros(nq, np.int64)
            nt = np.zeros(nq, np.int64)
            lo = np.zeros(nq, np.int64)
            allr = np.zeros(nq, bool)
            anyr = np.zeros(nq, bool)
            lat = {"first_pass": np.zeros(nq), "second_pass": np.zeros(nq)}
            for rep in ("first_pass", "second_pass"):
                for j in range(nq):
                    t0 = time.perf_counter()
                    oq = ORQ[j]
                    hq = np.asarray(hard[oq], np.int64) if len(oq) else np.zeros(0, np.int64)
                    up, first = np.unique(hq, return_index=True)
                    fe = np.full(K, len(oq), np.int64)
                    fe[up] = first
                    ro = np.lexsort((np.arange(K), fe))
                    b = K if NOEV[j] else (int(B0[j]) if K == K_REF else int(tab[B0[j]]))
                    mass = int(sz[ro[:b]].sum())
                    lat[rep][j] = time.perf_counter() - t0
                    if rep == "second_pass":
                        bP[j], cm[j], nt[j] = b, mass, len(up)
                        pr = RT.inv(ro)
                        gs = np.asarray(hard[GO[j]], np.int64) if len(GO[j]) else np.zeros(0, np.int64)
                        if len(gs):
                            pg = pr[gs]
                            lo[j] = int(pg.max()) + 1
                            allr[j], anyr[j] = bool((pg < b).all()), bool((pg < b).any())
            has_g = ngold > 0
            ent = {"B_P": q(bP), "B_P_over_K": q(bP / float(K)), "rows_B_P_eq_K": int((bP == K).sum()), "rows_no_evidence": int(NOEV.sum()),
                   "shards_with_router_evidence": q(nt), "contacted_mass_nodes": q(cm), "contacted_mass_over_N": q(cm / float(N)),
                   "bytes_touched_dense_fp16_MB": q(cm * DENSE_BYTES / 1048576.0), "shard_reach_LO (ES positions to cover every resolved gold)": q(lo[has_g]),
                   "rows_with_resolved_gold": int(has_g.sum()), "ALL_gold_shard_reach_rows": int(allr[has_g].sum()), "ANY_gold_shard_reach_rows": int(anyr[has_g].sum()),
                   "routing_latency_ms_second_pass": q(lat["second_pass"] * 1e3), "routing_latency_ms_first_pass": q(lat["first_pass"] * 1e3),
                   "block_sizes": {"min": int(sz.min()), "max": int(sz.max()), "mean": round(float(sz.mean()), 1)}}
            out["K"][str(K)] = ent
            arrays["%s__K%d__B_P" % (meth, K)], arrays["%s__K%d__mass" % (meth, K)] = bP, cm
            arrays["%s__K%d__LO" % (meth, K)] = lo
            log("%s K %d: B_P/K mean %.3f, contacted mass/N mean %.3f, ALL reach %d/%d rows, second-pass latency p50 %.2f ms" % (
                meth, K, ent["B_P_over_K"]["mean"], ent["contacted_mass_over_N"]["mean"], ent["ALL_gold_shard_reach_rows"], ent["rows_with_resolved_gold"], ent["routing_latency_ms_second_pass"]["p50"]))
        res["methods"][meth] = out
    np.savez_compressed(out_z, **arrays)
    res["npz"] = {"path": os.path.basename(out_z), "sha256": sha_file(out_z)}
    res["code"] = {"scratchpad/_fbx_scale_route.py": sha_file(os.path.abspath(__file__))}
    with io.open(out_j, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False))
    log("record -> %s %s" % (out_j, sha_file(out_j)[:12]))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    main()
