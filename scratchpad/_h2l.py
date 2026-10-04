"""FBX_SCALE Track B -- is a HIERARCHICAL (two-level) decomposition of the frozen partitioner recall-preserving?  (WebQSP, SK-to-SK, in memory.)

The architecture the user ruled for the out-of-core partitioner has to run the expensive global step on something that fits.  The simplest structure that bounds memory is nested:
  TOP  partition H4_SK into S super-blocks (the frozen Zoltan-PHG driver, NP 4, k = S, IMBALANCE_TOL 1.01),
  SUB  for every super-block, restrict the final-K hypergraph H4_SK(K) to the block's vertices (net splitting: a net keeps its pins inside the block, nets with < 2 pins are dropped, weights unchanged),
       partition it with the same driver into K / S blocks with Zoltan's tolerance set so that the largest block is within the contract bound ceil(1.03 N / K),
  and the K blocks are the union.  This module builds that two-level map; scripts/_h2l_eval.py scores it against the frozen flat PHG_SK under the addendum-1 gate.

  python -u scratchpad/_h2l.py CONTROL <K> [...]        # the hypergraph of this path at K must have the frozen H4_SK_K content digest
  python -u scratchpad/_h2l.py TOP <S> [...]            # the S-way top partition (write-once record + map)
  python -u scratchpad/_h2l.py SUB <S> <K> [...]        # the two-level K-way map from the top map (needs TOP <S>)
  python -u scratchpad/_h2l.py TEST                     # restriction identities on a random hypergraph (no Zoltan)
"""
import io
import json
import math
import os
import shutil
import sys
import time

import numpy as np

import _l1h_host as H

HG, P, FR, PR = H.HG, H.P, H.FR, H.PR
DS = "webqsp"
OUT = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "h2l", "parts")
H.PDIR = OUT
MODE = {"top": False}
TOP_TOL = "1.01"


def tag_of(K):
    return ("H4_H2LTOP_k%d" if MODE["top"] else "H4_SK_k%d") % int(K)


H.tag_of = tag_of


def arm_files(S, K):
    base = os.path.join(OUT, "%s__H4_H2L%d_k%d__PHG_con" % (DS, S, K))
    return base + ".npy", base + ".RUN.json", base + ".FAILED.json"


def ckpt_dir(npy):
    """per-arm block checkpoints (orchestration only): the arm's own output name, so two arms never share one."""
    return os.path.join(OUT, "_blockckpt", os.path.basename(npy)[:-4])


def ckpt_save(cdir, bi, key, rec_b, hb):
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, "b%02d.npz" % bi)
    tmp = fp[:-4] + ".tmp.npz"
    np.savez(tmp, hb=np.asarray(hb, np.int64), meta=np.array(json.dumps({"key": key, "rec_b": rec_b}, sort_keys=True)))
    os.replace(tmp, fp)


def ckpt_load(cdir, bi, key):
    """the stored result of block bi iff it was produced for exactly this key (top map, hypergraph, block shape, tolerance, driver parameters); else None."""
    fp = os.path.join(cdir, "b%02d.npz" % bi)
    if not os.path.exists(fp):
        return None
    try:
        z = np.load(fp, allow_pickle=False)
        meta, hb = json.loads(str(z["meta"])), z["hb"].astype(np.int64)
    except Exception:
        return None
    if meta["key"] != key or len(hb) != key["N"] or (len(hb) and (int(hb.min()) < 0 or int(hb.max()) >= key["kb"])):
        return None
    return meta["rec_b"], hb


def restrict(Hf, top, S):
    """per top block: the net-split sub-hypergraph with local vertex ids.  Returns [{'vertices', 'N', 'M', 'eptr', 'eidx', 'ew'}]."""
    eptr, eidx, ew, N, M = Hf["eptr"], Hf["eidx"], Hf["ew"], Hf["N"], Hf["M"]
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    blk = top[eidx]
    out = []
    for b in range(S):
        vb = np.flatnonzero(top == b)
        loc = np.full(N, -1, np.int64)
        loc[vb] = np.arange(len(vb))
        m = blk == b
        sh = hid[m]
        cnt = np.bincount(sh, minlength=M)
        nets = np.flatnonzero(cnt >= 2)
        keep = cnt[sh] >= 2
        pv = eidx[m][keep]
        ph = sh[keep]
        ep = np.zeros(len(nets) + 1, np.int64)
        np.cumsum(cnt[nets], out=ep[1:])
        assert len(pv) == int(ep[-1]) and bool((np.diff(ph) >= 0).all())
        out.append({"vertices": vb, "N": int(len(vb)), "M": int(len(nets)), "eptr": ep, "eidx": loc[pv], "ew": ew[nets]})
    return out


def tol_for(nb, N, K, kb):
    """Zoltan's tolerance so that its bound (tol x nb / kb) equals the contract bound 1.03 N / K, floored to 4 decimals; < 1 is infeasible."""
    t = 1.03 * (N / float(K)) * kb / float(nb)
    return math.floor(t * 1e4) / 1e4


def cmd_control(Ks):
    for K in Ks:
        MODE["top"] = False
        H.cmd_hg(DS, [K])
        m = json.load(io.open(H.hg_npz(DS, K)[:-4] + ".json", encoding="utf-8"))
        ref = json.load(io.open(os.path.join(H.REPO, "results", "L1_HOST", "parts", "%s__H4_SK_k%d.json" % (DS, K)), encoding="utf-8"))
        ok = m["content_digest"] == ref["content_digest"]
        H.log("CONTROL K %d: digest %s vs frozen %s -> %s" % (K, m["content_digest"][:16], ref["content_digest"][:16], "EQUAL" if ok else "DIFFERENT"))
        assert ok


def cmd_top(Ss):
    MODE["top"] = True
    base = list(P.PARAMS)
    P.PARAMS = [(k, TOP_TOL if k == "IMBALANCE_TOL" else v) for k, v in base]
    try:
        H.cmd_hg(DS, Ss)
        H.cmd_phg(DS, Ss)
    finally:
        P.PARAMS = base
        MODE["top"] = False


def cmd_sub(S, K):
    assert K % S == 0, "S must divide K"
    kb = K // S
    npy, runp, failp = arm_files(S, K)
    if os.path.exists(runp) or os.path.exists(failp):
        H.log("S %d K %d: exists" % (S, K))
        return
    MODE["top"] = True
    tp = H.phg_npy(DS, S)
    tr = json.load(io.open(tp[:-4] + ".RUN.json", encoding="utf-8"))
    assert tr["STATUS"] == "OK" and H.sha_file(tp) == tr["output"]["sha256"]
    top = np.load(tp).astype(np.int64)
    MODE["top"] = False
    t0 = time.time()
    H.cmd_hg(DS, [K])
    Hf = H.load_hg(DS, K)
    ref = json.load(io.open(os.path.join(H.REPO, "results", "L1_HOST", "parts", "%s__H4_SK_k%d.json" % (DS, K)), encoding="utf-8"))
    assert Hf["meta"]["content_digest"] == ref["content_digest"], "the K-way hypergraph of this path is not the frozen H4_SK_K"
    N, M = Hf["N"], Hf["M"]
    assert len(top) == N and int(top.max()) == S - 1
    subs = restrict(Hf, top, S)
    b = H.host_build()
    hard = np.full(N, -1, np.int64)
    blocks = []
    resumed = []
    cdir = ckpt_dir(npy)
    base = list(P.PARAMS)
    bad = None
    try:
        for bi, sb in enumerate(subs):
            nb = sb["N"]
            tol = tol_for(nb, N, K, kb)
            rec_b = {"block": bi, "N": nb, "M": sb["M"], "pins": int(len(sb["eidx"])), "zoltan_tolerance": tol}
            if tol < 1.0:
                rec_b["status"] = "INFEASIBLE_TOLERANCE"
                blocks.append(rec_b)
                bad = "top block %d has %d vertices > 1.03 N/S: no tolerance >= 1 meets the contract bound" % (bi, nb)
                break
            P.PARAMS = [(k, "%.4f" % tol if k == "IMBALANCE_TOL" else v) for k, v in base]
            key = {"top_sha256": tr["output"]["sha256"], "hg_content_digest": Hf["meta"]["content_digest"], "S": S, "K": K, "block": bi, "N": nb, "M": sb["M"], "pins": int(len(sb["eidx"])),
                   "zoltan_tolerance": tol, "kb": kb, "NP": P.NP, "params": [list(x) for x in P.PARAMS]}
            hit = ckpt_load(cdir, bi, key)
            if hit is not None:                       # a block finished by an earlier (preempted) run of this same arm: reuse its stored result, run nothing
                rec_b, hb = hit
                rec_b["resumed"] = True
                blocks.append(rec_b)
                hard[sb["vertices"]] = bi * kb + hb
                resumed.append(bi)
                H.log("S %d K %d: block %d restored from its checkpoint" % (S, K, bi))
                continue
            name = "H2L%d_k%d_b%02d" % (S, K, bi)
            sdir = os.path.join(H.CM.DATA, DS, "stream_%s" % name)
            shards, smf = H.write_shards({"N": nb, "M": sb["M"], "eptr": sb["eptr"], "eidx": sb["eidx"], "ew": sb["ew"]}, sdir)
            r, d = P.mpirun(DS, name, "partition", kb, smf, b, timeout=H.PHG_TIMEOUT_S)
            gq = r["global_from_queries"]
            assert gq["objects"] == nb and gq["pins"] == len(sb["eidx"]), ("Zoltan saw a different sub-hypergraph", gq)
            hb = P.assemble_partition(d, nb, kb)
            sizes = np.bincount(hb, minlength=kb)
            rec_b.update({"status": "OK", "seconds": r["wall_seconds_outer"], "max_block": int(sizes.max()), "min_block": int(sizes.min()), "peak_rss_kb_per_rank": r["memory"]["peak_rss_kb_per_rank_time_v"],
                          "zoltan_imbalance": r["zoltan_eval"]["imbalance"]})
            ckpt_save(cdir, bi, key, rec_b, hb)
            blocks.append(rec_b)
            hard[sb["vertices"]] = bi * kb + hb
            shutil.rmtree(sdir, ignore_errors=True)
            shutil.rmtree(d, ignore_errors=True)
    finally:
        P.PARAMS = base
    rec = {"dataset": DS, "arm": "H2L%d" % S, "S": S, "K": K, "kb": kb, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "top": {"file": H.rel(tp), "sha256": tr["output"]["sha256"], "tolerance": TOP_TOL, "record_sha256": H.sha_file(tp[:-4] + ".RUN.json")},
           "hypergraph": {"file": H.rel(Hf["npz"]), "sha256": Hf["npz_sha256"], "content_digest": Hf["meta"]["content_digest"], "N": N, "M": M, "P": Hf["P"]},
           "blocks": blocks, "blocks_resumed_from_checkpoint": resumed, "rule": "two-level: top S-way partition, net-split sub-hypergraphs, K/S-way sub-partitions with the tolerance that equals the contract bound (scratchpad/_h2l.py)",
           "driver": {"build_record_sha256": H.sha_file(H.BUILD_REC), "parameters_base": P.PARAMS, "NP": P.NP}, "placement": H.placement(), "code": dict(H.code_pins(), **{"scratchpad/_h2l.py": H.sha_file(os.path.abspath(__file__))})}
    if bad is None:
        assert bool((hard >= 0).all())
        v = P.validity(hard, N, K)
        m = FR.km1_metrics(Hf, hard)
        rec.update({"raw": {"validity": v, "km1": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"]}})
        H.log("S %d K %d: km1 %d, blocks used %d/%d empty %d, max %d (bound %d) -> %s" % (S, K, m["km1_weighted"], m["blocks"]["used"], K, m["blocks"]["empty"], m["blocks"]["max"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"]))
        if not (v["length_ok"] and v["ids_in_range"] and v["max_block"] <= v["contract_bound_ceil_1.03_N_over_k"]):
            bad = "largest block %d exceeds ceil(1.03 N / K) = %d" % (v["max_block"], v["contract_bound_ceil_1.03_N_over_k"])
    if bad is not None:
        rec.update({"STATUS": "PARTITION_INVALID", "reason": bad, "wall_seconds": round(time.time() - t0, 1),
                    "rule": "as _l1h_host.cmd_phg: a map whose largest block exceeds ceil(1.03 N / K) is refused; no map is written (never relaxed or replaced)"})
        H.wj(failp, rec)
        shutil.rmtree(cdir, ignore_errors=True)
        H.log("S %d K %d: PARTITION_INVALID (%s)" % (S, K, bad))
        return
    if m["blocks"]["empty"] > 0:
        hard2, empties, moves = PR.repair(dict(Hf, meta=Hf["meta"]), hard)
        rec["repair"] = {"rule": PR.REPAIR, "empty_blocks": empties, "moves": len(moves)}
        hard = hard2
        assert P.validity(hard, N, K)["gate"] == "PASS"
    else:
        rec["repair"] = None
    hard = hard.astype(np.int64)
    bal = H.balance(hard, N, K)
    assert bal["length_ok"] and bal["ids_in_range"] and bal["every_block_used"], bal
    tmp = npy[:-4] + ".tmp.npy"
    np.save(tmp, hard)
    os.replace(tmp, npy)
    rec.update({"STATUS": "OK", "balance": bal, "wall_seconds": round(time.time() - t0, 1), "output": {"file": H.rel(npy), "sha256": H.sha_file(npy), "n": int(len(hard))}})
    H.wj(runp, rec)
    shutil.rmtree(cdir, ignore_errors=True)
    H.log("S %d K %d: done %.0fs, sizes %d..%d -> %s" % (S, K, rec["wall_seconds"], bal["size_min"], bal["size_max"], H.rel(npy)))


def _test():
    rng = np.random.RandomState(5)
    N, M, S = 300, 400, 4
    sz = rng.randint(1, 9, M)
    eptr = np.zeros(M + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    eidx = np.concatenate([rng.choice(N, s, replace=False) for s in sz]).astype(np.int64)
    ew = rng.randint(1, 50, M).astype(np.int64)
    Hf = {"eptr": eptr, "eidx": eidx, "ew": ew, "N": N, "M": M}
    top = rng.randint(0, S, N)
    subs = restrict(Hf, top, S)
    assert sum(s["N"] for s in subs) == N
    hid = np.repeat(np.arange(M), sz)
    for b, s in enumerate(subs):
        vb = s["vertices"]
        assert bool((top[vb] == b).all()) and bool((np.diff(vb) > 0).all())
        for j in range(s["M"]):
            pins = s["eidx"][s["eptr"][j]:s["eptr"][j + 1]]
            assert len(pins) >= 2
        # the pins of the block, net by net, equal the original net's pins inside the block
        for e in rng.choice(M, 60, replace=False):
            orig = eidx[eptr[e]:eptr[e + 1]]
            inb = orig[top[orig] == b]
            if len(inb) >= 2:
                cand = [j for j in range(s["M"]) if s["ew"][j] == ew[e] and np.array_equal(vb[s["eidx"][s["eptr"][j]:s["eptr"][j + 1]]], inb)]
                assert cand, "net %d restricted to block %d not found" % (e, b)
    # KM1 identity: K-way KM1 = top-level KM1-part (nets cut by the top) + sum of the sub-problems' KM1 (each net's connectivity splits over the blocks it touches)
    Kb = 3
    sub_part = [rng.randint(0, Kb, s["N"]) for s in subs]
    hard = np.empty(N, np.int64)
    for b, s in enumerate(subs):
        hard[s["vertices"]] = b * Kb + sub_part[b]
    lam = np.array([len(set(hard[eidx[eptr[e]:eptr[e + 1]]].tolist())) for e in range(M)])
    km1_full = int((ew * (lam - 1)).sum())
    lam_top = np.array([len(set(top[eidx[eptr[e]:eptr[e + 1]]].tolist())) for e in range(M)])
    km1_top = int((ew * (lam_top - 1)).sum())
    km1_sub = 0
    for b, s in enumerate(subs):
        for j in range(s["M"]):
            pins = s["eidx"][s["eptr"][j]:s["eptr"][j + 1]]
            km1_sub += int(s["ew"][j]) * (len(set(sub_part[b][pins].tolist())) - 1)
    assert km1_full == km1_top + km1_sub, (km1_full, km1_top, km1_sub)
    assert tol_for(100, 400, 8, 2) == math.floor(1.03 * 50 * 2 / 100 * 1e4) / 1e4
    # block checkpoints: round trip is exact; any change of the key (top map, tolerance, shape, parameters), a wrong length or an out-of-range id is a miss, never a silent reuse
    import tempfile
    cd = tempfile.mkdtemp(prefix="h2l_ckpt_")
    try:
        key = {"top_sha256": "a" * 64, "hg_content_digest": "b" * 16, "S": 4, "K": 12, "block": 1, "N": 7, "M": 5, "pins": 11, "zoltan_tolerance": 1.0123, "kb": 3, "NP": 4, "params": [["IMBALANCE_TOL", "1.0123"]]}
        hb0 = np.array([0, 2, 1, 1, 0, 2, 2], np.int64)
        rec0 = {"block": 1, "N": 7, "status": "OK", "seconds": 1.5, "peak_rss_kb_per_rank": [1, 2, 3, 4]}
        assert ckpt_load(cd, 1, key) is None
        ckpt_save(cd, 1, key, rec0, hb0)
        r1, h1 = ckpt_load(cd, 1, key)
        assert r1 == rec0 and np.array_equal(h1, hb0) and h1.dtype == np.int64
        for ch in ({"top_sha256": "c" * 64}, {"zoltan_tolerance": 1.0124}, {"N": 8}, {"kb": 4}, {"params": [["IMBALANCE_TOL", "1.0124"]]}, {"K": 24}):
            assert ckpt_load(cd, 1, dict(key, **ch)) is None, ch
        ckpt_save(cd, 2, dict(key, block=2, kb=2), rec0, hb0)          # an id >= kb is refused
        assert ckpt_load(cd, 2, dict(key, block=2, kb=2)) is None
        with open(os.path.join(cd, "b03.npz"), "wb") as f:              # a torn / unreadable file is a miss
            f.write(b"not an npz")
        assert ckpt_load(cd, 3, key) is None
    finally:
        shutil.rmtree(cd, ignore_errors=True)
    H.log("TEST PASS: restriction = net-split sub-hypergraphs; K-way KM1 = top-level KM1 + sum of sub-problem KM1 (%d = %d + %d)" % (km1_full, km1_top, km1_sub))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a and a[0] == "TEST":
        _test()
    elif a and a[0] == "CONTROL":
        cmd_control([int(x) for x in a[1:]])
    elif a and a[0] == "TOP":
        H.log("placement:", json.dumps(H.placement()))
        cmd_top([int(x) for x in a[1:]])
    elif len(a) >= 3 and a[0] == "SUB":
        S = int(a[1])
        for K in a[2:]:
            cmd_sub(S, int(K))
    else:
        raise SystemExit(__doc__)
