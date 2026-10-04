"""PHASES 9-12 -- core-scored / halo-fetched.

The decisive overlap experiment, and the reason it is clean: halo nodes are NOT allowed to
influence ranking.  The cores, their scores and their identities are exactly today's, so the
selected top-50 core ids are bit-identical to O0_CORE by construction -- this module never
recomputes them, it reads them from the frozen Phase-C path and only changes what a selected
block PAYS OUT.

    ranking unit = balanced CORE      (bit-identical to today)
    fetch unit   = CORE + 1-hop HALO  (deduplicated across the 50 selected blocks)

That makes the trade-off exposure, not ranking regression: adding halo nodes can only add
coverage or do nothing, it can never remove a required node the core path already fetched.
This module measures exactly that trade-off.

Because a node can now sit in many blocks, coverage is evaluated on REQUIRED NODES directly
(Phase 10) rather than through a canonical partition id, and the old "number of distinct gold
partitions" is replaced by an exact MIN_BLOCK_COVER set cover (Phase 11).

  python scratchpad/_l1ov_eval.py <ds> [core_tag] [fam ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ep_pu as PU
import _l1ep_c as EC
import _l1ps_router as RT
import _l1kb_core as KB

OUT, ROOT, log = OV.OUT, EC.ROOT, OV.log
CFG = dict(KB.BASE_CFG)
P = 50


def selected_blocks(ds, hard, npart, log=log):
    """the frozen selection: BASE top-50 and F6 top-50 core ids, per query.

    Read from the same rebuild + router path Phase C uses, so these are the incumbent numbers;
    nothing about the halo can reach this function.
    """
    z0 = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}
    z.update(EC.rebuild(ds, hard, npart, z, meta, log))
    C = RT.build_cache(ds, z, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    nq = meta["n_dev_queries"]
    goldp = [set(int(x) for x in z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]])
             for qi in range(nq)]
    ind_base = np.array([int(goldp[qi] <= C["base50"][qi]) for qi in range(nq)], np.int8)
    ctxs = KB.contexts(z, meta, C, CFG["B"], CFG)
    # the frozen F6 lambda, character for character the one _l1ep_c.replay uses
    sel = lambda c, qi, extra: KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"],
                                            c["cpos"], CFG["B"])[0]
    r6 = KB.run_selector(ctxs, sel, goldp, ind_base)
    rb = KB.run_selector(ctxs, KB.sel_base, goldp, ind_base)
    base50 = [sorted(f) for f in rb["finals"]]
    f650 = [sorted(f) for f in r6["finals"]]
    assert all(len(s) == P for s in base50) and all(len(s) == P for s in f650)
    return z, meta, C, ctxs, base50, f650, ind_base, {
        "BASE_ALL_P50": round(float(ind_base.mean()), 4),
        "F6_ALL_P50": round(float(r6["ALL"]), 4)}


def cover_min_blocks(need, blocks_of, budget=20000):
    """PHASE 11 -- MIN_BLOCK_COVER: fewest retrieval blocks whose union holds every required node.

    Greedy gives an upper bound, then a branch-and-bound over the blocks of the least-covered
    required node proves optimality.  Overlap makes the instance genuinely harder than it was
    for hard partitions -- a node can sit in thousands of blocks -- so the search carries an
    explicit step budget and reports whether the answer is EXACT or the greedy upper bound.
    Returns (k, exact).
    """
    if not need:
        return 0, True
    sets = {}
    for g in need:
        for b in blocks_of(g):
            sets.setdefault(b, set()).add(g)
    covered = set()
    for v in sets.values():
        covered |= v
    if covered != set(need):
        return -1, True                 # not coverable by any number of blocks
    rem, k = set(need), 0
    while rem:
        b = max(sets, key=lambda b: len(sets[b] & rem))
        got = sets[b] & rem
        if not got:
            return -1, True
        rem -= got
        k += 1
    if k <= 1:
        return k, True
    best = [k]
    steps = [0]
    # per required node, its blocks ordered by how much they cover -- best-first branching
    bl = {g: sorted((b for b in blocks_of(g) if b in sets),
                    key=lambda b: -len(sets[b])) for g in need}

    def bb(rem, used):
        if steps[0] > budget:
            return
        steps[0] += 1
        if used + 1 >= best[0]:
            return
        # branch on the required node with the fewest covering blocks
        g = min(rem, key=lambda x: len(bl[x]))
        for b in bl[g]:
            nrem = rem - sets[b]
            if not nrem:
                best[0] = used + 1
                return
            bb(nrem, used + 1)
            if steps[0] > budget:
                return
    bb(frozenset(need), 0)
    return best[0], steps[0] <= budget


def evaluate(ds, core_tag="CURRENT", fams=("O0_CORE",) + tuple(OV.FAMS), betas=(), log=log):
    hard, npart = PU.load_assignment(ds, core_tag) if core_tag == "CURRENT" else (
        np.load(f"scratchpad/_l1ep/parts/{ds}__{core_tag}.npy"), None)
    hard = np.asarray(hard, np.int64)
    if npart is None:
        npart = int(hard.max()) + 1
    N = len(hard)
    z, meta, C, ctxs, base50, f650, ind_base, PAR = selected_blocks(ds, hard, npart, log)
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    core_sizes = np.bincount(hard, minlength=npart).astype(np.int64)

    R = {"ds": ds, "core_tag": core_tag, "N": int(N), "npart": int(npart), "nq": int(nq),
         "PARTITION_PARITY": PAR, "CELLS": {}}
    for fam in fams:
        variants = [(fam, None)] if fam == "O0_CORE" else \
                   [(fam, None)] + [(f"{fam}_b{b}", b) for b in betas]
        for name, beta in variants:
            t0 = time.time()
            if fam == "O0_CORE":
                nptr = np.zeros(N + 1, np.int64); nidx = np.zeros(0, np.int32)
                bptr = np.zeros(npart + 1, np.int64); bidx = np.zeros(0, np.int32)
            else:
                pairs = OV.halo_pairs(ds, hard, core_tag, fam, N, log)
                if beta is not None:
                    pk, mass = OV.boundary_mass(ds, hard, core_tag, fam, N, npart, log)
                    pairs = OV.bounded_pairs(pk, mass, hard, npart, N, beta)
                nptr, nidx = OV.to_node_csr(pairs, npart, N)
                bptr, bidx = OV.to_block_csr(pairs, npart, N)
            halo_sz = np.diff(bptr)

            rec = {"halo_memberships": int(len(bidx)),
                   "REPLICATION_FACTOR": round((N + len(bidx)) / N, 4)}
            for lane, SEL in (("BASE", base50), ("F6", f650)):
                if any(s is None for s in SEL):
                    continue
                rn_num = np.zeros(nq); rn_den = np.zeros(nq)
                allf = np.zeros(nq, np.int8); allcore = np.zeros(nq, np.int8)
                newh = np.zeros(nq, np.int64)
                expo = np.zeros(nq, np.int64); expo_core = np.zeros(nq, np.int64)
                mbc = np.full(nq, -2, np.int32)
                mbc_exact = np.ones(nq, np.int8)
                selmask = np.zeros(npart, bool)
                for qi in range(nq):
                    S = SEL[qi]
                    Ss = set(S)
                    selmask[:] = False
                    selmask[np.asarray(S, np.int64)] = True
                    # dedupe: webqsp lists some required nodes twice, and a set-valued
                    # "did we fetch all of them" must not count a node twice
                    need = sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]})
                    if not need:
                        rn_den[qi] = np.nan; continue
                    incore = [x for x in need if int(hard[x]) in Ss]
                    got = set(incore)
                    for x in need:
                        if x in got:
                            continue
                        bs = nidx[nptr[x]:nptr[x + 1]]
                        if len(bs) and Ss.intersection(bs.tolist()):
                            got.add(x)
                    rn_num[qi] = len(got); rn_den[qi] = len(need)
                    allf[qi] = int(len(got) == len(need))
                    allcore[qi] = int(len(incore) == len(need))
                    newh[qi] = len(got) - len(incore)
                    expo_core[qi] = int(core_sizes[S].sum())
                    if len(bidx):
                        u = np.unique(np.concatenate([bidx[bptr[j]:bptr[j + 1]] for j in S]))
                        # halo nodes already inside a selected core are not new exposure
                        expo[qi] = expo_core[qi] + int(np.count_nonzero(
                            ~selmask[hard[u.astype(np.int64)]]))
                    else:
                        expo[qi] = expo_core[qi]
                    if lane == "F6":
                        blocks_of = (lambda x: [int(hard[x])] +
                                     nidx[nptr[x]:nptr[x + 1]].tolist())
                        mbc[qi], ex = cover_min_blocks(need, blocks_of)
                        mbc_exact[qi] = int(ex)
                ok = ~np.isnan(rn_den)
                rec[lane] = {
                    "REQUIRED_NODE_RECALL": round(float((rn_num[ok] / rn_den[ok]).mean()), 4),
                    "ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4),
                    "ALL_REQUIRED_CORE_ONLY": round(float(allcore[ok].mean()), 4),
                    "NEW_REQUIRED_FROM_HALO": int(newh.sum()),
                    "queries_rescued": int(((allf == 1) & (allcore == 0)).sum()),
                    "UNIQUE_EXPOSURE_P50": round(float(expo[ok].mean()), 1),
                    "CORE_EXPOSURE_P50": round(float(expo_core[ok].mean()), 1),
                    "EXPOSURE_MULTIPLIER": round(float(expo[ok].mean() /
                                                       max(expo_core[ok].mean(), 1e-9)), 4),
                    "_ind_ALL": allf.tolist()}
                extra = float(expo[ok].mean() - expo_core[ok].mean())
                rec[lane]["required_per_1k_extra_exposed"] = (
                    round(float(newh.sum()) / nq / max(extra, 1e-9) * 1000, 4) if extra > 0 else None)
                if lane == "F6" and (mbc > -2).any():
                    v = mbc[(mbc > -2) & ok]
                    rec["MIN_BLOCK_COVER"] = {
                        "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                        "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)),
                        "max": int(v.max()), "uncoverable": int((v < 0).sum()),
                        "exact_fraction": round(float(mbc_exact[(mbc > -2) & ok].mean()), 4)}
                    if ds == "metaqa":
                        h = np.asarray(hops)[:nq]
                        rec["BY_HOP"] = {f"hop{k}": {
                            "ALL_REQUIRED_FETCHED": round(float(allf[(h == k) & ok].mean()), 4),
                            "REQUIRED_NODE_RECALL": round(float(
                                (rn_num[(h == k) & ok] / rn_den[(h == k) & ok]).mean()), 4),
                            "MIN_BLOCK_COVER_median": float(np.median(mbc[(h == k) & (mbc > -2)]))}
                            for k in (1, 2, 3) if ((h == k) & ok).any()}
            rec["seconds"] = round(time.time() - t0, 1)
            R["CELLS"][name] = rec
            f6 = rec.get("F6", {})
            log(f"  {ds:16s} {name:16s} R={rec['REPLICATION_FACTOR']:7.3f}  "
                f"ALLREQ {f6.get('ALL_REQUIRED_FETCHED')}  (core {f6.get('ALL_REQUIRED_CORE_ONLY')})"
                f"  expo x{f6.get('EXPOSURE_MULTIPLIER')}  rescued {f6.get('queries_rescued')}")
    return R


if __name__ == "__main__":
    ds = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "CURRENT"
    rest = sys.argv[3:]
    betas = tuple(float(a[1:]) for a in rest if a.startswith("b"))
    fams = tuple(a for a in rest if not a.startswith("b")) or (("O0_CORE",) + tuple(OV.FAMS))
    os.makedirs(f"{OUT}/overlap", exist_ok=True)
    fp = f"{OUT}/overlap/COVERAGE_{ds}.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    R = evaluate(ds, tag, tuple(fams), betas=betas)
    rec.setdefault(tag, {}).update(R["CELLS"])
    rec[tag]["_meta"] = {k: R[k] for k in ("ds", "N", "npart", "nq")}
    rec[tag]["_meta"]["PARTITION_PARITY"] = R["PARTITION_PARITY"]
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
