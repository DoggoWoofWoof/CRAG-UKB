"""L1 CANDIDATE ADMISSION PHASE -- STEPS 1-7 and 10.

The stage under test is ADMISSION, not selection: compress the already-visited partition universe
into a pool of EXACTLY the frozen per-query size K(q) that contains more of the missing required
partitions.  No new traversal, no new encoder, no learning, no wider production pool.

POOLS (production pools all have exactly K(q) members and all exclude the protected core)
    SAFE_POOL        the frozen F6 candidate list, bnd ++ chal          -- incumbent
    A1_SRC_ASSIGN    source-conditioned assignment over the SRC_* core-exit atoms alone
    A2_HYBRID_ADMIT  SAFE candidates carrying unique non-SRC evidence, then SRC-assignment fills
    SAFE_U_SRC       diagnostic union only -- NEVER a production pool

A1 is a bounded round robin, not a global ranking.  Sources are visited in canonical-rank order of
the SOURCE partition and each contributes its best not-yet-admitted target before any source gets a
second slot, so one dense core partition cannot consume the whole admission budget.

A2 is the safeguard STEP 11 of the previous phase demanded.  A SAFE candidate is RETAINED when it
wins at least one NON-SRC evidence atom outright inside prot | SAFE_POOL -- that is, when it is the
sole carrier of some dense / splade / canonical / structural / seed evidence.  Only the slots left
over by redundant SAFE candidates are offered to SRC assignment, and if SRC evidence is absent A2
falls back to exactly the SAFE set.  SRC never erases a channel.

SELECTORS.  `f6_on_pool` is the frozen equal-RRF rule re-expressed for an arbitrary pool and is
gated to reproduce KB.f6_select bit-for-bit on the SAFE pool.  `g4_on_pool` is the previous phase's
assignment objective with the tie-break made total (frozen order, then partition id).

  python scratchpad/_l1ca_admit.py <ds> [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1pp_core as PP
import _l1bc_core as BC
import _l1bc_ledger as LG

P, B, K0, MISS = BC.P, BC.B, BC.K0, BC.MISS
CAD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_ADMISSION"
POOLS = ["SAFE_POOL", "A1_SRC_ASSIGN", "A2_HYBRID_ADMIT", "A3_INTERLEAVE",
         "A4_SUBSUME", "SAFE_U_SRC"]
PROD = ["SAFE_POOL", "A1_SRC_ASSIGN", "A2_HYBRID_ADMIT", "A3_INTERLEAVE", "A4_SUBSUME"]
# A1 is an ADMISSION probe only: it is pure-SRC and is legitimately empty when a query has no
# core-exit evidence, so it cannot always fill B=6 and is never run through a selector.
MTX = ["SAFE_POOL", "A2_HYBRID_ADMIT", "A3_INTERLEAVE", "A4_SUBSUME"]
SELS = ["F6", "G4"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ evidence helpers
def wmat(r):
    n_atom = r["n_atom"]
    W = np.zeros((max(n_atom, 1), r["n"]))
    if n_atom:
        W[r["A_aid"], r["A_pix"]] = r["A_w"]
    return W


def src_atoms(r):
    """[(source partition, target row-indices in atom order)] over the SRC_* family."""
    a0, out = r["a0_src"], []
    aid, pix, w = r["A_aid"], r["A_pix"], r["A_w"]
    for k, Pi in enumerate(r["src_pid"]):
        m = aid == (a0 + k)
        if not m.any():
            continue
        o = np.argsort(-w[m], kind="mergesort")
        out.append((int(Pi), pix[m][o]))
    return out


# ------------------------------------------------------------------ admission
def a1_pool(r, cpos, K, exclude):
    """bounded round robin over the SRC atoms; deterministic, parameter-free."""
    at = src_atoms(r)
    at.sort(key=lambda t: (cpos.get(t[0], MISS), t[0]))
    pid = r["pid"]
    ptr = [0] * len(at)
    seen, out = set(exclude), []
    moved = True
    while len(out) < K and moved:
        moved = False
        for i, (_, tg) in enumerate(at):
            if len(out) >= K:
                break
            while ptr[i] < len(tg):
                p = int(pid[tg[ptr[i]]])
                ptr[i] += 1
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    moved = True
                    break
    return out


def a2_pool(r, ixp, cands, cpos, K, prot, a1):
    """retain SAFE candidates with unique NON-SRC evidence, fill the rest from SRC assignment."""
    W = wmat(r)
    a0 = r["a0_src"]
    Wn = W[:a0] if a0 else W[:0]
    pool_ix = np.array([ixp[p] for p in cands], np.int64)
    prot_ix = np.array([ixp[p] for p in prot if p in ixp], np.int64)
    keep = []
    if len(Wn) and len(pool_ix):
        allix = np.concatenate([prot_ix, pool_ix])
        Wa = Wn[:, allix]
        win = np.asarray(np.argmax(Wa, 1))
        bst = Wa[np.arange(len(Wa)), win]
        uniq = set()
        for e in range(len(Wa)):
            if bst[e] > 0 and win[e] >= len(prot_ix):
                uniq.add(cands[win[e] - len(prot_ix)])
        keep = [p for p in cands if p in uniq]
    out = list(keep)
    seen = set(out)
    for p in a1:
        if len(out) >= K:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    for p in cands:                       # never shrink: fall back to the frozen set
        if len(out) >= K:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:K], len(keep)


def a4_pool(r, ixp, cands, K, a1, safe_ord):
    """evict ONLY SAFE candidates whose evidence is subsumed by a better-ranked SAFE candidate.

    This is the non-degenerate form of the STEP-4 safeguard.  A2 tests uniqueness against the
    protected core, which already owns almost every atom, so it retains ~0.4 candidates per query and
    SRC silently takes the whole budget.  Here uniqueness is tested INSIDE the pool: a candidate is
    evictable only when some candidate the frozen admission already ranks above it carries a superset
    of its evidence.  No evidence present in the frozen pool is ever lost, and SRC can only fill
    slots that are provably redundant -- exactly "the old pipeline protects, the new one adds"."""
    W = wmat(r)
    sig = {}
    for p in cands:
        sig[p] = frozenset(np.nonzero(W[:, ixp[p]])[0].tolist())
    order = sorted(cands, key=lambda p: (safe_ord.get(p, MISS), p))
    dom, kept = set(), []
    for p in order:
        if any(sig[p] <= sig[q] for q in kept):
            dom.add(p)
        else:
            kept.append(p)
    out = [p for p in cands if p not in dom]
    seen = set(out)
    for src in (a1, [p for p in order if p in dom]):
        for p in src:
            if len(out) >= K:
                break
            if p not in seen:
                out.append(p)
                seen.add(p)
    return out[:K], len(dom)


def a3_pool(cands, K, a1):
    """equal 1:1 interleave of the two admission channels into one fixed-size pool.

    A2 as specified degenerates: the protected core wins almost every non-SRC atom, so the
    "unique existing SAFE evidence" test retains close to nothing and SRC takes the whole budget.
    This variant is the architecture the safeguard actually calls for -- the frozen admission and
    the SRC assignment alternate, so the head of the frozen pool always survives and SRC only ever
    contributes the slots the frozen channel has not yet claimed.  Equal alternation is the same
    parameter-free choice the frozen equal-RRF fusion already makes between channels."""
    out, seen = [], set()
    ia = ib = 0
    while len(out) < K and (ia < len(cands) or ib < len(a1)):
        for src in (cands, a1):
            i = ia if src is cands else ib
            while i < len(src) and src[i] in seen:
                i += 1
            if i < len(src) and len(out) < K:
                out.append(src[i])
                seen.add(src[i])
                i += 1
            if src is cands:
                ia = i
            else:
                ib = i
    for p in cands:
        if len(out) >= K:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:K]


# ------------------------------------------------------------------ selectors on an arbitrary pool
def f6_on_pool(pool, spos, rpos, cpos):
    sc = []
    for p in pool:
        s = 1.0 / (K0 + cpos[p]) if p in cpos else 0.0
        if p in spos:
            s += 1.0 / (K0 + spos[p])
        if p in rpos:
            s += 1.0 / (K0 + rpos[p])
        sc.append((-s, cpos.get(p, MISS), p))
    sc.sort()
    return [p for _, _, p in sc]


def g4_on_pool(r, ixp, prot, pool, safe_ord):
    W = wmat(r)
    pix = np.array([ixp[p] for p in pool], np.int64)
    prot_ix = np.array([ixp[p] for p in prot if p in ixp], np.int64)
    allix = np.concatenate([prot_ix, pix])
    Wa = W[:, allix]
    win = np.asarray(np.argmax(Wa, 1))
    bst = Wa[np.arange(len(Wa)), win]
    s = np.zeros(len(pool))
    for e in range(len(Wa)):
        if bst[e] > 0 and win[e] >= len(prot_ix):
            s[win[e] - len(prot_ix)] += bst[e]
    key = sorted(range(len(pool)),
                 key=lambda j: (-s[j], safe_ord.get(pool[j], MISS), pool[j]))
    return [pool[j] for j in key], s


# ------------------------------------------------------------------ driver
def run(ds, log=log):
    S = EV.substrate(ds)
    T = dict(np.load(f"{BC.BCD}/data/bc_{ds}.npz"))
    nq = int(T["nq"][0])
    goldp, ctxs = S["goldp"], S["ctxs"]
    hops = np.asarray(S["hops"])[:nq] if S.get("hops") is not None else np.zeros(nq, np.int32)

    rec = {k: np.zeros(nq, np.float64) for k in
           ["n_req", "n_miss", "K", "n_keep", "n_a1", "n_new_a2", "n_new_a3", "n_new_a4", "n_dom", "n_union",
            "miss_visited", "miss_has_src", "miss_admitted"]}
    padm = {p: np.zeros(nq, np.float64) for p in POOLS}      # missing-required admitted
    pall = {p: np.zeros(nq, np.float64) for p in POOLS}      # all-required admitted
    porc = {p: np.zeros(nq, np.int8) for p in POOLS}         # pool oracle at B6
    hit = {(p, s): np.zeros(nq, np.int8) for p in MTX for s in SELS}
    ms = {"frozen_admit": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0, "a4": 0.0, "F6": 0.0, "G4": 0.0}
    picks = {(p, s): [] for p in MTX for s in SELS}
    par = 0
    for qi in range(nq):
        c = ctxs[qi]
        r = BC.rows(T, qi)
        ixp = {int(p): i for i, p in enumerate(r["pid"])}
        prot = [int(p) for p in c["prot"]]
        pset = set(prot)
        cpos, spos, rpos = c["cpos"], c["spos"], c["rpos"]
        t = time.perf_counter()
        X, cands, sc = LG.safe_pick(c)
        ms["frozen_admit"] += time.perf_counter() - t
        cands = [int(p) for p in cands if int(p) in ixp]
        safe_ord = {int(p): i for i, (_, _, p) in enumerate(sc)}
        K = len(cands)
        REQ = set(int(p) for p in goldp[qi])
        sel = pset | set(int(p) for p in X)
        missing = REQ - sel
        reach = set(int(p) for p, f in zip(r["pid"], r["in_reach"]) if f)

        t = time.perf_counter()
        a1 = a1_pool(r, cpos, K, pset)
        ms["a1"] += time.perf_counter() - t
        t = time.perf_counter()
        a2, nkeep = a2_pool(r, ixp, cands, cpos, K, prot, a1)
        ms["a2"] += time.perf_counter() - t
        t = time.perf_counter()
        a3 = a3_pool(cands, K, a1)
        ms["a3"] += time.perf_counter() - t
        t = time.perf_counter()
        a4, ndom = a4_pool(r, ixp, cands, K, a1, safe_ord)
        ms["a4"] += time.perf_counter() - t
        uni = list(dict.fromkeys(cands + a1))
        pools = {"SAFE_POOL": cands, "A1_SRC_ASSIGN": a1, "A2_HYBRID_ADMIT": a2,
                 "A3_INTERLEAVE": a3, "A4_SUBSUME": a4, "SAFE_U_SRC": uni}
        for k in PROD:
            assert len(pools[k]) <= K and not (set(pools[k]) & pset), k
        assert len(a2) == K or K == 0
        assert len(a3) == K or K == 0
        assert len(a4) == K or K == 0

        # ---- STEP 1 facts about the missing required partitions
        srcset = set(int(r["pid"][j]) for _, tg in src_atoms(r) for j in tg)
        rec["n_req"][qi] = len(REQ); rec["n_miss"][qi] = len(missing); rec["K"][qi] = K
        rec["n_keep"][qi] = nkeep; rec["n_a1"][qi] = len(a1)
        rec["n_new_a2"][qi] = len(set(a2) - set(cands))
        rec["n_new_a3"][qi] = len(set(a3) - set(cands))
        rec["n_new_a4"][qi] = len(set(a4) - set(cands)); rec["n_dom"][qi] = ndom
        rec["n_union"][qi] = len(uni)
        rec["miss_visited"][qi] = sum(1 for p in missing if p in reach)
        rec["miss_has_src"][qi] = sum(1 for p in missing if p in srcset)
        rec["miss_admitted"][qi] = sum(1 for p in missing if p in set(cands))

        out = REQ - pset
        for k, pl in pools.items():
            ps = set(pl)
            padm[k][qi] = len(missing & ps)
            pall[k][qi] = len(out & ps)
            porc[k][qi] = int(len(out) <= B and out <= ps)

        # ---- STEP 6 the 2x2
        for k in MTX:
            pl = pools[k]
            t = time.perf_counter()
            o6 = f6_on_pool(pl, spos, rpos, cpos)[:B]
            ms["F6"] += time.perf_counter() - t
            t = time.perf_counter()
            og, _ = g4_on_pool(r, ixp, prot, pl, safe_ord)
            ms["G4"] += time.perf_counter() - t
            og = og[:B]
            if k == "SAFE_POOL":
                par += int(list(o6) == [int(x) for x in X])
            for s, pk in (("F6", o6), ("G4", og)):
                fs = pset | set(pk)
                assert len(fs) == P, f"{len(fs)} != {P}"
                hit[(k, s)][qi] = int(REQ <= fs)
                picks[(k, s)].append(list(pk))
        if (qi + 1) % 500 == 0:
            log(f"   {ds} admit {qi+1}/{nq}")
    assert par == nq, f"F6 PARITY BROKEN {par}/{nq}"
    return dict(ds=ds, nq=nq, hops=hops, rec=rec, padm=padm, pall=pall, porc=porc,
                hit=hit, ms=ms, picks=picks, parity=par)


def report(R):
    ds, nq = R["ds"], R["nq"]
    MS = LG.masks(ds, R["hops"], nq)
    out = {"ds": ds, "nq": nq, "F6_PARITY_ON_SAFE_POOL": f"{R['parity']}/{nq}",
           "pool_sizes": {}, "admission": {}, "pool_oracle": {}, "p50_2x2": {}, "latency_ms": {}}
    rc = R["rec"]
    out["pool_sizes"] = {"K_per_q": round(float(rc["K"].mean()), 2),
                         "A1_filled_per_q": round(float(rc["n_a1"].mean()), 2),
                         "A2_retained_SAFE_per_q": round(float(rc["n_keep"].mean()), 2),
                         "A2_new_vs_SAFE_per_q": round(float(rc["n_new_a2"].mean()), 2),
                         "A3_new_vs_SAFE_per_q": round(float(rc["n_new_a3"].mean()), 2),
                         "A4_new_vs_SAFE_per_q": round(float(rc["n_new_a4"].mean()), 2),
                         "A4_subsumed_evictable_per_q": round(float(rc["n_dom"].mean()), 2),
                         "UNION_per_q": round(float(rc["n_union"].mean()), 2)}
    for nm, m in MS:
        den = float(rc["n_miss"][m].sum())
        dena = float((rc["n_req"][m] - 0).sum())
        out["admission"][nm] = {
            "missing_total": int(den),
            "missing_visited": round(float(rc["miss_visited"][m].sum()) / den, 4) if den else 0.0,
            "missing_with_SRC_atom": round(float(rc["miss_has_src"][m].sum()) / den, 4) if den else 0.0,
            "missing_already_admitted": round(float(rc["miss_admitted"][m].sum()) / den, 4) if den else 0.0}
        for k in POOLS:
            out["admission"][nm][k + "_MISSING_RECALL"] = round(
                float(R["padm"][k][m].sum()) / den, 4) if den else 0.0
        out["pool_oracle"][nm] = {k: round(float(R["porc"][k][m].mean()), 4) for k in POOLS}
        out["p50_2x2"][nm] = {}
        base = R["hit"][("SAFE_POOL", "F6")]
        for k in MTX:
            for s in SELS:
                st = PP.mcnemar(R["hit"][(k, s)][m], base[m])
                out["p50_2x2"][nm][f"{k}+{s}"] = {
                    "acc": round(float(R["hit"][(k, s)][m].mean()), 4), "net": st["net"],
                    "gained": st["gained"], "lost": st["lost"],
                    "p": round(st["mcnemar_p"], 4), "sig": bool(st["sig"])}
    out["latency_ms"] = {k: round(v * 1000.0 / nq, 4) for k, v in R["ms"].items()}
    return out


if __name__ == "__main__":
    os.makedirs(f"{CAD}/diag", exist_ok=True)
    for ds in (sys.argv[1:] or ["metaqa"]):
        R = run(ds)
        rp = report(R)
        json.dump(rp, open(f"{CAD}/diag/admit_{ds}.json", "w"), indent=1)
        np.savez_compressed(f"{CAD}/diag/hits_{ds}.npz",
                            **{f"{k}__{s}": R["hit"][(k, s)] for k in MTX for s in SELS},
                            hops=R["hops"])
        log(ds, json.dumps({k: rp[k] for k in ("F6_PARITY_ON_SAFE_POOL", "pool_sizes",
                                               "pool_oracle", "latency_ms")}, indent=1))
        log(ds, "2x2 ALL", json.dumps(rp["p50_2x2"]["ALL"], indent=1))
