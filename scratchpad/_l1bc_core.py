"""L1 BACKWARD CAUSAL OPTIMIZATION PHASE -- STEP 0 + STEP 3 substrate.

One replay pass per corpus over the FROZEN bounded traversal (parity asserted), producing for every
query the complete inference-safe candidate/evidence table and the partition x evidence-atom matrix.
No new graph traversal, no new encoder, no learned weight, no gold anywhere in the inputs.

CANDIDATE UNIVERSE, per query, with a flag for each membership:
    prot   the protected canonical top-(P-B) = 44
    bnd    the canonical boundary 44..50
    chal   the frozen F6 challenger list  = SF \\ base50  ++  RF \\ base50
    cont   the canonical continuation base_rank[50:]
    reach  everything the bounded machinery touched at all: canonical 200, the dense / SPLADE /
           RRF continuations, every structurally admitted node, and every edge endpoint the beam
           inspected.  This is the ceiling for O3/O4 and the UNREACHABLE test in the ledger.

EVIDENCE ATOMS.  Under a hard partitioning a per-NODE atom is covered by exactly one partition, so
a node-atom coverage function is modular and cannot express redundancy at all.  The atoms therefore
have to be things SEVERAL partitions can carry, which is also what makes coverage submodular:

    CH_*    one atom per retrieval/structural channel (dense, splade, retrieval continuation,
            canonical, structural).  Many partitions rank inside each.
    SEED_*  one atom per dense retrieval seed.  Many partitions are structurally reached from the
            same seed.  This is the "Seed A / Seed B" family.
    SRC_*   one atom per PROTECTED-CORE source partition Pi.  Many partitions are reached by an
            actual graph transition out of the same Pi.  This is the core-exit family, and it is
            the one quantity the node collapse destroys.

Every atom is rank-normalised identically -- W[p, e] = 1 / (K0 + rank of p within atom e), and 0
when p is not in that atom's list.  No channel weight, no learned weight, no tuning.

  python scratchpad/_l1bc_core.py <ds> [nq]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1sr_eval as EV
import _l1tp_core as TP

BCD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_BACKWARD"
K0 = EV.K0
P, B = EV.P, EV.B
MISS = 10 ** 6
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# per-partition rank columns.  ALL oriented so that SMALLER IS BETTER, MISS when absent, which is
# what the Pareto dominance test needs and what the ranking columns already mean.
RCOL = ["r_canon", "r_dense", "r_splade", "r_retc", "r_struct", "r_sfirst", "r_scount",
        "r_ssdir", "r_seedsup", "r_coreexit"]
FCOL = ["in_prot", "in_bnd", "in_chal", "in_cont", "in_reach"]


def _rank_of_partitions(nodes, hard, limit=None):
    """first-arrival index of each partition in a node ordering -> a dense 0..n-1 rank."""
    seen = {}
    for jj, v in enumerate(nodes if limit is None else nodes[:limit]):
        v = int(v)
        if v < 0:
            break
        p = int(hard[v])
        if p >= 0 and p not in seen:
            seen[p] = jj
    return {p: r for r, (p, _) in enumerate(sorted(seen.items(), key=lambda kv: (kv[1], kv[0])))}


def build(ds, nq_max=None, log=log):
    C = TP.ctx(ds)
    S = EV.substrate(ds)
    nq = S["nq"] if nq_max is None else min(int(nq_max), S["nq"])
    hard = np.asarray(S["z"]["hard"], np.int64)
    z = S["z"]
    ctxs = S["ctxs"]
    npart = int(hard.max()) + 1
    dense, splade, rrf = z["ret_dense"], z["ret_splade"], z["ret_rrf"]
    seeds = z["seeds"]

    cols = {k: [] for k in RCOL + FCOL}
    pid, qptr = [], [0]
    A_aid, A_pix, A_w, A_qptr, A_names = [], [], [], [0], []
    SRCP = []
    par_ok = 0
    for qi in range(nq):
        c = ctxs[qi]
        E, rq, par, st = TP.replay(C, qi)
        par_ok += int(par)

        # ---------------- membership sets
        prot, bnd = set(c["prot"]), set(c["bnd"])
        b50 = c["base50"]
        SF = [int(p) for p in c["spos"].keys()] if isinstance(c["spos"], dict) else []
        spos = c["spos"]; rpos_full = {p: r for r, p in enumerate(c["_RF"])}
        chal = set(p for p in spos if p not in b50) | set(p for p in rpos_full if p not in b50)
        cont = set(int(p) for p in c["_CONT"])
        u, v = E["u"], E["v"]
        # base_rank is ALREADY a partition ranking; every other array is a NODE ranking.  The flag
        # is explicit because `arr is not z["base_rank"][qi]` re-indexes and builds a fresh view, so
        # an identity test silently sends the canonical partitions through hard[] as doc ids.
        reach = set()
        for arr, is_part in ((z["base_rank"][qi], True), (dense[qi], False), (splade[qi], False),
                             (rrf[qi], False), (z["s_node"][qi], False)):
            for x in arr:
                x = int(x)
                if x < 0:
                    continue
                pp = x if is_part else int(hard[x])
                if pp >= 0:
                    reach.add(int(pp))
        if len(v):
            reach |= set(int(x) for x in np.unique(hard[np.concatenate([u, v])]) if x >= 0)
        reach.discard(-1)
        # the visited universe must contain everything already offered to the selector, or the
        # O1 <= O3 rung of the oracle chain is not a rung at all
        assert prot <= reach and bnd <= reach and chal <= reach, "reach lost an offered candidate"

        univ = sorted(prot | bnd | chal | cont | reach)
        ix = {p: i for i, p in enumerate(univ)}
        n = len(univ)

        # ---------------- rank columns
        r_dense = _rank_of_partitions(dense[qi], hard)
        r_splade = _rank_of_partitions(splade[qi], hard)
        sagg = c["sagg"]
        s_first = {p: a[0] for p, a in sagg.items()}
        s_cnt = {p: a[1] for p, a in sagg.items()}
        s_sdir = {p: a[4] for p, a in sagg.items()}
        rk = lambda d, rev: {p: r for r, (p, _) in enumerate(
            sorted(d.items(), key=lambda kv: ((-kv[1]) if rev else kv[1], kv[0])))}
        r_sfirst, r_scount, r_ssdir = rk(s_first, False), rk(s_cnt, True), rk(s_sdir, True)

        # seed provenance and core-exit provenance, both from the replayed edges
        seedsup = {}
        ce = {}
        if len(v):
            pj = hard[v]; pi = hard[u]
            ok = (pj >= 0)
            sd = E["seed"][ok]; pjo = pj[ok]; pio = pi[ok]; t0 = E["T0_OFFSET"][ok]
            for a, bq in zip(pjo.tolist(), sd.tolist()):
                seedsup.setdefault(a, set()).add(bq)
            inc = np.fromiter((int(x) in prot for x in pio), bool, len(pio))
            if inc.any():
                for a, w_ in zip(pjo[inc].tolist(), t0[inc].tolist()):
                    if w_ > ce.get(a, -np.inf):
                        ce[a] = w_
        r_seedsup = rk({p: len(sset) for p, sset in seedsup.items()}, True)
        r_coreexit = rk(ce, True)

        src = {"r_canon": c["cpos"], "r_dense": r_dense, "r_splade": r_splade,
               "r_retc": rpos_full, "r_struct": spos, "r_sfirst": r_sfirst,
               "r_scount": r_scount, "r_ssdir": r_ssdir, "r_seedsup": r_seedsup,
               "r_coreexit": r_coreexit}
        for k in RCOL:
            d = src[k]
            cols[k].append(np.array([d.get(p, MISS) for p in univ], np.int32))
        for k, st_ in (("in_prot", prot), ("in_bnd", bnd), ("in_chal", chal),
                       ("in_cont", cont), ("in_reach", reach)):
            cols[k].append(np.fromiter((p in st_ for p in univ), np.int8, n))
        pid.append(np.asarray(univ, np.int32)); qptr.append(qptr[-1] + n)

        # ---------------- evidence atoms.  identical rank normalisation for every atom.
        atoms = []
        for nm, d in (("CH_CANON", c["cpos"]), ("CH_DENSE", r_dense), ("CH_SPLADE", r_splade),
                      ("CH_RETC", rpos_full), ("CH_STRUCT", spos)):
            atoms.append((nm, sorted(d.items(), key=lambda kv: (kv[1], kv[0]))))
        for si, sv in enumerate([int(x) for x in seeds[qi] if x >= 0]):
            mem = {p: min(s_first.get(p, MISS), MISS) for p, ss in seedsup.items() if sv in ss}
            atoms.append((f"SEED_{si}", sorted(mem.items(), key=lambda kv: (kv[1], kv[0]))))
        if len(v):
            pjq = hard[v]; piq = hard[u]; okq = pjq >= 0
            pjo, pio, t0o = pjq[okq], piq[okq], E["T0_OFFSET"][okq]
            for Pi in sorted(prot):
                m2 = pio == Pi
                if not m2.any():
                    continue
                best = {}
                for a, w_ in zip(pjo[m2].tolist(), t0o[m2].tolist()):
                    if w_ > best.get(a, -np.inf):
                        best[a] = w_
                atoms.append((f"SRC_{Pi}",
                              sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))))
        SRCP.append(np.array([int(nm[4:]) for nm, _ in atoms if nm.startswith("SRC_")], np.int32))
        A_names.append((sum(1 for nm, _ in atoms if nm.startswith("CH_")),
                        sum(1 for nm, _ in atoms if nm.startswith("SEED_")),
                        sum(1 for nm, _ in atoms if nm.startswith("SRC_"))))
        for ai, (nm, lst) in enumerate(atoms):
            for r, (p, _) in enumerate(lst):
                j = ix.get(int(p))
                if j is None:
                    continue
                A_aid.append(ai); A_pix.append(j); A_w.append(1.0 / (K0 + r))
        A_qptr.append(len(A_aid))
        if (qi + 1) % 250 == 0:
            log(f"   {ds} {qi+1}/{nq}  parity {par_ok}/{qi+1}  univ {n}  atoms {len(atoms)}")

    assert par_ok == nq, f"PARITY BROKEN {par_ok}/{nq}"
    D = {k: np.concatenate(cols[k]) for k in RCOL + FCOL}
    D["pid"] = np.concatenate(pid)
    D["qptr"] = np.asarray(qptr, np.int64)
    D["A_aid"] = np.asarray(A_aid, np.int32)
    D["A_pix"] = np.asarray(A_pix, np.int32)
    D["A_w"] = np.asarray(A_w, np.float64)
    D["A_qptr"] = np.asarray(A_qptr, np.int64)
    D["A_fam"] = np.asarray(A_names, np.int32)          # per query: (n_CH, n_SEED, n_SRC)
    D["A_srcpid"] = np.concatenate(SRCP) if SRCP else np.zeros(0, np.int32)
    D["nq"] = np.array([nq]); D["npart"] = np.array([npart])
    os.makedirs(f"{BCD}/data", exist_ok=True); os.makedirs(f"{BCD}/diag", exist_ok=True)
    np.savez_compressed(f"{BCD}/data/bc_{ds}.npz", **D)
    log(f"[{ds}] PARITY EXACT {par_ok}/{nq}  universe/q {len(D['pid'])/nq:.1f}  "
        f"atom entries/q {len(A_aid)/nq:.0f}")
    return D


def rows(T, qi):
    a, b = int(T["qptr"][qi]), int(T["qptr"][qi + 1])
    d = {k: T[k][a:b] for k in RCOL + FCOL}
    d["pid"] = T["pid"][a:b]
    d["n"] = b - a
    c, e = int(T["A_qptr"][qi]), int(T["A_qptr"][qi + 1])
    d["A_aid"] = T["A_aid"][c:e]; d["A_pix"] = T["A_pix"][c:e]; d["A_w"] = T["A_w"][c:e]
    d["n_atom"] = int(d["A_aid"].max()) + 1 if e > c else 0
    d["fam"] = T["A_fam"][qi]
    ns = T["A_fam"][:, 2].astype(np.int64)
    o = int(ns[:qi].sum())
    d["src_pid"] = T["A_srcpid"][o:o + int(ns[qi])]      # source partition of each SRC_* atom
    d["a0_src"] = int(T["A_fam"][qi][0]) + int(T["A_fam"][qi][1])   # first SRC atom index
    return d


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
          int(sys.argv[2]) if len(sys.argv) > 2 else None)
