"""L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- STEPS 2-11, one replay pass per corpus.

For every query the SAME frozen retrieval seeds and the SAME frozen residual are handed to each
traversal substrate in turn (`_l1kn_sub.SUBS`).  Nothing upstream of the graph changes: seeds,
embeddings, partitions, hop depth, beam width, DEG_CAP, edge budget, M_struct read depth, B and P
are the frozen contract.  Only the adjacency changes.

Recorded per query x substrate:
  reach       nodes/partitions VISITED (everything the bounded search scored), ADMITTED (`added`,
              M_MAX=256) and READ (`added[:M_struct]`, what the S4 aggregation actually sees)
  needed      for every gold partition, whether each of those three levels contains it
  work        edges inspected, scope, wall time
  selection   SF -> exact P50 under F6 with that substrate SUBSTITUTED for the frozen structural
              channel (diagnostic), and the candidate oracle over the resulting F6 pool

STEP 10 keeps the frozen channels untouched and adds the kNN ranking as a FOURTH symmetric RRF
channel, in the two parameter-free forms that separate scoring from admission:
  K_CHAN_VOTE  the kNN channel votes on the FROZEN candidate list only (no new candidates)
  K_CHAN_FULL  the kNN channel also contributes challengers

STEP 7 path provenance is taken from T2_FULL_UNION, the only substrate where all families compete:
every scored edge carries the family sequence of the path that produced it (`parent_tid` chain).

  python scratchpad/_l1kn_run.py <ds> [stride]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1sr_eval as EV
import _l1ss_core as SS
import _l1bm_core as BM
import _l1bc_ledger as LG
import _l1kb_core as KB
import _l1kn_sub as SUB

KND = SUB.KND
P, B, K0, MSTR = EV.P, EV.B, EV.K0, EV.M_STRUCT
SUBS = SUB.SUBS
LEVELS = ["VIS", "ADD", "READ"]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def _sf(sn, sh, sd, sc, hard):
    return EV.sf_from_cache({"s_node": sn[None], "s_hop": sh[None], "s_sdir": sd[None],
                             "s_cnt": sc[None]}, 0, hard, "P0_S4", MSTR)


def _f6(bnd, chal, spos, rpos, cpos, kpos=None):
    """the frozen selector, optionally with a FOURTH symmetric RRF channel.  No weight, no gate."""
    cands = bnd + [p for p in dict.fromkeys(chal) if p not in bnd]
    out = []
    for p in cands:
        s = 1.0 / (K0 + cpos[p]) if p in cpos else 0.0
        if p in spos:
            s += 1.0 / (K0 + spos[p])
        if p in rpos:
            s += 1.0 / (K0 + rpos[p])
        if kpos is not None and p in kpos:
            s += 1.0 / (K0 + kpos[p])
        out.append((-s, cpos.get(p, 10 ** 6), p))
    out.sort()
    return [p for _, _, p in out[:B]], cands


def _oracle(prot_set, cands, REQ):
    """could ANY B-subset of this candidate pool cover the query?  (O2 in the backward ledger)"""
    need = REQ - prot_set
    return int(len(need) <= B and need <= set(cands))


def run(ds, stride=1, log=log):
    S = EV.substrate(ds)
    z = S["z"]
    nq = S["nq"]
    hard = np.asarray(z["hard"], np.int64)
    ctxs, goldp = S["ctxs"], S["goldp"]
    hops = np.asarray(S["hops"]) if S.get("hops") is not None else np.zeros(nq, np.int32)
    import _l1bm_run as RUN
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    npart = int(hard.max()) + 1
    TOPO = {t: SUB.substrate(ds, t, log)[:3] for t in SUBS}
    LAB = SUB.Lab(ds, log)
    qs = list(range(0, nq, stride))
    nQ = len(qs)

    # ---------------- per-query x substrate arrays
    A = {t: {k: np.zeros(nQ, np.float64) for k in
             ["vis_nodes", "vis_parts", "add_parts", "read_parts", "edges", "scope", "ms",
              "p50", "orac", "npool"]} for t in SUBS}
    for t in SUBS:
        for L in LEVELS:
            A[t]["need_" + L] = np.zeros(nQ, np.float64)      # needed partitions reached at level L
    need_tot = np.zeros(nQ, np.float64)
    need_missed = np.zeros(nQ, np.float64)                    # needed & NOT in frozen final50
    base = np.zeros(nQ, np.int8)
    hq = np.zeros(nQ, np.int32)
    # STEP 3 cross-classification of MISSED needed partitions (VIS level, and READ level)
    CLS = {L: {k: 0 for k in ["STRUCT_ONLY", "KNN_ONLY", "BOTH", "NEITHER", "NERX_ONLY",
                              "KNN_OR_NERX_NOT_STRUCT"]} for L in LEVELS}
    CLSH = {}                                                  # same, per metaqa hop
    # STEP 4 redundancy of KNN-derived partitions
    RED = {k: 0 for k in ["knn_parts", "in_dense", "in_splade", "in_canon200", "in_RF", "in_CONT",
                          "in_struct_T0", "novel", "novel_needed", "dup_needed", "needed"]}
    # STEP 7 path provenance (T2)
    PATH = {}
    # STEP 8 knn evidence channel audit
    K8 = {"needed": [], "nuis": []}
    # STEP 10
    TEN = {v: np.zeros(nQ, np.int8) for v in ["K_CHAN_VOTE", "K_CHAN_FULL"]}
    # STEP 2 parity: T0 must reproduce the frozen structural channel and the frozen P50 exactly
    PAR = {"spos_exact": 0, "p50_exact": 0, "seeds": 0, "hops_reached": np.zeros(4)}

    for j, qi in enumerate(qs):
        c = ctxs[qi]
        sd0 = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd0, Xn)
        REQ = set(int(p) for p in goldp[qi])
        hq[j] = int(hops[qi])
        need_tot[j] = len(REQ)
        Xsel, cands0, sc0 = LG.safe_pick(c)
        fin = c["prot_set"] | set(int(p) for p in Xsel)
        base[j] = int(REQ <= fin)
        MISS = REQ - fin
        need_missed[j] = len(MISS)
        cpos, rpos = c["cpos"], c["rpos"]
        RF = [int(p) for p in c["_RF"]]
        b50, bnd = c["base50"], c["bnd"]

        R = {}
        for t in SUBS:
            ap, ai, dg = TOPO[t]
            t0 = time.perf_counter()
            added, vmeta, st, FE = SS.expand_feat(sd0, rq, ap, ai, dg, Xn, W, want_future=False,
                                                  want_visited=True, want_edges=(t == "T2_FULL_UNION"))
            el = time.perf_counter() - t0
            vis = np.concatenate([np.asarray(sd0, np.int64), FE["node"], FE["VIS_PRUNED"]])
            vp = np.unique(hard[vis]); vp = vp[vp >= 0]
            adp = np.unique(hard[FE["node"]]) if len(FE["node"]) else np.zeros(0, np.int64)
            adp = adp[adp >= 0]
            sn, sh, sdd, scn = BM.arrays_of(added, vmeta)
            SF = _sf(sn, sh, sdd, scn, hard)
            spos = {p: r for r, p in enumerate(SF)}
            chal = [p for p in SF if p not in b50] + [p for p in RF if p not in b50]
            pick, pool = _f6(bnd, chal, spos, rpos, cpos)
            fs = c["prot_set"] | set(pick)
            assert len(fs) == P
            a = A[t]
            a["vis_nodes"][j] = len(np.unique(vis)); a["vis_parts"][j] = len(vp)
            a["add_parts"][j] = len(adp); a["read_parts"][j] = len(SF)
            a["edges"][j] = st["edges"]; a["scope"][j] = st["scope"]; a["ms"][j] = 1000 * el
            a["p50"][j] = int(REQ <= fs); a["orac"][j] = _oracle(c["prot_set"], pool, REQ)
            a["npool"][j] = len(pool)
            SV, SA, SR = set(int(x) for x in vp), set(int(x) for x in adp), set(spos)
            a["need_VIS"][j] = len(REQ & SV); a["need_ADD"][j] = len(REQ & SA)
            a["need_READ"][j] = len(REQ & SR)
            R[t] = {"VIS": SV, "ADD": SA, "READ": SR, "spos": spos, "EDGES": FE.get("EDGES")}
            if t == "T0_FROZEN":
                PAR["spos_exact"] += int(spos == {int(k2): int(v2) for k2, v2 in c["spos"].items()})
                PAR["p50_exact"] += int(a["p50"][j] == base[j])
                PAR["seeds"] += len(sd0)
                PAR["hops_reached"][min(len(st["cands"]), 3)] += 1

        # ---------------- STEP 3: unique reach over the MISSED needed partitions
        for L in LEVELS:
            s0, s1, s4 = R["T0_FROZEN"][L], R["T1_KNN_ONLY"][L], R["T4_NERX_ONLY"][L]
            for p in MISS:
                a, b_, d = p in s0, p in s1, p in s4
                k = ("BOTH" if (a and b_) else "STRUCT_ONLY" if a else "KNN_ONLY" if b_
                     else "NEITHER")
                CLS[L][k] += 1
                if d and not a:
                    CLS[L]["NERX_ONLY"] += 1
                if (b_ or d) and not a:
                    CLS[L]["KNN_OR_NERX_NOT_STRUCT"] += 1
            if ds == "metaqa":
                h = CLSH.setdefault(f"hop{int(hops[qi])}", {x: {y: 0 for y in CLS[x]}
                                                            for x in LEVELS})
                for p in MISS:
                    a, b_, d = p in s0, p in s1, p in s4
                    k = ("BOTH" if (a and b_) else "STRUCT_ONLY" if a else "KNN_ONLY" if b_
                         else "NEITHER")
                    h[L][k] += 1
                    if d and not a:
                        h[L]["NERX_ONLY"] += 1
                    if (b_ or d) and not a:
                        h[L]["KNN_OR_NERX_NOT_STRUCT"] += 1

        # ---------------- STEP 4: is a KNN-derived partition already known to another channel?
        dparts = set(int(hard[int(x)]) for x in z["ret_dense"][qi] if int(x) >= 0
                     and int(hard[int(x)]) >= 0)
        sparts = set(int(hard[int(x)]) for x in z["ret_splade"][qi] if int(x) >= 0
                     and int(hard[int(x)]) >= 0)
        can200 = set(int(p) for p in z["base_rank"][qi][:200])
        rfs, cts = set(RF), set(int(p) for p in c["_CONT"])
        for p in R["T1_KNN_ONLY"]["VIS"]:
            RED["knn_parts"] += 1
            i_d, i_s = p in dparts, p in sparts
            i_c, i_r, i_t = p in can200, p in rfs, p in R["T0_FROZEN"]["VIS"]
            RED["in_dense"] += i_d; RED["in_splade"] += i_s; RED["in_canon200"] += i_c
            RED["in_RF"] += i_r; RED["in_CONT"] += (p in cts); RED["in_struct_T0"] += i_t
            nov = not (i_d or i_s or i_c or i_r or i_t)
            RED["novel"] += nov
            if p in REQ:
                RED["needed"] += 1
                RED["novel_needed" if nov else "dup_needed"] += 1

        # ---------------- STEP 7: family sequence of every path T2 actually explored
        E = R["T2_FULL_UNION"]["EDGES"]
        if E is not None and len(E["u"]):
            fam, okf = LAB.of(E["u"], E["v"])
            ptid = E["parent_tid"]
            tp = hard[E["v"]]
            seq1 = fam
            par1 = np.where(ptid >= 0, fam[np.clip(ptid, 0, len(fam) - 1)], 0)
            gpt = np.where(ptid >= 0, ptid[np.clip(ptid, 0, len(ptid) - 1)], -1)
            par2 = np.where(gpt >= 0, fam[np.clip(gpt, 0, len(fam) - 1)], 0)
            # vectorised: the sequence is a 9-bit code (par2, par1, self), so bincount does the
            # counting and one np.unique over code*npart+partition does the distinct-partition sets.
            # Same aggregates as the per-edge loop, but it survives a 400k-edge budget.
            code = (par2.astype(np.int64) << 6) | (par1.astype(np.int64) << 3) | seq1.astype(np.int64)
            okt = tp >= 0
            ndm = okt & np.isin(tp, np.fromiter(REQ, np.int64, len(REQ))) if REQ else \
                np.zeros(len(tp), bool)
            n9 = 512
            PC = np.bincount(code, minlength=n9)
            TC = np.bincount(code[okt], minlength=n9)
            NC = np.bincount(code[ndm], minlength=n9)
            kk = np.unique(code[okt] * np.int64(npart) + tp[okt]) if okt.any() else \
                np.zeros(0, np.int64)
            nk = np.unique(code[ndm] * np.int64(npart) + tp[ndm]) if ndm.any() else \
                np.zeros(0, np.int64)
            for cx in np.flatnonzero(PC):
                d = PATH.setdefault(int(cx), {"paths": 0, "needed_targets": 0, "targets": 0,
                                              "needed_parts": set(), "parts": set()})
                d["paths"] += int(PC[cx]); d["targets"] += int(TC[cx])
                d["needed_targets"] += int(NC[cx])
            for k2 in kk:
                PATH[int(k2 // npart)]["parts"].add(int(k2 % npart))
            for k2 in nk:
                PATH[int(k2 // npart)]["needed_parts"].add(int(k2 % npart))

        # ---------------- STEP 8: the kNN evidence channel, as its own family
        kp = R["T1_KNN_ONLY"]["spos"]
        for p, r in kp.items():
            (K8["needed"] if p in REQ else K8["nuis"]).append(r)

        # ---------------- STEP 10: kNN as a FOURTH channel on the FROZEN candidate list
        sposF = c["spos"]
        chalF = [p for p in sorted(sposF, key=lambda x: sposF[x]) if p not in b50] + \
                [p for p in RF if p not in b50]
        for v in ("K_CHAN_VOTE", "K_CHAN_FULL"):
            ch = chalF if v == "K_CHAN_VOTE" else \
                chalF + [p for p in sorted(kp, key=lambda x: kp[x]) if p not in b50]
            pk, _ = _f6(bnd, ch, sposF, rpos, cpos, kpos=kp)
            f2 = c["prot_set"] | set(pk)
            assert len(f2) == P
            TEN[v][j] = int(REQ <= f2)

        if (j + 1) % 200 == 0:
            log(f"   {ds} {j+1}/{nQ}")

    nmf = SUB.FNAME
    PATH2 = {}
    for k, d in PATH.items():
        key = (nmf[(k >> 6) & 7] + ">" + nmf[(k >> 3) & 7] + ">" + nmf[k & 7]).replace("NONE>", "")
        d["needed_parts"] = len(d["needed_parts"]); d["parts"] = len(d["parts"])
        PATH2[key] = d
    PATH = PATH2
    out = {"ds": ds, "nq": nQ, "npart": npart, "stride": stride,
           "A": {t: {k: v.tolist() for k, v in A[t].items()} for t in SUBS},
           "base": base.tolist(), "hq": hq.tolist(), "need_tot": need_tot.tolist(),
           "need_missed": need_missed.tolist(), "CLS": CLS, "CLSH": CLSH, "RED": RED,
           "PATH": PATH, "TEN": {k: v.tolist() for k, v in TEN.items()},
           "K8": {k: [int(x) for x in v] for k, v in K8.items()},
           "PAR": {"spos_exact": PAR["spos_exact"], "p50_exact": PAR["p50_exact"],
                   "nq": nQ, "seeds_per_q": round(PAR["seeds"] / max(nQ, 1), 3),
                   "hops_reached": PAR["hops_reached"].tolist(),
                   "T0_PARITY": ("EXACT" if PAR["spos_exact"] == nQ and PAR["p50_exact"] == nQ
                                 else "MISMATCH")}}
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    os.makedirs(f"{KND}/diag", exist_ok=True)
    R = run(ds, stride)
    fp = f"{SUB.CACHE}/run_{ds}.json"
    json.dump(R, open(fp, "w"))
    log("wrote", fp, f"nq={R['nq']}")
    for t in SUBS:
        a = R["A"][t]
        log(f"  {t:20s} edges/q {np.mean(a['edges']):9,.0f}  visP {np.mean(a['vis_parts']):8.1f}"
            f"  readP {np.mean(a['read_parts']):6.1f}  p50 {np.mean(a['p50']):.4f}"
            f"  orac {np.mean(a['orac']):.4f}  {np.mean(a['ms']):7.1f} ms")
    log("  PARITY " + json.dumps(R["PAR"]))
    log("  CLS[VIS] " + json.dumps(R["CLS"]["VIS"]))
    log("  CLS[READ] " + json.dumps(R["CLS"]["READ"]))
