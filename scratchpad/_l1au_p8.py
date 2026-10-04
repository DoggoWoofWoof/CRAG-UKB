"""PHASE 8 -- current H4 partition family attribution, WITHOUT repartitioning.  Diagnostic, not
causal (the spec's own framing): single-hop direct-edge explainability is a lower bound on what a
global hypergraph-partitioning objective can exploit, not a claim that only directly-edged pairs
benefit from co-location.

Part A -- per-core internal/cut pin counts by family (STRUCT/NERX/KNN, same residualized definition
KS.keysets uses everywhere else in this audit -- KNN=A\\STRUCT, NERX=NER\\STRUCT).  "Successful" core
= selected AND contains >=1 required node for at least one query; success-weighted STRUCT-share is
compared against the unweighted (all-core) STRUCT-share to test "do successful cores skew STRUCT?"

Part B -- for every query, every pair of REQUIRED nodes co-located in the SAME selected core is
tested for a DIRECT edge in each family (key = min(x,y)*N + max(x,y), exactly KS.keysets' own
convention).  A pair with no direct edge in any family is still co-located (NONE) -- that is the
partitioner's global objective doing work no single-hop check can see, reported as its own bucket,
not folded into any family's credit.

  python scratchpad/_l1au_p8.py run <ds>
  python scratchpad/_l1au_p8.py run_all
  python scratchpad/_l1au_p8.py report
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS
import _l1hu_hard as HH
import _l1hu_global as GL
import _l1hu_s as S

OUT = HH.OUT
AOUT = OUT + "/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = HH.DS
BIT = {"STRUCT": 1, "NERX": 2, "KNN": 4}
LABELMAP = {0: "NONE", 1: "STRUCT", 2: "NERX", 3: "STRUCT+NERX", 4: "KNN", 5: "STRUCT+KNN",
           6: "NERX+KNN", 7: "STRUCT+NERX+KNN"}


def core_family_stats(ds, hard, npart, N, log=log):
    _, S_, K_, X_ = KS.keysets(ds, log)
    stats = {}
    struct_share_num = np.zeros(npart)
    total_internal = np.zeros(npart)
    for fam, keys in (("STRUCT", S_), ("NERX", X_), ("KNN", K_)):
        u = (keys // np.int64(N)).astype(np.int64)
        v = (keys % np.int64(N)).astype(np.int64)
        hu, hv = hard[u], hard[v]
        internal = hu == hv
        internal_cnt = np.bincount(hu[internal], minlength=npart).astype(np.int64)
        cut = ~internal
        cut_cnt = np.bincount(np.concatenate([hu[cut], hv[cut]]), minlength=npart).astype(np.int64)
        stats[fam] = {"internal_per_core": internal_cnt, "cut_per_core": cut_cnt,
                     "total_internal": int(internal.sum()), "total_cut": int(cut.sum())}
        total_internal += internal_cnt
        if fam == "STRUCT":
            struct_share_num = internal_cnt.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        struct_share = np.where(total_internal > 0, struct_share_num / np.maximum(total_internal, 1), np.nan)
    return stats, struct_share, total_internal


def run(ds, log=log):
    hard, npart, N = GL.load_core(ds)
    E = S._setup(ds, log)
    need, nq = E["need"], E["nq"]
    SEL = E["f650"]

    stats, struct_share, total_internal = core_family_stats(ds, hard, npart, N, log)
    has_any = total_internal > 0
    unweighted_mean_struct_share = float(np.nanmean(struct_share[has_any])) if has_any.any() else None

    success = np.zeros(npart, np.int64)
    for qi in range(nq):
        nd_set = set(need[qi])
        if not nd_set:
            continue
        for j in SEL[qi]:
            success[j] += 1
    core_has_req = np.zeros(npart, np.int64)
    for qi in range(nq):
        nd = need[qi]
        if not nd:
            continue
        Sl = SEL[qi]; Ss = set(Sl)
        cores_with_req = {int(hard[x]) for x in nd if int(hard[x]) in Ss}
        for j in cores_with_req:
            core_has_req[j] += 1
    total_success_weight = int(core_has_req.sum())
    if total_success_weight > 0 and has_any.any():
        w = core_has_req.astype(np.float64)
        valid = has_any & (w > 0)
        success_weighted_struct_share = float(np.sum(w[valid] * struct_share[valid]) / np.sum(w[valid]))
    else:
        success_weighted_struct_share = None
    log("  %s: Part A done (%d/%d cores with any internal edge; unweighted struct-share=%s "
        "success-weighted=%s)" % (ds, int(has_any.sum()), npart, unweighted_mean_struct_share,
                                  success_weighted_struct_share))

    _, S_, K_, X_ = KS.keysets(ds, log)
    fam_arrays = {"STRUCT": S_, "NERX": X_, "KNN": K_}
    xs, ys = [], []
    for qi in range(nq):
        nd = need[qi]
        if len(nd) < 2:
            continue
        Sl = SEL[qi]; Ss = set(Sl)
        req_by_core = {}
        for x in nd:
            hx = int(hard[x])
            if hx in Ss:
                req_by_core.setdefault(hx, []).append(x)
        for core_j, members in req_by_core.items():
            if len(members) < 2:
                continue
            m = sorted(set(members))
            for i in range(len(m)):
                for j2 in range(i + 1, len(m)):
                    xs.append(m[i]); ys.append(m[j2])
    n_pairs = len(xs)
    counts = {lab: 0 for lab in LABELMAP.values()}
    if n_pairs:
        xs = np.array(xs, np.int64); ys = np.array(ys, np.int64)
        keys = xs * np.int64(N) + ys
        sig = np.zeros(n_pairs, np.int64)
        for fam, keyarr in fam_arrays.items():
            idx = np.searchsorted(keyarr, keys)
            idxc = np.minimum(idx, len(keyarr) - 1)
            hit = (idx < len(keyarr)) & (keyarr[idxc] == keys)
            sig |= np.where(hit, BIT[fam], 0)
        for code, lab in LABELMAP.items():
            counts[lab] = int((sig == code).sum())
    log("  %s: Part B done, %d co-located required pairs" % (ds, n_pairs))

    frac = {lab: (round(counts[lab] / n_pairs, 4) if n_pairs else None) for lab in counts}
    struct_combo_note = ("STRUCT+NERX / STRUCT+KNN / STRUCT+NERX+KNN are structurally GUARANTEED to be "
                         "0 for every corpus, not an empirical finding: NERX=NER\\STRUCT and "
                         "KNN=A\\STRUCT are defined as set-differences against STRUCT everywhere in "
                         "this codebase, so any pair with a STRUCT edge is by construction excluded "
                         "from the NERX/KNN key arrays. The only meaningful buckets are STRUCT, NERX, "
                         "KNN, NERX+KNN, and NONE.")
    with_struct = counts["STRUCT"] + counts["STRUCT+NERX"] + counts["STRUCT+KNN"] + counts["STRUCT+NERX+KNN"]
    with_nerx = counts["NERX"] + counts["STRUCT+NERX"] + counts["NERX+KNN"] + counts["STRUCT+NERX+KNN"]
    with_knn = counts["KNN"] + counts["STRUCT+KNN"] + counts["NERX+KNN"] + counts["STRUCT+NERX+KNN"]

    rec = {"ds": ds, "npart": npart,
          "PART_A": {"n_cores_with_any_internal_edge": int(has_any.sum()), "npart": npart,
                     "unweighted_mean_struct_internal_share": unweighted_mean_struct_share,
                     "success_weighted_struct_internal_share": success_weighted_struct_share,
                     "total_internal_edges": {f: stats[f]["total_internal"] for f in stats},
                     "total_cut_edges": {f: stats[f]["total_cut"] for f in stats}},
          "PART_B": {"n_colocated_required_pairs": n_pairs, "counts": counts, "frac": frac,
                    "NOTE_STRUCT_COMBOS": struct_combo_note,
                    "frac_with_struct_any": round(with_struct / n_pairs, 4) if n_pairs else None,
                    "frac_with_nerx_any": round(with_nerx / n_pairs, 4) if n_pairs else None,
                    "frac_with_knn_any": round(with_knn / n_pairs, 4) if n_pairs else None,
                    "frac_no_direct_edge": frac["NONE"]}}
    os.makedirs("%s/partition_attribution" % AOUT, exist_ok=True)
    fp = "%s/partition_attribution/P8_%s.json" % (AOUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
    return rec


def run_all():
    for ds in DS:
        run(ds)


def report():
    rows = []
    for ds in DS:
        fp = "%s/partition_attribution/P8_%s.json" % (AOUT, ds)
        if os.path.exists(fp):
            rows.append(json.load(open(fp)))
    os.makedirs(AOUT, exist_ok=True)
    json.dump(rows, open("%s/PHASE8_PARTITION_ATTRIBUTION_REPORT.json" % AOUT, "w"), indent=1)
    for r in rows:
        a, b = r["PART_A"], r["PART_B"]
        print("\n== %s ==" % r["ds"])
        print("  PART A: unweighted_struct_share=%s  success_weighted_struct_share=%s"
              % (a["unweighted_mean_struct_internal_share"], a["success_weighted_struct_internal_share"]))
        print("    total_internal: %s   total_cut: %s" % (a["total_internal_edges"], a["total_cut_edges"]))
        print("  PART B: n_pairs=%d  frac_with_struct=%s  frac_with_nerx=%s  frac_with_knn=%s  "
              "frac_no_direct_edge=%s" % (b["n_colocated_required_pairs"], b["frac_with_struct_any"],
                                         b["frac_with_nerx_any"], b["frac_with_knn_any"],
                                         b["frac_no_direct_edge"]))
        print("    signature breakdown: %s" % b["frac"])
    print("\nwrote %s/PHASE8_PARTITION_ATTRIBUTION_REPORT.json" % AOUT)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "run_all":
        run_all()
    elif a and a[0] == "report":
        report()
    else:
        print(__doc__)
