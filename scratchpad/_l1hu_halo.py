"""PHASE B -- does a better hypergraph CORE compose with the bounded FULL-C HALO?

The two mechanisms attack different failures.  The core changes WHERE the ranking puts things;
the halo changes WHAT a selected block pays out, so it repairs required nodes that any hard
boundary splits off.  Phase B measures whether they add, overlap or interfere.

Exactly three cores, per B0:

  C0_METIS                  production, unchanged
  C1_HYPER_UNIVERSAL        the Phase-A representation that passed the universal gate
  C2_HYPER_BEST_DIAGNOSTIC  the strongest Phase-A representation, if it is a different one

The halo itself is untouched from the phase that validated it: FULL_C one-hop, static boundary
mass ranking, |H_j| <= beta * |C_j| with ONE beta for every corpus, beta in {0, 0.25, 0.5}.
Core ranking is never recomputed -- _l1ov_eval reads the selected block ids from the frozen path
and the halo only changes the fetch payload, which B3 re-verifies rather than assumes.

  python scratchpad/_l1hu_halo.py cores <C1_rule> [C2_rule]     # register + copy assignments
  python scratchpad/_l1hu_halo.py eval <ds> <core> [<core> ...]
  python scratchpad/_l1hu_halo.py depth <ds> <core>
  python scratchpad/_l1hu_halo.py report
"""
import os, sys, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ov_eval as OVE
import _l1ov_matched as OVM
import _l1ep_pu as PU
import _l1ep_c as EC
import _l1hu_hard as HH

OUT = HH.OUT
PREV = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
EPARTS = "scratchpad/_l1ep/parts"
BETAS = (0.25, 0.5)
FAMS = ("O0_CORE", "O4_FULL_C")
DEPTHS = (1, 5, 10, 25, 50)
log = HH.log


def register(c1, c2=None):
    """copy the chosen Phase-A assignments under stable core names _l1ov_eval can read."""
    names = {"C1_HYPER_UNIVERSAL": c1}
    if c2 and c2 != c1:
        names["C2_HYPER_BEST_DIAGNOSTIC"] = c2
    got = {}
    for core, rule in names.items():
        for ds in HH.DS:
            src = "%s/%s__%s.npy" % (HH.PARTS, ds, rule)
            if not os.path.exists(src):
                log("  %s %s MISSING (%s)" % (ds, core, rule))
                continue
            shutil.copyfile(src, "%s/%s__%s.npy" % (EPARTS, ds, core))
            got.setdefault(core, []).append(ds)
    rec = {"C0_METIS": "CURRENT", **names, "copied": got,
           "NOTE": "C0_METIS is the production assignment, read through PU.load_assignment"}
    os.makedirs("%s/halo" % OUT, exist_ok=True)
    json.dump(rec, open("%s/halo/CORES.json" % OUT, "w"), indent=1)
    log("registered %s" % json.dumps(names))
    return rec


def coverage_path(ds):
    return "%s/halo/COVERAGE_%s.json" % (OUT, ds)


def evaluate(ds, cores):
    os.makedirs("%s/halo" % OUT, exist_ok=True)
    fp = coverage_path(ds)
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for core in cores:
        tag = "CURRENT" if core == "C0_METIS" else core
        # C0_METIS was already measured under the identical code path in the previous phase;
        # re-running it would burn CPU to reproduce numbers that are on disk.
        prev = "%s/overlap/COVERAGE_%s.json" % (PREV, ds)
        if core == "C0_METIS" and os.path.exists(prev):
            src = json.load(open(prev)).get("CURRENT", {})
            if all(c in src for c in ("O0_CORE", "O4_FULL_C_b0.25", "O4_FULL_C_b0.5")):
                rec[core] = {k: v for k, v in src.items()
                             if k in ("O0_CORE", "O4_FULL_C", "O4_FULL_C_b0.25",
                                      "O4_FULL_C_b0.5", "_meta")}
                rec[core]["_reused_from"] = prev
                json.dump(rec, open(fp, "w"), indent=1)
                log("  %s %s reused from the previous phase" % (ds, core))
                continue
        R = OVE.evaluate(ds, tag, FAMS, betas=BETAS, log=log)
        rec[core] = dict(R["CELLS"])
        rec[core]["_meta"] = {k: R[k] for k in ("ds", "N", "npart", "nq")}
        rec[core]["_meta"]["PARTITION_PARITY"] = R["PARTITION_PARITY"]
        json.dump(rec, open(fp, "w"), indent=1)
    return rec


def b3_gate(ds):
    """B3 -- the halo must not touch core ranking.  Verified, not assumed.

    Two independent checks: the core-only coverage of every beta cell must equal the beta=0 cell
    exactly (same selected blocks, same core payload), and the beta=0 core coverage must equal the
    partition-level ALL@50 that the Phase-A replay computed through a completely separate path.
    """
    rec = json.load(open(coverage_path(ds)))
    rp = "%s/partitions/REPLAY_%s.json" % (OUT, ds)
    rr = json.load(open(rp)) if os.path.exists(rp) else {}
    cores = json.load(open("%s/halo/CORES.json" % OUT))
    out = {}
    for core, cells in rec.items():
        o = {"core_only_identical_across_beta": True, "replay_agrees": None}
        base = cells.get("O0_CORE", {})
        for name, c in cells.items():
            if name.startswith("_") or name == "O0_CORE":
                continue
            for lane in ("BASE", "F6"):
                if c.get(lane, {}).get("ALL_REQUIRED_CORE_ONLY") != \
                        base.get(lane, {}).get("ALL_REQUIRED_CORE_ONLY"):
                    o["core_only_identical_across_beta"] = False
        rtag = "CURRENT" if core == "C0_METIS" else cores.get(core)
        R = rr.get(rtag)
        if R and base:
            o["replay_agrees"] = bool(
                abs(base["F6"]["ALL_REQUIRED_FETCHED"] - R["F6_ALL_P50"]) < 1e-9
                and abs(base["BASE"]["ALL_REQUIRED_FETCHED"] - R["BASE_ALL_P50"]) < 1e-9)
            o["replay"] = {"BASE": R["BASE_ALL_P50"], "SAFE": R["F6_ALL_P50"]}
            o["halo_path_beta0"] = {"BASE": base["BASE"]["ALL_REQUIRED_FETCHED"],
                                    "SAFE": base["F6"]["ALL_REQUIRED_FETCHED"]}
        o["CORE_RANKING_BIT_IDENTICAL"] = bool(o["core_only_identical_across_beta"]
                                               and o["replay_agrees"] is not False)
        out[core] = o
    return out


def depth(ds, core):
    """B5 -- required-node coverage at P1/P5/P10/P25/P50, core-only and core+halo.

    Depth is defined on the BASE block ranking, which is the only ranked object in the frozen
    contract: the selector is set-valued at exactly P=50, so it has no P<50 prefix.
    """
    tag = "CURRENT" if core == "C0_METIS" else core
    hard, npart = PU.load_assignment(ds, "CURRENT") if tag == "CURRENT" else (
        np.load("%s/%s__%s.npy" % (EPARTS, ds, tag)), None)
    hard = np.asarray(hard, np.int64)
    npart = int(npart) if npart is not None else int(hard.max()) + 1
    N = len(hard)
    z0 = np.load("%s/runs/cache_%s.npz" % (EC.ROOT, ds), allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files}
    z.update(EC.rebuild(ds, hard, npart, z, meta, log))
    br = np.asarray(z["base_rank"])
    g, gptr, rows, hops = PU.gold_rows(ds)
    nq = meta["n_dev_queries"]
    sizes = np.bincount(hard, minlength=npart).astype(np.int64)
    need = [sorted({int(x) for x in g[gptr[q]:gptr[q + 1]]}) for q in range(nq)]

    out = {}
    for cell, beta in [("O0_CORE", None)] + [("O4_FULL_C_b%g" % b, b) for b in BETAS]:
        if cell == "O0_CORE":
            nptr, nidx = np.zeros(N + 1, np.int64), np.zeros(0, np.int32)
            bptr, bidx = np.zeros(npart + 1, np.int64), np.zeros(0, np.int32)
        else:
            pk, mass = OV.boundary_mass(ds, hard, tag, "O4_FULL_C", N, npart, log)
            pairs = OV.bounded_pairs(pk, mass, hard, npart, N, beta)
            nptr, nidx = OV.to_node_csr(pairs, npart, N)
            bptr, bidx = OV.to_block_csr(pairs, npart, N)
        per = {}
        for P in DEPTHS:
            allf = np.zeros(nq, np.int8)
            rn = np.zeros(nq)
            expo = np.zeros(nq, np.int64)
            for q in range(nq):
                if not need[q]:
                    continue
                S = br[q, :P]
                Ss = set(int(x) for x in S)
                got = {x for x in need[q] if int(hard[x]) in Ss}
                for x in need[q]:
                    if x in got:
                        continue
                    bs = nidx[nptr[x]:nptr[x + 1]]
                    if len(bs) and Ss.intersection(bs.tolist()):
                        got.add(x)
                allf[q] = int(len(got) == len(need[q]))
                rn[q] = len(got) / len(need[q])
                e = int(sizes[S].sum())
                if len(bidx):
                    u = np.unique(np.concatenate([bidx[bptr[j]:bptr[j + 1]] for j in S]))
                    m = np.zeros(npart, bool)
                    m[S] = True
                    e += int(np.count_nonzero(~m[hard[u.astype(np.int64)]]))
                expo[q] = e
            ok = np.array([bool(n) for n in need])
            per["P%d" % P] = {"ALL_REQUIRED_FETCHED": round(float(allf[ok].mean()), 4),
                              "REQUIRED_NODE_RECALL": round(float(rn[ok].mean()), 4),
                              "UNIQUE_EXPOSURE": round(float(expo[ok].mean()), 1)}
            if ds == "metaqa":
                h = np.asarray(hops)[:nq]
                per["P%d" % P]["by_hop"] = {
                    "hop%d" % k: round(float(allf[(h == k) & ok].mean()), 4)
                    for k in (1, 2, 3) if ((h == k) & ok).any()}
            log("  %s %s %-16s P%-3d ALLREQ %.4f  expo %s"
                % (ds, core, cell, P, per["P%d" % P]["ALL_REQUIRED_FETCHED"],
                   HH.__dict__.get("cm", lambda x: x)(int(per["P%d" % P]["UNIQUE_EXPOSURE"]))))
        out[cell] = per
    fp = "%s/halo/DEPTH_%s.json" % (OUT, ds)
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    rec[core] = out
    json.dump(rec, open(fp, "w"), indent=1)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "cores":
        register(a[1], a[2] if len(a) > 2 else None)
    elif a and a[0] == "eval":
        evaluate(a[1], a[2:])
    elif a and a[0] == "depth":
        depth(a[1], a[2])
    elif a and a[0] == "b3":
        print(json.dumps({d: b3_gate(d) for d in HH.DS
                          if os.path.exists(coverage_path(d))}, indent=1))
    else:
        print(__doc__)
