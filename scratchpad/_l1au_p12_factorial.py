"""PHASE 12 -- pure-STRUCT / edge-family partition-BUILD causal factorial (metaqa first).

Directly answers what Phase 8 could only diagnose indirectly (Phase 8's own docstring: "diagnostic,
not causal" -- direct-edge co-location is a lower bound on what the GLOBAL hypergraph objective can
exploit transitively, so a low frac_with_struct_any does NOT prove KNN/NER caused the co-location).
This module actually REPARTITIONS metaqa's H4_SPLIT_PRESERVE hyperedges under 4 different family
sets, with node universe / k / epsilon / seed / split-preserve rule / hyperedge weighting held
bit-identical (only the hyperedge FAMILY SET input to Mt-KaHyPar varies), and scores each resulting
partition through the SAME online serving stack (bit-identical G1 neighbor graph, G3 hyperedge
substrate, SAFE selector, SP1 ranker -- the online edge-family question is ALREADY SETTLED
(STRUCT-only G3 ~= full G3 on 6/6 under SP1) and is deliberately NOT reopened here).

  P0_S    STRUCT only
  P1_SK   STRUCT+KNN   (shipped H4_SPLIT_PRESERVE, the reference cell)
  P2_SN   STRUCT+NER   (canonical/raw NER -- see _l1kn_sub.raw_ner_keys, NOT residualized NERX)
  P3_SKN  STRUCT+KNN+NER

x4 evaluation cells per partition variant (all _ind_ALL arrays are per-dev-query ALL_REQUIRED_FETCHED):
  E0  BASE Dense+SPLADE P50            OVE.evaluate  fam=O0_CORE      lane=BASE
  E1  historical SAFE (pre-SP1)        OVE.evaluate  fam=O4_FULL_C_b0.5  lane=F6
  E2  SP1 + FULL_C halo (shipped)      _l1au_halo.run4  pool=H0_FULL_C     core_lane=SAFE kq=MATCHED
  E3  SP1 + STRUCT-only halo           _l1au_halo.run4  pool=H3_STRUCT_ONLY core_lane=SAFE kq=MATCHED

METAQA GATE is keyed to E2 (the actual shipped online mechanism): FAIL if any of S/SN/SKN shows a
significant (McNemar p<0.05 AND |delta|>HH.FLOOR[ds]) REGRESSION vs SK on E2; otherwise PASS.

  python scratchpad/_l1au_p12_factorial.py stage <ds>
  python scratchpad/_l1au_p12_factorial.py run <ds>
  python scratchpad/_l1au_p12_factorial.py report <ds>
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_eval as OVE
import _l1ep_pu as PU
import _l1hu_hard as HH
import _l1au_halo as AH

OUT = HH.OUT
AOUT = OUT + "/audit"
EPARTS = "scratchpad/_l1ep/parts"
PARTS_SRC = "scratchpad/_l1hu/parts"
MANIFEST_FP = "%s/hypergraph_build/BUILD_MANIFEST.json" % OUT
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)

FAMSETS = ("S", "SK", "SN", "SKN")
RULE = "H4_SPLIT_PRESERVE"
REF_FAM = "SK"


def tag_of(fam):
    return "H4FAM_%s" % fam


def stage(ds, famsets=FAMSETS, log=log):
    """copy _l1hu/parts/{ds}__H4_SPLIT_PRESERVE__{fam}.npy -> _l1ep/parts under a FRESH per-family
    tag, so OVE.evaluate / _l1au_halo.run4 score this variant instead of the shipped C1_HYPER_UNIVERSAL
    one.  Every downstream boundary-mass cache is keyed by this tag (see _l1hu_e.boundary_mass_keys /
    _l1au_halo._setup4's docstrings) so a fresh tag per family is required, not optional."""
    os.makedirs(EPARTS, exist_ok=True)
    out = {}
    for fam in famsets:
        src = "%s/%s__%s__%s.npy" % (PARTS_SRC, ds, RULE, fam)
        tag = tag_of(fam)
        dst = "%s/%s__%s.npy" % (EPARTS, ds, tag)
        hard = np.load(src)
        np.save(dst, hard)
        npart = int(hard.max()) + 1
        out[fam] = {"tag": tag, "npart": npart, "N": len(hard)}
        log("  staged %s -> %s  npart=%d N=%d" % (src, dst, npart, len(hard)))
    return out


def colocation(hard, need_by_q):
    """selection-independent partition-quality check: of every pair of gold-required nodes that
    co-occur in the same query, what fraction land in the same `hard` block at all (no core-
    selection/halo involved) -- a simple, comparable-across-variants clustering-quality signal."""
    xs, ys = [], []
    for nd in need_by_q:
        if len(nd) < 2:
            continue
        m = sorted(set(nd))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                xs.append(m[i]); ys.append(m[j])
    if not xs:
        return {"n_pairs": 0, "frac_colocated": None}
    xs = np.array(xs, np.int64); ys = np.array(ys, np.int64)
    same = hard[xs] == hard[ys]
    return {"n_pairs": len(xs), "frac_colocated": round(float(same.mean()), 4)}


def run_one(ds, fam, log=log):
    tag = tag_of(fam)
    t0 = time.time()
    hard = np.load("%s/%s__%s.npy" % (EPARTS, ds, tag))
    hard = np.asarray(hard, np.int64)
    npart = int(hard.max()) + 1
    N = len(hard)

    R = OVE.evaluate(ds, core_tag=tag, fams=("O0_CORE", "O4_FULL_C"), betas=(0.5,), log=log)
    cov_fp = "%s/overlap/COVERAGE_%s.json" % (OUT, ds)
    os.makedirs("%s/overlap" % OUT, exist_ok=True)
    rec_all = json.load(open(cov_fp)) if os.path.exists(cov_fp) else {}
    rec_all.setdefault(tag, {}).update(R["CELLS"])
    rec_all[tag]["_meta"] = {k: R[k] for k in ("ds", "N", "npart", "nq")}
    json.dump(rec_all, open(cov_fp, "w"), indent=1)
    e0 = R["CELLS"]["O0_CORE"]["BASE"]
    e1 = R["CELLS"]["O4_FULL_C_b0.5"]["F6"]

    e2 = AH.run4(ds, "SAFE", "H0_FULL_C", "MATCHED", hard=hard, npart=npart, tag=tag, log=log)
    e3 = AH.run4(ds, "SAFE", "H3_STRUCT_ONLY", "MATCHED", hard=hard, npart=npart, tag=tag, log=log)

    g, gptr, rows, hops = PU.gold_rows(ds)
    need = [sorted({int(x) for x in g[gptr[qi]:gptr[qi + 1]]}) for qi in range(gptr.shape[0] - 1)]
    coloc = colocation(hard, need)

    manifest = json.load(open(MANIFEST_FP))
    build_stats = manifest.get("%s__%s__%s" % (ds, RULE, fam), {})
    stats_fp = "%s/%s__%s__%s.stats.json" % (PARTS_SRC, ds, RULE, fam)
    part_stats = json.load(open(stats_fp)) if os.path.exists(stats_fp) else {}

    block_sizes = np.bincount(hard, minlength=npart)
    balance = {"target": build_stats.get("target_block_size"),
              "min": int(block_sizes.min()), "mean": round(float(block_sizes.mean()), 2),
              "p90": float(np.percentile(block_sizes, 90)), "max": int(block_sizes.max())}

    def _cell(c):
        return {"ALL_REQUIRED_FETCHED": c["ALL_REQUIRED_FETCHED"], "_ind_ALL": c["_ind_ALL"],
               "hop": {hk: c.get("hop%d_ALL_REQUIRED_FETCHED" % hk) for hk in (1, 2, 3)}}

    rec = {"ds": ds, "fam": fam, "tag": tag, "N": N, "npart": npart,
          "E0_BASE_P50": {"ALL_REQUIRED_FETCHED": e0["ALL_REQUIRED_FETCHED"], "_ind_ALL": e0["_ind_ALL"]},
          "E1_HIST_SAFE": {"ALL_REQUIRED_FETCHED": e1["ALL_REQUIRED_FETCHED"], "_ind_ALL": e1["_ind_ALL"]},
          "E2_SP1_FULL_C": _cell(e2["CELLS"]["SP1"]),
          "E3_SP1_STRUCT_HALO": _cell(e3["CELLS"]["SP1"]),
          "BUILD": {"hyperedges": build_stats.get("hyperedges"), "pins": build_stats.get("pins"),
                    "pin_retention_total": build_stats.get("pin_retention_total"),
                    "weight_sum": build_stats.get("weight_sum")},
          "PARTITION": {"objective_km1": part_stats.get("objective_km1"),
                       "imbalance": part_stats.get("imbalance"),
                       "wall_seconds": part_stats.get("wall_seconds"),
                       "peak_rss_mb": part_stats.get("peak_rss_mb")},
          "BALANCE": balance,
          "COLOCATION": coloc,
          "runtime_s": round(time.time() - t0, 1)}
    os.makedirs("%s/halo_family_factorial" % AOUT, exist_ok=True)
    fp = "%s/halo_family_factorial/P12_CELL_%s__%s.json" % (AOUT, ds, fam)
    json.dump(rec, open(fp, "w"), indent=1)
    log("wrote %s  (%.1fs)" % (fp, time.time() - t0))
    return rec


def run(ds, famsets=FAMSETS, log=log):
    stage(ds, famsets, log)
    for fam in famsets:
        run_one(ds, fam, log)


EFF_LABELS = [("KNN_ON_STRUCT", "SK", "S"), ("NER_ON_STRUCT", "SN", "S"),
             ("NER_ON_SK", "SKN", "SK"), ("KNN_ON_SN", "SKN", "SN")]


def _load(ds, fam):
    fp = "%s/halo_family_factorial/P12_CELL_%s__%s.json" % (AOUT, ds, fam)
    return json.load(open(fp))


def report(ds, famsets=FAMSETS, log=log):
    cells = {fam: _load(ds, fam) for fam in famsets}
    ref = cells[REF_FAM]
    floor = HH.FLOOR.get(ds, 0.0)
    ECELLS = ["E0_BASE_P50", "E1_HIST_SAFE", "E2_SP1_FULL_C", "E3_SP1_STRUCT_HALO"]

    table = {}
    for ek in ECELLS:
        row = {"SK_ref": ref[ek]["ALL_REQUIRED_FETCHED"]}
        for fam in famsets:
            if fam == REF_FAM:
                continue
            v = cells[fam][ek]
            g_, l_, p_ = HH.mcnemar(ref[ek]["_ind_ALL"], v["_ind_ALL"])
            d = round(v["ALL_REQUIRED_FETCHED"] - ref[ek]["ALL_REQUIRED_FETCHED"], 4)
            row[fam] = {"ALL_REQUIRED": v["ALL_REQUIRED_FETCHED"], "delta_vs_SK": d,
                       "gained": g_, "lost": l_, "p": p_,
                       "sig": bool(p_ < 0.05 and abs(d) > floor),
                       "sig_regression": bool(p_ < 0.05 and d < -floor)}
            if "hop" in v:
                row[fam]["hop"] = v["hop"]
        table[ek] = row

    effects = {}
    for ek in ECELLS:
        acc = {fam: cells[fam][ek]["ALL_REQUIRED_FETCHED"] for fam in famsets}
        effects[ek] = {name: round(acc[a] - acc[b], 4) for name, a, b in EFF_LABELS
                      if a in acc and b in acc}
        if all(f in acc for f in ("S", "SK", "SN", "SKN")):
            effects[ek]["FAMILY_INTERACTION"] = round(
                acc["SKN"] - acc["SK"] - acc["SN"] + acc["S"], 4)

    gate_regressions = [fam for fam in famsets if fam != REF_FAM
                        and table["E2_SP1_FULL_C"][fam]["sig_regression"]]
    gate = "FAIL" if gate_regressions else "PASS"

    best_by_ecell = {ek: max(famsets, key=lambda f: cells[f][ek]["ALL_REQUIRED_FETCHED"])
                     for ek in ECELLS}

    rec = {"ds": ds, "REF_FAM": REF_FAM,
          "SUMMARY_ALL_REQUIRED": {fam: {ek: cells[fam][ek]["ALL_REQUIRED_FETCHED"] for ek in ECELLS}
                                   for fam in famsets},
          "MCNEMAR_VS_SK": table,
          "FACTORIAL_EFFECTS": effects,
          "BEST_FAM_PER_ECELL": best_by_ecell,
          "BUILD": {fam: cells[fam]["BUILD"] for fam in famsets},
          "PARTITION": {fam: cells[fam]["PARTITION"] for fam in famsets},
          "BALANCE": {fam: cells[fam]["BALANCE"] for fam in famsets},
          "COLOCATION": {fam: cells[fam]["COLOCATION"] for fam in famsets},
          "METAQA_GATE" if ds == "metaqa" else "GATE": gate,
          "GATE_REGRESSIONS_ON_E2": gate_regressions}
    os.makedirs("%s/halo_family_factorial" % AOUT, exist_ok=True)
    fp = "%s/halo_family_factorial/P12_REPORT_%s.json" % (AOUT, ds)
    json.dump(rec, open(fp, "w"), indent=1)

    print("\n== PHASE 12 FACTORIAL: %s ==" % ds)
    for ek in ECELLS:
        print("  %s  SK=%.4f" % (ek, ref[ek]["ALL_REQUIRED_FETCHED"]))
        for fam in famsets:
            if fam == REF_FAM:
                continue
            v = table[ek][fam]
            print("    %-4s ALL_REQUIRED=%.4f delta=%+.4f gained=%-3d lost=%-3d p=%-10s sig=%-5s regression=%s" %
                 (fam, v["ALL_REQUIRED"], v["delta_vs_SK"], v["gained"], v["lost"], v["p"], v["sig"],
                  v["sig_regression"]))
    print("\n  factorial effects (point deltas on ALL_REQUIRED_FETCHED):")
    for ek in ECELLS:
        print("    %s: %s" % (ek, effects[ek]))
    print("\n  BEST_FAM_PER_ECELL: %s" % best_by_ecell)
    print("  GATE (keyed to E2_SP1_FULL_C): %s   regressions=%s" % (gate, gate_regressions))
    print("\nwrote", fp)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "stage":
        stage(a[1])
    elif a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "report":
        report(a[1])
    else:
        print(__doc__)
