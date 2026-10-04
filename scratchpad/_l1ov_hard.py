"""PHASES 1-5 -- hard-partition utility and the universal hard-partition gate.

Takes the assignments the Modal CPU partitioners produced, replays the FROZEN L1 contract on
each, and asks the only question that matters for promotion: does any single recipe improve
retrieval on some corpus without significantly regressing any other?

Balance and cut are reported, but selection is on retrieval utility, exactly as the program says.

  python scratchpad/_l1ov_hard.py pull                  # fetch assignments off the volume
  python scratchpad/_l1ov_hard.py score <ds> [tag ...]  # replay the frozen contract
  python scratchpad/_l1ov_hard.py gate                  # the universal gate + P4 table
"""
import os, sys, json, math, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ep_pu as PU
import _l1ep_c as EC
import _l1kn_sub as KS

OUT, log = OV.OUT, OV.log
PARTS = "scratchpad/_l1ep/parts"
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "webqsp", "hotpotqa_clean"]
# METIS reseed noise floors measured in the previous program: the same graph and the same
# algorithm at a different seed moves F6_ALL by this much, so nothing smaller is a real effect.
FLOOR = {"metaqa": 0.0021, "2wiki_clean": 0.0025, "musique_clean": 0.0029,
         "squad_clean": 0.0013, "hotpotqa_clean": 0.0025, "webqsp": 0.0077}


def mcnemar(a, b):
    """exact two-sided McNemar on paired per-query indicators."""
    a = np.asarray(a, np.int8)
    b = np.asarray(b, np.int8)
    gained = int(((b == 1) & (a == 0)).sum())
    lost = int(((b == 0) & (a == 1)).sum())
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0
    k = min(gained, lost)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return gained, lost, p


def _env():
    return dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "swathihrao28"),
                PYTHONUTF8="1", MSYS_NO_PATHCONV="1")


def pull():
    os.makedirs(PARTS, exist_ok=True)
    os.makedirs(f"{OUT}/hard_partitions", exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "crag-partition", "parts"],
                       capture_output=True, text=True, env=_env())
    names = sorted({w.strip("| ") for line in r.stdout.splitlines() for w in line.split()
                    if w.endswith(".npy")})
    log(f"volume has {len(names)} assignments")
    got = []
    for base in names:
        stem = os.path.basename(base)[:-4]
        parts = stem.split("__")
        if len(parts) != 3:
            log(f"  skip unparseable {stem}")
            continue
        ds, graph, method = parts
        tag = f"{method}__{graph}"
        dst = f"{PARTS}/{ds}__{tag}.npy"
        if os.path.exists(dst):
            got.append([ds, tag])
            continue
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "crag-partition",
                        f"parts/{os.path.basename(base)}", dst, "--force"],
                       capture_output=True, text=True, env=_env())
        if os.path.exists(dst):
            got.append([ds, tag])
            log(f"  pulled {ds} {tag}")
        else:
            log(f"  FAILED {stem}")
    json.dump(sorted(got), open(f"{OUT}/hard_partitions/PULLED.json", "w"), indent=1)
    return got


def cut_stats(ds, hard):
    N, S, K, X = KS.keysets(ds)
    C = np.unique(np.concatenate([S, X, K]))
    u = (C // np.int64(N)).astype(np.int64)
    v = (C % np.int64(N)).astype(np.int64)
    cross = hard[u] != hard[v]
    bnd = np.zeros(len(hard), bool)
    bnd[u[cross]] = True
    bnd[v[cross]] = True
    return {"edges": int(len(C)), "edge_cut": int(cross.sum()),
            "edge_cut_fraction": round(float(cross.mean()), 4),
            "boundary_fraction": round(float(bnd.mean()), 4)}


def score(ds, tags):
    os.makedirs(f"{OUT}/hard_partitions", exist_ok=True)
    fp = f"{OUT}/hard_partitions/REPLAY_{ds}.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for tag in tags:
        if tag in rec:
            log(f"  {ds} {tag} cached")
            continue
        if tag == "CURRENT":
            hard, npart = PU.load_assignment(ds, "CURRENT")
        else:
            p = f"{PARTS}/{ds}__{tag}.npy"
            if not os.path.exists(p):
                log(f"  {ds} {tag} MISSING")
                continue
            hard = np.load(p)
            npart = int(np.asarray(hard).max()) + 1
        hard = np.asarray(hard, np.int64)
        sizes = np.bincount(hard, minlength=npart).astype(np.int64)
        t0 = time.time()
        R = EC.replay(ds, hard, npart, tag, log)
        R["BALANCE"] = {"npart": int(npart), "blocks_used": int((sizes > 0).sum()),
                        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                        "size_mean": round(float(sizes.mean()), 2),
                        "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4),
                        "max_over_mean": round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4)}
        R["CUT"] = cut_stats(ds, hard)
        R["seconds"] = round(time.time() - t0, 1)
        rec[tag] = R
        json.dump(rec, open(fp, "w"), indent=1)
        log(f"  {ds:16s} {tag:32s} F6 {R['F6_ALL_P50']:.4f}  BASE {R['BASE_ALL_P50']:.4f}  "
            f"cut {R['CUT']['edge_cut_fraction']:.4f}  bnd {R['CUT']['boundary_fraction']:.4f}  "
            f"max/mean {R['BALANCE']['max_over_mean']:.3f}")
    return rec


def gate():
    tab, cells = {}, {}
    have = [d for d in DS if os.path.exists(f"{OUT}/hard_partitions/REPLAY_{d}.json")]
    for ds in have:
        rec = json.load(open(f"{OUT}/hard_partitions/REPLAY_{ds}.json"))
        base = rec.get("CURRENT")
        if not base:
            continue
        for tag, R in rec.items():
            if tag == "CURRENT":
                continue
            gained, lost, p = mcnemar(base["_ind_F6"], R["_ind_F6"])
            d = round(R["F6_ALL_P50"] - base["F6_ALL_P50"], 4)
            fl = FLOOR.get(ds, 0.003)
            row = {"delta": d, "F6": R["F6_ALL_P50"], "base_F6": base["F6_ALL_P50"],
                   "gained": gained, "lost": lost, "p": round(p, 6),
                   "sig": bool(p < 0.05 and abs(d) > fl),
                   "noise_floor": fl, "sd_units": round(abs(d) / fl, 1),
                   "max_over_mean": R["BALANCE"]["max_over_mean"],
                   "edge_cut_fraction": R["CUT"]["edge_cut_fraction"],
                   "boundary_fraction": R["CUT"]["boundary_fraction"]}
            if "by_hop" in R:
                row["by_hop"] = {h: v["F6_ALL"] for h, v in R["by_hop"].items()}
                if "by_hop" in base:
                    row["by_hop_delta"] = {h: round(v["F6_ALL"] - base["by_hop"][h]["F6_ALL"], 4)
                                           for h, v in R["by_hop"].items()}
            cells.setdefault(tag, {})[ds] = row
    for tag, per in cells.items():
        ds_list = sorted(per)
        dl = [per[d]["delta"] for d in ds_list]
        regs = [d for d in ds_list if per[d]["sig"] and per[d]["delta"] < 0]
        tab[tag] = {"corpora": ds_list, "n_corpora": len(ds_list),
                    "WORST_DELTA": round(min(dl), 4), "MACRO_DELTA": round(float(np.mean(dl)), 4),
                    "SIG_REGRESSIONS": len(regs), "regressed_on": regs,
                    "n_sig_gains": sum(1 for d in ds_list
                                       if per[d]["sig"] and per[d]["delta"] > 0),
                    "METAQA_H3": (per.get("metaqa", {}).get("by_hop") or {}).get("hop3"),
                    "per_corpus": {d: per[d]["delta"] for d in ds_list}}
    order = sorted(tab, key=lambda t: (tab[t]["SIG_REGRESSIONS"], -tab[t]["WORST_DELTA"],
                                       -tab[t]["MACRO_DELTA"]))
    passed = [t for t in order if tab[t]["SIG_REGRESSIONS"] == 0
              and tab[t]["n_sig_gains"] > 0 and tab[t]["n_corpora"] == len(have)]
    res = {"CORPORA": have, "CELLS": cells, "SUMMARY": tab, "ORDER": order,
           "UNIVERSAL_HARD_GATE_PASSED": passed, "HARD_CROSS_CORPUS_SAFE": bool(passed)}
    json.dump(res, open(f"{OUT}/hard_partitions/HARD_GATE.json", "w"), indent=1)
    hdr = (f"{'PARTITIONER x GRAPH':36s} {'WORST':>8s} {'MACRO':>8s} {'SIGREG':>6s} "
           f"{'GAINS':>5s} {'H3':>7s} {'n':>3s}")
    print(f"corpora scored: {have}\n")
    print(hdr)
    print("-" * len(hdr))
    for t in order:
        v = tab[t]
        print(f"{t:36s} {v['WORST_DELTA']:+8.4f} {v['MACRO_DELTA']:+8.4f} "
              f"{v['SIG_REGRESSIONS']:6d} {v['n_sig_gains']:5d} "
              f"{(v['METAQA_H3'] or 0):7.4f} {v['n_corpora']:3d}")
    print(f"\nUNIVERSAL_HARD_GATE_PASSED = {passed}")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if cmd == "pull":
        pull()
    elif cmd == "scoreall":
        import collections
        byds = collections.defaultdict(list)
        for f in sorted(os.listdir(PARTS)):
            # only this program's P1 matrix (H0..H4); the PARTS dir also holds the previous
            # program's P0/P2/P3/P4/PM cells, which are not part of this search.
            if f.endswith(".npy") and "__H" in f:
                d, t = f[:-4].split("__", 1)
                byds[d].append(t)
        for d in DS:
            if d in byds:
                score(d, ["CURRENT"] + sorted(byds[d]))
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3:])
    else:
        gate()
