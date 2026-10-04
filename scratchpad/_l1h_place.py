"""L1 HOST LANE -- placement test (declared in results/L1_HOST/HOST_LANE_DECLARATION.json), run on the laptop from fetched host records.

For metaqa, squad, musique at the native K and every grid K it compares the host's work with the laptop's, cell by cell:
  HG      host hypergraph content digest == the laptop's (grid K: results/L1_COVPART/parts/<ds>__H4_SK_k<K>.json; native K: also the frozen
          data/l1_canonical/<ds>/hypergraph/H4_SK.json), plus cap / hyperedges / pins / anchor duplication
  SHARDS  host PHG shard sha256 list == the laptop run's list (results/L1_COVPART/parts/<ds>__H4_SK_k<K>__PHG_con.RUN.json)
  OUTCOME host STATUS (OK / PARTITION_INVALID) == the laptop's (RUN.json / FAILED.json)
  PHG     host vector == the laptop vector position by position; at the native K also == the served vector; raw KM1 and repair moves compared
Verdict (the declaration's rule): BIT_IDENTICAL when every comparison is equal, else NOT_IDENTICAL (with where and by how much).
Read-only on everything but its own record: results/L1_HOST/PLACEMENT_TEST__v1.json (write-once; nothing under data/ is written).

    python -u scratchpad/_l1h_place.py           print the table; write the record only when every declared cell is present
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HOST = os.path.join(REPO, "results", "L1_HOST")
LAP = os.path.join(REPO, "results", "L1_COVPART", "parts")
OUTP = os.path.join(HOST, "PLACEMENT_TEST__v1.json")
GRID = (100, 250, 500, 1000, 2000, 5000)
NATIVE = {"metaqa": 432, "squad": 202, "musique": 1175}
SERVED = {"metaqa": "LOWMEM__PHG_REPAIR1_con", "squad": "LOWMEM__PHG_con", "musique": "LOWMEM__PHG_C1_con"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rj(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def hg_stats(m):
    return {k: m.get(k) for k in ("content_digest", "cap", "hyperedges", "pins", "anchor_duplication_pins")}


def vec_cmp(a, b):
    """equal position by position; else how far apart (differing positions, equal up to a relabelling of the blocks)."""
    if a.shape != b.shape:
        return {"equal": False, "shape": [list(a.shape), list(b.shape)]}
    if np.array_equal(a, b):
        return {"equal": True}
    pair = np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b.astype(np.int64))
    ka, kb = len(np.unique(a)), len(np.unique(b))
    return {"equal": False, "differing_positions": int((a != b).sum()), "n": int(len(a)),
            "equal_up_to_relabel": bool(len(pair) == ka == kb), "distinct_label_pairs": int(len(pair)), "blocks_host": ka, "blocks_laptop": kb}


def cell(ds, K):
    tag = "H4_SK_k%d" % K
    out = {"dataset": ds, "K": K, "native": K == NATIVE[ds]}
    # ---------------------------------------------------------------- HG
    hh = rj(os.path.join(HOST, "parts", "%s__%s.json" % (ds, tag)))
    lh = rj(os.path.join(LAP, "%s__%s.json" % (ds, tag)))
    if hh is None:
        out["missing"] = "host HG record"
        return out
    out["HG"] = {"host": hg_stats(hh), "laptop": hg_stats(lh) if lh else None,
                 "equal": bool(lh is not None and hg_stats(hh) == hg_stats(lh))}
    if K == NATIVE[ds]:
        fz = rj(os.path.join(REPO, "data", "l1_canonical", ds, "hypergraph", "H4_SK.json"))
        out["HG"]["frozen_H4_SK"] = hg_stats(fz)
        out["HG"]["equal_frozen"] = hg_stats(hh) == hg_stats(fz)
        if lh is None:                                # the native cell may have no scale build: the frozen record is the reference
            out["HG"]["equal"] = out["HG"]["equal_frozen"]
    out["HG"]["host_placement"] = {k: hh["placement"].get(k) for k in ("machine", "python", "numpy", "rx_job")}
    # ---------------------------------------------------------------- PHG
    base = os.path.join(HOST, "parts", "%s__%s__PHG_con" % (ds, tag))
    hr, hf = rj(base + ".RUN.json"), rj(base + ".FAILED.json")
    lbase = os.path.join(LAP, "%s__%s__PHG_con" % (ds, tag))
    lr, lf = rj(lbase + ".RUN.json"), rj(lbase + ".FAILED.json")
    if hr is None and hf is None:
        out["missing"] = "host PHG record"
        return out
    hs = "OK" if hr else hf["STATUS"]
    ls = "OK" if lr else (lf["STATUS"] if lf else None)
    out["OUTCOME"] = {"host": hs, "laptop": ls, "equal": hs == ls}
    h = hr or hf
    run = h.get("run", {})
    out["host_run"] = {"rx_job": h["placement"].get("rx_job"), "wall_seconds": h.get("wall_seconds"), "mpirun_seconds": run.get("wall_seconds_outer"),
                       "peak_rss_mb_per_rank": [round(x / 1024.0, 1) for x in run.get("memory", {}).get("peak_rss_kb_per_rank_time_v", [])],
                       "zoltan_cutl_vs_python_km1_rel": h.get("raw", {}).get("zoltan_cutl_vs_python_km1_rel")}
    lsrc = lr or lf
    if lr is not None:
        hsh = [c["sha256"] for c in h["shards"]]
        lsh = [c["sha256"] for c in lr["shards"]]
        out["SHARDS"] = {"host": len(hsh), "laptop": len(lsh), "equal": hsh == lsh}
    if hr and lr:
        out["RAW"] = {"km1_host": hr["raw"]["km1"], "km1_laptop": lr["raw"]["km1"], "km1_equal": hr["raw"]["km1"] == lr["raw"]["km1"],
                      "max_block_host": hr["raw"]["validity"]["max_block"], "max_block_laptop": lr["raw"]["validity"]["max_block"],
                      "repair_moves_host": (hr.get("repair") or {}).get("moves", 0), "repair_moves_laptop": (lr.get("repair") or {}).get("moves", 0)}
        hv = np.load(base + ".npy").astype(np.int64)
        assert sha_file(base + ".npy") == hr["output"]["sha256"]
        lv = np.load(lbase + ".npy").astype(np.int64)
        assert sha_file(lbase + ".npy") == lr["output"]["sha256"]
        out["PHG"] = dict(vec_cmp(hv, lv), host_sha256=hr["output"]["sha256"], laptop_sha256=lr["output"]["sha256"])
        if K == NATIVE[ds]:
            sp = os.path.join(REPO, "data", "l1_canonical", ds, "parts", SERVED[ds] + ".npy")
            out["PHG"]["served"] = {"partition": SERVED[ds], "sha256": sha_file(sp)}
            out["PHG"]["served"].update(vec_cmp(hv, np.load(sp).astype(np.int64)))
    elif hf and lf:
        g = hf.get("gate", {})
        lg = lf.get("gate")
        lg = lg if isinstance(lg, dict) else None
        out["RAW"] = {"host_gate": {k: g.get(k) for k in ("max_block", "contract_bound_ceil_1.03_N_over_k", "gate")}, "laptop_gate_record": lsrc["STATUS"]}
    ok = [out["HG"]["equal"], out["OUTCOME"]["equal"]]
    if "SHARDS" in out:
        ok.append(out["SHARDS"]["equal"])
    if "PHG" in out:
        ok.append(out["PHG"]["equal"])
        if "served" in out["PHG"]:
            ok.append(out["PHG"]["served"]["equal"])
    out["identical"] = bool(all(ok))
    return out


def main():
    rows, missing = [], []
    for ds in ("metaqa", "squad", "musique"):
        for K in sorted(set(GRID) | {NATIVE[ds]}):
            c = cell(ds, K)
            rows.append(c)
            if "missing" in c:
                missing.append("%s K %d: %s" % (ds, K, c["missing"]))
                print("%-8s K %5d  MISSING %s" % (ds, K, c["missing"]))
                continue
            p = c.get("PHG", {})
            print("%-8s K %5d%s  HG %-5s  shards %-5s  outcome %s/%s  km1 %s/%s  vector %s%s  served %s  -> %s" % (
                ds, K, "n" if c["native"] else " ", c["HG"]["equal"], c.get("SHARDS", {}).get("equal"), c["OUTCOME"]["host"], c["OUTCOME"]["laptop"],
                c.get("RAW", {}).get("km1_host"), c.get("RAW", {}).get("km1_laptop"), p.get("equal"),
                "" if p.get("equal", True) else " (diff %s/%s, relabel %s)" % (p.get("differing_positions"), p.get("n"), p.get("equal_up_to_relabel")),
                p.get("served", {}).get("equal"), "IDENTICAL" if c["identical"] else "DIFFERS"))
    if missing:
        print("not written: %d cells missing" % len(missing))
        return
    verdict = "BIT_IDENTICAL" if all(c["identical"] for c in rows) else "NOT_IDENTICAL"
    b = rj(os.path.join(HOST, "PHG_BUILD_HOST.json"))
    rec = {"RECORD": "L1_HOST_PLACEMENT_TEST", "version": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "VERDICT": verdict,
           "rule": rj(os.path.join(HOST, "HOST_LANE_DECLARATION.json"))["placement_test"],
           "declaration": {"path": rel(os.path.join(HOST, "HOST_LANE_DECLARATION.json")), "sha256": sha_file(os.path.join(HOST, "HOST_LANE_DECLARATION.json"))},
           "host_build": {"path": rel(os.path.join(HOST, "PHG_BUILD_HOST.json")), "sha256": sha_file(os.path.join(HOST, "PHG_BUILD_HOST.json")),
                          "binary_sha256": b["binary"]["sha256"], "binary_equals_laptop": b["binary_equals_laptop"], "wrapper_equals_laptop": b["wrapper_equals_laptop"],
                          "toolchain_and_libs": b["toolchain_and_libs"], "placement": b["placement"]},
           "cells": rows,
           "counts": {"cells": len(rows), "identical": sum(c["identical"] for c in rows), "HG_equal": sum(c["HG"]["equal"] for c in rows),
                      "shards_equal": sum(c.get("SHARDS", {}).get("equal", False) for c in rows), "shards_compared": sum("SHARDS" in c for c in rows),
                      "vectors_equal": sum(c.get("PHG", {}).get("equal", False) for c in rows), "vectors_compared": sum("PHG" in c for c in rows),
                      "outcomes_equal": sum(c["OUTCOME"]["equal"] for c in rows)},
           "code": {"placement_module": {"path": rel(os.path.abspath(__file__)), "sha256": sha_file(os.path.abspath(__file__))},
                    "host_module": {"path": "scratchpad/_l1h_host.py", "sha256": sha_file(os.path.join(REPO, "scratchpad", "_l1h_host.py"))}}}
    assert not os.path.exists(OUTP), "write-once"
    with io.open(OUTP + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    os.replace(OUTP + ".tmp", OUTP)
    print(verdict, rec["counts"], "->", rel(OUTP), sha_file(OUTP))


if __name__ == "__main__":
    main()
