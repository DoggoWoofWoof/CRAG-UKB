"""FREEBASE_SCALE stage 2 -- can the frozen PHG contract partition the Freebase STRUCT hypergraph on this host?

A calculation, not a launch: it reads the measured PHG cells of the six datasets (results/L1_HOST/parts/*RUN.json,
40 cells with a memory record) and the stage-1 pin count, fits memory and time against pins, and compares the
prediction with what the shared host offers.  Nothing is run on the host; nothing could exhaust the shared WSL VM.

  python scratchpad/_fbx_feasibility.py            (write-once: results/FREEBASE_SCALE/FBX_PARTITION_FEASIBILITY__v1.json)
"""
import glob
import hashlib
import io
import json
import os
import time

import numpy as np

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/FBX_PARTITION_FEASIBILITY__v1.json"
STATS = "results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rnd(x, n=1):
    return round(float(x), n)


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    st = json.load(io.open(STATS, encoding="utf-8"))["stats"]
    P_FB = st["H4_STRUCT_only"]["pins_precap"]
    N_FB = st["nodes"]
    rows, pins_used = [], {}
    for f in sorted(glob.glob("results/L1_HOST/parts/*__PHG_con.RUN.json")):
        J = json.load(io.open(f, encoding="utf-8"))
        r = J["run"]
        mem = r.get("memory") or {}
        pk = mem.get("ru_maxrss_kb_per_rank") or mem.get("peak_rss_kb_per_rank_time_v")
        tm = r.get("timing") or {}
        if not pk or not tm.get("partition_wall_seconds"):
            continue
        h = J["hypergraph"]
        rep = J.get("repair") or {}
        rows.append({"dataset": J["dataset"], "K": J["K"], "N": h["N"], "pins": h["P"], "peak_sum_gb": sum(pk) / 1e6, "peak_max_rank_gb": max(pk) / 1e6,
                     "partition_s": tm["partition_wall_seconds"], "driver_s": tm.get("driver_total_seconds_rank0"), "repair_moves": rep.get("moves", 0),
                     "whole_step_s": J.get("wall_seconds")})
        pins_used[os.path.basename(f)] = sha(f)
    P = np.array([x["pins"] for x in rows], float)
    M = np.array([x["peak_sum_gb"] for x in rows], float)
    T = np.array([x["partition_s"] for x in rows], float)
    big = P >= 25e6                                   # webqsp / hotpotqa / 2wiki cells: the regime closest to Freebase
    bpp = M[big] * 1e9 / P[big]
    A = np.vstack([np.log(P), np.ones(len(P))]).T
    cm = np.linalg.lstsq(A, np.log(M), rcond=None)[0]
    ct = np.linalg.lstsq(A, np.log(T), rcond=None)[0]
    Ab = np.vstack([np.log(P[big]), np.ones(big.sum())]).T
    cmb = np.linalg.lstsq(Ab, np.log(M[big]), rcond=None)[0]
    ctb = np.linalg.lstsq(Ab, np.log(T[big]), rcond=None)[0]
    share_max = float(np.mean([x["peak_max_rank_gb"] / x["peak_sum_gb"] for x in rows if x["pins"] >= 25e6]))
    pred_mem_lin_lo, pred_mem_lin_hi = bpp.min() * P_FB / 1e9, bpp.max() * P_FB / 1e9
    pred_mem_fit = float(np.exp(cm[1]) * P_FB ** cm[0])
    pred_mem_fit_big = float(np.exp(cmb[1]) * P_FB ** cmb[0])
    ref = max((x for x in rows if x["dataset"] == "2wiki" and x["K"] == 2000), key=lambda x: x["pins"])
    host = {
        "wsl_vm_total_gb": 94.3, "wsl_available_gb_at_10:05_local_2026-09-30": 81.6, "wsl_swap_gb": 32.0,
        "windows_ram_gb": 127.7, "windows_ram_in_use_by_other_users_gb_at_10:03": 46.9,
        "c_drive_free_gb": 386.8, "cpu_logical": 32,
        "source": "rx exec free -m in Ubuntu-24.04 and scratchpad/util.py, 2026-09-30 10:03-10:05 local (read-only)",
    }
    avail = host["wsl_available_gb_at_10:05_local_2026-09-30"]
    rec = {
        "stage": "FBX_SCALE / stage 2 (feasibility of the frozen PHG contract; a calculation, no host job)",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "DEVELOPMENT (systems); no partition attempted",
        "inputs": {"stage1_stats": STATS, "stage1_stats_sha256": sha(STATS), "phg_cells_used": len(rows), "cell_record_sha256": pins_used},
        "contract": "H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 + PHG_NONEMPTY_REPAIR_V1 (NP = 4 frozen; a different rank count needs a ruling); hypergraph = H4_SPLIT_PRESERVE, STRUCT family only (no KNN: no vectors exist for 302M nodes)",
        "freebase_input": {"nodes": N_FB, "anchors": st["anchors_deg_ge_1"], "unique_undirected_pairs": st["unique_undirected_pairs_F"], "pins": P_FB,
                           "pins_vs_2wiki_native_cell": rnd(P_FB / ref["pins"], 1), "pins_vs_largest_measured_cell": rnd(P_FB / P.max(), 1),
                           "eidx_int32_gb": rnd(P_FB * 4 / 1e9, 1)},
        "measured_cells": {"n": len(rows), "pins_min": int(P.min()), "pins_max": int(P.max()),
                           "bytes_per_pin_all": [rnd((M * 1e9 / P).min(), 1), rnd((M * 1e9 / P).max(), 1)],
                           "bytes_per_pin_cells_ge_25M_pins": [rnd(bpp.min(), 1), rnd(bpp.max(), 1), "n=%d" % big.sum()],
                           "largest_rank_share_of_peak_sum": rnd(share_max, 3)},
        "memory_prediction_gb (sum over the 4 ranks)": {
            "linear_at_measured_bytes_per_pin_range": [rnd(pred_mem_lin_lo, 0), rnd(pred_mem_lin_hi, 0)],
            "power_law_fit_all_cells": {"exponent": rnd(cm[0], 3), "at_freebase": rnd(pred_mem_fit, 0)},
            "power_law_fit_cells_ge_25M": {"exponent": rnd(cmb[0], 3), "at_freebase": rnd(pred_mem_fit_big, 0)},
        },
        "per_rank_gb_at_NP4": [rnd(pred_mem_lin_lo * share_max, 0), rnd(pred_mem_lin_hi * share_max, 0)],
        "host": host,
        "ratio_prediction_over_wsl_available": [rnd(pred_mem_lin_lo / avail, 1), rnd(pred_mem_lin_hi / avail, 1)],
        "time_prediction_hours (partition step only, if the memory existed)": {
            "linear_in_pins_from_2wiki_K2000": rnd(ref["partition_s"] * P_FB / ref["pins"] / 3600, 1),
            "power_law_fit_all_cells": {"exponent": rnd(ct[0], 3), "hours": rnd(np.exp(ct[1]) * P_FB ** ct[0] / 3600, 1)},
            "power_law_fit_cells_ge_25M": {"exponent": rnd(ctb[0], 3), "hours": rnd(np.exp(ctb[1]) * P_FB ** ctb[0] / 3600, 1)},
        },
        "other_blockers_recorded_not_measured": [
            "the split rule ranks every member v of an oversize anchor u by |N(u) n N(v)| (H4 build); at K = 5,000 the %d oversize anchors hold %d pins (%.1f%% of all pins), at K = 1,000 %d anchors hold %d"
            % (st["oversize_hyperedges_by_K"]["5000"]["oversize_anchors"], st["oversize_hyperedges_by_K"]["5000"]["pins_in_oversize"],
               100 * st["oversize_hyperedges_by_K"]["5000"]["share_of_pins"], st["oversize_hyperedges_by_K"]["1000"]["oversize_anchors"],
               st["oversize_hyperedges_by_K"]["1000"]["pins_in_oversize"]),
            "PHG_NONEMPTY_REPAIR_V1 cost 7 s per move at N = 2.6M (WebQSP, 326 moves = 38 min); the per-move cost at N = 302M is not measured",
            "the frozen builder holds the whole STRUCT adjacency as int64 (2F = %.1f billion entries, %.1f GB) before the hypergraph is written" % (2 * st["unique_undirected_pairs_F"] / 1e9, 2 * st["unique_undirected_pairs_F"] * 8 / 1e9),
        ],
        "verdict": "FBX_PHG_NP4_PREDICTED_INFEASIBLE_ON_HOST" if pred_mem_lin_lo > avail else "FBX_PHG_NP4_FITS_PREDICTED",
        "verdict_basis": "memory prediction %.0f-%.0f GB against %.1f GB available in the shared WSL VM; the %d measured cells above 25M pins have %.0f-%.0f bytes per pin, and no measured cell is above %.0fM pins; NOT ATTEMPTED because a launch that exceeds the VM would harm other users' jobs; NP was not changed"
                         % (pred_mem_lin_lo, pred_mem_lin_hi, avail, big.sum(), bpp.min(), bpp.max(), P.max() / 1e6),
        "what_would_change_it": ["a machine with >= about 0.7 TB RAM for the frozen recipe, or", "a user ruling on a lower-memory route (not chosen here; the standing rule forbids a silent swap)"],
        "code": {"scratchpad/_fbx_feasibility.py": sha("scratchpad/_fbx_feasibility.py")},
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT))
    print(json.dumps({k: rec[k] for k in ("freebase_input", "measured_cells", "memory_prediction_gb (sum over the 4 ranks)", "per_rank_gb_at_NP4",
                                            "ratio_prediction_over_wsl_available", "time_prediction_hours (partition step only, if the memory existed)", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
