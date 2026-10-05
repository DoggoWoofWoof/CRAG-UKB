"""FBX_SCALE addendum 17 / E7 report assembler: the Freebase STRUCT-only V-cycle systems measurement, built ONLY from the stored stage records (no new measurement, no gold, no recall claim).

  python -u scratchpad/_ooc_e7_report.py            # writes results/FREEBASE_SCALE/E7_REPORT__fbx__PARTIAL_L4-1.json (+ .md) while level 0 is missing, E7_REPORT__fbx.json (+ .md) once E6_VCYCLE__fbx.json exists

Inputs (all under results/FREEBASE_SCALE): E1_CENSUS__fbx, E3_LADDER__fbx, E4_SURROGATE_BUILD__fbx, E4_SURROGATE__fbx, E6_VCYCLE_L4-2__fbx, E6_VCYCLE_L1-1__fbx and, when level 0 has run, E6_VCYCLE__fbx.
Reports exactly what the addendum asks: V_l / M_l / P_l per level, surrogate pins vs the 0.39e9 budget / 0.45e9 stop rule + Zoltan RSS / wall, per-stage wall / peak RSS / peak disk, the KM1 trajectory, block validity.
"""
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "results", "FREEBASE_SCALE")
BUDGET_PINS, STOP_PINS = 0.39e9, 0.45e9
RAM_GB, DISK_GB = 70.0, 55.0


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    e1, e3 = load("E1_CENSUS__fbx"), load("E3_LADDER__fbx")
    sb, sz = load("E4_SURROGATE_BUILD__fbx"), load("E4_SURROGATE__fbx")
    v42, v11, vfull = load("E6_VCYCLE_L4-2__fbx"), load("E6_VCYCLE_L1-1__fbx"), load("E6_VCYCLE__fbx")
    N = e3["N"]
    q1 = e1["quotient_level1_census"]

    # ---- V_l, M_l, P_l of the exact quotient at every level (level 0 = the implicit STRUCT-only hypergraph itself)
    levels = {0: {"V": N, "M": q1["M_in"], "P": q1["P_in"], "source": "E1 (implicit level 0: net u = {u} U N(u))"}}
    for l in ("1", "2", "3", "4"):
        q = e3["levels"][l]
        levels[int(l)] = {"V": q["V"], "M": q["M"], "P": q["P"], "source_vertices_V_in": q.get("agg", {}).get("V"), "nets_dropped_singleton": q["nets_dropped_singleton"], "nets_merged_identical": q["nets_merged_identical"], "max_vertex_weight": q["max_vertex_weight"], "weight_conserved_vertex": q["weight_conserved_vertex"]}
    for l, v in levels.items():
        v["V_reduction_vs_level0"] = round(N / v["V"], 3)
        v["P_reduction_vs_level0"] = round(levels[0]["P"] / v["P"], 3)

    # ---- surrogate (cap 4, level 4) against the budget / stop rule + Zoltan
    sp = sb["surrogate"]["P"]
    z = sz["zoltan"]["phg"]
    surrogate = {"V": sb["surrogate"]["V"], "M": sb["surrogate"]["M"], "P": sp, "pins_budget": BUDGET_PINS, "pins_stop_rule": STOP_PINS,
                 "budget_ratio": round(sp / BUDGET_PINS, 4), "stop_rule_triggered": sp > STOP_PINS, "within_budget": sp <= BUDGET_PINS,
                 "shards": sb["shards"]["shards"], "shard_bytes": sb["shards"]["bytes"],
                 "zoltan": {"np": z["np"], "rc": z["rc"], "wall_seconds_job": z["timing"]["job_wall_seconds"], "partition_wall_seconds": z["timing"]["partition_wall_seconds"],
                            "peak_rss_gb_max_rank": round(z["memory"]["max_rank_peak_kb"] / 1e6, 3), "peak_rss_gb_sum_of_rank_peaks": round(z["memory"]["sum_of_rank_peaks_kb"] / 1e6, 3),
                            "cutl_global_surrogate": z["zoltan_eval"]["cutl_global"], "imbalance": z["zoltan_eval"]["imbalance"]}}

    # ---- per stage wall / peak RSS / peak disk
    stages = [{"stage": "E1 twins (level-1 contraction)", "wall_s": e1["twins"]["wall_seconds"], "peak_rss_gb": e1["twins"]["peak_rss_gb"]},
              {"stage": "E1 quotient level 1", "wall_s": q1["wall_seconds"], "peak_rss_gb": q1["peak_rss_gb"]},
              {"stage": "E1 surrogate-at-level-1 census", "wall_s": e1["surrogate_cap4_level1_census"]["wall_seconds"], "peak_rss_gb": e1["surrogate_cap4_level1_census"]["peak_rss_gb"]}]
    for l in ("1", "2", "3", "4"):
        q = e3["levels"][l]
        stages.append({"stage": "E2/E3 ladder level %s quotient" % l, "wall_s": q["wall_seconds"], "peak_rss_gb": q["peak_rss_gb"]})
        if "agg" in q:
            stages.append({"stage": "E3 ladder level %s aggregation pass" % l, "wall_s": q["agg"]["wall_seconds"], "peak_rss_gb": q["agg"]["peak_rss_gb"]})
    stages += [{"stage": "E4 surrogate quotient (cap 4, level 4)", "wall_s": sp and sb["surrogate"]["wall_seconds"], "peak_rss_gb": sb["surrogate"]["peak_rss_gb"]},
               {"stage": "E4 Zoltan-PHG top (NP 4)", "wall_s": surrogate["zoltan"]["wall_seconds_job"], "peak_rss_gb_sum_of_ranks": surrogate["zoltan"]["peak_rss_gb_sum_of_rank_peaks"]}]
    traj = []
    for rec in (v42, v11, vfull):
        if rec:
            for t in rec["trajectory"]:
                if not any(x["level"] == t["level"] for x in traj):
                    traj.append(t)
    traj.sort(key=lambda t: -t["level"])
    for t in traj:
        if "quot_seconds" in t:
            stages.append({"stage": "E5/E6 level %d: quotient + transpose + FM2 refine" % t["level"], "wall_s": round(t["quot_seconds"] + t["transpose_seconds"] + t["fm_seconds"], 1), "fm_peak_rss_gb": t["fm_peak_rss_gb"], "disk_bytes_work": t["disk_bytes_work"]})
        else:  # level 0 is implicit (the original hypergraph): no quotient / transpose, and the stage recorded no work-dir size
            stages.append({"stage": "E6 level %d: implicit FM2 refine (no quotient / transpose)" % t["level"], "wall_s": t["fm_seconds"], "fm_peak_rss_gb": t["fm_peak_rss_gb"], "disk_bytes_work": None})
    peak_rss = max([s.get("peak_rss_gb", 0) for s in stages] + [s.get("fm_peak_rss_gb", 0) for s in stages])
    peak_disk = max([t["disk_bytes_work"] for t in traj if t.get("disk_bytes_work") is not None] + [sb["disk_bytes_work"]]) / 1e9
    no_disk_levels = [t["level"] for t in traj if t.get("disk_bytes_work") is None]

    # ---- KM1 trajectory
    km1 = [{"level": t["level"], "V": t["V"], "km1_in": t["km1_in"], "km1_out": t["km1_out"], "moves": t["moves"], "passes_run": t["passes_run"], "max_load_after": t["max_load_after"], "min_load_after": t["min_load_after"]} for t in traj]
    first_in = km1[0]["km1_in"]
    last_out = km1[-1]["km1_out"]
    done_levels = sorted(t["level"] for t in traj)
    complete = vfull is not None and 0 in done_levels

    out = {"stage": "FBX_SCALE addendum 17 / E7: Freebase STRUCT-only V-cycle systems measurement (VCQ512L4N4FM2)", "status": "COMPLETE" if complete else "PARTIAL: levels %s done, level 0 pending (waiting for a 41 GB scheduler slot; mpr has priority)" % done_levels[::-1],
           "N": N, "S": 25, "cap_per_block": (vfull or v42)["cap"], "levels": {str(k): v for k, v in sorted(levels.items())}, "surrogate": surrogate, "stages": stages,
           "peaks": {"max_stage_rss_gb": round(peak_rss, 2), "ram_budget_gb": RAM_GB, "ram_ok": peak_rss <= RAM_GB, "max_work_disk_gb": round(peak_disk, 2), "disk_budget_gb": DISK_GB, "disk_ok": peak_disk <= DISK_GB, "disk_levels_without_record": no_disk_levels},
           "km1_trajectory": km1, "km1_first_in": first_in, "km1_last_out": last_out, "km1_last_over_first": round(last_out / first_in, 4),
           "validity": (vfull or {}).get("validity"), "not_claimed": "no recall at Freebase scale (no Freebase gold; the WebQSP bridge proxy is a later addendum), no K-way sub-partition, no SK arm; the verdict transferred from WebQSP is ON the pass line (7 rows, F = 8), one Zoltan seed"}
    name = "E7_REPORT__fbx" + ("" if complete else "__PARTIAL_L4-1")
    json.dump(out, io.open(os.path.join(R, name + ".json"), "w", encoding="utf-8"), indent=1)

    L = ["# E7 -- Freebase STRUCT-only V-cycle (VCQ512L4N4FM2), systems measurement -- %s\n" % ("COMPLETE" if complete else "PARTIAL (levels %s done; level 0 pending)" % done_levels[::-1]),
         "N = %s nodes, S = 25, per-block cap %s (ceil(1.03 N / 25)).  Nothing here is a recall claim.\n" % ("{:,}".format(N), "{:,}".format(out["cap_per_block"])),
         "## Levels (exact quotients)\n", "| level | V | M (nets) | P (pins) | V reduction | P reduction |", "|---|---|---|---|---|---|"]
    for l in range(0, 5):
        v = levels[l]
        L.append("| %d | %s | %s | %s | %.2fx | %.2fx |" % (l, "{:,}".format(v["V"]), "{:,}".format(v["M"]), "{:,}".format(v["P"]), v["V_reduction_vs_level0"], v["P_reduction_vs_level0"]))
    L += ["\n## Surrogate (cap 4, level 4) vs budget\n", "pins %s = %.1f%% of the 0.39e9 budget (stop rule 0.45e9: %s); Zoltan NP 4 wall %.0f s, peak RSS %.2f GB summed over ranks (%.2f GB max rank); surrogate cut %s, imbalance %.4f.\n" % (
        "{:,}".format(sp), 100 * sp / BUDGET_PINS, "NOT triggered" if sp <= STOP_PINS else "TRIGGERED", surrogate["zoltan"]["wall_seconds_job"], surrogate["zoltan"]["peak_rss_gb_sum_of_rank_peaks"], surrogate["zoltan"]["peak_rss_gb_max_rank"], "{:,.0f}".format(surrogate["zoltan"]["cutl_global_surrogate"]), surrogate["zoltan"]["imbalance"]),
          "## KM1 trajectory (full STRUCT-only hypergraph, projected top refined level by level)\n", "| level | V | km1 in | km1 out | moves | load max / min after |", "|---|---|---|---|---|---|"]
    for t in km1:
        L.append("| %d | %s | %s | %s | %s | %s / %s |" % (t["level"], "{:,}".format(t["V"]), "{:,}".format(t["km1_in"]), "{:,}".format(t["km1_out"]), "{:,}".format(t["moves"]), "{:,}".format(t["max_load_after"]), "{:,}".format(t["min_load_after"])))
    L += ["\nkm1 last out / first in = %.4f (%.2f%% reduction through %d refined levels).\n" % (out["km1_last_over_first"], 100 * (1 - out["km1_last_over_first"]), len(km1)),
          "## Peaks\n", "max stage RSS %.1f GB (budget %.0f GB: %s); max recorded crag work-dir disk %.1f GB (budget %.0f GB: %s)%s.\n" % (peak_rss, RAM_GB, "ok" if out["peaks"]["ram_ok"] else "EXCEEDED", peak_disk, DISK_GB, "ok" if out["peaks"]["disk_ok"] else "EXCEEDED", ("; level(s) %s recorded no work-dir size (implicit level, label files only; the work dir measured 2.81 GB after the stage, the peak during the stage was not recorded)" % no_disk_levels) if no_disk_levels else "")]
    if complete:
        v = out["validity"]
        L += ["## Validity\n", "```\n%s\n```\n" % json.dumps(v, indent=1)]
    else:
        L += ["## Not yet measured\n", "Level 0 (the full 301,977,131-node FM2 refinement, peak RSS ~41 GB at level 1 so reserved 41 GB) and the final block-validity check are pending; they need a host capacity slot that mpr jobs currently hold.\n"]
    L += ["## Not claimed\n", out["not_claimed"] + "\n"]
    io.open(os.path.join(R, name + ".md"), "w", encoding="utf-8").write("\n".join(L))
    print(name, "written;", out["status"])
    print("surrogate pins %s (%.1f%% of budget), max stage RSS %.1f GB, max disk %.1f GB, km1 %s -> %s (%.4f)" % ("{:,}".format(sp), 100 * sp / BUDGET_PINS, peak_rss, peak_disk, "{:,}".format(first_in), "{:,}".format(last_out), out["km1_last_over_first"]))


if __name__ == "__main__":
    main()
