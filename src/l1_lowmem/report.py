"""H4_SK_LOW_MEMORY_PARTITIONING_REPORT -- assembles every record of the validation into one report + table.

    python -u src/l1_lowmem/report.py            -> results/L1_LOWMEM/H4_SK_LOW_MEMORY_PARTITIONING_REPORT.{json,md}

Inputs (all optional; missing ones are reported as PENDING, never inferred):
  results/L1_LOWMEM/H4_SK_CONTRACT_FREEZE.json                  the frozen contract (never inferred)
  data/l1_lowmem/<ds>/H4_SK.hgr.json                            ORIGINAL_STRUCTURE_SHA256
  data/l1_lowmem/<ds>/stream/RECONSTRUCTION_GATE.json           digest + bytes gates
  data/l1_lowmem/<ds>/H4_SK.official.netl.json                  official converter run
  results/L1_LOWMEM/FREIGHT_BUILD.json, FREIGHT_RUNS_<ds>.json  experiment 1 (exact restart equivalence) + metrics
  results/L1_LOWMEM/L1_DOWNSTREAM_<ds>.json                     experiment 2 (paired canonical L1 evaluation)
  results/L1_LOWMEM/AUDIT_ZOLTAN_PHG.json, AUDIT_MTKAHYPAR_OUTOFCORE.json
Classification (exactly one of STRUCTURE_MISMATCH / OBJECTIVE_INCOMPATIBLE / LOW_MEMORY_BUT_QUALITY_FAIL /
LOW_MEMORY_L1_EQUIVALENT / NEEDS_MORE_EVIDENCE) follows the rule written into the record; the 5 % objective
threshold is NOT preregistered and therefore never gates the classification -- the numbers are reported.
"""
import io
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, LOCAL_SETS, log, rj, wj, ds_dir, pin  # noqa: E402
from src.l1_lowmem import stream as ST  # noqa: E402

LOCAL_RAM_GB = {"host_windows_total": 15.69, "wsl_vm_total": 7.0}   # measured this session (Win32_OperatingSystem / free -g)


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%%.%df" % nd) % x
    return str(x)


def mb(kb):
    return "-" if kb is None else "%.0f MB" % (kb / 1024.0)


def p50_stats(arm):
    """P50 post-processing statistics of one replay arm (the numbers l1_eval reports after the partition: router scope,
    coverage, SAFE selector additions, block-size balance).  Copied, never recomputed."""
    if not arm:
        return None
    c, bal = arm.get("CORR") or {}, arm.get("BALANCE") or {}
    return {"BASE_ANY_P50": arm.get("BASE_ANY_P50"), "SAFE_ANY_P50": arm.get("SAFE_ANY_P50"),
            "BASE_SCOPE_NODES": arm.get("BASE_SCOPE_NODES"), "SAFE_SCOPE_NODES": arm.get("SAFE_SCOPE_NODES"), "BND_ALL_P50": arm.get("BND_ALL_P50"),
            "SAFE_additions": {k: c.get(k) for k in ("dALL", "gained", "lost", "net", "mcnemar_p", "sig", "gold_admitted", "gold_evicted",
                                                     "churn_per_query", "churn_max", "queries_with_zero_churn")},
            "BALANCE": {k: bal.get(k) for k in ("npart", "blocks_used", "size_min", "size_median", "size_mean", "size_p90", "size_p99", "size_max",
                                                "size_cv", "max_over_mean", "ELIGIBLE")},
            "by_hop": arm.get("by_hop")}


def dataset_block(ds):
    B = {"dataset": ds}
    hrec = rj(os.path.join(ds_dir(ds), "H4_SK.hgr.json"))
    gate = rj(os.path.join(ST.stream_dir(ds), "RECONSTRUCTION_GATE.json"))
    off = rj(os.path.join(ds_dir(ds), "H4_SK.official.netl.json"))
    runs = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))
    down = rj(os.path.join(OUT, "L1_DOWNSTREAM_%s.json" % ds))
    B["structure"] = {"ORIGINAL_STRUCTURE_SHA256": hrec and hrec["ORIGINAL_STRUCTURE_SHA256"], "counts": hrec and hrec["counts"],
                      "gate_digest(shards)": gate and gate["gate_digest"], "STREAM_STRUCTURE_SHA256": gate and gate.get("STREAM_STRUCTURE_SHA256"),
                      "gate_bytes(official net-list)": gate and gate["gate_bytes"],
                      "official_converter": off and {"peak_rss_kb": off["time"].get("peak_rss_kb"), "wall_seconds": off["time"].get("wall_seconds"),
                                                     "bytes": off["output"]["bytes"], "sha256": off["output"]["sha256"]}}
    if gate is None:
        B["structure_exact"] = None                                   # PENDING: no gate record yet
    elif gate["gate_digest"] == "FAIL" or gate["gate_bytes"] == "FAIL":
        B["structure_exact"] = False
    elif gate["gate_digest"] == "PASS" and gate["gate_bytes"] == "PASS":
        B["structure_exact"] = True
    else:
        B["structure_exact"] = None                                   # digest PASS, official-bytes gate still PENDING
    # arm A: canonical Mt-KaHyPar
    A = {"method": "Mt-KaHyPar DETERMINISTIC_QUALITY KM1 eps 0.03 seed 0 (canonical H4_SK)", "status": "PENDING"}
    if runs and runs.get("baseline_mtkahypar"):
        bm = runs["baseline_mtkahypar"]
        A["status"] = bm.get("status")
        if bm.get("status") == "OK":
            ws, gd = bm["manifest"]["worker_stats"], bm["manifest"]["guard"] or {}
            A.update({"peak_rss_kb": int(ws.get("peak_rss_mb", 0) * 1024) if ws.get("peak_rss_mb") else gd.get("peak_rss_kb"),
                      "wall_seconds": ws.get("wall_seconds"), "objective_km1": bm["km1_weighted"], "objective_km1_worker": ws.get("objective_km1"),
                      "python_km1_equals_worker": bm.get("python_km1_equals_mtkahypar_objective"), "cut_weighted": bm["cut_weighted"],
                      "blocks": bm["blocks"], "family_cuts": bm["family_cuts"]})
        else:
            A["record"] = bm.get("record")
            retry = rj(os.path.join(OUT, "%s_MTKAHYPAR_RETRY.json" % ds.upper()))
            if retry:                                                  # spec step 7: one isolated retry, then classify
                A["status"] = retry["CLASSIFICATION"]
                A["retry"] = {"CLASSIFICATION": retry["CLASSIFICATION"], "ruling": retry["ruling"],
                              "attempts": [{"utc": retry["previous_attempt"]["utc"], "threads": retry["previous_attempt"]["threads"],
                                            "wall_seconds": retry["previous_attempt"]["wall_seconds"], "peak_rss_kb_at_kill": retry["previous_attempt"]["peak_rss_kb"],
                                            "host_available_gb_at_start": retry["previous_attempt"]["host_available_gb_at_start"]},
                                           {"utc": retry["retry"]["guard"] and rj(os.path.join(REPO, retry["retry"]["record"]["path"]))["utc"],
                                            "threads": retry["retry"]["threads"], "wall_seconds": retry["retry"]["wall_seconds"],
                                            "peak_rss_kb_at_kill": retry["retry"]["peak_rss_kb_at_kill"],
                                            "host_available_gb_at_start": retry["retry"]["guard"]["host_available_gb_at_start"]}],
                              "rss_cap_gb": retry["retry"]["guard"]["rss_cap_gb"], "isolation": retry["retry"]["isolation"], "record": retry["retry"]["record"]}
    # arm B: FREIGHT
    F = {"method": "FREIGHT freight_con (connectivity = weighted KM1), imbalance 3 %, seed 0, 1 pass, canonical node order, sharded + checkpointed",
         "status": "PENDING"}
    if runs:
        rB, rA = runs["runs"]["B"], runs["runs"]["A"]
        F.update({"status": "OK", "experiment_1": runs["VERDICT_EXPERIMENT_1"], "runs_identical_to_A": {k: v["identical_to_A"] for k, v in runs["EQUIVALENCE"]["compared"].items()},
                  "refusals_all_refused": runs["refusals"]["ALL_REFUSED"], "kill_test": runs["runs"]["C_kill"].get("status", "KILLED_AND_RESUMED" if runs["runs"]["C_kill"].get("killed") else "?"),
                  "two_pass_smoke": (runs.get("two_pass_smoke") or {}).get("VERDICT"),
                  "peak_rss_kb": rB["time"].get("peak_rss_kb"), "wall_seconds": rB["time"].get("wall_seconds"),
                  "peak_rss_kb_stock_monolithic": rA["time"].get("peak_rss_kb"), "wall_seconds_stock_monolithic": rA["time"].get("wall_seconds"),
                  "objective_km1": runs["metrics_B"]["km1_weighted"], "cut_weighted": runs["metrics_B"]["cut_weighted"], "blocks": runs["metrics_B"]["blocks"],
                  "family_cuts": runs["metrics_B"]["family_cuts"], "input_bytes": runs["inputs"]["input_bytes"], "shards": runs["inputs"]["stream_manifest"]["shards"],
                  "freight_connectivity_evaluator_skipped": runs["EQUIVALENCE"]["freight_connectivity_evaluator_skipped"],
                  "python_km1_equals_freight_connectivity": runs["EQUIVALENCE"]["python_km1_equals_freight_connectivity"],
                  "peak_rss_kb_checkpointed_B0": runs["runs"]["B0"]["time"].get("peak_rss_kb"), "wall_seconds_checkpointed_B0": runs["runs"]["B0"]["time"].get("wall_seconds"),
                  "checkpoint_files": (runs["runs"]["B0"].get("ckpt_dir") or {}).get("files"), "A_eq_A_forklib": runs["EQUIVALENCE"].get("A_eq_A_forklib"),
                  "exact_metrics_per_arm": {k: v.get("ALL_EXACT") for k, v in runs["EQUIVALENCE"]["compared"].items()}})
    B["A"], B["B"] = A, F
    # quality relation (reported, not gated: the 5 % threshold is not preregistered)
    if A.get("objective_km1") and F.get("objective_km1"):
        B["objective_ratio_B_over_A"] = round(F["objective_km1"] / float(A["objective_km1"]), 4)
        B["struct_cut_delta_B_minus_A"] = round(F["family_cuts"]["STRUCT"]["edge_cut_fraction"] - A["family_cuts"]["STRUCT"]["edge_cut_fraction"], 4)
        B["knn_cut_delta_B_minus_A"] = round(F["family_cuts"]["KNN"]["edge_cut_fraction"] - A["family_cuts"]["KNN"]["edge_cut_fraction"], 4)
    # downstream
    B["downstream"] = {"status": "PENDING"}
    if down:
        B["downstream"] = {"status": down["status"], "identical_population": down.get("identical_population"), "nq": down.get("population_B", {}).get("nq"),
                           "B": {k: down["B"].get(k) for k in ("BASE_ALL_P50", "SAFE_ALL_P50", "BASE_ANY_P50", "SAFE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_SCOPE_NODES", "BND_ALL_P50")},
                           "B_safe_additions": down["B"]["CORR"], "B_by_hop": down["B"].get("by_hop"), "B_p50_postprocessing": p50_stats(down["B"])}
        if down["status"] == "PAIRED":
            B["downstream"].update({"A": {k: down["A"].get(k) for k in ("BASE_ALL_P50", "SAFE_ALL_P50", "BASE_ANY_P50", "SAFE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_SCOPE_NODES", "BND_ALL_P50")},
                                    "A_safe_additions": down["A"]["CORR"], "A_by_hop": down["A"].get("by_hop"), "A_p50_postprocessing": p50_stats(down["A"]),
                                    "paired": down["paired"], "by_hop_paired": down.get("by_hop_paired"),
                                    "coverage_paired": down.get("coverage"), "safe_additions_paired": down.get("safe_additions")})
    # per-dataset gates
    g = {"1_structure_exact": B["structure_exact"],
         "2_feasibility_peak_rss_le_local_ram": (None if F.get("peak_rss_kb") is None else bool(F["peak_rss_kb"] / 1048576.0 <= LOCAL_RAM_GB["wsl_vm_total"])),
         "2b_exact_restart_equivalence": (None if not runs else runs["VERDICT_EXPERIMENT_1"] == "EXACT_RESTART_EQUIVALENCE"),
         "3_partition_valid": (None if not F.get("blocks") else bool(F["blocks"]["empty"] == 0 and F["blocks"]["within_eps_0.03"])),
         "4_quality_reported_not_gated": {"objective_ratio_B_over_A": B.get("objective_ratio_B_over_A"), "struct_cut_delta": B.get("struct_cut_delta_B_minus_A"),
                                          "knn_cut_delta": B.get("knn_cut_delta_B_minus_A"), "threshold": "NOT PREREGISTERED (5 % proposed, not frozen)"},
         "5_downstream": (None if B["downstream"]["status"] != "PAIRED" else
                          {"SAFE_delta": B["downstream"]["paired"]["SAFE_ALL_P50"]["delta_B_minus_A"], "SAFE_p": B["downstream"]["paired"]["SAFE_ALL_P50"]["mcnemar_p"],
                           "SAFE_sig": B["downstream"]["paired"]["SAFE_ALL_P50"]["sig"], "BASE_delta": B["downstream"]["paired"]["BASE_ALL_P50"]["delta_B_minus_A"],
                           "BASE_p": B["downstream"]["paired"]["BASE_ALL_P50"]["mcnemar_p"], "BASE_sig": B["downstream"]["paired"]["BASE_ALL_P50"]["sig"]})}
    B["gates"] = g
    return B


def classify(blocks):
    """exactly one label; the rule is part of the record."""
    rule = ("STRUCTURE_MISMATCH if any local dataset fails gate 1; OBJECTIVE_INCOMPATIBLE if the FREIGHT objective family is not connectivity/KM1 "
            "(it is: freight_con; the Python KM1 recomputation equals FREIGHT's connectivity wherever FREIGHT's evaluator runs); "
            "LOW_MEMORY_BUT_QUALITY_FAIL if gates 1-3 pass and any paired dataset shows a SIGNIFICANT (exact McNemar p<0.05) SAFE_ALL_P50 or BASE_ALL_P50 loss; "
            "LOW_MEMORY_L1_EQUIVALENT if gates 1-3 and exact restart equivalence pass on all three local datasets, all three are paired, and no paired loss is "
            "significant; NEEDS_MORE_EVIDENCE otherwise (a pending stage, an unpaired dataset, or an unregistered threshold decision left to the user).  "
            "The 5 % objective-degradation threshold is not preregistered and does not gate.")
    if any(b["structure_exact"] is False for b in blocks):
        return "STRUCTURE_MISMATCH", rule
    complete = all(b["structure_exact"] is True and b["B"]["status"] == "OK" and b["gates"]["2b_exact_restart_equivalence"] and b["gates"]["3_partition_valid"]
                   and b["gates"]["2_feasibility_peak_rss_le_local_ram"] for b in blocks)
    paired = [b for b in blocks if b["downstream"]["status"] == "PAIRED"]
    sig_loss = [b["dataset"] for b in paired if (b["gates"]["5_downstream"]["SAFE_sig"] and b["gates"]["5_downstream"]["SAFE_delta"] < 0)
                or (b["gates"]["5_downstream"]["BASE_sig"] and b["gates"]["5_downstream"]["BASE_delta"] < 0)]
    if complete and sig_loss:
        return "LOW_MEMORY_BUT_QUALITY_FAIL", rule
    if complete and len(paired) == len(blocks) and not sig_loss:
        return "LOW_MEMORY_L1_EQUIVALENT", rule
    return "NEEDS_MORE_EVIDENCE", rule


def sx(b):
    return {True: "yes", False: "NO"}.get(b["structure_exact"], "PENDING (digest %s, bytes %s)" % (b["structure"]["gate_digest(shards)"], str(b["structure"]["gate_bytes(official net-list)"]).split(" ")[0]))


def table(blocks):
    hdr = "| Dataset | Method | Structure exact | Peak RSS | Time | Objective (KM1) | Imbalance (max/ceil(N/k)-1) | STRUCT cut | KNN cut | BASE | SAFE |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for b in blocks:
        d = b["downstream"]
        for arm, key in (("Mt-KaHyPar (canonical)", "A"), ("FREIGHT con (sharded+ckpt)", "B")):
            m = b[key]
            if m.get("status") != "OK":
                if m.get("retry"):                                     # killed at the RSS cap on both attempts
                    last = m["retry"]["attempts"][-1]
                    rows.append("| %s | %s | %s | >%s (killed at %.1f GB cap, %d attempts) | %s s (killed) | %s | | | | | |" % (
                        b["dataset"], arm, sx(b), mb(last["peak_rss_kb_at_kill"]), m["retry"]["rss_cap_gb"], len(m["retry"]["attempts"]), fmt(last["wall_seconds"], 1), m["status"]))
                else:
                    rows.append("| %s | %s | %s | %s | | | | | | | |" % (b["dataset"], arm, sx(b), m.get("status")))
                continue
            base = safe = "-"
            if key == "A" and d.get("A"):
                base, safe = fmt(d["A"]["BASE_ALL_P50"]), fmt(d["A"]["SAFE_ALL_P50"])
            if key == "B" and d.get("B"):
                base, safe = fmt(d["B"]["BASE_ALL_P50"]), fmt(d["B"]["SAFE_ALL_P50"])
            rows.append("| %s | %s | %s | %s | %s s | %s | %s | %s | %s | %s | %s |" % (
                b["dataset"], arm, sx(b), mb(m.get("peak_rss_kb")), fmt(m.get("wall_seconds"), 1), fmt(m.get("objective_km1")),
                fmt(m["blocks"]["eps_actual(max/ceil(N/k)-1)"]), fmt(m["family_cuts"]["STRUCT"]["edge_cut_fraction"]), fmt(m["family_cuts"]["KNN"]["edge_cut_fraction"]),
                base, safe))
    return "\n".join(rows)


def table_p50(blocks):
    """P50 post-processing statistics per arm: block balance (from the replay), router scope / coverage, SAFE additions."""
    hdr = ("| Dataset | Method | Blocks used/k | Block size min/p50/p95/max | max/mean | BASE ALL | BASE ANY | BASE scope | SAFE ALL | SAFE ANY | SAFE scope | "
           "SAFE additions +g/-l (net) | SAFE p | churn/q |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for b in blocks:
        d = b["downstream"]
        for arm, key in (("Mt-KaHyPar (canonical)", "A"), ("FREIGHT con (sharded+ckpt)", "B")):
            s, m = d.get(key + "_p50_postprocessing"), b[key]
            if not s:
                rows.append("| %s | %s | %s | | | | | | | | | | | |" % (b["dataset"], arm, "downstream " + str(d.get("status")) if m.get("status") == "OK" else m.get("status")))
                continue
            bl, sa, blk = s["BALANCE"], s["SAFE_additions"], m.get("blocks") or {}
            rows.append("| %s | %s | %s/%s | %s/%s/%s/%s | %s | %s | %s | %s | %s | %s | %s | +%s/-%s (%+d) | %s | %s |" % (
                b["dataset"], arm, bl["blocks_used"], bl["npart"], bl["size_min"], fmt(blk.get("p50"), 0), fmt(blk.get("p95"), 0), bl["size_max"],
                fmt(bl["max_over_mean"]), fmt(d[key]["BASE_ALL_P50"]), fmt(d[key]["BASE_ANY_P50"]), fmt(d[key]["BASE_SCOPE_NODES"], 1),
                fmt(d[key]["SAFE_ALL_P50"]), fmt(d[key]["SAFE_ANY_P50"]), fmt(d[key]["SAFE_SCOPE_NODES"], 1),
                sa["gained"], sa["lost"], sa["net"] or 0, sa["mcnemar_p"], fmt(sa["churn_per_query"], 3)))
    return "\n".join(rows)


def canonical_untouched():
    """Evidence that the validation modified nothing canonical: the served H4_SK partition and production replay cache of every
    local dataset still hash to what their own records pin (musique's canonical partition is absent by its FAILED record)."""
    from src.l1_lowmem.common import sha_file
    out = {}
    for ds in LOCAL_SETS:
        dd = os.path.join(REPO, "data", "l1_canonical", ds)
        row = {}
        part = os.path.join(dd, "parts", "H4_SK.npy")
        prod = rj(os.path.join(REPO, "results", "L1_CANONICAL", "L1_REPLAY_%s.json" % ds))
        if os.path.exists(part):
            row["H4_SK_partition_sha256"] = sha_file(part)
            row["H4_SK_partition_matches_production_replay_pin"] = None if prod is None else bool(prod["FROZEN_AGAINST"]["partition_hash"] == row["H4_SK_partition_sha256"])
        else:
            failed = rj(os.path.join(dd, "parts", "H4_SK.FAILED.json"))
            row["H4_SK_partition"] = "ABSENT (%s)" % (failed and failed.get("STATUS"))
        cache = os.path.join(dd, "replay_cache.npz")
        cman = rj(cache[:-4] + ".json")
        row["production_replay_cache_matches_manifest"] = None if not (cman and os.path.exists(cache)) else bool(cman["sha256"] == sha_file(cache))
        row["experimental_files_written"] = ["parts/LOWMEM__FREIGHT_con.npy", "parts/LOWMEM__FREIGHT_con.json", "replay_cache__LOWMEM__FREIGHT_con.npz",
                                             "replay_cache__LOWMEM__FREIGHT_con.json"]
        out[ds] = row
    return out


def review_items(blocks, label):
    """Facts the reviewer needs next to the single classification -- nothing here changes the label."""
    items = []
    for b in blocks:
        d, g5 = b["downstream"], b["gates"]["5_downstream"]
        if d["status"] == "PAIRED":
            p = d["paired"]
            items.append("%s: paired on %d identical query IDs; SAFE %+.4f (p=%s, sig %s, +%d/-%d), BASE %+.4f (p=%s, sig %s, +%d/-%d)%s" % (
                b["dataset"], p["nq"], p["SAFE_ALL_P50"]["delta_B_minus_A"], p["SAFE_ALL_P50"]["mcnemar_p"], p["SAFE_ALL_P50"]["sig"],
                p["SAFE_ALL_P50"]["B_only_covered"], p["SAFE_ALL_P50"]["A_only_covered"], p["BASE_ALL_P50"]["delta_B_minus_A"], p["BASE_ALL_P50"]["mcnemar_p"],
                p["BASE_ALL_P50"]["sig"], p["BASE_ALL_P50"]["B_only_covered"], p["BASE_ALL_P50"]["A_only_covered"],
                " -- this is the classification trigger" if (g5["SAFE_sig"] and g5["SAFE_delta"] < 0) or (g5["BASE_sig"] and g5["BASE_delta"] < 0) else ""))
            if d.get("by_hop_paired"):
                sig_hops = ["%s SAFE %+.4f p=%s" % (h, v["SAFE"]["delta_B_minus_A"], v["SAFE"]["mcnemar_p"]) for h, v in d["by_hop_paired"].items() if v["SAFE"]["sig"]]
                items.append("%s hop-wise (reported, not gated by the preregistered rule): significant hops = %s; others n.s." % (b["dataset"], sig_hops or "none"))
        else:
            items.append("%s: %s -- FREIGHT-arm numbers only (BASE %s, SAFE %s); the Mt-KaHyPar baseline is %s" % (
                b["dataset"], d["status"], fmt(d.get("B", {}).get("BASE_ALL_P50")), fmt(d.get("B", {}).get("SAFE_ALL_P50")), b["A"]["status"]))
        if b.get("objective_ratio_B_over_A"):
            items.append("%s objective diagnostic: FREIGHT KM1 = %.3fx Mt-KaHyPar (STRUCT cut delta %+.4f, KNN cut delta %+.4f); not a gate" % (
                b["dataset"], b["objective_ratio_B_over_A"], b["struct_cut_delta_B_minus_A"], b["knn_cut_delta_B_minus_A"]))
    def rss_a(b):
        if b["A"].get("peak_rss_kb"):
            return mb(b["A"]["peak_rss_kb"])
        return ">%s (killed)" % mb(b["A"]["retry"]["attempts"][-1]["peak_rss_kb_at_kill"]) if b["A"].get("retry") else str(b["A"].get("status"))
    items.append("structure exact on %s; exact restart equivalence on %s; refusals all refused on %s; peak RSS FREIGHT %s vs Mt-KaHyPar %s" % (
        [b["dataset"] for b in blocks if b["structure_exact"] is True], [b["dataset"] for b in blocks if b["gates"]["2b_exact_restart_equivalence"]],
        [b["dataset"] for b in blocks if b["B"].get("refusals_all_refused")], " / ".join(mb(b["B"].get("peak_rss_kb")) for b in blocks),
        " / ".join(rss_a(b) for b in blocks)))
    items.append("classification %s follows the rule as written before the downstream numbers existed; no threshold was added or moved afterwards" % label)
    return items


def main():
    blocks = [dataset_block(ds) for ds in LOCAL_SETS]
    label, rule = classify(blocks)
    R = {"RECORD": "H4_SK_LOW_MEMORY_PARTITIONING_REPORT", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "scope": "experimental validation on the three local datasets (squad, metaqa, musique); canonical L1 manifests / partitions untouched; "
                  "STOP_FOR_REVIEW before WebQSP / HotpotQA / 2Wiki",
         "contract_freeze": pin(os.path.join(OUT, "H4_SK_CONTRACT_FREEZE.json")),
         "build": rj(os.path.join(OUT, "FREIGHT_BUILD.json")) and {k: v for k, v in rj(os.path.join(OUT, "FREIGHT_BUILD.json")).items() if k in ("toolchain", "upstream", "binaries", "utc")},
         "local_ram_gb": LOCAL_RAM_GB,
         "datasets": blocks,
         "CLASSIFICATION": label, "classification_rule": rule,
         "audits": {"zoltan_phg": rj(os.path.join(OUT, "AUDIT_ZOLTAN_PHG.json")) or "PENDING",
                    "mtkahypar_out_of_core": rj(os.path.join(OUT, "AUDIT_MTKAHYPAR_OUTOFCORE.json")) or "PENDING"},
         "invariants": ["identical H4_SK hypergraph (structure digest, byte-identical net-list)", "objective family connectivity / KM1", "k = max(1, N//100)",
                        "imbalance 0.03", "canonical node-position stream order, no shuffle", "downstream mechanism unchanged (canonical replay + l1_eval numerics)",
                        "block labels never compared", "no tuning from evaluation outcomes"],
         "mtkahypar_retry_records": {ds: rj(os.path.join(OUT, "%s_MTKAHYPAR_RETRY.json" % ds.upper())) for ds in LOCAL_SETS
                                     if rj(os.path.join(OUT, "%s_MTKAHYPAR_RETRY.json" % ds.upper()))},
         "table_markdown": table(blocks), "table_p50_postprocessing_markdown": table_p50(blocks),
         "canonical_untouched": canonical_untouched(),
         "promotion": "NOT PROMOTED -- FREIGHT stays an EXPERIMENTAL arm (parts/LOWMEM__FREIGHT_con.*); the canonical L1 partitioner contract is unchanged",
         "STOP_FOR_REVIEW": True,
         "review_items": review_items(blocks, label)}
    wj(os.path.join(OUT, "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json"), R)
    md = ["# H4_SK low-memory partitioning validation (FREIGHT vs canonical Mt-KaHyPar)", "", "Generated %s -- CLASSIFICATION: **%s**" % (R["utc"], label), "",
          R["table_markdown"], "", "P50 post-processing statistics (canonical replay + l1_eval numerics; nothing tuned):", "", R["table_p50_postprocessing_markdown"], "",
          "Rule: " + rule, "",
          "Not promoted: FREIGHT stays EXPERIMENTAL (parts/LOWMEM__FREIGHT_con.*); canonical H4_SK manifests and partitions are unchanged. "
          "STOP_FOR_REVIEW before WebQSP / HotpotQA / 2Wiki.", "", "Review items:", ""]
    md += ["- " + it for it in R["review_items"]]
    md += ["", "Canonical untouched (hashes re-read at report time): %s" % R["canonical_untouched"], ""]
    for b in blocks:
        md.append("## %s" % b["dataset"])
        md.append("- structure: ORIGINAL %s, shards gate %s, official-bytes gate %s" % ((b["structure"]["ORIGINAL_STRUCTURE_SHA256"] or "-")[:16],
                                                                                        b["structure"]["gate_digest(shards)"], b["structure"]["gate_bytes(official net-list)"]))
        if b["A"].get("retry"):
            rt = b["A"]["retry"]
            md.append("- Mt-KaHyPar baseline: %s -- %s; attempts (threads / wall s / peak RSS kB at kill / host avail GB at start): %s; cap %.1f GB" % (
                rt["CLASSIFICATION"], rt["ruling"], ["%s / %s / %s / %s" % (a["threads"], a["wall_seconds"], a["peak_rss_kb_at_kill"], a["host_available_gb_at_start"])
                                                     for a in rt["attempts"]], rt["rss_cap_gb"]))
        md.append("- experiment 1 (exact restart equivalence): %s; runs identical to A: %s; refusals: %s; kill test: %s; 2-pass: %s" % (
            b["B"].get("experiment_1"), b["B"].get("runs_identical_to_A"), b["B"].get("refusals_all_refused"), b["B"].get("kill_test"), b["B"].get("two_pass_smoke")))
        md.append("- gates: %s" % {k: v for k, v in b["gates"].items() if k != "4_quality_reported_not_gated"})
        md.append("- quality (reported, not gated): %s" % b["gates"]["4_quality_reported_not_gated"])
        for arm in ("A", "B"):
            s = b["downstream"].get(arm + "_p50_postprocessing")
            if s:
                bl, sa = s["BALANCE"], s["SAFE_additions"]
                md.append("- P50 post-processing %s: blocks used %s/%s, size min/median/mean/p90/p99/max %s/%s/%s/%s/%s/%s, cv %s, max/mean %s, ELIGIBLE %s; "
                          "BASE ALL/ANY %s/%s scope %s; SAFE ALL/ANY %s/%s scope %s; BND %s; SAFE additions +%s/-%s net %s p=%s sig=%s, gold admitted/evicted %s/%s, "
                          "churn/q %s (max %s, zero-churn queries %s)" % (
                              {"A": "Mt-KaHyPar", "B": "FREIGHT"}[arm], bl["blocks_used"], bl["npart"], bl["size_min"], bl["size_median"], bl["size_mean"], bl["size_p90"],
                              bl["size_p99"], bl["size_max"], bl["size_cv"], bl["max_over_mean"], bl["ELIGIBLE"], fmt(b["downstream"][arm]["BASE_ALL_P50"]),
                              fmt(s["BASE_ANY_P50"]), s["BASE_SCOPE_NODES"], fmt(b["downstream"][arm]["SAFE_ALL_P50"]), fmt(s["SAFE_ANY_P50"]), s["SAFE_SCOPE_NODES"],
                              fmt(s["BND_ALL_P50"]), sa["gained"], sa["lost"], sa["net"], sa["mcnemar_p"], sa["sig"], sa["gold_admitted"], sa["gold_evicted"],
                              sa["churn_per_query"], sa["churn_max"], sa["queries_with_zero_churn"]))
                if s.get("by_hop"):
                    md.append("  - hop-wise %s: %s" % ({"A": "Mt-KaHyPar", "B": "FREIGHT"}[arm], "; ".join("%s (n %d) BASE %.4f SAFE %.4f" % (h, v["n"], v["BASE_ALL"], v["SAFE_ALL"])
                                                                                                   for h, v in s["by_hop"].items())))
        if b["downstream"].get("paired"):
            p = b["downstream"]["paired"]
            md.append("- downstream paired (nq %d): SAFE A %.4f B %.4f d %+.4f p=%s sig=%s (+%d/-%d); BASE A %.4f B %.4f d %+.4f p=%s" % (
                p["nq"], p["SAFE_ALL_P50"]["A"], p["SAFE_ALL_P50"]["B"], p["SAFE_ALL_P50"]["delta_B_minus_A"], p["SAFE_ALL_P50"]["mcnemar_p"], p["SAFE_ALL_P50"]["sig"],
                p["SAFE_ALL_P50"]["B_only_covered"], p["SAFE_ALL_P50"]["A_only_covered"], p["BASE_ALL_P50"]["A"], p["BASE_ALL_P50"]["B"],
                p["BASE_ALL_P50"]["delta_B_minus_A"], p["BASE_ALL_P50"]["mcnemar_p"]))
            cv, sa = b["downstream"].get("coverage_paired") or {}, b["downstream"].get("safe_additions_paired") or {}
            if cv and sa:
                md.append("- candidate coverage A vs B: BASE ANY %s vs %s, SAFE ANY %s vs %s, BASE scope %s vs %s, SAFE scope %s vs %s; SAFE additions A +%s/-%s (net %s) vs B +%s/-%s (net %s)" % (
                    fmt(cv["BASE_ANY"]["A"]), fmt(cv["BASE_ANY"]["B"]), fmt(cv["SAFE_ANY"]["A"]), fmt(cv["SAFE_ANY"]["B"]), cv["BASE_SCOPE_NODES"]["A"], cv["BASE_SCOPE_NODES"]["B"],
                    cv["SAFE_SCOPE_NODES"]["A"], cv["SAFE_SCOPE_NODES"]["B"], sa["A"]["gained"], sa["A"]["lost"], sa["A"]["net"], sa["B"]["gained"], sa["B"]["lost"], sa["B"]["net"]))
            if b["downstream"].get("by_hop_paired"):
                for h, v in b["downstream"]["by_hop_paired"].items():
                    md.append("  - %s (n %d): SAFE A %.4f B %.4f d %+.4f p=%s; BASE d %+.4f p=%s" % (h, v["n"], v["SAFE"]["A"], v["SAFE"]["B"], v["SAFE"]["delta_B_minus_A"],
                                                                                                 v["SAFE"]["mcnemar_p"], v["BASE"]["delta_B_minus_A"], v["BASE"]["mcnemar_p"]))
        else:
            md.append("- downstream: %s" % b["downstream"].get("status"))
        md.append("")
    with io.open(os.path.join(OUT, "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("CLASSIFICATION", label)
    print(R["table_markdown"])
    return R


if __name__ == "__main__":
    main()
