"""Follow-up arm H4_SK_FREIGHT_RESTREAM_VCYCLE_2PASS -- one fixed configuration, no sweeps (spec: user's decision tree after the
frozen LOW_MEMORY_BUT_QUALITY_FAIL classification).

Reason (preregistered, verbatim): single-pass streaming produced substantially worse partition objective/cut quality; this follow-up
tests FREIGHT's documented restreaming refinement mechanism.

    python -u src/l1_lowmem/freight_restream.py prereg               -> results/L1_LOWMEM/FREIGHT_RESTREAM_PREREG.json (before any run)
    python -u src/l1_lowmem/freight_restream.py run squad metaqa     -> results/L1_LOWMEM/FREIGHT_RESTREAM_RUNS_<ds>.json (+ downstream when the
                                                                        partition differs from the frozen 1-pass arm)
    python -u src/l1_lowmem/freight_restream.py report               -> results/L1_LOWMEM/FREIGHT_RESTREAM_REPORT.{json,md}
Everything of the frozen experiment (results/L1_LOWMEM/H4_SK_LOW_MEMORY_PARTITIONING_REPORT.*, FREIGHT_RUNS_<ds>.json, L1_DOWNSTREAM_<ds>.json,
data/l1_lowmem/<ds>/freight/*, parts/LOWMEM__FREIGHT_con.*) is read, never written.  New files use the suffix / tag `restream2`.
"""
import io
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, sha_file, rj, wj, pin, ds_dir, wsl_path, wsl  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402

ARM = "H4_SK_FREIGHT_RESTREAM_VCYCLE_2PASS"
PASSES = 2
EXTRA_FLAGS = "--restream_vcycle"
TAG2 = "LOWMEM__FREIGHT_con_restream2"
SUFFIX = "__restream2"
SUBDIR = "freight_restream2"
CKPT_HOME2 = "~/l1_lowmem_ckpt_restream2"
DATASETS = ["squad", "metaqa"]
FROZEN = ["H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json", "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.md", "FREIGHT_RUNS_squad.json", "FREIGHT_RUNS_metaqa.json",
          "FREIGHT_RUNS_musique.json", "L1_DOWNSTREAM_squad.json", "L1_DOWNSTREAM_metaqa.json", "L1_DOWNSTREAM_musique.json", "FREIGHT_BUILD.json",
          "MUSIQUE_MTKAHYPAR_RETRY.json"]

STOP_RULE = ("FREIGHT_CLOSED if, on squad or metaqa, (a) the restream partition is byte-identical to the frozen 1-pass FREIGHT partition (the frozen "
             "paired result then stands unchanged), or (b) the canonical paired downstream evaluation shows a SIGNIFICANT (exact two-sided McNemar "
             "p<0.05) SAFE_ALL_P50 or BASE_ALL_P50 loss vs Mt-KaHyPar, or (c) the output violates the imbalance contract (max block > ceil(1.03 N/k)).  "
             "FREIGHT_RESTREAM_QUALITY_PASS if both datasets are paired with no significant loss.  EXACTNESS_FAIL (engineering, separate from quality) "
             "if A != A_forklib or the sharded / checkpointed / killed / aborted runs differ from A.  Objective and cut ratios vs Mt-KaHyPar are reported "
             "as diagnostics; no numeric objective threshold is preregistered (whether the gap is 'materially closed' is the reviewer's call).  "
             "No pass-count, seed or order sweep follows a failure.")


def prereg():
    pre_path = os.path.join(OUT, "FREIGHT_RESTREAM_PREREG.json")
    if os.path.exists(pre_path):
        raise RuntimeError("preregistration already exists -- it is written once, before any run")
    b = FR.build_record()
    rec = {"RECORD": "FREIGHT_RESTREAM_PREREG", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": ARM,
           "reason": "single-pass streaming produced substantially worse partition objective/cut quality; this follow-up tests FREIGHT's documented "
                     "restreaming refinement mechanism",
           "not_the_reason": "not chosen because SQuAD lost accuracy; the configuration is the smallest documented quality-enhancing step beyond the one that failed",
           "config": {"families": "STRUCT + KNN (frozen H4_SK, unchanged)", "objective": "connectivity / KM1 (freight_con, unchanged)", "k": "max(1, N//100) (unchanged)",
                      "imbalance": "3 % = eps 0.03 (unchanged)", "node_order": "canonical node-position order, H4_SK_STREAM_V1 shards (unchanged)", "seed": FR.SEED,
                      "num_streams_passes": PASSES, "restream_vcycle": True, "flags": "--num_streams_passes=%d %s" % (PASSES, EXTRA_FLAGS)},
           "datasets": DATASETS, "runs": ["A (stock, monolithic)", "A_forklib", "B (sharded, no checkpoints)", "B0 (checkpoint every shard)",
                                          "C_kill (SIGKILL in pass 1 at the middle shard, resume)", "C_abort (mid-shard abort in pass 1, resume)"],
           "no_sweep": "no 2/3/4/8-pass sweep, no seed sweep, no order sweep; exactly this one arm on exactly these two datasets",
           "stop_rule": STOP_RULE,
           "binaries": {k: b["binaries"][k] for k in ("freight_con", "freight_con_ckpt", "fork_freight_con_unpatched_driver")}, "upstream_commit": b["upstream"]["commit"],
           "source_facts_at_pinned_commit": {
               "num_streams_passes_implies_restream_vcycle": "code_for_hypergraphs/app/parse_parameters.h:1695-1698 -- `if (num_streams_passes->count > 0) "
                                                             "{ partition_config.restream_vcycle = true; partition_config.num_streams_passes = ...; }`",
               "restream_vcycle_flag": "parse_parameters.h:1704-1711 sets the same flag (refused with --full_stream_mode); README lines 92-93 document both",
               "flag_use_in_hypergraph_code": "`restream_vcycle` is not referenced in code_for_hypergraphs/app/freight.cpp or lib/ (only parse/config); restreaming "
                                              "is driven by restream_number: pass p>0 removes each node from its previous block before re-scoring "
                                              "(freight.cpp:196-206), keeps the per-net last-pin block carry-over in connectivity mode (graph_io_stream.h:531-534), "
                                              "evaluates each pass and RESTORES THE BEST PASS (freight.cpp:272-331); non-final passes use a relaxed upper bound "
                                              "ceil((1+1.5*imb) N/k) (graph_io_stream.h:552-558), the final pass ceil((1+imb) N/k)",
               "consequence": "the arm is exactly --num_streams_passes=2 (+ the explicit, implied flag); the imbalance contract must be checked on the OUTPUT "
                              "because a restored pass-0 partition was built under the relaxed bound"},
           "known_prior": {"note": "the frozen experiment's 2-pass smoke test on squad (P2_A: stock, --num_streams_passes=2, same seed/k/imbalance) already exists",
                           "P2_A_partition_sha256": sha_file(os.path.join(ds_dir("squad"), "freight", "P2_A", "partition.txt")),
                           "one_pass_A_partition_sha256": sha_file(os.path.join(ds_dir("squad"), "freight", "A", "partition.txt")),
                           "read_before_running": "the two files hash equal -> on squad the 2-pass output is expected to be byte-identical to the 1-pass output; "
                                                  "the arm is still run as specified (and the explicit flag added); metaqa is new"},
           "frozen_records": {fn: pin(os.path.join(OUT, fn)) for fn in FROZEN if os.path.exists(os.path.join(OUT, fn))},
           "frozen_rule": "the frozen classification LOW_MEMORY_BUT_QUALITY_FAIL is permanent; none of the frozen records is rewritten or reinterpreted by this arm"}
    wj(pre_path, rec)
    log("preregistered", ARM, "->", os.path.relpath(pre_path, REPO))
    return rec


def run_dir2(ds, name):
    import shutil
    d = os.path.join(ds_dir(ds), SUBDIR, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def args2(k, out):
    return "%s %s" % (FR.base_args(k, PASSES, out), EXTRA_FLAGS)


def ckpt_meta(path):
    """key=value META lines of one checkpoint file (text block right after the magic)."""
    r = wsl("head -c 2000 %s | strings" % path)
    out = {}
    for ln in r.stdout.split("\n"):
        m = re.match(r"^([A-Za-z0-9_.]+)=(.*)$", ln.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def run(ds):
    pre = rj(os.path.join(OUT, "FREIGHT_RESTREAM_PREREG.json"))
    if pre is None:
        raise RuntimeError("run `prereg` first")
    if ds not in DATASETS:
        raise RuntimeError("%s is not in the preregistered dataset list %s" % (ds, DATASETS))
    b = FR.build_record()
    for name in ("freight_con", "freight_con_ckpt", "fork_freight_con_unpatched_driver"):
        if b["binaries"][name]["sha256"] != pre["binaries"][name]["sha256"]:
            raise RuntimeError("binary %s changed since preregistration" % name)
    R1 = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))
    if R1 is None:
        raise RuntimeError("%s: frozen 1-pass record missing" % ds)
    H, man, gate, off, netl = FR.load_inputs(ds)
    N, k, M, P = H["N"], H["k"], H["M"], H["P"]
    stock, fork = b["binaries"]["freight_con"], b["binaries"]["freight_con_ckpt"]
    smf = os.path.join(ds_dir(ds), "freight", "stream_manifest.txt")
    if sha_file(smf) != R1["inputs"]["stream_manifest"]["sha256"]:
        raise RuntimeError("%s: stream manifest changed since the frozen run" % ds)
    h = FR.home()
    ckhome = "%s/%s" % (CKPT_HOME2.replace("~", h), ds)
    FR.sh("rm -rf %s && mkdir -p %s" % (ckhome, ckhome), quiet=True)
    meta = {"dataset": ds, "binary_sha256": fork["sha256"], "stream_manifest_sha256": R1["inputs"]["stream_manifest"]["sha256"],
            "structure_sha256": man["ORIGINAL_STRUCTURE_SHA256"], "upstream_commit": b["upstream"]["commit"], "arm": ARM}
    shard_first = [c["first_position"] for c in man["shards"]]
    S = len(man["shards"])
    T = S // 2
    abort_node = shard_first[T] + man["shard_nodes"] // 2
    log("=== %s %s: N %d M %d P %d k %d shards %d (pass-1 crash target shard %d, abort node %d) ===" % (ARM, ds, N, M, P, k, S, T, abort_node))
    R = {"RECORD": "FREIGHT_RESTREAM_RUNS", "arm": ARM, "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "preregistration": pin(os.path.join(OUT, "FREIGHT_RESTREAM_PREREG.json")),
         "inputs": {"N": N, "M": M, "P": P, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl": off["output"],
                    "stream_manifest": R1["inputs"]["stream_manifest"], "frozen_one_pass_record": pin(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))},
         "contract": {"objective": "connectivity (weighted KM1)", "imbalance_percent": FR.IMBALANCE, "seed": FR.SEED, "num_streams_passes": PASSES,
                      "restream_vcycle": True, "flags": "--num_streams_passes=%d %s" % (PASSES, EXTRA_FLAGS), "k": k,
                      "binaries": {"stock": stock, "fork": fork}, "upstream_commit": b["upstream"]["commit"]},
         "runs": {}, "crash": {"pass": 1, "target_shard": T, "abort_after_node": abort_node}}
    parts = {}
    d = run_dir2(ds, "A"); pA = os.path.join(d, "partition.txt")
    R["runs"]["A"] = FR.exec_run("A", "%s %s %s" % (stock["path"], wsl_path(netl), args2(k, pA)), d)
    parts["A"] = FR.read_partition(pA, N, k)
    d = run_dir2(ds, "A_forklib"); pAf = os.path.join(d, "partition.txt")
    R["runs"]["A_forklib"] = FR.exec_run("A_forklib", "%s %s %s" % (b["binaries"]["fork_freight_con_unpatched_driver"]["path"], wsl_path(netl), args2(k, pAf)), d)
    parts["A_forklib"] = FR.read_partition(pAf, N, k)
    d = run_dir2(ds, "B"); pB = os.path.join(d, "partition.txt"); ck = "%s/B" % ckhome
    FR.sh("mkdir -p %s" % ck, quiet=True)
    R["runs"]["B"] = FR.exec_run("B", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=0 %s" % (
        fork["path"], wsl_path(smf), args2(k, pB), ck, FR.meta_args(meta)), d)
    R["runs"]["B"]["arm"] = "sharded stream (H4_SK_STREAM_V1 shards), same persistent process/state, no checkpoints written"
    parts["B"] = FR.read_partition(pB, N, k)
    d = run_dir2(ds, "B0"); pB0 = os.path.join(d, "partition.txt"); ck0 = "%s/B0" % ckhome
    FR.sh("mkdir -p %s" % ck0, quiet=True)
    R["runs"]["B0"] = FR.exec_run("B0", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 %s" % (
        fork["path"], wsl_path(smf), args2(k, pB0), ck0, FR.meta_args(meta)), d)
    R["runs"]["B0"]["ckpt_dir"] = FR.ckpt_listing(ck0)
    parts["B0"] = FR.read_partition(pB0, N, k)
    # per-pass diagnostics from the B0 checkpoints (META text): last pass-0 and last pass-1 checkpoint
    passes_meta = {}
    for pss in range(PASSES):
        names = sorted(f["name"] for f in R["runs"]["B0"]["ckpt_dir"]["files"] if re.match(r"checkpoint_%d_\d+\.bin$" % pss, f["name"]))
        if names:
            mm = ckpt_meta("%s/%s" % (ck0, names[-1]))
            passes_meta["pass%d" % pss] = {"checkpoint": names[-1], **{kk: mm.get(kk) for kk in ("next_node", "nodes_moved", "best_objective", "stream_total_upperbound",
                                                                                              "block0_capacity", "processed_pins")}}
    R["per_pass_from_checkpoints"] = passes_meta
    # C_kill: SIGKILL once LATEST names a pass-1 checkpoint with shard index >= T, then resume
    d = run_dir2(ds, "C_kill"); pCk = os.path.join(d, "partition.txt"); ckk = "%s/C_kill" % ckhome
    watcher = ("rm -rf {ck} && mkdir -p {ck}; ( exec {bin} {smf} {args} --ckpt_manifest --ckpt_dir={ck} --ckpt_every_shards=1 {meta} "
               "> {so} 2> {se} ) & WP=$!; killed=0; seen=; "
               "while kill -0 $WP 2>/dev/null; do if [ -f {ck}/LATEST ]; then name=$(cut -d' ' -f1 {ck}/LATEST); ps=${{name#checkpoint_}}; ps=${{ps%%_*}}; "
               "idx=${{name##*_}}; idx=${{idx%.bin}}; idx=$((10#$idx)); seen=${{ps}}:${{idx}}; if [ $ps -ge 1 ] && [ $idx -ge {T} ]; then kill -9 $WP 2>/dev/null && killed=1; break; fi; fi; "
               "sleep 0.005; done; wait $WP; echo wrapper_rc=$? killed=$killed last_seen=$seen").format(
        ck=ckk, bin=fork["path"], smf=wsl_path(smf), args=args2(k, pCk), meta=FR.meta_args(meta), so=wsl_path(os.path.join(d, "stdout_killed.txt")),
        se=wsl_path(os.path.join(d, "stderr_killed.txt")), T=T)
    r = wsl(watcher, timeout=7200)
    kl = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
    m = re.match(r"wrapper_rc=(\d+) killed=(\d) last_seen=(\S*)", kl)
    R["runs"]["C_kill"] = {"name": "C_kill", "watcher_result": kl, "killed": bool(m and m.group(2) == "1"),
                           "killed_stdout": FR.parse_freight_stdout(os.path.join(d, "stdout_killed.txt")), "ckpt_dir_after_kill": FR.ckpt_listing(ckk)}
    if R["runs"]["C_kill"]["killed"]:
        R["runs"]["C_kill"]["resume"] = FR.exec_run("C_kill", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (
            fork["path"], wsl_path(smf), args2(k, pCk), ckk, FR.meta_args(meta)), d)
        parts["C_kill"] = FR.read_partition(pCk, N, k)
    else:
        R["runs"]["C_kill"]["status"] = "KILL_MISSED (process finished before LATEST reached pass 1 shard %d)" % T
        log("  C_kill    KILL MISSED (%s)" % kl)
    # C_abort: deterministic mid-shard abort in pass 1, then resume
    d = run_dir2(ds, "C_abort"); pCa = os.path.join(d, "partition.txt"); cka = "%s/C_abort" % ckhome
    FR.sh("rm -rf %s && mkdir -p %s" % (cka, cka), quiet=True)
    da = os.path.join(d, "aborted"); os.makedirs(da)
    R["runs"]["C_abort"] = {"name": "C_abort"}
    R["runs"]["C_abort"]["aborted"] = FR.exec_run("C_abort~", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_abort_after_node=%d --ckpt_abort_in_pass=1 %s" % (
        fork["path"], wsl_path(smf), args2(k, pCa), cka, abort_node, FR.meta_args(meta)), da, expect_rc=137)
    R["runs"]["C_abort"]["ckpt_dir_after_abort"] = FR.ckpt_listing(cka)
    R["runs"]["C_abort"]["partition_written_by_aborted_run"] = os.path.exists(pCa)
    R["runs"]["C_abort"]["resume"] = FR.exec_run("C_abort", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (
        fork["path"], wsl_path(smf), args2(k, pCa), cka, FR.meta_args(meta)), d)
    parts["C_abort"] = FR.read_partition(pCa, N, k)
    # exactness
    E = {"reference": "A", "compared": {}, "ALL_IDENTICAL": True}
    mA = FR.km1_metrics(H, parts["A"])
    for nm, arr in parts.items():
        same = bool(np.array_equal(arr, parts["A"]))
        E["compared"][nm] = {"identical_to_A": same, "n_diff": 0 if same else int((arr != parts["A"]).sum())}
        E["ALL_IDENTICAL"] &= same
    E["A_eq_A_forklib"] = E["compared"]["A_forklib"]["identical_to_A"]
    R["EQUIVALENCE"] = E
    R["VERDICT_EXACTNESS"] = "EXACT_RESTART_EQUIVALENCE" if E["ALL_IDENTICAL"] and "C_kill" in parts else ("KILL_MISSED" if E["ALL_IDENTICAL"] else "EXACTNESS_FAIL")
    # metrics of the arm and the two frozen references (read-only)
    mA["family_cuts"] = FR.family_cuts(ds, parts["A"])
    R["metrics"] = mA
    R["partition_sha256"] = sha_file(pA)
    one = {"partition_sha256": sha_file(os.path.join(ds_dir(ds), "freight", "A", "partition.txt")), "km1_weighted": R1["metrics_B"]["km1_weighted"],
           "cut_weighted": R1["metrics_B"]["cut_weighted"], "blocks": R1["metrics_B"]["blocks"], "family_cuts": R1["metrics_B"]["family_cuts"],
           "peak_rss_kb_B": R1["runs"]["B"]["time"].get("peak_rss_kb"), "wall_seconds_B": R1["runs"]["B"]["time"].get("wall_seconds")}
    R["frozen_one_pass_arm"] = one
    R["identical_to_frozen_one_pass_partition"] = bool(R["partition_sha256"] == one["partition_sha256"]
                                                       and np.array_equal(parts["A"], FR.read_partition(os.path.join(ds_dir(ds), "freight", "A", "partition.txt"), N, k)))
    bm = R1.get("baseline_mtkahypar") or {}
    R["mtkahypar_baseline"] = ({"status": bm.get("status"), "km1_weighted": bm.get("km1_weighted"), "cut_weighted": bm.get("cut_weighted"), "blocks": bm.get("blocks"),
                                "family_cuts": bm.get("family_cuts")} if bm.get("status") == "OK" else {"status": bm.get("status")})
    if bm.get("status") == "OK":
        R["diagnostic_vs_mtkahypar"] = {"objective_ratio": round(mA["km1_weighted"] / float(bm["km1_weighted"]), 4),
                                        "one_pass_objective_ratio": round(one["km1_weighted"] / float(bm["km1_weighted"]), 4),
                                        "struct_cut_delta": round(mA["family_cuts"]["STRUCT"]["edge_cut_fraction"] - bm["family_cuts"]["STRUCT"]["edge_cut_fraction"], 4),
                                        "knn_cut_delta": round(mA["family_cuts"]["KNN"]["edge_cut_fraction"] - bm["family_cuts"]["KNN"]["edge_cut_fraction"], 4)}
    R["diagnostic_vs_one_pass"] = {"objective_ratio_restream_over_one_pass": round(mA["km1_weighted"] / float(one["km1_weighted"]), 6),
                                   "n_positions_changed": int((parts["A"] != FR.read_partition(os.path.join(ds_dir(ds), "freight", "A", "partition.txt"), N, k)).sum())}
    R["imbalance_contract_on_output"] = {"max_block": mA["blocks"]["max"], "bound_ceil_1.03_N_over_k": int(np.ceil(1.03 * N / float(k))), "within": mA["blocks"]["within_eps_0.03"]}
    log("%-7s %s: exactness %s; km1 %d (1-pass %d, x%.4f; Mt-KaHyPar %s); identical to 1-pass partition: %s; max block %d (bound %d); pass1 nodes_moved %s" % (
        ds, ARM, R["VERDICT_EXACTNESS"], mA["km1_weighted"], one["km1_weighted"], R["diagnostic_vs_one_pass"]["objective_ratio_restream_over_one_pass"],
        bm.get("km1_weighted"), R["identical_to_frozen_one_pass_partition"], mA["blocks"]["max"], R["imbalance_contract_on_output"]["bound_ceil_1.03_N_over_k"],
        (passes_meta.get("pass1") or {}).get("nodes_moved")))
    # downstream: only when the partition differs from the frozen 1-pass arm (otherwise the frozen paired result is exactly the result)
    if R["identical_to_frozen_one_pass_partition"]:
        D1 = rj(os.path.join(OUT, "L1_DOWNSTREAM_%s.json" % ds))
        R["downstream"] = {"status": "IDENTICAL_PARTITION_NOT_RERUN", "cites": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s.json" % ds)),
                           "frozen_paired": D1 and D1.get("paired"), "frozen_status": D1 and D1.get("status")}
    elif E["ALL_IDENTICAL"] and mA["blocks"]["within_eps_0.03"]:
        from src.l1_lowmem import l1_downstream as LD
        R["import"] = FR.import_partition(ds, H, parts["A"], {"metrics_B": mA, "runs": R["runs"], "VERDICT_EXPERIMENT_1": R["VERDICT_EXACTNESS"],
                                                              "EQUIVALENCE": E, "contract": R["contract"], "inputs": R["inputs"]}, b, off, man, tag=TAG2)
        R["import"]["arm"] = ARM
        R["import"]["contract"]["algorithm"] = ("two-pass restreaming FREIGHT (--num_streams_passes=2 --restream_vcycle): pass 1 removes each node from its "
                                                "pass-0 block and re-scores it; the pass with the lower objective is restored")
        R["import"]["contract"]["worker_note"] = "fork binary (freight_con_ckpt) driven by src/l1_lowmem/freight_restream.py; A == A_forklib == B position-for-position"
        wj(os.path.join(H["d"].derived_dir, "parts", "%s.json" % TAG2), R["import"])
        D = LD.run(ds, tag=TAG2, suffix=SUFFIX)
        R["downstream"] = {"status": D["status"], "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (ds, SUFFIX))), "paired": D.get("paired"),
                           "B": D.get("B"), "A": D.get("A"), "by_hop_paired": D.get("by_hop_paired"), "coverage": D.get("coverage"), "safe_additions": D.get("safe_additions")}
    else:
        R["downstream"] = {"status": "NOT_RUN (exactness fail or imbalance contract violated)"}
    fp = os.path.join(OUT, "FREIGHT_RESTREAM_RUNS_%s.json" % ds)
    wj(fp, R)
    log("wrote", os.path.relpath(fp, REPO))
    return R


def decide(recs):
    """the preregistered stop rule, applied verbatim."""
    reasons = []
    exact_fail = [ds for ds, R in recs.items() if R["VERDICT_EXACTNESS"] == "EXACTNESS_FAIL"]
    for ds, R in recs.items():
        if R["identical_to_frozen_one_pass_partition"]:
            reasons.append("%s: restream partition byte-identical to the frozen 1-pass partition (rule a)" % ds)
        if not R["imbalance_contract_on_output"]["within"]:
            reasons.append("%s: output violates the imbalance contract (rule c)" % ds)
        p = (R.get("downstream") or {}).get("paired")
        if p:
            for key in ("SAFE_ALL_P50", "BASE_ALL_P50"):
                if p[key]["sig"] and p[key]["delta_B_minus_A"] < 0:
                    reasons.append("%s: significant paired %s loss %+.4f (p=%s) (rule b)" % (ds, key, p[key]["delta_B_minus_A"], p[key]["mcnemar_p"]))
    if reasons:
        label = "FREIGHT_CLOSED"
    elif all((R.get("downstream") or {}).get("status") == "PAIRED" for R in recs.values()) and len(recs) == len(DATASETS):
        label = "FREIGHT_RESTREAM_QUALITY_PASS"
    else:
        label = "INCOMPLETE"
    return label, reasons, exact_fail


def report():
    pre = rj(os.path.join(OUT, "FREIGHT_RESTREAM_PREREG.json"))
    recs = {ds: rj(os.path.join(OUT, "FREIGHT_RESTREAM_RUNS_%s.json" % ds)) for ds in DATASETS}
    recs = {ds: R for ds, R in recs.items() if R}
    label, reasons, exact_fail = decide(recs)
    rows = ["| Dataset | Arm | Exactness | Peak RSS | Time | KM1 | ratio vs Mt-KaHyPar | ratio vs 1-pass | STRUCT cut | KNN cut | max block (bound) | pass-1 nodes moved | Downstream |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, R in recs.items():
        m, dv, dm = R["metrics"], R["diagnostic_vs_one_pass"], R.get("diagnostic_vs_mtkahypar") or {}
        ds_status = R["downstream"]["status"]
        if ds_status == "PAIRED":
            p = R["downstream"]["paired"]
            ds_status = "PAIRED: SAFE %+.4f p=%s, BASE %+.4f p=%s" % (p["SAFE_ALL_P50"]["delta_B_minus_A"], p["SAFE_ALL_P50"]["mcnemar_p"],
                                                                      p["BASE_ALL_P50"]["delta_B_minus_A"], p["BASE_ALL_P50"]["mcnemar_p"])
        elif ds_status == "IDENTICAL_PARTITION_NOT_RERUN" and R["downstream"].get("frozen_paired"):
            p = R["downstream"]["frozen_paired"]
            ds_status = "identical partition -> frozen paired result stands: SAFE %+.4f p=%s, BASE %+.4f p=%s" % (
                p["SAFE_ALL_P50"]["delta_B_minus_A"], p["SAFE_ALL_P50"]["mcnemar_p"], p["BASE_ALL_P50"]["delta_B_minus_A"], p["BASE_ALL_P50"]["mcnemar_p"])
        rows.append("| %s | %s | %s | %.0f MB | %.1f s | %d | %s | %.4f | %.4f | %.4f | %d (%d) | %s | %s |" % (
            ds, "2-pass + restream_vcycle", R["VERDICT_EXACTNESS"], R["runs"]["B"]["time"].get("peak_rss_kb", 0) / 1024.0, R["runs"]["B"]["time"].get("wall_seconds", 0),
            m["km1_weighted"], dm.get("objective_ratio", "-"), dv["objective_ratio_restream_over_one_pass"], m["family_cuts"]["STRUCT"]["edge_cut_fraction"],
            m["family_cuts"]["KNN"]["edge_cut_fraction"], m["blocks"]["max"], R["imbalance_contract_on_output"]["bound_ceil_1.03_N_over_k"],
            (R["per_pass_from_checkpoints"].get("pass1") or {}).get("nodes_moved"), ds_status))
        bm = R.get("mtkahypar_baseline") or {}
        if bm.get("status") == "OK":
            rows.append("| %s | Mt-KaHyPar (frozen) | - | - | - | %d | 1.0 | - | %.4f | %.4f | %d (%d) | - | - |" % (
                ds, bm["km1_weighted"], bm["family_cuts"]["STRUCT"]["edge_cut_fraction"], bm["family_cuts"]["KNN"]["edge_cut_fraction"], bm["blocks"]["max"],
                R["imbalance_contract_on_output"]["bound_ceil_1.03_N_over_k"]))
        one = R["frozen_one_pass_arm"]
        rows.append("| %s | 1-pass (frozen) | - | %.0f MB | %.1f s | %d | %s | 1.0 | %.4f | %.4f | %d (%d) | - | frozen |" % (
            ds, (one.get("peak_rss_kb_B") or 0) / 1024.0, one.get("wall_seconds_B") or 0, one["km1_weighted"], dm.get("one_pass_objective_ratio", "-"),
            one["family_cuts"]["STRUCT"]["edge_cut_fraction"], one["family_cuts"]["KNN"]["edge_cut_fraction"], one["blocks"]["max"],
            R["imbalance_contract_on_output"]["bound_ceil_1.03_N_over_k"]))
    # integrity: nothing frozen was rewritten by this arm, the canonical partitions are the bytes the frozen report pinned
    frozen_report = rj(os.path.join(OUT, "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json")) or {}
    cu = frozen_report.get("canonical_untouched") or {}
    integrity = {"frozen_records_changed_since_prereg": [fn for fn, pn in pre["frozen_records"].items()
                                                         if not os.path.exists(os.path.join(OUT, fn)) or sha_file(os.path.join(OUT, fn)) != pn["sha256"]],
                 "canonical_H4_SK_partition": {}, "new_files_written_by_this_arm": []}
    from src.l1_canonical.adapter import CanonicalDataset
    for ds in recs:
        d = CanonicalDataset(ds)
        npy = os.path.join(d.derived_dir, "parts", "H4_SK.npy")
        now = sha_file(npy) if os.path.exists(npy) else None
        integrity["canonical_H4_SK_partition"][ds] = {"sha256_now": now, "sha256_pinned_by_frozen_report": (cu.get(ds) or {}).get("H4_SK_partition_sha256"),
                                                      "unchanged": bool(now and now == (cu.get(ds) or {}).get("H4_SK_partition_sha256"))}
        for fn in ("parts/%s.npy" % TAG2, "parts/%s.json" % TAG2):
            p = os.path.join(d.derived_dir, fn)
            if os.path.exists(p):
                integrity["new_files_written_by_this_arm"].append(os.path.relpath(p, REPO).replace("\\", "/"))
        for fn in ("replay_cache__%s.npz" % TAG2, "replay_cache__%s.json" % TAG2):
            p = os.path.join(ds_dir(ds), fn)
            if os.path.exists(p):
                integrity["new_files_written_by_this_arm"].append(os.path.relpath(p, REPO).replace("\\", "/"))
        integrity["new_files_written_by_this_arm"].append("data/l1_lowmem/%s/%s/{A,A_forklib,B,B0,C_kill,C_abort}/" % (ds, SUBDIR))
    for fn in ("FREIGHT_RESTREAM_PREREG.json", "FREIGHT_RESTREAM_RUNS_squad.json", "FREIGHT_RESTREAM_RUNS_metaqa.json", "L1_DOWNSTREAM_metaqa__restream2.json",
               "L1_REPLAY_metaqa__%s.json" % TAG2):
        if os.path.exists(os.path.join(OUT, fn)):
            integrity["new_files_written_by_this_arm"].append("results/L1_LOWMEM/" + fn)
    integrity["OK"] = not integrity["frozen_records_changed_since_prereg"] and all(v["unchanged"] for v in integrity["canonical_H4_SK_partition"].values())
    rep = {"RECORD": "FREIGHT_RESTREAM_REPORT", "arm": ARM, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "preregistration": pin(os.path.join(OUT, "FREIGHT_RESTREAM_PREREG.json")), "stop_rule": STOP_RULE,
           "DECISION": label, "reasons": reasons, "exactness_failures": exact_fail, "integrity": integrity,
           "post_hoc_observation_not_a_gate": (
               "metaqa: pass 1 lowered the H4-weighted KM1 by 4.2 % (57,790,340 -> 55,390,296) by cutting fewer KNN nets (KNN edge-cut 0.937 -> 0.894) "
               "at the price of more STRUCT cut (0.803 -> 0.814), and L1 coverage fell significantly, concentrated in hop3 (SAFE -0.066, BASE -0.069, sig; "
               "hop1/hop2 n.s.).  H4 weights max(1, rint(1000/(|e|-1))) make the small KNN nets worth ~1000 per cut and the large STRUCT nets a few units, "
               "so the streaming objective is dominated by KNN locality while the P50 coverage of multi-hop queries tracks STRUCT locality; Mt-KaHyPar "
               "reaches both (KNN 0.631, STRUCT 0.819).  Recorded as an observation after seeing the numbers; it changes no rule and motivates no re-run."),
           "datasets": {ds: {k2: R[k2] for k2 in ("VERDICT_EXACTNESS", "partition_sha256", "identical_to_frozen_one_pass_partition", "metrics", "frozen_one_pass_arm",
                                                  "mtkahypar_baseline", "diagnostic_vs_mtkahypar", "diagnostic_vs_one_pass", "imbalance_contract_on_output",
                                                  "per_pass_from_checkpoints", "downstream") if k2 in R} for ds, R in recs.items()},
           "frozen_classification_unchanged": "LOW_MEMORY_BUT_QUALITY_FAIL (results/L1_LOWMEM/H4_SK_LOW_MEMORY_PARTITIONING_REPORT.*, pinned in the preregistration)",
           "next_per_decision_tree": ("close FREIGHT permanently; next candidate = Zoltan-PHG on SQuAD / MetaQA (AUDIT_ZOLTAN_PHG.json: FEASIBLE_NOT_EXECUTED)"
                                      if label == "FREIGHT_CLOSED" else "user review"),
           "table_markdown": "\n".join(rows)}
    wj(os.path.join(OUT, "FREIGHT_RESTREAM_REPORT.json"), rep)
    md = ["# %s -- follow-up arm (one fixed configuration, no sweeps)" % ARM, "", "Generated %s -- DECISION: **%s**" % (rep["utc"], label), "",
          "Reason (preregistered): %s" % pre["reason"], "", rep["table_markdown"], "", "Stop rule (preregistered): " + STOP_RULE, "", "Reasons:", ""]
    md += ["- " + r for r in reasons] or ["- none"]
    md += ["", "Per-pass facts (fork checkpoints META, B0 run):"]
    for ds, R in recs.items():
        md.append("- %s: %s" % (ds, R["per_pass_from_checkpoints"]))
    md += ["", "Integrity: frozen records changed since preregistration = %s; canonical H4_SK partitions unchanged = %s; new files written by this arm: %s" % (
        integrity["frozen_records_changed_since_prereg"] or "none", all(v["unchanged"] for v in integrity["canonical_H4_SK_partition"].values()),
        ", ".join(integrity["new_files_written_by_this_arm"])), "", "Post-hoc observation (not a gate): " + rep["post_hoc_observation_not_a_gate"],
        "", "Frozen classification unchanged: LOW_MEMORY_BUT_QUALITY_FAIL.  Next per the decision tree: %s" % rep["next_per_decision_tree"], ""]
    with io.open(os.path.join(OUT, "FREIGHT_RESTREAM_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("DECISION", label, "| reasons:", reasons)
    print(rep["table_markdown"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "prereg":
        prereg()
    elif a[0] == "run":
        for ds in a[1:]:
            run(ds)
    elif a[0] == "report":
        report()
    else:
        print(__doc__)
