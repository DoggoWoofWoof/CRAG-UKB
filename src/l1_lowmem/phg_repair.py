"""PHG_NONEMPTY_REPAIR_V1 -- the one preregistered PHG follow-up: a generic deterministic adapter that converts an otherwise valid PHG k-way
assignment into the representation canonical L1 requires (every one of the k part ids non-empty), applied to the already frozen PHG R1
vectors.  PHG is NOT rerun and NOT retuned.

    python -u src/l1_lowmem/phg_repair.py prereg          -> results/L1_LOWMEM/PHG_REPAIR_PREREG.json (written once, before any repair)
    python -u src/l1_lowmem/phg_repair.py amend           -> supersedes PHG_RUNS_squad.json with the terminology amendment
                                                             (PHG_L1_NONINFERIOR -> PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS; original kept in _history/)
    python -u src/l1_lowmem/phg_repair.py run squad metaqa -> results/L1_LOWMEM/PHG_REPAIR_RUNS_<ds>.json
                                                             squad: expected no-op (0 moves, byte-identical vector)
                                                             metaqa: repair -> experimental import parts/LOWMEM__PHG_REPAIR1_con.* -> replay
                                                             cache -> paired canonical L1 vs the frozen Mt-KaHyPar replay (same 1,998 ids)
    python -u src/l1_lowmem/phg_repair.py report          -> supersedes PHG_REPORT.{json,md} (originals kept in _history/); STOP_FOR_REVIEW
Algorithm (for each empty block id in ascending order): candidates = nodes whose current (donor) block has >= 2 nodes; for each candidate
the exact weighted H4_SK KM1 delta of moving that single node into the empty block; pick the minimum delta, ties by canonical node position,
then donor block id; move exactly that node; update block sizes and hyperedge connectivity state; continue.  The repair reads the frozen
H4_SK hypergraph and the frozen PHG vector only -- never queries, answers, hop labels, BASE, SAFE, coverage or any downstream result.
"""
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, sha_file, rj, wj, pin, ds_dir  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402

REPAIR = "PHG_NONEMPTY_REPAIR_V1"
TAG_R = "LOWMEM__PHG_REPAIR1_con"
SUFFIX_R = "__phg_repair1"
DATASETS = ["squad", "metaqa"]
OLD_LABEL, NEW_LABEL = "PHG_L1_NONINFERIOR", "PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS"
OLD_VERDICT, NEW_VERDICT = "NONINFERIOR", "NO_SIGNIFICANT_PAIRED_LOSS"
HIST = os.path.join(OUT, "_history")
THIS = os.path.abspath(__file__)
ALGORITHM = [
    "input: the frozen PHG R1 k-way vector of the dataset (assembled from data/l1_lowmem/<ds>/phg/R1/part_rank*.txt, sha-pinned in PHG_RUNS_<ds>.json) "
    "and the frozen H4_SK hypergraph (weighted nets over canonical positions)",
    "for each empty block id b, in ascending order:",
    "  1. candidates = nodes v whose current (donor) block a = part[v] has size >= 2",
    "  2. delta(v) = exact change of the weighted KM1 = sum over nets e containing v of w_e * (1 - [count(e, a) == 1]) -- b is empty so every net of v "
    "gains one block; a net loses one block iff v was its only pin in a",
    "  3. choose the candidate with the minimum delta",
    "  4. ties: smaller canonical node position, then smaller donor block id (a node has exactly one donor block, so the third key can never decide "
    "between distinct nodes; implemented as a lexicographic sort on (delta, position, donor) for fidelity)",
    "  5. move exactly that node: part[v] = b",
    "  6. recompute block sizes and the per-(net, block) pin counts / connectivity before the next empty block (exact, from scratch)",
    "moves == number of empty blocks of the input (one move per empty block); no other node changes block",
    "postconditions: same node universe (length N, every position assigned exactly once), same k, block ids in [0, k), 0 empty blocks, "
    "max block <= ceil(1.03 N/k) (the canonical bound; a move can only shrink a donor of size >= 2 and create a singleton), exact H4_SK unchanged "
    "(npz sha + H4_SK_STRUCTURE_V1 digest == ORIGINAL), KM1_after == KM1_before + sum of the recorded deltas (full recomputation)",
    "must NOT read: queries, answers, hop labels, BASE, SAFE, coverage, any replay/eval result; no parameter search; no minimum-block-size rule"]


# ----------------------------------------------------------------------------- repair
def km1_state(H, hard):
    """per-pin count of pins in the same (net, block), connectivity lambda per net, weighted KM1 -- exact, from scratch."""
    eptr, eidx, ew, k, M = H["eptr"], H["eidx"], H["ew"], H["k"], H["M"]
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    keys = hid * np.int64(k) + hard[eidx]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    lam = np.bincount(uniq // np.int64(k), minlength=M).astype(np.int64)
    return hid, counts[inv], int((ew * (lam - 1)).sum())


def deltas_into_empty(H, hid, c_pin):
    """exact weighted KM1 delta, per node, of moving that node alone into an EMPTY block."""
    term = H["ew"][hid].astype(np.float64) * (c_pin != 1)
    d = np.bincount(H["eidx"], weights=term, minlength=H["N"])
    if not np.all(d == np.rint(d)):
        raise RuntimeError("non-integral delta")
    return d.astype(np.int64)


def repair(H, hard_in):
    N, k = H["N"], H["k"]
    hard = hard_in.astype(np.int64).copy()
    empties = [int(b) for b in np.where(np.bincount(hard, minlength=k) == 0)[0]]     # ascending block id
    moves = []
    for b in empties:
        sizes = np.bincount(hard, minlength=k)
        if sizes[b] != 0:
            raise RuntimeError("block %d is no longer empty" % b)
        hid, c_pin, km1_before = km1_state(H, hard)
        delta = deltas_into_empty(H, hid, c_pin)
        cand = sizes[hard] >= 2
        big = np.iinfo(np.int64).max
        dm = np.where(cand, delta, big)
        order = np.lexsort((hard, np.arange(N, dtype=np.int64), dm))           # primary delta, then position, then donor block
        v = int(order[0])
        if not cand[v]:
            raise RuntimeError("no candidate node with a donor block of size >= 2")
        a = int(hard[v])
        n_tied = int(((dm == dm[v]) & cand).sum())
        hard[v] = b
        _, _, km1_after = km1_state(H, hard)
        if km1_after - km1_before != int(delta[v]):
            raise RuntimeError("incremental delta %d != recomputed %d" % (int(delta[v]), km1_after - km1_before))
        moves.append({"empty_block": b, "node_position": v, "donor_block": a, "donor_size_before": int(sizes[a]), "donor_size_after": int(sizes[a] - 1),
                      "delta_km1_weighted": int(delta[v]), "min_delta_candidates_tied": n_tied, "incident_nets": int((H["eidx"] == v).sum()),
                      "km1_before": km1_before, "km1_after": km1_after, "candidates": int(cand.sum())})
        log("  move %d: node %d  block %d -> %d (empty)  delta KM1 %+d  (ties at the minimum %d, candidates %d)  KM1 %d -> %d" % (
            len(moves), v, a, b, int(delta[v]), n_tied, int(cand.sum()), km1_before, km1_after))
    return hard, empties, moves


def vec_sha(hard):
    return hashlib.sha256(np.ascontiguousarray(hard.astype(np.int64)).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- records
def load_r1(ds, R):
    d = os.path.join(ds_dir(ds), "phg", "R1")
    pins = R["stages"]["3_partition"]["partition_sha256_txt"]
    for i in range(P.NP):
        if sha_file(os.path.join(d, "part_rank%d.txt" % i)) != pins[str(i)]:
            raise RuntimeError("%s: R1 part_rank%d.txt changed since PHG_RUNS record" % (ds, i))
    return P.assemble_partition(d, R["inputs"]["N"], R["inputs"]["k"])


def prereg(supersede_reason=None):
    pp = os.path.join(OUT, "PHG_REPAIR_PREREG.json")
    moved = None
    if os.path.exists(pp):
        if not supersede_reason or any(os.path.exists(os.path.join(OUT, "PHG_REPAIR_RUNS_%s.json" % ds)) for ds in DATASETS):
            raise RuntimeError("repair preregistration already exists -- written once (supersession only before any repair result, with a reason)")
        moved = supersede("PHG_REPAIR_PREREG.json")
    runs = {ds: rj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds)) for ds in DATASETS}
    if any(R is None for R in runs.values()):
        raise RuntimeError("PHG_RUNS records missing")
    inputs = {}
    for ds, R in runs.items():
        v = R["stages"]["3_partition"]["validity"]
        inputs[ds] = {"PHG_RUNS_record": pin(os.path.join(OUT, "PHG_RUNS_%s.json" % ds)), "DECISION_as_recorded": R["DECISION"],
                      "R1_part_rank_txt_sha256": R["stages"]["3_partition"]["partition_sha256_txt"], "N": R["inputs"]["N"], "k": R["inputs"]["k"],
                      "M": R["inputs"]["M"], "P": R["inputs"]["P"], "ORIGINAL_STRUCTURE_SHA256": R["inputs"]["ORIGINAL_STRUCTURE_SHA256"],
                      "H4_SK_npz_sha256": R["stages"]["1_input_verified"]["H4_SK_npz_sha256"], "hypergraph_content_digest": R["stages"]["1_input_verified"]["hypergraph_content_digest"],
                      "empty_blocks_recorded": v["empty_blocks"], "max_block_recorded": v["max_block"], "contract_bound": v["contract_bound_ceil_1.03_N_over_k"],
                      "km1_recorded": R["metrics"]["km1_weighted"], "mtkahypar_km1": R["mtkahypar_baseline"]["km1_weighted"]}
    inputs["squad"]["frozen_import"] = {"npy": pin(os.path.join(REPO, "data", "l1_canonical", "squad", "parts", "LOWMEM__PHG_con.npy")),
                                        "json": pin(os.path.join(REPO, "data", "l1_canonical", "squad", "parts", "LOWMEM__PHG_con.json")),
                                        "downstream": pin(os.path.join(OUT, "L1_DOWNSTREAM_squad__phg.json"))}
    rec = {"RECORD": "PHG_REPAIR_PREREG", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "adapter": REPAIR, "tag": TAG_R, "suffix": SUFFIX_R,
           "purpose": "adapt an otherwise valid PHG k-way assignment to the existing canonical L1 requirement that all k part ids are non-empty; the "
                      "MetaQA PARTITION_INVALID was an interface/validity mismatch between PHG (upper-only balance) and the canonical contract, not "
                      "evidence about retrieval -- L1 was never reached",
           "not_done": "PHG is not rerun and not retuned; no parameter search; no rank-count change; no minimum-block-size rule (the frozen contract "
                       "rejected empty blocks only, so the two singleton blocks the repair creates are allowed and L1 gets the final say)",
           "algorithm": ALGORITHM, "program": pin(THIS), "phg_module": pin(P.__file__),
           "inputs": inputs,
           "expectations": {"squad": "no empty block recorded -> 0 moves; repaired vector byte-identical to the frozen import parts/LOWMEM__PHG_con.npy "
                                     "(no-op correctness test; the existing squad downstream record stands, nothing is re-evaluated)",
                            "metaqa": "2 empty blocks recorded -> exactly 2 moves (unless the frozen input proves otherwise), then import + replay cache + "
                                      "paired canonical L1 on the identical 1,998 dev query ids vs the frozen Mt-KaHyPar replay"},
           "postconditions": ["same node universe: length N, every position assigned exactly once", "same k", "block ids in [0, k)", "0 empty blocks",
                              "max block <= ceil(1.03 N/k)", "exact H4_SK unchanged (npz sha + structure digest == ORIGINAL)", "moves == empty blocks of the input",
                              "KM1_after == KM1_before + sum(delta) by full recomputation"],
           "recorded_per_run": ["moved canonical node ids", "source / destination blocks", "per-move weighted KM1 delta", "KM1 before / after", "STRUCT cut before / after",
                                "KNN cut before / after", "partition sha before / after (vector bytes and npy file)"],
           "downstream": {"path": "normal experimental canonical L1 path (l1_downstream.run: replay cache -> l1_eval numerics -> paired vs results/L1_CANONICAL replay)",
                          "population": "identical dev query ids (metaqa 1,998) to the frozen Mt-KaHyPar replay; the sealed test split is never read",
                          "primary": "final SAFE_ALL_P50 paired exact two-sided McNemar: PHG_L1_SIG_LOSS iff delta(PHG - MtK) < 0 and p < 0.05, else PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS",
                          "secondary_reported": "BASE_ALL_P50 (same rule, not the decision)",
                          "diagnostics_no_rule": ["candidate coverage (ANY)", "SAFE additions", "hop1 / hop2 / hop3 paired", "scope nodes", "balance",
                                                  "KM1 before / after repair and the 2-move delta", "STRUCT / KNN cut before / after"],
                          "no_tuning": "nothing is tuned on these results"},
           "labels": {"repair": ["REPAIR_NOOP_IDENTICAL (squad expectation)", "REPAIR_APPLIED", "PHG_REPAIR_POSTCONDITION_FAIL"],
                      "L1": [NEW_LABEL, "PHG_L1_SIG_LOSS"]},
           "terminology_amendment": {"from": OLD_LABEL, "to": NEW_LABEL, "verdict_field": {"from": OLD_VERDICT, "to": NEW_VERDICT},
                                     "why": "a non-significant paired McNemar difference is not formal non-inferiority; no non-inferiority margin was "
                                            "preregistered in PHG_PREREG.json, so no non-inferiority claim is made (the squad point estimate is -0.003, "
                                            "not 'same or better')",
                                     "rule_unchanged": "SIG_LOSS iff delta < 0 and p < 0.05; otherwise the renamed label -- identical to PHG_PREREG.json gate 3",
                                     "applied_by": "supersession of PHG_RUNS_squad.json and PHG_REPORT.{json,md}: originals moved unchanged to results/L1_LOWMEM/_history/, "
                                                   "superseding records carry `supersedes` pointers; PHG_PREREG.json itself is never edited",
                                     "user_spellings_mapped": ["PHG_L1_NO_SIGNIFICANT_LOSS", "NO_SIGNIFICANT_PAIRED_LOSS"]},
           "stop_rule": "STOP_FOR_REVIEW after the repaired MetaQA downstream result; MuSiQue / WebQSP / HotpotQA / 2Wiki untouched; no promotion",
           "frozen_read_only": {fn: pin(os.path.join(OUT, fn)) for fn in P.FROZEN_READ_ONLY + ["PHG_PREREG.json", "PHG_BUILD.json", "PHG_RUNS_metaqa.json",
                                                                                                "L1_DOWNSTREAM_squad__phg.json", "L1_REPLAY_squad__LOWMEM__PHG_con.json"]
                                if os.path.exists(os.path.join(OUT, fn))}}
    if moved:
        rec["supersedes"] = {"original": moved, "reason": supersede_reason, "algorithm_unchanged": True, "results_before_supersession": "none (no PHG_REPAIR_RUNS record existed)"}
    wj(pp, rec)
    log("preregistered", REPAIR, "->", os.path.relpath(pp, REPO))
    return rec


def relabel(obj, path, hits):
    if isinstance(obj, dict):
        return {kk: relabel(vv, path + [kk], hits) for kk, vv in obj.items()}
    if isinstance(obj, list):
        return [relabel(vv, path + [str(i)], hits) for i, vv in enumerate(obj)]
    if isinstance(obj, str) and obj in (OLD_LABEL, OLD_VERDICT):
        hits.append({"field": "/".join(path), "from": obj, "to": NEW_LABEL if obj == OLD_LABEL else NEW_VERDICT})
        return NEW_LABEL if obj == OLD_LABEL else NEW_VERDICT
    return obj


def supersede(fn):
    """move results/L1_LOWMEM/<fn> unchanged to _history/<stem>.<sha8><ext>; returns the pin of the moved original."""
    src = os.path.join(OUT, fn)
    if not os.path.exists(src):
        return None
    os.makedirs(HIST, exist_ok=True)
    sha = sha_file(src)
    stem, ext = os.path.splitext(fn)
    dst = os.path.join(HIST, "%s.%s%s" % (stem, sha[:8], ext))
    if os.path.exists(dst):
        raise RuntimeError("history file exists: %s" % dst)
    shutil.move(src, dst)
    if sha_file(dst) != sha:
        raise RuntimeError("history copy sha mismatch")
    return {"path": os.path.relpath(dst, REPO).replace("\\", "/"), "sha256": sha, "bytes": os.path.getsize(dst)}


def amend():
    pre = rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json"))
    if pre is None:
        raise RuntimeError("run prereg first")
    fp = os.path.join(OUT, "PHG_RUNS_squad.json")
    R = rj(fp)
    if R.get("terminology_amendment"):
        log("PHG_RUNS_squad.json already amended"); return R
    hits = []
    R2 = relabel(R, [], hits)
    moved = supersede("PHG_RUNS_squad.json")
    R2["supersedes"] = moved
    R2["terminology_amendment"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preregistration": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json")),
                                   "from": OLD_LABEL, "to": NEW_LABEL, "relabelled_fields": hits,
                                   "content_otherwise": "identical to the superseded record (all numbers, pins and stage artifacts unchanged)",
                                   "DECISION_as_originally_recorded": R["DECISION"]}
    wj(fp, R2)
    log("superseded PHG_RUNS_squad.json: %s -> %s (%d fields; original -> %s)" % (R["DECISION"], R2["DECISION"], len(hits), moved["path"]))
    return R2


# ----------------------------------------------------------------------------- run
def run(ds):
    pre = rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json"))
    if pre is None:
        raise RuntimeError("run prereg first")
    if ds not in DATASETS:
        raise RuntimeError("%s not preregistered" % ds)
    if pin(THIS)["sha256"] != pre["program"]["sha256"]:
        raise RuntimeError("phg_repair.py changed since preregistration")
    Rp = rj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds))
    if ds == "squad" and not Rp.get("terminology_amendment"):
        raise RuntimeError("apply `amend` first")
    ins = pre["inputs"][ds]
    # ---- frozen inputs verified: H4_SK (npz sha, structure digest == ORIGINAL), R1 vector (part-file shas)
    H, man, gate_rec, off, netl = FR.load_inputs(ds)
    if H["npz_sha256"] != ins["H4_SK_npz_sha256"] or man["ORIGINAL_STRUCTURE_SHA256"] != ins["ORIGINAL_STRUCTURE_SHA256"]:
        raise RuntimeError("%s: H4_SK changed since the PHG run" % ds)
    dig, dstats = HG.structure_digest(H["eptr"], H["eidx"], H["ew"], H["N"])
    if dig != man["ORIGINAL_STRUCTURE_SHA256"]:
        raise RuntimeError("%s: H4_SK structure digest %s != ORIGINAL" % (ds, dig))
    hard_in = load_r1(ds, Rp)
    N, k = H["N"], H["k"]
    log("=== %s %s: N %d k %d  R1 vector %s  H4_SK digest %s == ORIGINAL ===" % (REPAIR, ds, N, k, vec_sha(hard_in)[:16], dig[:16]))
    v_in = P.validity(hard_in, N, k)
    m_in = FR.km1_metrics(H, hard_in); m_in["family_cuts"] = FR.family_cuts(ds, hard_in)
    if m_in["km1_weighted"] != ins["km1_recorded"] or v_in["empty_blocks"] != ins["empty_blocks_recorded"]:
        raise RuntimeError("%s: R1 vector does not reproduce the recorded KM1 / empty-block count" % ds)
    t = time.time()
    hard_out, empties, moves = repair(H, hard_in)
    t_rep = round(time.time() - t, 3)
    v_out = P.validity(hard_out, N, k)
    m_out = FR.km1_metrics(H, hard_out); m_out["family_cuts"] = FR.family_cuts(ds, hard_out)
    changed = np.where(hard_out != hard_in)[0]
    post = {"same_N": len(hard_out) == N, "assigned_once": bool(len(hard_out) == N and hard_out.min() >= 0 and hard_out.max() < k), "same_k": k == ins["k"],
            "empty_blocks_after": v_out["empty_blocks"], "max_block_after": v_out["max_block"], "contract_bound": v_out["contract_bound_ceil_1.03_N_over_k"],
            "H4_SK_unchanged": True, "moves": len(moves), "empty_blocks_before": len(empties), "moves_equal_empties": len(moves) == len(empties),
            "nodes_changed": int(len(changed)), "nodes_changed_equal_moves": int(len(changed)) == len(moves),
            "km1_consistent": m_out["km1_weighted"] == m_in["km1_weighted"] + sum(mv["delta_km1_weighted"] for mv in moves)}
    post["PASS"] = bool(post["same_N"] and post["assigned_once"] and post["empty_blocks_after"] == 0 and post["max_block_after"] <= post["contract_bound"]
                        and post["moves_equal_empties"] and post["nodes_changed_equal_moves"] and post["km1_consistent"])
    rdir = os.path.join(ds_dir(ds), "phg_repair1")
    os.makedirs(rdir, exist_ok=True)
    npy = os.path.join(rdir, "repaired.npy")
    np.save(npy, hard_out.astype(np.int64))
    R = {"RECORD": "PHG_REPAIR_RUNS", "adapter": REPAIR, "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "preregistration": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json")), "program": pin(THIS),
         "inputs": {"PHG_RUNS_record": pin(os.path.join(OUT, "PHG_RUNS_%s.json" % ds)), "R1_part_rank_txt_sha256": Rp["stages"]["3_partition"]["partition_sha256_txt"],
                    "vector_sha256_in": vec_sha(hard_in), "H4_SK_npz_sha256": H["npz_sha256"], "structure_digest": dig, "N": N, "k": k, "M": H["M"], "P": H["P"]},
         "before": {"validity": v_in, "km1_weighted": m_in["km1_weighted"], "cut_weighted": m_in["cut_weighted"], "blocks": m_in["blocks"], "family_cuts": m_in["family_cuts"]},
         "empty_blocks": empties, "moves": moves, "repair_seconds": t_rep,
         "after": {"validity": v_out, "km1_weighted": m_out["km1_weighted"], "cut_weighted": m_out["cut_weighted"], "blocks": m_out["blocks"], "family_cuts": m_out["family_cuts"],
                   "delta_km1_total": m_out["km1_weighted"] - m_in["km1_weighted"],
                   "delta_km1_relative": round((m_out["km1_weighted"] - m_in["km1_weighted"]) / float(m_in["km1_weighted"]), 8),
                   "struct_cut_delta": round(m_out["family_cuts"]["STRUCT"]["edge_cut_fraction"] - m_in["family_cuts"]["STRUCT"]["edge_cut_fraction"], 6),
                   "knn_cut_delta": round(m_out["family_cuts"]["KNN"]["edge_cut_fraction"] - m_in["family_cuts"]["KNN"]["edge_cut_fraction"], 6),
                   "vector_sha256_out": vec_sha(hard_out), "repaired_npy": pin(npy)},
         "postconditions": post, "mtkahypar_km1": Rp["mtkahypar_baseline"]["km1_weighted"],
         "km1_ratio_vs_mtkahypar": {"before": round(m_in["km1_weighted"] / float(Rp["mtkahypar_baseline"]["km1_weighted"]), 4),
                                    "after": round(m_out["km1_weighted"] / float(Rp["mtkahypar_baseline"]["km1_weighted"]), 4)}}
    log("%-7s repair: empties %s  moves %d  KM1 %d -> %d (%+d)  STRUCT %.4f -> %.4f  KNN %.4f -> %.4f  max %d -> %d  empty %d -> %d  post %s" % (
        ds, empties, len(moves), m_in["km1_weighted"], m_out["km1_weighted"], R["after"]["delta_km1_total"], m_in["family_cuts"]["STRUCT"]["edge_cut_fraction"],
        m_out["family_cuts"]["STRUCT"]["edge_cut_fraction"], m_in["family_cuts"]["KNN"]["edge_cut_fraction"], m_out["family_cuts"]["KNN"]["edge_cut_fraction"],
        v_in["max_block"], v_out["max_block"], v_in["empty_blocks"], v_out["empty_blocks"], "PASS" if post["PASS"] else "FAIL"))
    fp = os.path.join(OUT, "PHG_REPAIR_RUNS_%s.json" % ds)
    if not post["PASS"]:
        R["DECISION"] = "PHG_REPAIR_POSTCONDITION_FAIL"
        wj(fp, R); log(ds, "DECISION", R["DECISION"]); return R
    if ds == "squad":
        fi = pre["inputs"]["squad"]["frozen_import"]["npy"]
        R["noop_check"] = {"moves": len(moves), "vector_identical_to_input": bool(np.array_equal(hard_in, hard_out)),
                           "repaired_npy_sha256": R["after"]["repaired_npy"]["sha256"], "frozen_import_npy_sha256": fi["sha256"],
                           "byte_identical_to_frozen_import": R["after"]["repaired_npy"]["sha256"] == fi["sha256"] == sha_file(os.path.join(REPO, fi["path"]))}
        ok = len(moves) == 0 and R["noop_check"]["vector_identical_to_input"] and R["noop_check"]["byte_identical_to_frozen_import"]
        R["DECISION"] = "REPAIR_NOOP_IDENTICAL" if ok else "PHG_REPAIR_POSTCONDITION_FAIL"
        R["downstream"] = {"note": "no-op: the frozen squad import parts/LOWMEM__PHG_con.npy IS the repaired partition; L1_DOWNSTREAM_squad__phg.json stands "
                                   "(label per the terminology amendment)", "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_squad__phg.json"))}
        wj(fp, R); log(ds, "DECISION", R["DECISION"], R["noop_check"]); return R
    # ---- metaqa: import the repaired vector through the normal experimental canonical L1 path
    b = rj(os.path.join(OUT, "PHG_BUILD.json"))
    b["packages_versions"] = rj(os.path.join(OUT, "PHG_PACKAGES.json"))["after"]["dpkg_versions"]
    Rp2 = dict(Rp); Rp2["metrics"] = m_out           # the manifest's worker_stats / post_checks describe the REPAIRED vector
    extra = {"repair": {"adapter": REPAIR, "program": R["program"], "preregistration": R["preregistration"], "input_R1_part_rank_txt_sha256": R["inputs"]["R1_part_rank_txt_sha256"],
                        "vector_sha256_in": R["inputs"]["vector_sha256_in"], "vector_sha256_out": R["after"]["vector_sha256_out"], "empty_blocks": empties,
                        "moves": moves, "km1_before": m_in["km1_weighted"], "km1_after": m_out["km1_weighted"], "postconditions": post,
                        "note": "PHG R1 (frozen, PARTITION_INVALID: empty blocks) + PHG_NONEMPTY_REPAIR_V1 = this vector; PHG was not rerun"}}
    rec = P.import_partition(ds, H, hard_out, Rp2, b, off, man, Rp["stages"]["2_structure_gate"], tag=TAG_R, extra=extra)
    mp = os.path.join(H["d"].derived_dir, "parts", "%s.json" % TAG_R)
    rec = rj(mp)
    rec["contract"]["repair"] = "%s applied to the frozen PHG R1 vector (program sha %s): one minimum-delta-KM1 node move per empty block" % (REPAIR, R["program"]["sha256"][:16])
    rec["arm"] = P.ARM + "__" + REPAIR
    wj(mp, rec)
    R["import"] = rec
    from src.l1_lowmem import l1_downstream as LD
    D = LD.run(ds, tag=TAG_R, suffix=SUFFIX_R)
    R["downstream"] = {"status": D["status"], "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (ds, SUFFIX_R))), "paired": D.get("paired"),
                       "B_phg_repaired": D.get("B"), "A_mtkahypar": D.get("A"), "by_hop_paired": D.get("by_hop_paired"), "coverage": D.get("coverage"),
                       "safe_additions": D.get("safe_additions"), "population": D.get("population_B")}
    p = D.get("paired") or {}
    if D["status"] != "PAIRED":
        R["DECISION"] = "PHG_L1_UNPAIRED (%s)" % D["status"]
    else:
        s, bsec = p["SAFE_ALL_P50"], p["BASE_ALL_P50"]
        R["l1_primary_SAFE"] = {"delta_phg_minus_mtk": s["delta_B_minus_A"], "p": s["mcnemar_p"], "sig": s["sig"], "phg_only": s["B_only_covered"], "mtk_only": s["A_only_covered"],
                                "verdict": "SIG_LOSS" if (s["sig"] and s["delta_B_minus_A"] < 0) else NEW_VERDICT}
        R["l1_secondary_BASE"] = {"delta_phg_minus_mtk": bsec["delta_B_minus_A"], "p": bsec["mcnemar_p"], "sig": bsec["sig"], "phg_only": bsec["B_only_covered"],
                                  "mtk_only": bsec["A_only_covered"], "verdict": "SIG_LOSS" if (bsec["sig"] and bsec["delta_B_minus_A"] < 0) else NEW_VERDICT}
        R["DECISION"] = "PHG_L1_" + R["l1_primary_SAFE"]["verdict"]
    wj(fp, R)
    log("%-7s DECISION %s  (SAFE %s p=%s; BASE %s p=%s) -> %s" % (ds, R["DECISION"], (R.get("l1_primary_SAFE") or {}).get("delta_phg_minus_mtk"),
                                                                  (R.get("l1_primary_SAFE") or {}).get("p"), (R.get("l1_secondary_BASE") or {}).get("delta_phg_minus_mtk"),
                                                                  (R.get("l1_secondary_BASE") or {}).get("p"), os.path.relpath(fp, REPO)))
    return R


# ----------------------------------------------------------------------------- report (supersedes PHG_REPORT)
def pf(pr, key):
    x = (pr or {}).get(key)
    return "%.4f -> %.4f (%+.4f, p=%s, +%d/-%d)" % (x["A"], x["B"], x["delta_B_minus_A"], x["mcnemar_p"], x["B_only_covered"], x["A_only_covered"]) if x else "-"


def report():
    pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    prr = rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json"))
    runs = {ds: rj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds)) for ds in DATASETS}
    reps = {ds: rj(os.path.join(OUT, "PHG_REPAIR_RUNS_%s.json" % ds)) for ds in DATASETS}
    reps = {ds: R for ds, R in reps.items() if R}
    moved = {fn: supersede(fn) for fn in ("PHG_REPORT.json", "PHG_REPORT.md")}
    # integrity: every pinned frozen record unchanged, canonical H4_SK partitions unchanged
    from src.l1_canonical.adapter import CanonicalDataset
    frozen = dict(pre["frozen_read_only"]); frozen.update(prr["frozen_read_only"])
    integ = {"frozen_records_changed": [fn for fn, pn in frozen.items() if not os.path.exists(os.path.join(OUT, fn)) or sha_file(os.path.join(OUT, fn)) != pn["sha256"]],
             "canonical_H4_SK_partition": {}, "superseded_to_history": {"PHG_RUNS_squad.json": runs["squad"].get("supersedes"), **moved}}
    cu = (rj(os.path.join(OUT, "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json")) or {}).get("canonical_untouched") or {}
    for ds in DATASETS:
        d = CanonicalDataset(ds)
        now = sha_file(os.path.join(d.derived_dir, "parts", "H4_SK.npy"))
        integ["canonical_H4_SK_partition"][ds] = {"sha256_now": now, "pinned": (cu.get(ds) or {}).get("H4_SK_partition_sha256"), "unchanged": now == (cu.get(ds) or {}).get("H4_SK_partition_sha256")}
    # final per-dataset decision = PHG lane decision, with the repair follow-up applied where it ran
    final = {}
    for ds in DATASETS:
        R, Q = runs[ds], reps.get(ds)
        final[ds] = {"phg_lane": R["DECISION"], "repair": Q["DECISION"] if Q else None,
                     "final": (Q["DECISION"] if (Q and Q["DECISION"].startswith("PHG_L1_")) else R["DECISION"])}
    rows = ["| Dataset | Arm | KM1 | KM1 / MtK | STRUCT cut | KNN cut | blocks min/p50/max (bound) | empty | SAFE (paired, dev) | BASE (paired, dev) | Decision |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        R, Q = runs[ds], reps.get(ds)
        m, bm, fr = R["metrics"], R["mtkahypar_baseline"], R.get("freight_one_pass_frozen") or {}
        pr = (R.get("downstream") or {}).get("paired")
        rows.append("| %s | PHG R1 (NP=4) | %d | %.4f | %.4f | %.4f | %d/%.0f/%d (%d) | %d | %s | %s | %s |" % (
            ds, m["km1_weighted"], R["diagnostic_vs_mtkahypar"]["km1_ratio"], m["family_cuts"]["STRUCT"]["edge_cut_fraction"], m["family_cuts"]["KNN"]["edge_cut_fraction"],
            m["blocks"]["min"], m["blocks"]["p50"], m["blocks"]["max"], R["stages"]["3_partition"]["validity"]["contract_bound_ceil_1.03_N_over_k"], m["blocks"]["empty"],
            pf(pr, "SAFE_ALL_P50"), pf(pr, "BASE_ALL_P50"), R["DECISION"]))
        if Q and Q["DECISION"] != "REPAIR_NOOP_IDENTICAL":
            a = Q["after"]; qr = (Q.get("downstream") or {}).get("paired")
            rows.append("| %s | PHG R1 + %s (%d moves) | %d | %.4f | %.4f | %.4f | %d/%.0f/%d (%d) | %d | %s | %s | %s |" % (
                ds, REPAIR, len(Q["moves"]), a["km1_weighted"], Q["km1_ratio_vs_mtkahypar"]["after"], a["family_cuts"]["STRUCT"]["edge_cut_fraction"],
                a["family_cuts"]["KNN"]["edge_cut_fraction"], a["blocks"]["min"], a["blocks"]["p50"], a["blocks"]["max"], a["validity"]["contract_bound_ceil_1.03_N_over_k"],
                a["blocks"]["empty"], pf(qr, "SAFE_ALL_P50"), pf(qr, "BASE_ALL_P50"), Q["DECISION"]))
        elif Q:
            rows.append("| %s | PHG R1 + %s | (no-op: 0 moves, vector byte-identical) | | | | | | | | %s |" % (ds, REPAIR, Q["DECISION"]))
        rows.append("| %s | Mt-KaHyPar (canonical, frozen) | %d | 1.0 | %.4f | %.4f | %d/%.0f/%d | %d | (reference) | (reference) | |" % (
            ds, bm["km1_weighted"], bm["family_cuts"]["STRUCT"]["edge_cut_fraction"], bm["family_cuts"]["KNN"]["edge_cut_fraction"], bm["blocks"]["min"], bm["blocks"]["p50"],
            bm["blocks"]["max"], bm["blocks"]["empty"]))
        if fr.get("km1_weighted"):
            rows.append("| %s | FREIGHT 1-pass (frozen, CLOSED) | %d | %.4f | %.4f | %.4f | %d/%.0f/%d | %d | (frozen record) | (frozen record) | FREIGHT_CLOSED |" % (
                ds, fr["km1_weighted"], fr["km1_weighted"] / float(bm["km1_weighted"]), fr["family_cuts"]["STRUCT"]["edge_cut_fraction"], fr["family_cuts"]["KNN"]["edge_cut_fraction"],
                fr["blocks"]["min"], fr["blocks"]["p50"], fr["blocks"]["max"], fr["blocks"]["empty"]))
    # FREIGHT vs PHG: objective gap vs L1 outcome, from the frozen records
    def dsn(fn):
        D = rj(os.path.join(OUT, fn)) or {}
        pr = D.get("paired") or {}
        s, b_ = pr.get("SAFE_ALL_P50"), pr.get("BASE_ALL_P50")
        return ("SAFE %+.4f (p=%s%s), BASE %+.4f (p=%s%s)" % (s["delta_B_minus_A"], s["mcnemar_p"], ", sig" if s["sig"] else "", b_["delta_B_minus_A"], b_["mcnemar_p"],
                                                            ", sig" if b_["sig"] else "")) if s else "-"
    fr_s, fr_m = rj(os.path.join(OUT, "FREIGHT_RUNS_squad.json")), rj(os.path.join(OUT, "FREIGHT_RUNS_metaqa.json"))
    fr_m2 = rj(os.path.join(OUT, "FREIGHT_RESTREAM_RUNS_metaqa.json")) or {}
    cmp_rows = ["| Arm | Dataset | Objective gap (KM1 / Mt-KaHyPar) | Paired L1 vs Mt-KaHyPar (same dev ids) | Outcome |", "|---|---|---|---|---|",
                "| FREIGHT 1-pass | squad | %.3fx | %s | significant loss (LOW_MEMORY_BUT_QUALITY_FAIL) |" % (fr_s["metrics_B"]["km1_weighted"] / float(runs["squad"]["mtkahypar_baseline"]["km1_weighted"]), dsn("L1_DOWNSTREAM_squad.json")),
                "| PHG R1 | squad | %.3fx | %s | %s |" % (runs["squad"]["diagnostic_vs_mtkahypar"]["km1_ratio"], dsn("L1_DOWNSTREAM_squad__phg.json"), runs["squad"]["DECISION"]),
                "| FREIGHT 1-pass | metaqa | %.3fx | %s | n.s. overall (hop1 sig loss) |" % (fr_m["metrics_B"]["km1_weighted"] / float(runs["metaqa"]["mtkahypar_baseline"]["km1_weighted"]), dsn("L1_DOWNSTREAM_metaqa.json")),
                "| FREIGHT 2-pass restream | metaqa | %.3fx | %s | significant loss (FREIGHT_CLOSED) |" % ((fr_m2.get("metrics") or {}).get("km1_weighted", 0) / float(runs["metaqa"]["mtkahypar_baseline"]["km1_weighted"]), dsn("L1_DOWNSTREAM_metaqa__restream2.json")),
                "| PHG R1 | metaqa | %.3fx | not evaluated (2 empty blocks) | PHG_PARTITION_INVALID |" % runs["metaqa"]["diagnostic_vs_mtkahypar"]["km1_ratio"]]
    if reps.get("metaqa") and reps["metaqa"]["DECISION"].startswith("PHG_L1_"):
        cmp_rows.append("| PHG R1 + %s | metaqa | %.3fx | %s | %s |" % (REPAIR, reps["metaqa"]["km1_ratio_vs_mtkahypar"]["after"], dsn("L1_DOWNSTREAM_metaqa%s.json" % SUFFIX_R), reps["metaqa"]["DECISION"]))
    hop_rows = ["| Dataset | Arm | hop | n | SAFE MtK -> PHG | BASE MtK -> PHG |", "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for label, src in (("PHG R1", runs[ds]), ("PHG R1 + repair", reps.get(ds) or {})):
            for h, x in ((src.get("downstream") or {}).get("by_hop_paired") or {}).items():
                hop_rows.append("| %s | %s | %s | %d | %.4f -> %.4f (%+.4f, p=%s) | %.4f -> %.4f (%+.4f, p=%s) |" % (
                    ds, label, h, x["n"], x["SAFE"]["A"], x["SAFE"]["B"], x["SAFE"]["delta_B_minus_A"], x["SAFE"]["mcnemar_p"], x["BASE"]["A"], x["BASE"]["B"],
                    x["BASE"]["delta_B_minus_A"], x["BASE"]["mcnemar_p"]))
    cov_rows = ["| Dataset | Arm | SAFE_ANY MtK / PHG | BASE_ANY MtK / PHG | SAFE scope nodes MtK / PHG | SAFE additions MtK / PHG (gained/lost) | balance max/mean MtK / PHG |", "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for label, src in (("PHG R1", runs[ds]), ("PHG R1 + repair", reps.get(ds) or {})):
            dsn_ = src.get("downstream") or {}
            c, sa = dsn_.get("coverage"), dsn_.get("safe_additions")
            A, B = dsn_.get("A_mtkahypar") or {}, dsn_.get("B_phg") or dsn_.get("B_phg_repaired") or {}
            if c:
                cov_rows.append("| %s | %s | %.4f / %.4f | %.4f / %.4f | %.1f / %.1f | %d/%d / %d/%d | %s / %s |" % (
                    ds, label, c["SAFE_ANY"]["A"], c["SAFE_ANY"]["B"], c["BASE_ANY"]["A"], c["BASE_ANY"]["B"], c["SAFE_SCOPE_NODES"]["A"], c["SAFE_SCOPE_NODES"]["B"],
                    sa["A"]["gained"], sa["A"]["lost"], sa["B"]["gained"], sa["B"]["lost"], (A.get("BALANCE") or {}).get("max_over_mean"), (B.get("BALANCE") or {}).get("max_over_mean")))
    rep_rows = ["| Dataset | empty blocks | moves | moved node -> block (from donor, donor size) | delta KM1 per move | KM1 before -> after | STRUCT cut before -> after | KNN cut before -> after | vector sha in -> out | postconditions | decision |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, Q in reps.items():
        b_, a = Q["before"], Q["after"]
        rep_rows.append("| %s | %s | %d | %s | %s | %d -> %d (%+d, %+.2e rel) | %.4f -> %.4f | %.4f -> %.4f | %s -> %s | %s | %s |" % (
            ds, Q["empty_blocks"], len(Q["moves"]), "; ".join("%d -> %d (from %d, size %d)" % (mv["node_position"], mv["empty_block"], mv["donor_block"], mv["donor_size_before"]) for mv in Q["moves"]) or "-",
            ", ".join("%+d" % mv["delta_km1_weighted"] for mv in Q["moves"]) or "-", b_["km1_weighted"], a["km1_weighted"], a["delta_km1_total"], a["delta_km1_relative"],
            b_["family_cuts"]["STRUCT"]["edge_cut_fraction"], a["family_cuts"]["STRUCT"]["edge_cut_fraction"], b_["family_cuts"]["KNN"]["edge_cut_fraction"], a["family_cuts"]["KNN"]["edge_cut_fraction"],
            Q["inputs"]["vector_sha256_in"][:12], a["vector_sha256_out"][:12], "PASS" if Q["postconditions"]["PASS"] else "FAIL", Q["DECISION"]))
    rep = {"RECORD": "PHG_REPORT", "version": 2, "supersedes": moved, "arm": P.ARM, "follow_up": REPAIR, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "preregistrations": {"lane": pin(os.path.join(OUT, "PHG_PREREG.json")), "repair": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json"))},
           "STATUS": "STOP_FOR_REVIEW", "decisions": final, "terminology_amendment": prr["terminology_amendment"],
           "l1_primary_SAFE": {ds: {"phg_R1": runs[ds].get("l1_primary_SAFE"), "phg_R1_repaired": (reps.get(ds) or {}).get("l1_primary_SAFE")} for ds in DATASETS},
           "l1_secondary_BASE": {ds: {"phg_R1": runs[ds].get("l1_secondary_BASE"), "phg_R1_repaired": (reps.get(ds) or {}).get("l1_secondary_BASE")} for ds in DATASETS},
           "gates": pre["gates"], "decision_rule": pre["decision_rule"], "repair_algorithm": prr["algorithm"], "integrity": integ,
           "repairs": {ds: {kk: Q.get(kk) for kk in ("DECISION", "empty_blocks", "moves", "before", "after", "postconditions", "noop_check", "km1_ratio_vs_mtkahypar")} for ds, Q in reps.items()},
           "datasets": {ds: {kk: runs[ds].get(kk) for kk in ("DECISION", "metrics", "mtkahypar_baseline", "freight_one_pass_frozen", "diagnostic_vs_mtkahypar", "repeat", "downstream", "l1_primary_SAFE", "l1_secondary_BASE")} for ds in DATASETS},
           "repaired_downstream": {ds: (Q.get("downstream")) for ds, Q in reps.items()},
           "stage3_summaries": {ds: {kk: (runs[ds]["stages"].get("3_partition") or {}).get(kk) for kk in ("validity", "memory", "timing", "python_km1_vs_zoltan_cutl")} for ds in DATASETS},
           "structure_gates": {ds: {kk: runs[ds]["stages"]["2_structure_gate"].get(kk) for kk in ("gate", "N_returned", "M_with_pins", "P_returned", "weight_sum", "digest_from_queries", "ORIGINAL_STRUCTURE_SHA256", "partition_run_dumps_identical", "zoltan_removed_or_warning_lines", "weight_reception")} for ds in DATASETS},
           "table_markdown": "\n".join(rows), "comparison_markdown": "\n".join(cmp_rows), "repair_table_markdown": "\n".join(rep_rows),
           "hop_table_markdown": "\n".join(hop_rows), "coverage_table_markdown": "\n".join(cov_rows),
           "not_done": "MuSiQue / WebQSP / HotpotQA / 2Wiki not run; no promotion; no tuning; PHG not rerun; FREIGHT stays FREIGHT_CLOSED"}
    # observations (not gates, not claims): exact p behind the frozen reporter's 5-decimal rounding; a significant paired GAIN is not a preregistered hypothesis
    from math import comb

    def exact_p(g, l):
        n = g + l
        return 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, i) for i in range(min(g, l) + 1)) / (2.0 ** n))
    obs = {}
    for ds in DATASETS:
        for label, src in (("PHG R1", runs[ds]), ("PHG R1 + repair", reps.get(ds) or {})):
            dsn_ = src.get("downstream") or {}
            pr, c = dsn_.get("paired"), dsn_.get("coverage")
            if not pr:
                continue
            o = {"exact_mcnemar_p": {kk: exact_p(pr[kk]["B_only_covered"], pr[kk]["A_only_covered"]) for kk in ("SAFE_ALL_P50", "BASE_ALL_P50")},
                 "exact_mcnemar_p_by_hop": {h: {m_: exact_p(x[m_]["B_only_covered"], x[m_]["A_only_covered"]) for m_ in ("SAFE", "BASE")} for h, x in (dsn_.get("by_hop_paired") or {}).items()},
                 "reporter_rounding": "the frozen bridge.mcnemar rounds p to 5 decimals, so a printed p=0.0 means p < 5e-6",
                 "scope_nodes_SAFE": {"mtkahypar": c["SAFE_SCOPE_NODES"]["A"], "phg": c["SAFE_SCOPE_NODES"]["B"]} if c else None}
            if pr["SAFE_ALL_P50"]["sig"] and pr["SAFE_ALL_P50"]["delta_B_minus_A"] > 0:
                o["note"] = ("the preregistered rule only separates 'significant paired loss' from 'no significant paired loss'; the observed SAFE delta is a "
                             "significant paired GAIN, which was not a preregistered hypothesis -- recorded as an observation, not a claim; the PHG P50 scope "
                             "holds more nodes (fuller, more uniform blocks) and whether the gain is scope size or block composition is not tested here")
            obs["%s / %s" % (ds, label)] = o
    rep["post_hoc_observations_not_gates"] = obs
    wj(os.path.join(OUT, "PHG_REPORT.json"), rep)
    s3 = rep["stage3_summaries"]
    md = ["# %s -- Zoltan-PHG substitution validation (SQuAD + MetaQA) + %s" % (P.ARM, REPAIR), "",
          "Generated %s (supersedes %s) -- STATUS: **STOP_FOR_REVIEW** -- final decisions: %s" % (rep["utc"], moved["PHG_REPORT.json"]["path"] if moved["PHG_REPORT.json"] else "-",
                                                                                                 ", ".join("%s = **%s**" % (ds, f["final"]) for ds, f in final.items())), "",
          "Terminology amendment: `%s` -> `%s` (rule identical: SIG_LOSS iff delta < 0 and p < 0.05; no non-inferiority margin was preregistered, so no non-inferiority "
          "claim is made; the superseded records are in results/L1_LOWMEM/_history/)." % (OLD_LABEL, NEW_LABEL), "",
          "Primary question: does canonical L1 with the PHG partition show no significant paired SAFE loss against Mt-KaHyPar on the identical frozen H4_SK input and identical "
          "dev query ids?  Partition equality is not required; KM1 / cuts are diagnostics.", "", rep["table_markdown"], "",
          "Objective gap vs L1 outcome (FREIGHT records frozen, PHG this lane):", "", rep["comparison_markdown"], "",
          "%s (preregistered adapter; PHG not rerun):" % REPAIR, "", rep["repair_table_markdown"], "",
          "Hop-wise (diagnostic, no rule):", "", rep["hop_table_markdown"], "", "Coverage / SAFE additions / balance (diagnostic):", "", rep["coverage_table_markdown"], "",
          "Structure gates (exact):"]
    for ds, g in rep["structure_gates"].items():
        md.append("- %s: %s -- N %s M %s P %s weight sum %s; digest from the query functions %s == ORIGINAL %s; weight reception (Zoltan cutl vs Python KM1 of the rank ownership): %s; "
                  "partition-run dumps identical: %s; Zoltan removal/warning lines: %s" % (ds, g["gate"], g["N_returned"], g["M_with_pins"], g["P_returned"], g["weight_sum"],
                                                                                          (g["digest_from_queries"] or "-")[:16], (g["ORIGINAL_STRUCTURE_SHA256"] or "-")[:16],
                                                                                          (g.get("weight_reception") or {}).get("relative_diff"), g["partition_run_dumps_identical"],
                                                                                          g["zoltan_removed_or_warning_lines"] or "none"))
    md += ["", "PHG runs (4 local MPI ranks; validation of implementation/quality only -- no aggregate-RAM claim at scale):"]
    for ds, s in s3.items():
        md.append("- %s: peak RSS per rank %s MB (max %.0f, sum %.0f = upper bound); partition wall %.2f s, job wall %s s; validity %s (max %d, bound %d, empty %d); Python KM1 vs Zoltan cutl %s; repeat run identical %s" % (
            ds, [round(x / 1024.0, 1) for x in s["memory"]["peak_rss_kb_per_rank_time_v"]], s["memory"]["max_rank_peak_kb"] / 1024.0, s["memory"]["sum_of_rank_peaks_kb"] / 1024.0,
            s["timing"]["partition_wall_seconds"], s["timing"]["job_wall_seconds"], s["validity"]["gate"], s["validity"]["max_block"], s["validity"]["contract_bound_ceil_1.03_N_over_k"],
            s["validity"]["empty_blocks"], s["python_km1_vs_zoltan_cutl"]["relative_diff"], runs[ds]["repeat"]["identical_partition"]))
    md += ["", "Observations (not gates, not claims):"]
    for key, o in obs.items():
        md.append("- %s: exact McNemar p SAFE %.2e / BASE %.2e (printed p=0.0 means p < 5e-6); by hop: %s; SAFE scope nodes Mt-KaHyPar %s vs PHG %s.%s" % (
            key, o["exact_mcnemar_p"]["SAFE_ALL_P50"], o["exact_mcnemar_p"]["BASE_ALL_P50"],
            ", ".join("%s SAFE %.1e / BASE %.1e" % (h, v["SAFE"], v["BASE"]) for h, v in o["exact_mcnemar_p_by_hop"].items()) or "n/a",
            (o["scope_nodes_SAFE"] or {}).get("mtkahypar"), (o["scope_nodes_SAFE"] or {}).get("phg"), ("  " + o["note"]) if o.get("note") else ""))
    md += ["", "Gates / decision rule (PHG_PREREG.json, label renamed): " + pre["decision_rule"], "",
           "Repair algorithm (PHG_REPAIR_PREREG.json): " + " | ".join(prr["algorithm"]), "",
           "Integrity: frozen records changed = %s; canonical H4_SK partitions unchanged = %s; superseded originals: %s" % (
               integ["frozen_records_changed"] or "none", all(v["unchanged"] for v in integ["canonical_H4_SK_partition"].values()),
               {k: (v or {}).get("path") for k, v in integ["superseded_to_history"].items()}),
           "", "Caveats: " + pre["rank_count_caveat"] + "  " + pre["reproducibility"], "", rep["not_done"], ""]
    with io.open(os.path.join(OUT, "PHG_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("STOP_FOR_REVIEW", {ds: f["final"] for ds, f in final.items()})
    print(rep["table_markdown"]); print(); print(rep["repair_table_markdown"]); print(); print(rep["comparison_markdown"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "prereg":
        prereg(" ".join(a[1:]) or None)
    elif a[0] == "amend":
        amend()
    elif a[0] == "run":
        for ds in a[1:]:
            run(ds)
    elif a[0] == "report":
        report()
    else:
        print(__doc__)
