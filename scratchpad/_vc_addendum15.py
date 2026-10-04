"""FBX_SCALE declaration addendum 15 (write-once): ORCHESTRATION ONLY -- the nested K-way stage (scratchpad/_h2l.py cmd_sub) saves each finished block and reuses it after a preemption.

Addendum 14's gate arms run the K-way stage as 25 independent Zoltan blocks per (arm, K), about 3 - 5 minutes each (about 87 minutes per arm at K = 500).  The shared host preempts crag jobs for mpr (rc 75 / 15,
auto-requeued by the yield waiter) at intervals of 1 - 60 minutes, and cmd_sub writes the arm's record only at the very end, so every preemption restarted the arm at block 0: 11 consecutive K = 500 attempts
(gens 5 - 11 of the arm-3 job) never completed it, and the last one held 3 of 25 blocks when it was preempted.  This addendum changes how a block's result survives a preemption and nothing else.
It changes no gate, no parameter, no partition rule and no earlier record.   Usage: python scratchpad/_vc_addendum15.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_15.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11", "12", "13", "14")]
CODE = ["scratchpad/_h2l.py", "scratchpad/_vc_sub.py", "scratchpad/_vc_subk.py", "scratchpad/_vc_eval.py", "scratchpad/_h2lt.py", "scratchpad/_host_yield.py",
        "scratchpad/_l1h_host.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py"]
OLD_H2L = "e578fc4b01ff29ec9a96047a7a0875fd19cff9d0e63b862a019ee48d34a46dd1"      # the sha pinned by addendum 14 (and by every arm record written before this addendum)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    rec = {
        "stage": "FBX_SCALE / addendum 15: block-level resume of the nested K-way stage (orchestration only), declared before any arm of addendum 14 that needs it is completed",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "status": "ORCHESTRATION (no science): nothing here selects a mechanism, changes a gate, a threshold, a parameter or a record",
        "user_ruling": "2026-10-02, in chat: 'cancel the waiter and make blocks resumable' (answer to the proposal of the previous turn)",
        "problem": "scratchpad/_h2l.py cmd_sub(S, K) loops over the S top blocks, partitions each one with the frozen Zoltan-PHG driver and writes the arm's RUN record only after the last block, so a preempted job (rc 75 / 15) loses every completed block "
                   "and the requeued job starts at block 0.  The vc-sub K = 500 job (arms VCQ512L6FM2 and VCQ512L6N4LP2 outstanding; about 87 minutes of uninterrupted compute per arm) was preempted in every attempt; its last attempt "
                   "had 3 of 25 blocks done after 13 m 42 s.  The gate of addendum 14 cannot be evaluated until the K = 500 maps exist.",
        "change": {
            "file": "scratchpad/_h2l.py (cmd_sub and three helpers: ckpt_dir, ckpt_save, ckpt_load; _test extended)",
            "rule": "after block b is partitioned and its record built, save (local block ids hb, the block record) to results/FREEBASE_SCALE/h2l/parts/_blockckpt/<arm output name>/b<NN>.npz (written to a temp name and os.replace'd); "
                    "before partitioning a block, load that file and reuse it iff its stored KEY equals the key of this call.  KEY = top-map sha256, hypergraph content digest, S, K, block index, block N / M / pins, the Zoltan tolerance, kb, NP and the full driver parameter list "
                    "(with the block's tolerance applied).  Any mismatch, a torn or unreadable file, a wrong length or an id outside [0, kb) is a MISS and the block is computed as before.  The checkpoint directory is named after the arm's own "
                    "output file, so two arms never share one; it is deleted when the arm's RUN or FAILED record has been written.",
            "record_additions": "the arm record gains 'blocks_resumed_from_checkpoint' (the block indices reused) and every reused block record carries 'resumed': true; 'wall_seconds' of an arm that was resumed counts only the final job, "
                                "and a reused block's 'seconds' / 'peak_rss_kb_per_rank' are those of the run that computed it.",
        },
        "unchanged": ["the top maps and their records", "restrict() (net splitting)", "tol_for() and the Zoltan tolerance of every block", "the driver, its parameters and NP", "the validity rule, the empty-block repair, the balance check",
                      "the arm record's other fields", "_vc_sub.py, _vc_subk.py, _h2lt.py, _vc_eval.py (not edited)", "addendum 14's gate, its decision table and its four arms"],
        "why_this_does_not_change_a_measurement": "block b's partition is a function of its inputs only (the sub-hypergraph, kb, the tolerance, the driver parameters, NP) and blocks do not share state: hard[vertices_b] = b * kb + hb is the only coupling and it is "
                                                  "applied identically to a computed or a restored hb.  The sub-hypergraph itself is determined by the top map, the hypergraph and S, all of which are in the key.  A restored block is therefore the stored output of the SAME call made in an earlier job; "
                                                  "an uninterrupted job is one run of that call and nothing in the pipeline assumes Zoltan is bit-reproducible across runs (the earlier arms were produced by single uninterrupted jobs).  The gate reads the final map only.",
        "verification": {"local": "python scratchpad/_h2l.py TEST (restriction identities and the KM1 identity, unchanged) now also asserts: exact save / load round trip of (hb, record); a miss on a changed top sha, tolerance, N, kb, parameters or K; "
                                  "a miss on an out-of-range id and on a torn file.  PASS on the laptop before the file was installed.",
                         "host": "the first resumed arm is checked from its own record: 'blocks_resumed_from_checkpoint', the per-block 'resumed' flags, and the final KM1 / validity lines in its log."},
        "records_and_code_pins": {"old_sha_of__h2l.py": OLD_H2L, "scope": "arm records written before this addendum (K = 100 and 250 for all four arms, K = 500 for VCQ512L6N4FM2 and VCQ512L6N8FM2) carry the old sha in their code pins; "
                                                                    "arm records written after it carry the new one.  Both are listed here and in addendum 14's pin table; neither record set is edited.",
                                 "discarded": "the blocks computed by the preempted K = 500 attempts before this change are lost (they were never saved); the arms restart at block 0 once."},
        "not_claimed": ["no change to any result already written", "no claim that a resumed arm is bit-identical to an uninterrupted one (it is not claimed for two uninterrupted runs either)", "nothing about preemption policy or priorities"],
        "rollback": "restore results/FREEBASE_SCALE/_history/_h2l.py.pre_addendum15__e578fc4b (sha256 = old_sha_of__h2l.py; the file was untracked, so this copy was rebuilt by undoing the edits and accepted only on an exact sha match) -- not needed unless a resumed arm fails its own checks",
        "execution_order": ["cancel the vc-sub k500 waiter (done, user-approved) so no job runs the old file against the new one", "python scratchpad/_h2l.py TEST", "write this addendum",
                            "rx run (under _host_yield.py): python -u scratchpad/_vc_subk.py 500 25 VCQ512L6N4FM2 VCQ512L6N8FM2 VCQ512L6FM2 VCQ512L6N4LP2   (arms 1 and 2 are skipped as 'exists')",
                            "python -u scratchpad/_vc_eval.py EVAL vc_v1 v1 top_v1 H2LT_VCQ512L6N4FM2_25,H2LT_VCQ512L6N8FM2_25,H2LT_VCQ512L6FM2_25,H2LT_VCQ512L6N4LP2_25   (the only step that reads gold)"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
