"""FBX_SCALE declaration addendum 21 (write-once): the level-0 chain job (addendum 19) reached its 51 GB window at 2026-10-04 19:05 and FAILED IN 27 s in its WebQSP-regression step on a missing host input (not an engine result, not a
regression); the cause is the host workspace, the fix is the rx push list, and the chain is re-armed with an unchanged reservation after the same regression passed stand-alone on the patched binary.  It changes no gate, no algorithm,
no engine source, no reservation and no earlier record, and reads no gold label.
Usage: python scratchpad/_ooc21_addendum.py   (refuses to overwrite)"""
import hashlib
import io
import json
import os
import shutil
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_21.json"
A19 = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_19.json"
LOGS = {"work/_vc0_launch_fail_job.log": FB + "/E_VC0_LAUNCH_FAIL__fbx_ooc.log", "work/_wqvc_pre2_job.log": FB + "/E_WQVC_PRE__fbx_ooc.log"}
CODE = ["scratchpad/_ooc2.cpp", "scratchpad/_ooc_vc.py", "scratchpad/_ooc_wq_chain.py", "scratchpad/_vc0_chain.py", "scratchpad/_host_yield.py", "scratchpad/_rx_gate.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    a19 = json.load(io.open(A19, encoding="utf-8"))
    for c in ("scratchpad/_ooc2.cpp", "scratchpad/_ooc_vc.py", "scratchpad/_ooc_wq_chain.py", "scratchpad/_vc0_chain.py", "scratchpad/_host_yield.py"):
        assert a19["code_pinned"][c] == sha(c), "%s changed since addendum 19 pinned it" % c
    for src, dst in LOGS.items():
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
    rec = {
        "stage": "FBX_SCALE / addendum 21: level-0 chain job launch failure (job 261004-190517-ooc-fb-vc0-0705, 27 s): a host workspace input was missing; fixed in the push list; regression re-verified stand-alone",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {A19: sha(A19)},
        "measured": {
            "window": "the capacity gate admitted the 51 GB job at 2026-10-04 19:05 (scheduler 57.7 GB free) and submitted job 261004-190517-ooc-fb-vc0-0705 (4 cpu / 51 GB, anon_gb 50, --mem-hard 63)",
            "failure": "rc 1 after 27 s in step 1 of scratchpad/_vc0_chain.py (the WebQSP regression, _ooc_wq_chain.py VC): FileNotFoundError for results/FREEBASE_SCALE/ml2/clusters__M2_W512_L1.npy in the host workspace (log %s, sha %s). "
                       "The Freebase level was NOT run (the chain's stop rule worked as declared); no engine, no cap and no memory result exists from this job; peak memory was not reached" % (LOGS["work/_vc0_launch_fail_job.log"], sha(LOGS["work/_vc0_launch_fail_job.log"])),
            "second_finding": "stand-alone retry 261004-191009-ooc-wq-vc-pre-37eb (3 cpu / 4 GB, 1m11s, rc 1): after the ml2 files were declared, the pushed record results/FREEBASE_SCALE/h2l/parts/webqsp__H4_H2LTOP_k25.json made _l1h_host.cmd_hg assert the hash of the stored top-hypergraph npz "
                              "(which is not kept; the npz is REBUILT when its record is absent, as on 2026-10-03) -> FileNotFoundError; the record was removed from the push list and from the host workspace",
            "regression_verified_standalone": {"job": "261004-191245-ooc-wq-vc-pre2-6346", "reservation": "3 cpu / 4 GB, --mem-hard 8", "runtime": "7m06s", "peak_mem_gb": 1.04, "rc": 0,
                                               "results": "E4 PASS (surrogate shards byte-identical; Zoltan top reproduces the stored top map sha d9c3c4aab8b9), E6 PASS (the V-cycle reproduces the stored top map byte for byte, sha 4c043f0cd704, and the D1 trajectory and V / P of levels 1..4), validity PASS (max block 104,752 <= bound 106,828), WQ_CHAIN PASS: VC",
                                               "on_the_patched_binary": "compiled source hash d7d48c711233 (addendum 19)", "log": LOGS["work/_wqvc_pre2_job.log"], "log_sha256": sha(LOGS["work/_wqvc_pre2_job.log"])},
        },
        "cause": {
            "what": "the host workspace did not contain the stored WebQSP records that scratchpad/_ooc_vc.py test_webqsp_vc compares against (results/FREEBASE_SCALE/ml2/clusters__M2_W512_L1..4.npy; results/FREEBASE_SCALE/h2l/parts/webqsp__H4_H2LT_Q512L4N4_top_k25__PHG_con.{RUN.json,npy}); "
                    "they were not in the rx push list (the 2026-10-03 regression runs had them present, evidently pushed per job), and addendum 19 did not verify the chain's inputs on the host before arming the 51 GB window",
            "lesson": "a heavy stage's PRE-STEP inputs are verified by a cheap stand-alone run of that pre-step on the host before the heavy window is spent; done here before the re-arm",
        },
        "fix": {"what": "rx.toml push list only: the four ml2 cluster maps, COARSEN__M2_W512.json, the Q512L4N4 top RUN.json / npy and the vc/ D1.json / npy the test reads (the top-hypergraph record is deliberately NOT pushed)",
                "no_change_to": ["the engine source", "the chain script", "the algorithm", "any gate", "the reservation (rx 4 cpu / 51 GB, anon_gb 50, --mem-hard 63)", "any earlier record"]},
        "rearmed": "scratchpad/_rx_gate.py --name ooc-fb-vc0 --mem 51 --cpus 4 --hard 63 -- python -u scratchpad/_vc0_chain.py anon_gb=50 (gate log work/_vc0_gate3.log); the chain still re-runs the regression itself before level 0 (addendum 19 stop rule)",
        "stop_rules": "unchanged from addendum 19: peak RSS above 70 GB stops and reports; a further failure of the stage is diagnosed from the per-pass peak-RSS lines and enlarged only by a new addendum; a regression failure blocks the Freebase level; no gate is relaxed",
        "not_done_here": ["any Freebase level-0 result", "any change to the algorithm or the reservation", "any claim of recall at Freebase scale"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
