"""FBX_SCALE declaration addendum 19 (write-once): the level-0 V-cycle stage (E6, level 0) failed a SECOND time at the addendum-18 reservation; the cause is identified in the engine code, fixed by a 2-line memory-LIFETIME change of
scratchpad/_ooc2.cpp (bit-identity against the frozen reference re-verified), and the stage is re-sized from the corrected model.  It changes no gate, no algorithm, no earlier record, and reads no gold label.
Usage: python scratchpad/_ooc19_addendum.py   (refuses to overwrite)"""
import hashlib
import io
import json
import os
import shutil
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
HIST = FB + "/_history"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_19.json"
A17 = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_17.json"
A18 = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_18.json"
OLD_SRC = "work/_ooc2.cpp.pre_addendum19"
NEW_SRC = "scratchpad/_ooc2.cpp"
FMLOG_SRC = "work/_fmtest_a19.log"
FMLOG = FB + "/E_FM19__fbx_ooc.log"
CODE = ["scratchpad/_ooc.cpp", "scratchpad/_ooc2.cpp", "scratchpad/_ooc_lib.py", "scratchpad/_ooc_vc.py", "scratchpad/_ooc_wq_chain.py", "scratchpad/_vc0_chain.py", "scratchpad/_host_yield.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    old_sha = sha(OLD_SRC)
    a18 = json.load(io.open(A18, encoding="utf-8"))
    assert a18["code_pinned"]["scratchpad/_ooc2.cpp"] == old_sha, "the saved pre-change source is not the one addendum 18 pinned"
    os.makedirs(HIST, exist_ok=True)
    keep = HIST + "/_ooc2.cpp.pre_addendum19__%s" % old_sha[:8]
    if not os.path.exists(keep):
        shutil.copyfile(OLD_SRC, keep)
    if not os.path.exists(FMLOG):
        shutil.copyfile(FMLOG_SRC, FMLOG)
    rec = {
        "stage": "FBX_SCALE / addendum 19: level-0 V-cycle (E6, level 0) second failure (job 261004-093615-ooc-fb-vc0-392a): cause found in the engine, memory-lifetime fix, re-sized reservation",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {A17: sha(A17), A18: sha(A18)},
        "measured": {
            "job": "261004-093615-ooc-fb-vc0-392a (addendum-18 reservation: rx 4 cpu / 47 GB, anon_gb=46), runtime 24m31s, rc 1 (engine exit 134, std::bad_alloc under RLIMIT_DATA 46 GB; the engine runs as wsl.exe bash -c 'ulimit -d ...')",
            "phases_with_peak_rss": {"implicit level 0 / split built": "16.38 GB at 34.9 s", "phi built": "30.16 GB at 164.3 s", "uint16 gain cache built": "46.53 GB at 697.7 s (identical to the first run)",
                                     "pass 0 candidate build": "291,726,641 candidate vertices, peak RSS 52.83 GB at 832.5 s (the first run died HERE at the 40 GB cap)",
                                     "pass 0 moves": "done at 1051.5 s: 1,355,841 tried / 1,355,341 kept, gain 123,323,517, heap peak 407,094,339 entries (1.35 V)",
                                     "after pass 0": "std::bad_alloc inside the pass-1 candidate build; nothing written (no lab0.i32, no E6 record)"},
            "what_it_shows": "raising the cap 40 -> 46 GB moved the failure from pass 0 to pass 1 of the same stage: the failure is the engine's own anonymous-memory cap (RLIMIT_DATA), not a host-level allocation failure (the discriminating test promised in addendum 18); "
                             "addendum 18's model was incomplete, it omitted what pass 0 leaves allocated",
        },
        "cause_in_the_code": {
            "mechanism": "scratchpad/_ooc2.cpp fm2 pass loop: fm.heap.reserve(tot) takes exactly the 291.7M candidates (4.67 GB at 16 B / Ent); pass 0's moves push the heap to 407.1M entries so std::vector doubles its capacity to 583.4M = 9.33 GB (old + new alive during the copy); "
                         "the next pass's `fm.heap.clear()` keeps that 9.33 GB capacity (pages stay resident) while the per-thread candidate vectors (sum of sizes 4.67 GB, capacities up to 2x = 9.3 GB, plus one doubling copy) are built on top of it",
            "model_pass_k_ge_1_before_the_fix": "26.6 GB (V*63 B + phi) + 9.33 (retained heap capacity) + up to 9.7 (candidate vector capacities) + ~1.2 (one doubling copy) = up to ~46.8 GB: above the 46 GB cap, which is where the job died; the same model reproduces the first failure (40 GB cap vs 41.1 GB in pass 0)",
            "model_after_the_fix": "26.6 + 4.67 (heap.reserve) + up to 9.7 (candidate vectors still alive while inserting) = ~41 GB; the pass-0 move loop is unchanged: 26.6 + 4.67 + 9.33 (reserve + doubling copy) = ~40.6 GB; so anon <= ~42 GB in every pass",
        },
        "engine_change": {
            "what": "two lines of scratchpad/_ooc2.cpp: `fm.heap.clear();` -> `std::vector<Ent>().swap(fm.heap);` (releases the previous pass's capacity) and the `pass N done` log line also prints the peak RSS (progress text only)",
            "why_results_are_unchanged": "heap order is the strict total order (gain desc, vertex asc, block asc) and reserve / capacity never enter push_heap / pop_heap / make_heap; the fix touches allocation lifetime only; no algorithm, tie-break, move, label or statistic can change",
            "verified": "TEST_FM on the patched binary (compiled source sha d7d48c711233): 5 random explicit hypergraphs and 5 implicit closed-neighbourhood / split-preserve cases, 6 passes each, labels / km1 / moves / loads / re-pushes bit-identical to the frozen sequential reference scratchpad/_vc_ref2.c mode 1 "
                        "(log %s, sha %s)" % (FMLOG, sha(FMLOG)),
            "also_verified_by_the_stage_job_itself": "the stage job re-runs the WebQSP end-to-end regression (E4 surrogate + Zoltan top, E6 V-cycle: top map byte for byte, D1 trajectory, V / P) on the patched binary BEFORE the Freebase level; a failure stops the job",
            "pre_change_source": {"kept_as": keep, "sha256": old_sha, "pinned_by": "addendum 18 code_pinned"},
            "resume": "the V-cycle's step markers do not depend on the engine hash: the stored levels 1..4 (E6_VCYCLE_L1-1 / L4-2) stay valid; level 0 alone is run",
        },
        "new_reservation": {"engine_anonymous_cap": "anon_gb=50 (RLIMIT_DATA; ~8 GB over the ~42 GB predicted anonymous peak, because two successive models were each ~1-6 GB short; still a hard safeguard)",
                            "rx": "4 cpu, 51 GB (anon cap + 1 GB, the level-1 precedent), --mem-hard 63, under scratchpad/_host_yield.py, ONE job = regression chain then level 0 (scratchpad/_vc0_chain.py), launched only when the scheduler admits 51 GB (capacity gate, no queued job)",
                            "budget_check": "predicted peak RSS ~42 anonymous + ~20 file-backed mmap = ~62 GB <= 70 GB (addendum 17)"},
        "stop_rules": ["peak RSS above 70 GB: the stage stops and reports (user's call)",
                       "a further bad_alloc / failure of this stage: stop, diagnose from the per-pass peak-RSS lines now logged, and enlarge only by a new addendum",
                       "the WebQSP regression inside the job fails: the Freebase level is NOT run",
                       "no gate is relaxed; no gold label is read; no earlier record is edited (the changed source's predecessor is kept under _history/)"],
        "not_done_here": ["any change to the algorithm, the heap order, the move sequence or the gain cache", "any claim of recall at Freebase scale", "the SK arm"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
