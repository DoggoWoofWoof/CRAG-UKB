"""FBX_SCALE declaration addendum 18 (write-once): the level-0 V-cycle stage (E6, level 0) is RE-SIZED from a measured failure.

Addendum 17 declared the out-of-core V-cycle engine and a stage budget of 70 GB peak RSS ('a stage reserves what it measured, not more').  The first level-0 run (job 261004-041235-ooc-fb-vc0-a7f9, 2026-10-04) used a 41 GB reservation
(engine anonymous-memory cap anon_gb=40 = ulimit -d) and died in the FM2 pass-0 candidate build with std::bad_alloc: the 41 GB was an ESTIMATE made before level 0 had ever been run, not a measurement.  This addendum records that
measurement and the new reservation; it changes no gate, no engine code, no earlier record, and reads no gold label.   Usage: python scratchpad/_ooc18_addendum.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_18.json"
A17 = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_17.json"
REFS = [FB + "/E6_VCYCLE_L1-1__fbx.json", FB + "/E6_VCYCLE_L4-2__fbx.json", FB + "/E_STRESS__fbx_ooc.json"]
CODE = ["scratchpad/_ooc.cpp", "scratchpad/_ooc2.cpp", "scratchpad/_ooc_lib.py", "scratchpad/_ooc_vc.py", "scratchpad/_host_yield.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    rec = {
        "stage": "FBX_SCALE / addendum 18: level-0 V-cycle (E6, level 0) reservation re-sized from the measured pass-0 failure of job 261004-041235-ooc-fb-vc0-a7f9",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {A17: sha(A17)},
        "measured": {
            "job": "261004-041235-ooc-fb-vc0-a7f9 (python scratchpad/_ooc_vc.py VCYCLE ... 4 levels=0-0 anon_gb=40; rx 4 cpu 41 GB), runtime 13m36s, rc 1 (engine exit 134)",
            "phases_with_peak_rss": {"implicit level 0 / split built (V 301,977,131, M 301,926,716, S 25)": "16.36 GB at 32.8 s", "phi built (7.55 GB phi8, 109,833 big nets)": "30.16 GB at 132.3 s",
                                     "uint16 gain cache built (k 25: V x 25 x 2 B = 15.1 GB)": "46.53 GB at 497.8 s", "pass 0 candidate build (parallel eval of every vertex with a cut net)": "std::bad_alloc under RLIMIT_DATA 40 GB"},
            "what_it_shows": "most likely the engine's own anonymous-memory cap tripped (RLIMIT_DATA 40 GB counts anonymous memory only; the 46.53 GB peak RSS includes file-backed pages): the 41 GB reservation was an estimate made before level 0 had ever been run.  NOT excluded: a host-level allocation failure (host RAM was 92 % used at 04:21 while the job ran); the rerun logs peak RSS per phase and the engine cap differs, which discriminates the two",
            "nothing_written": "no lab0.i32 and no E6 record: the failed job produced no result; the stage is simply to be rerun with the reservation below",
        },
        "model_for_the_remaining_allocation": {
            "measured_at_level_1": "PROGRESS #5 of the state file: level 1 (job 261003-200055-ooc-fb-vc1-60a7, V 178,333,613) held anon_gb=26 with a 27 GB reservation; fm_peak_rss 40.5 GB of which ~20 GB file-backed mmap; anon ~20.6 GB = V*63 B + phi 4.41 GB + heap*16 B (heap peak 311.7M entries = 1.75 V)",
            "level0_at_the_gain_cache": "V*63 = 19.03 GB + phi 7.55 GB = 26.6 GB anon; peak RSS 46.53 GB at that point = 26.6 anon + ~20 GB file-backed, the same file-backed share as level 1",
            "level0_pass0_candidate_build": "per-thread candidate vectors (capacity up to 2x their size by push_back doubling) + the heap copy: 26.6 + 4.8 (heap, <= V entries x 16 B) + <= 9.7 (vector capacities) = <= 41.1 GB: ABOVE the 40 GB cap, which is where the job died",
            "level0_move_loop": "heap grows to 1.75 x 302M x 16 B = 8.5 GB; the doubling from the 4.8 GB reserved keeps old + new alive: 26.6 + 14.4 = 41.0 GB",
            "predicted_peak": "anon ~41-42 GB; peak RSS ~ 42 + 20 = ~62 GB, below the 70 GB stage budget of addendum 17",
        },
        "new_reservation": {"engine_anonymous_cap": "anon_gb=46 (RLIMIT_DATA; ~4 GB margin over the predicted 41-42 GB; still a hard safeguard against a runaway)",
                            "rx": "4 cpu, 47 GB (= anon cap + 1 GB, the level-1 precedent: 26 cap / 27 GB reserved with file-backed pages not charged to the reservation), under scratchpad/_host_yield.py, resumable at level granularity",
                            "budget_check": "predicted peak RSS ~62 GB <= 70 GB (addendum 17); the reservation is the measured need plus margin, not a push-through of a failing stage",
                            "host_fit": "the rx scheduler had 119 / 119.7 GB reserved at 2026-10-04 04:40 (mpr 102 GB): the job is launched only when 47 GB is admitted; no waiting job is queued without the user's say-so"},
        "withdrawn_draft": "a first draft of this addendum (66 GB reservation / anon_gb=54, sized from RSS instead of anonymous memory) was withdrawn before any use and is kept under results/FREEBASE_SCALE/_history/",
        "stop_rules": ["if the rerun's measured peak RSS exceeds 70 GB the stage stops and reports (user's call); no further enlargement is made without a new addendum",
                       "a stage that fails its WebQSP regression blocks the next stage (addendum 17, unchanged)",
                       "no gate is relaxed; no gold label is read; no earlier record is edited (supersession only)"],
        "not_done_here": ["any change to the engine code (hash pinned below), the heap representation or the move sequence", "any claim of recall at Freebase scale", "the SK arm"],
        "records_read": {p: sha(p) for p in REFS if os.path.exists(p)},
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
