"""Pre-registration of the DEV_B confirmation for the L1_STATIC lane (write-once; run BEFORE any DEV_B number
of a candidate arm is read).  Freezes: the candidate (module _l1s_candidate.py), the code that evaluates it,
the populations, the recorded DEV_A numbers the confirmation must reproduce, and the decision rule."""
import hashlib
import io
import json
import os
import shutil
import sys
import time

import _l1s_core as S
import _l1s_candidate as CAND
import _l1x90_core as X

OUT = S.OUT
PRE = os.path.join(OUT, "PREREGISTRATION_DEV_B.json")
if os.path.exists(PRE):
    sys.exit("REFUSED: %s exists (write-once; supersede by hash, never edit)" % PRE)
SNAP = os.path.join(OUT, "code_snapshot")
os.makedirs(SNAP, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ["_l1s_candidate.py", "_l1s_core.py", "_l1s_devB_confirm.py", "_l1x90_core.py", "_ta_prepartition.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


code = {}
for fn in CODE:
    p = os.path.join(HERE, fn)
    code[fn] = sha(p)
    shutil.copyfile(p, os.path.join(SNAP, fn))
caches = {name: {"path": os.path.relpath(X.CACHES[name], S.REPO), "sha256": sha(X.CACHES[name])} for name in ("metaqa", "squad", "musique")}
freeze = os.path.join(S.REPO, "data", "final_canonical", "CANONICAL_FREEZE.json")
devA = {}
for name in ("metaqa", "squad", "musique"):
    r = json.load(io.open(os.path.join(OUT, "radius_A_%s.json" % name), encoding="utf-8"))
    devA[name] = {"BASE": r["BASE_A"]["ALL"]}
    for arm, sub in CAND.ARMS.items():
        v = r["arms"]["+".join(sub)]
        devA[name][arm] = {"ALL": v["ALL"], "gained": v["gained"], "lost": v["lost"], "p": v["p"], "per_hop": v["per_hop"]}
rec = {
    "record": "PREREGISTRATION_DEV_B",
    "lane": "L1_STATIC",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "research_question": "How much of the gap between the frozen static L1 router (BASE, P50) and the dynamic-reasoning "
                         "ceiling can a purely static, parameter-free L1 router recover, uniformly across datasets?",
    "candidate": {"module": "_l1s_candidate.py", "definition": CAND.__doc__, "arms": {k: list(v) for k, v in CAND.ARMS.items()},
                  "frozen_numerics": {"K0": S.K0, "K_LOCK": S.K_LOCK, "P_MAIN": S.P_MAIN,
                                      "partition_ranking": "_ta_prepartition.partition_ranking (imported, unchanged)",
                                      "rrf": "_ta_prepartition.rrf_partitions (imported, unchanged)"},
                  "inputs": "served dense top-100 and served SPLADE top-100 node hits (replay caches), the frozen H4_SK "
                            "partition, the structural family of the canonical adapter (directed and undirected CSR); "
                            "nothing is walked at query time, no parameter is fitted, no dataset-specific rule"},
    "populations": {"exploration": "DEV_A = replay-cache dev rows with sha1(query_id)[:8] & 1 == 0 (all exploration read DEV_A only)",
                    "confirmation": "DEV_B = the complementary rows (sha1(query_id)[:8] & 1 == 1); never read for any candidate arm before this record",
                    "caches": caches, "test_split": "never read"},
    "recorded_DEV_A": devA,
    "decision_rule": {
        "primary_passes_iff": [
            "on EACH of metaqa, squad, musique DEV_B: NOT (lost > gained and exact McNemar p < 0.05)   [no significant loss anywhere]",
            "on AT LEAST ONE of the three DEV_B populations: gained > lost and exact McNemar p < 0.01     [a significant gain]"],
        "secondary": "evaluated in the same run under the same rule; adopted only if the PRIMARY fails; reported regardless",
        "reproduction_guard": "the confirmation run must reproduce the recorded DEV_A ALL / gained / lost of both arms exactly, else it aborts",
        "per_hop": "reported, never used for the decision",
        "full_dev": "A u B numbers reported after the DEV_B decision, for the record only",
        "no_other_arm": "no other arm, table subset, weight kernel or partition variant is evaluated on DEV_B",
        "one_shot": "the confirmation script refuses to run if DEV_B_CONFIRMATION.json exists",
    },
    "code_sha256": code,
    "code_snapshot_dir": os.path.relpath(SNAP, S.REPO),
    "canonical_freeze_sha256": sha(freeze) if os.path.exists(freeze) else None,
    "adoption_note": "a PASS is a lane result on the canonical replay caches; adopting the candidate as the served L1 "
                     "selector changes the L1 contract (new selector module) and needs a ruling -- no cache is rebuilt "
                     "(the candidate reads only served caches + the structural family)",
}
S.wj(PRE, rec)
print("wrote", os.path.relpath(PRE, S.REPO))
print("record sha256", sha(PRE))
for k, v in code.items():
    print("  %-22s %s" % (k, v))
for k, v in caches.items():
    print("  cache %-8s %s" % (k, v["sha256"]))
for name, d in devA.items():
    print("  DEV_A %-8s BASE %.4f  PRIMARY %.4f (+%d/-%d p=%.1e)  SECONDARY %.4f (+%d/-%d p=%.1e)" % (
        name, d["BASE"], d["PRIMARY"]["ALL"], d["PRIMARY"]["gained"], d["PRIMARY"]["lost"], d["PRIMARY"]["p"],
        d["SECONDARY"]["ALL"], d["SECONDARY"]["gained"], d["SECONDARY"]["lost"], d["SECONDARY"]["p"]))
