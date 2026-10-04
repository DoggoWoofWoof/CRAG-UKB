# -*- coding: utf-8 -*-
"""
LOCKED_FREEBASE_INFERENCE_V1
============================
The freeze record for the Freebase generated-name layer: an inventory of every part with its
content hash, the counts that define it, the gates that were run with their verdicts, and the
limitations that a consumer would otherwise discover the hard way.

WHAT IS AND IS NOT IN SCOPE
    IN   inferred_name_v1, inference_overlay_v1, inference_admissibility_v1, the v2 amendment
         and the name-hierarchy contract.
    OUT  overlay_v1 (CRAG_FREEBASE_RESOLUTION_OVERLAY_V1) and the canonical universe itself.
         Both were frozen earlier with their own hashes and are referenced, not re-frozen --
         re-hashing 8.3 GB to restate a number that is already on record adds nothing, and a
         second hash of a frozen artifact invites the question of which one is authoritative.

WHY THE ADMISSIBILITY SIDECAR IS THE INTERESTING ARTIFACT
    The generated names are only as good as the rule that decides which of them are names at
    all. Two rejection reasons are frozen here:

        1  BOOKKEEPING_RELATION_NAME    (v1)  2,015,907 rows
        2  UNREADABLE_IDENTITY_PAYLOAD  (v2)         36 rows

    Both share one justification: a rejected node falls back to its frozen display name, which
    IS the baseline. Rejection returns a node to the floor and can never push it below.

    The honest cost of reason 1 is stated rather than buried: 1,298,608 apparent "upgrades"
    (40.9% of Group E) are withdrawn, because they were never coverage -- every one of them
    read "Last referenced by of X".
"""
import glob
import hashlib
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FB = "data/final_canonical/freebase_v3"
LAYERS = ["inferred_name_v1", "inference_overlay_v1", "inference_admissibility_v1"]
VAL1 = "scratchpad/fb/validate_inference.json"
VAL2 = "scratchpad/fb/validate_inference2.json"
V2 = FB + "/inference_admissibility_v2_amendment.json"
CONTRACT = FB + "/NAME_HIERARCHY_CONTRACT_V1.json"
OUT = FB + "/LOCKED_FREEBASE_INFERENCE_V1.json"
BUF = 8 << 20


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(p):
    if not os.path.exists(p):
        sys.exit("missing prerequisite: %s" % p)
    return json.load(io.open(p, encoding="utf-8"))


def main():
    t0 = time.time()
    d1, d2, v2, con = load(VAL1), load(VAL2), load(V2), load(CONTRACT)

    rec = {"RECORD": "LOCKED_FREEBASE_INFERENCE_V1",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "STATUS": "FROZEN",
           "SCOPE": ("the Freebase generated-name layer. overlay_v1 and the canonical universe "
                     "are referenced by their existing frozen records, not re-frozen here."),
           "REFERENCES_NOT_FROZEN_HERE": {
               "resolution_overlay": {
                   "record": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1", "path": FB + "/overlay_v1",
                   "rows": int(d2["checks"]["E_FLOOR_INTACT"]["frozen_overlay_rows"]),
                   "sha256": ("25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27"
                              "d865"),
                   "note": "frozen and must not be modified"}},
           "layers": {}}

    for lay in LAYERS:
        parts = sorted(glob.glob("%s/%s/*.parquet" % (FB, lay)))
        if not parts:
            sys.exit("layer has no parts: %s" % lay)
        ent, tot = [], 0
        for p in parts:
            b = os.path.getsize(p)
            tot += b
            ent.append({"path": p.replace("\\", "/"), "bytes": b, "sha256": sha256(p)})
        roll = hashlib.sha256("".join(e["sha256"] for e in ent).encode()).hexdigest()
        rec["layers"][lay] = {"n_parts": len(ent), "bytes": tot, "layer_hash": roll,
                              "parts": ent}
        print("  %-30s %3d parts  %7.2f GB  %s  %.0fs"
              % (lay, len(ent), tot / 1e9, roll[:16], time.time() - t0), flush=True)

    rec["COUNTS"] = {
        "inferred_name_v1_rows": int(d1["checks"]["E_FLOOR_INTACT"]["name_layer_rows"]),
        "inference_overlay_v1_rows": int(d1["checks"]["A_STRUCTURE"]["total_rows"]),
        "admissible_v1": int(d1["ADMISSIBLE_N"]),
        "rejected_v1": int(d1["REJECTED_N"]),
        "rejected_v2_amendment": int(v2["SCALE"]["rows_rejected_by_this_amendment"]),
        "admissible_effective": int(d1["ADMISSIBLE_N"])
                                - int(v2["SCALE"]["rows_rejected_by_this_amendment"])}

    rec["GATES"] = {
        "A_STRUCTURE": {"verdict": "PASS", "claim": "parts, order and per-part row counts"},
        "B_REPRODUCIBLE": {
            "verdict": "PASS",
            "claim": ("all 64,038,024 admissibility decisions re-derived from the overlay's own "
                      "text, row for row, 0 mismatches. This matters more than it looks: the "
                      "sidecar carries no node_uid, so its alignment to the overlay is "
                      "positional and would otherwise be an assumption rather than a fact.")},
        "C_NO_BOOKKEEPING": {"verdict": "PASS",
                             "claim": "no admissible row has a bookkeeping relation head"},
        "D_UID_UNIQUE": {"verdict": "PASS", "claim": "node_uid unique across the layer"},
        "E_FLOOR_INTACT": {
            "verdict": "PASS",
            "claim": ("no inferred node already carried an original name, so tiers 1 and 2 are "
                      "disjoint and precedence is well defined"),
            "positive_control": ("64,038,024/64,038,024 inferred node_uids found in the frozen "
                                 "overlay. The control is what makes the zero admissible: the "
                                 "first attempt read is_original_name from inferred_name_v1, "
                                 "where it is false on all 69,243,435 rows by construction, "
                                 "and that vacuous zero was discarded rather than counted.")},
        "F_NAME_SANITY": {
            "verdict": "DISPOSITIONED",
            "claim": ("110 admissible names contain U+FFFD. 36 whose identity payload is "
                      "entirely replacement characters are rejected by the v2 amendment; 74 "
                      "that remain readable are deliberately kept."),
            "record": V2},
        "G_ACCOUNTING": {"verdict": "PASS",
                         "claim": "per-source accounting reconciles with the frozen record"},
        "evidence": [VAL1, VAL2]}

    for k, p in [("v2_amendment", V2), ("name_hierarchy_contract", CONTRACT)]:
        rec.setdefault("records", {})[k] = {
            "path": p, "bytes": os.path.getsize(p), "sha256": sha256(p)}

    rec["NAME_HIERARCHY"] = {"record": "CRAG_FREEBASE_NAME_HIERARCHY_V1",
                             "coverage": con["COVERAGE"],
                             "precedence": [t["name"] for t in con["PRECEDENCE"]]}

    rec["KNOWN_LIMITATIONS"] = [
        ("tier 2 names are GENERATED, not recovered. They are deterministic renderings of a "
         "node's own relations and must never be presented as authentic Freebase labels. The "
         "tiers are kept distinguishable so a consumer can always ask which one a name came "
         "from."),
        ("rejecting bookkeeping names withdraws 1,298,608 apparent upgrades, 40.916% of Group "
         "E. That is not lost coverage -- every one of them read 'Last referenced by of X', so "
         "they were never coverage in the first place. The number is stated because a "
         "40% withdrawal looks like a regression unless its cause is on the record."),
        ("the 74 kept mojibake names carry damage inherited from neighbour titles that were "
         "ALREADY corrupt in the source. The template is intact; the inference did not "
         "introduce the damage and cannot repair it."),
        ("82.5M nodes resolve to the floor and are reported as such. There is no claim that "
         "302M nodes have 302M original Freebase proper names -- that information did not "
         "exist.")]

    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    body = io.open(OUT, "rb").read()
    rec["RECORD_SHA256"] = hashlib.sha256(body).hexdigest()
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nadmissible effective %s  (v1 %s minus %d)"
          % (format(rec["COUNTS"]["admissible_effective"], ","),
             format(rec["COUNTS"]["admissible_v1"], ","),
             rec["COUNTS"]["rejected_v2_amendment"]))
    print("record sha256 (pre-stamp) %s" % rec["RECORD_SHA256"])
    print("wrote %s  %.1fs" % (OUT, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
