# -*- coding: utf-8 -*-
"""
CRAG_FREEBASE_NAME_HIERARCHY_V1
===============================
How to resolve a display name for any node in the 302M-node canonical Freebase universe, and
what each tier is and is not worth.

    tier 1  ORIGINAL NAME            overlay_v1 where is_original_name -- source-derived
    tier 2  ADMISSIBLE INFERRED      inference_overlay_v1, admissible under v1 AND not
                                     rejected by the v2 amendment
    tier 3  FLOOR                    overlay_v1's display name, "Unnamed Freebase entity"
                                     for nodes that never had one

THE TIERS ARE DISJOINT, AND THAT IS MEASURED RATHER THAN ASSUMED
    A precedence rule is only meaningful if tier 2 never silently overwrites tier 1. Check E
    established that ZERO of the 64,038,024 inferred nodes already carried an original name,
    and did so behind a positive control: all 64,038,024 were found in the frozen overlay
    first. Without the control the zero would be the same shape as a zero from an empty column,
    which is exactly how the first attempt at that check failed.

THE INVARIANT
    Applying this hierarchy can never make a node's name LESS specific than the floor. Every
    rejection -- bookkeeping name in v1, unreadable identity payload in v2 -- returns the node
    to its frozen display name. The baseline is the floor, not something the inference layer is
    permitted to degrade.

WHAT THIS IS NOT
    This is NOT a claim that 302M nodes have 302M original Freebase proper names. That
    information did not exist, and no amount of processing creates it. 82.5M nodes resolve to
    the floor and are reported as such rather than dressed up with a generated string.

    Tier 2 names are GENERATED, not recovered. They are deterministic renderings of a node's
    own relations, useful as retrieval and display surfaces, and must never be presented as
    authentic Freebase labels. The two tiers are kept distinguishable in the data for exactly
    that reason -- a consumer can always ask which tier a name came from.

EVERY NUMBER BELOW IS COPIED FROM A VERIFIED RECORD, WITH ITS SOURCE NAMED
    Nothing here is recomputed or estimated; this script assembles a contract from artifacts
    that were each measured and gated elsewhere, and fails loudly if they disagree.
"""
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
V2 = FB + "/inference_admissibility_v2_amendment.json"
VAL1 = "scratchpad/fb/validate_inference.json"
VAL2 = "scratchpad/fb/validate_inference2.json"
OUT = FB + "/NAME_HIERARCHY_CONTRACT_V1.json"


def load(p):
    if not os.path.exists(p):
        sys.exit("missing prerequisite: %s" % p)
    return json.load(io.open(p, encoding="utf-8"))


def main():
    t0 = time.time()
    v2 = load(V2)
    d2 = load(VAL2)
    e = d2["checks"]["E_FLOOR_INTACT"]

    total = int(e["frozen_overlay_rows"])
    original = int(e["frozen_overlay_original_named_nodes"])
    inferred = int(e["inferred_nodes"])
    overlap = int(e["inferred_nodes_that_already_had_an_original_name"])
    if not e["POSITIVE_CONTROL_PASS"]:
        sys.exit("refusing to write a precedence contract on an unverified join")
    if overlap != 0:
        sys.exit("tiers 1 and 2 overlap on %d nodes; precedence is not well defined" % overlap)

    # read the exact count, never reconstruct it: as_fraction_of_admissible is rounded, and
    # dividing 110 by it lands ~12 rows off -- close enough to look right in a contract and
    # wrong enough to be a lie.
    d1 = load(VAL1)
    adm_v1 = int(d1["ADMISSIBLE_N"])
    rej_v1 = int(d1["REJECTED_N"])
    if adm_v1 + rej_v1 != inferred:
        sys.exit("v1 admissible+rejected (%d) != inferred rows (%d)"
                 % (adm_v1 + rej_v1, inferred))
    rejected_v2 = int(v2["SCALE"]["rows_rejected_by_this_amendment"])
    adm_eff = adm_v1 - rejected_v2
    floor = total - original - adm_eff
    if floor < 0:
        sys.exit("tier arithmetic is inconsistent: floor would be negative")

    rec = {
        "RECORD": "CRAG_FREEBASE_NAME_HIERARCHY_V1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "STATUS": "FROZEN",
        "SCOPE": ("resolution of a display name for any node in the canonical Freebase "
                  "universe. Does not change any frozen artifact; it states how they compose."),
        "PRECEDENCE": [
            {"tier": 1, "name": "ORIGINAL_NAME",
             "source": FB + "/overlay_v1 (is_original_name = true)",
             "nodes": original,
             "what_it_is": "a name carried by the source. Authentic.",
             "authority": "SOURCE_DERIVED"},
            {"tier": 2, "name": "ADMISSIBLE_INFERRED_NAME",
             "source": (FB + "/inference_overlay_v1, gated by inference_admissibility_v1 "
                        "(reject_reason 0) and then by the v2 amendment"),
             "nodes": adm_eff,
             "what_it_is": ("a deterministic rendering of the node's own relations. Useful as "
                            "a retrieval and display surface."),
             "authority": "GENERATED -- never to be presented as an authentic Freebase label"},
            {"tier": 3, "name": "FLOOR",
             "source": FB + "/overlay_v1 display name",
             "nodes": floor,
             "what_it_is": ("the frozen display name, 'Unnamed Freebase entity' where the node "
                            "never had one. Reported honestly rather than filled in."),
             "authority": "FLOOR"}],
        "RESOLUTION_RULE": (
            "display_name(node) = original_name if is_original_name "
            "else admissible_inferred_name if reject_reason == 0 and node not in v2 amendment "
            "else frozen floor"),
        "EFFECTIVE_REJECT_REASON": (
            "v2_amendment.get(node_uid) or inference_admissibility_v1[row]. v1 is frozen and "
            "unedited; the amendment is an overlay with its own hash."),
        "TIERS_ARE_DISJOINT": {
            "claim": "no node can be resolved by two tiers, so precedence never overwrites a "
                     "source-derived name with a generated one",
            "inferred_nodes": inferred,
            "inferred_nodes_that_already_had_an_original_name": overlap,
            "POSITIVE_CONTROL_inferred_nodes_found_in_overlay":
                int(e["POSITIVE_CONTROL_inferred_nodes_found_in_overlay"]),
            "why_the_control_matters": (
                "a zero overlap is only evidence if the join reached the population. The first "
                "attempt at this check read is_original_name from inferred_name_v1, where it "
                "is false on all 69,243,435 rows by construction, and returned zero for the "
                "same reason an empty table does. That result was discarded, not counted."),
            "evidence": VAL2},
        "INVARIANT_THE_FLOOR_IS_NEVER_LOWERED": (
            "every rejection path -- bookkeeping relation name (v1 reason 1), unreadable "
            "identity payload (v2 reason 2) -- returns the node to its frozen display name. "
            "The existing baseline is the floor, not something a new pass may degrade."),
        "COVERAGE": {
            "nodes_total": total,
            "named_by_tier_1_or_2": original + adm_eff,
            "pct_named": round(100.0 * (original + adm_eff) / total, 4),
            "resolve_to_floor": floor,
            "pct_floor": round(100.0 * floor / total, 4)},
        "WHAT_THIS_IS_NOT": [
            ("NOT a claim that 302M nodes have 302M original Freebase proper names. That "
             "information did not exist and processing does not create it."),
            ("tier 2 names are GENERATED, not recovered. They are deliberately kept "
             "distinguishable from tier 1 so a consumer can always ask which tier a name came "
             "from, and so a generated string is never mistaken for an authentic label."),
            ("NOT a reconstruction of the official final Google Freebase dump. The universe is "
             "constructed from the freebase-rdf-latest.gz snapshot preserved by the Internet "
             "Archive, whose downloaded artifact was verified against the archive-published "
             "size and checksums.")],
        "PROVENANCE": {
            "tier_1_and_totals": VAL2 + " :: checks.E_FLOOR_INTACT",
            "tier_2_admissible_v1": VAL1 + " :: ADMISSIBLE_N (exact, cross-checked against "
                                           "REJECTED_N summing to the layer's row count)",
            "tier_2_rejections_v2": V2,
            "v1_admissibility_gates": VAL1,
            "note": "no number in this record is recomputed here; each is copied from a "
                    "measured, gated artifact and cross-checked for arithmetic consistency"},
        "elapsed_s": None}

    rec["elapsed_s"] = round(time.time() - t0, 2)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    body = io.open(OUT, "rb").read()
    rec["RECORD_SHA256"] = hashlib.sha256(body).hexdigest()
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("tier 1 ORIGINAL           %15s  %6.2f%%" % (format(original, ","),
                                                       100.0 * original / total))
    print("tier 2 ADMISSIBLE INFERRED%15s  %6.2f%%   (v1 %s minus %d rejected by v2)"
          % (format(adm_eff, ","), 100.0 * adm_eff / total, format(adm_v1, ","), rejected_v2))
    print("tier 3 FLOOR              %15s  %6.2f%%" % (format(floor, ","),
                                                       100.0 * floor / total))
    print("                          %15s  total" % format(total, ","))
    print("\nnamed by tier 1 or 2: %.4f%%" % rec["COVERAGE"]["pct_named"])
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
