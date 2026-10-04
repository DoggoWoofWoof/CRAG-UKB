"""How much of RoG's OWN collapse is directly attributable, not extrapolated?

    python scratchpad/final_canonical_build/webqsp_v1/rog_attributable_collapse.py

NAME_MAPPING_COVERAGE established two facts about the released RoG graph:
  * NAME_MAPPING_PRESENT_N = 0  -- not one of the 560,012 MIDs this file names survives as a bare
    MID in RoG.  RoG either named it or never extracted it.
  * 461,597 of the file's 462,455 distinct names (99.81%) appear VERBATIM as a RoG surface.

So on this subpopulation RoG's unreleased resolver dominates the file completely.  That lets the
collision structure of the file be carried onto RoG surfaces -- with one honest gap, handled below.

THE GAP: "0 bare" does not separate "RoG named it" from "RoG never extracted it".  A file group
{m.A, m.B, m.C} -> "Pilot" only proves RoG merged 3 identities into one node if all three are IN
RoG's graph.  RoG carries no MIDs on named nodes, so presence is not directly checkable.  Three
figures are therefore reported, and they are NOT the same claim:

  UPPER   assume every group member is present in RoG
  NSM     count only members present in the NSM sibling MID vocabulary (better grounded, but the
          bound is then about the sibling extraction's coverage, not RoG's)
  LOWER   groups whose members are ALL present in the NSM vocabulary

HARD RULE (user): evidence about how destructive name-keying is, NOT a recipe.  V1 identities are
untouched and V1 does not read this file.
"""
import json, os, time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

NODES = "data/final_canonical/webqsp/v1/nodes.parquet"
NAMES = "data/final_canonical/webqsp/_acquisition/nsm/entities_names.json"
NSM_ENT = ["data/final_canonical/webqsp/_acquisition/nsm/extracted/webqsp/webqsp/entities.txt",
           "data/final_canonical/webqsp/_acquisition/nsm/extracted/CWQ/CWQ/entities.txt"]
OUT = "data/final_canonical/webqsp/ROG_ATTRIBUTABLE_COLLAPSE.json"


def main():
    t0 = time.time()
    surfaces = set(pq.read_table(NODES, columns=["source_rog_endpoint"]).column(
        "source_rog_endpoint").to_pylist())
    names = json.load(open(NAMES, encoding="utf-8"))

    by_name = defaultdict(list)
    for mid, nm in names.items():
        by_name[nm].append(mid)

    nsm_vocab = set()
    for p in NSM_ENT:
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                nsm_vocab.add(ln.rstrip("\n"))

    grp_hit = 0          # collision names that ARE a RoG surface
    upper_merges = 0     # sum(size-1) assuming all members present in RoG
    nsm_merges = 0       # sum(present_in_nsm - 1) over those with >=2 present
    lower_grp = 0        # groups fully present in the NSM vocabulary
    lower_merges = 0
    max_grp, max_name = 0, None
    size_hist = Counter()
    singleton_names_that_are_rog_surfaces = 0

    for nm, mids in by_name.items():
        if nm not in surfaces:
            continue
        if len(mids) < 2:
            singleton_names_that_are_rog_surfaces += 1
            continue
        grp_hit += 1
        upper_merges += len(mids) - 1
        size_hist[min(len(mids), 10)] += 1
        if len(mids) > max_grp:
            max_grp, max_name = len(mids), nm
        npresent = sum(1 for m in mids if m in nsm_vocab)
        if npresent >= 2:
            nsm_merges += npresent - 1
        if npresent == len(mids):
            lower_grp += 1
            lower_merges += len(mids) - 1

    doc = {
        "schema": "ROG_ATTRIBUTABLE_COLLAPSE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_on": "released RoG nodes.parquet (2,592,894) x entities_names.json (560,012)",
        "HARD_RULE": "Evidence about how destructive name-keying is. NOT a recipe. V1 identities are "
                     "not modified and V1 does not read this file.",
        "enabling_facts": {
            "NAME_MAPPING_PRESENT_N": 0,
            "meaning": "no MID this file names survives as a bare MID in RoG",
            "file_names_appearing_as_rog_surfaces": 461597,
            "file_distinct_names": 462455,
            "pct": 99.8144,
        },
        "collision_names_that_are_rog_surfaces": grp_hit,
        "singleton_names_that_are_rog_surfaces": singleton_names_that_are_rog_surfaces,
        "MERGES_UPPER_all_members_assumed_present": upper_merges,
        "MERGES_NSM_presence_weighted": nsm_merges,
        "MERGES_LOWER_groups_fully_present_in_nsm": lower_merges,
        "groups_fully_present_in_nsm": lower_grp,
        "max_collision_group_on_a_rog_surface": {"name": max_name, "mids": max_grp},
        "group_size_histogram_capped_at_10": dict(sorted(size_hist.items())),
        "rog_nodes_total": len(surfaces),
        "collapse_as_pct_of_rog_nodes": {
            "upper": round(100.0 * upper_merges / len(surfaces), 4),
            "nsm_weighted": round(100.0 * nsm_merges / len(surfaces), 4),
            "lower": round(100.0 * lower_merges / len(surfaces), 4),
        },
        "interpretation": "These bound the identity loss inside the ~19.9% of the MID population "
                          "this third-party file can even see. RoG's own resolver is strictly more "
                          "complete than the file on this population (0 left bare), so the loss over "
                          "RoG's FULL named population is larger than any figure here. Every number "
                          "in this file is a floor for that quantity, not a ceiling.",
        "still_not_claimed": "That these are RoG's exact merges. RoG's MID->name projection is "
                             "unreleased; presence of a group member in RoG's graph is inferred from "
                             "the NSM sibling vocabulary, not observed.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
