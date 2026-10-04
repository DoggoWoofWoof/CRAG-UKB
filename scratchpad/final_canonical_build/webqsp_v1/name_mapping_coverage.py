"""Owed NAME_MAPPING_* fields -- coverage of entities_names.json over the REAL RoG population.

    python scratchpad/final_canonical_build/webqsp_v1/name_mapping_coverage.py

The naming audit could only report structural bounds because the persisted RoG endpoint list did
not exist yet.  V1 section 2 produced it, so the per-population measurement the pre-registration
demands is now computable.

Pre-registered rule (NSM_ACQUISITION_PREREGISTRATION.json): absence from entities_names.json is a
statement about the FILE, never about Freebase.  Every field below is named accordingly.
"""
import json, os, time
import pyarrow.parquet as pq

NODES = "data/final_canonical/webqsp/v1/nodes.parquet"
NAMES = "data/final_canonical/webqsp/_acquisition/nsm/entities_names.json"
OUT = "data/final_canonical/webqsp/NAME_MAPPING_COVERAGE.json"

PRIOR = {"total_bare_mids": 1652618, "m": 1247261, "g": 405357,
         "band_lo": 19566, "band_hi": 62375, "nameless_mediator": 1588085}


def main():
    t0 = time.time()
    eps = pq.read_table(NODES, columns=["source_rog_endpoint"]).column(
        "source_rog_endpoint").to_pylist()
    names = json.load(open(NAMES, encoding="utf-8"))
    name_values = set(names.values())

    bare_m, bare_g, present_m, present_g = 0, 0, 0, 0
    readable = 0
    readable_matching_a_name = 0
    for s in eps:
        p = s[:2]
        if p == "m.":
            bare_m += 1
            if s in names:
                present_m += 1
        elif p == "g.":
            bare_g += 1
            if s in names:
                present_g += 1
        else:
            readable += 1
            if s in name_values:
                readable_matching_a_name += 1

    bare = bare_m + bare_g
    present = present_m + present_g
    doc = {
        "schema": "NAME_MAPPING_COVERAGE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_on": "data/final_canonical/webqsp/v1/nodes.parquet (released RoG, 2,592,894 nodes)",
        "naming_resource": {"file": "entities_names.json", "entries": len(names),
                            "sha256": "93ce094acccb06b1906db8a4534175481efe55bb9fa8ed711fedd1f8b37d2f98",
                            "role": "SUPPLEMENTAL_NAMING_RESOURCE_ONLY"},
        "rog_population": {
            "nodes_total": len(eps),
            "bare_mid_nodes": bare, "bare_mid_m": bare_m, "bare_mid_g": bare_g,
            "non_mid_surface_nodes": readable,
        },
        "reproduces_prior_audit": {
            "prior_total_bare_mids": PRIOR["total_bare_mids"], "measured": bare,
            "delta": bare - PRIOR["total_bare_mids"],
            "prior_m": PRIOR["m"], "measured_m": bare_m, "delta_m": bare_m - PRIOR["m"],
            "prior_g": PRIOR["g"], "measured_g": bare_g, "delta_g": bare_g - PRIOR["g"],
        },
        "NAME_MAPPING_PRESENT_N": present,
        "NAME_MAPPING_MISSING_N": bare - present,
        "NAME_MAPPING_PRESENT_BY_CLASS": {"m.": present_m, "g.": present_g},
        "NAME_MAPPING_MISSING_BY_CLASS": {"m.": bare_m - present_m, "g.": bare_g - present_g},
        "NAME_MAPPING_COVERAGE_OF_ALL_BARE_MIDS": {
            "denominator": bare, "numerator": present,
            "pct": round(100.0 * present / bare, 4) if bare else None,
        },
        "NAME_MAPPING_COVERAGE_OF_NAME_RESOLVABLE_BAND": {
            "band": [PRIOR["band_lo"], PRIOR["band_hi"]],
            "status": "EXACT_INTERSECTION_NOT_COMPUTABLE_YET",
            "why": "The prior audit reported the band as COUNTS. It never persisted the member node "
                   "ids, so the intersection with entities_names.json cannot be computed without "
                   "re-deriving the classification -- which is exactly the section 3 job.",
            "hard_upper_bound": min(present, PRIOR["band_hi"]),
            "hard_upper_bound_note": "The band cannot intersect the naming file in more nodes than "
                                     "the file names in the whole bare-MID population, nor in more "
                                     "than the band's own size.",
            "structural_expectation": "The audit states the band leaves are OVERWHELMINGLY g. machine "
                                      "ids, and this file has ZERO g. entries. So the true "
                                      "intersection is expected near the low end. Recorded as an "
                                      "expectation, NOT as a measurement.",
            "owed_at": "section 3, once node_kind is assigned and the band exists as a node list",
        },
        "NAME_MAPPING_APPLIED_TO_CVT_N": 0,
        "NAME_MAPPING_APPLIED_TO_CVT_NOTE": "Trivially 0: no name from this file has been applied to "
                                            "any V1 node. nodes.parquet display fields are all null "
                                            "with resolution_status=PENDING_SECTION_3. This field "
                                            "MUST be re-measured in section 3 and must stay 0.",
        "rog_naming_shortfall": {
            "what": "Bare MIDs still visible in the RELEASED RoG graph that this third-party file "
                    "CAN name. RoG converted MIDs to names wherever its own (unreleased) resolver "
                    "succeeded, so every node counted here is one RoG left unresolved that a "
                    "commonly-available resource resolves.",
            "n": present, "by_class": {"m.": present_m, "g.": present_g},
            "reading": "A lower bound on RoG's naming shortfall, not an upper bound: this file "
                       "covers only a fraction of Freebase.",
        },
        "surface_consistency": {
            "what": "RoG non-MID surfaces that exactly equal some name in this file.",
            "non_mid_surface_nodes": readable,
            "matching_a_name_in_file": readable_matching_a_name,
            "pct_of_surfaces": round(100.0 * readable_matching_a_name / readable, 4) if readable else None,
            "reading": "Consistency evidence that the two naming vocabularies are the same kind of "
                       "string. NOT proof that RoG used this file -- it did not; RoG's resolver is "
                       "unreleased and predates it.",
        },
        "forbidden_inference": "MID absent from entities_names.json => the MID has no Freebase name. "
                               "That inference is prohibited by the pre-registration and is not made "
                               "anywhere in this file.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
