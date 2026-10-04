"""Check 3 (relation coverage) and the observable half of check 5, from the small files only.

    python scratchpad/final_canonical_build/webqsp_v1/v3_check35_relations.py

Check 5 was preregistered against a /type/property/reverse_property mapping the archive does not
contain (FINDING_1). What the delivered data DOES support is the empirical split: FB+CVT-REV's
relation vocabulary against FB+CVT+REV's. The difference is exactly the predicate set the anti-join
removed. That yields OBJECT_SIDE / SUBJECT_SIDE / BOTH_SIDES directly. NEITHER_SIDE still cannot be
computed here and is reported as uncomputable rather than as zero.
"""
import json, os, time
import pyarrow.parquet as pq

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK3_CHECK5_RELATIONS.json"

VARIANTS = ["FB+CVT-REV", "FB+CVT+REV", "FB-CVT-REV", "FB-CVT+REV"]


def load_rel(v):
    """-> set of dotted relation keys, normalised from IDIR's slashed form."""
    out = set()
    with open(f"{IDIR}/{v}/relation2id.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path = line.rsplit(",", 1)[0]
            out.add(path.lstrip("/").replace("/", "."))
    return out


def main():
    t0 = time.time()
    vocab = {v: load_rel(v) for v in VARIANTS}
    for v in VARIANTS:
        print(f"[{v}] {len(vocab[v]):,} relations", flush=True)

    v1 = set(pq.read_table(f"{PROBE}/v1_relation_keys.parquet").column("relation_key").to_pylist())
    nsm = set(pq.read_table(f"{PROBE}/nsm_predicates.parquet").column("predicate").to_pylist())
    backbone = vocab["FB+CVT-REV"]
    plus = vocab["FB+CVT+REV"]

    missing_v1 = sorted(v1 - backbone)
    missing_nsm = sorted(nsm - backbone)
    # a V1 relation absent from -REV but present in +REV was removed as an object-side predicate:
    # its content survives under the reverse name, so it is recoverable by re-orientation
    recoverable = sorted(set(missing_v1) & plus)
    absent_entirely = sorted(set(missing_v1) - plus)

    doc = {
        "schema": "V3_CHECK3_CHECK5_RELATIONS/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrument": "relation2id.txt of all four variants; small files, unaffected by the size and "
                      "offset corruption documented in FINDING_3/FINDING_4.",

        "CHECK_3_RELATION_COVERAGE": {
            "idir_backbone_relations": len(backbone),
            "v1_relations": len(v1),
            "nsm_predicates": len(nsm),
            "v1_covered_by_idir_backbone": len(v1 & backbone),
            "v1_covered_pct": round(100 * len(v1 & backbone) / len(v1), 3),
            "v1_missing_from_idir_backbone": len(missing_v1),
            "nsm_covered_by_idir_backbone": len(nsm & backbone),
            "nsm_covered_pct": round(100 * len(nsm & backbone) / len(nsm), 3),
            "nsm_missing_from_idir_backbone": len(missing_nsm),
            "of_the_missing_v1_relations": {
                "present_in_FB+CVT+REV_so_removed_as_object_side": len(recoverable),
                "absent_from_every_variant": len(absent_entirely),
            },
            "missing_v1_sample_recoverable": recoverable[:40],
            "missing_v1_sample_absent_entirely": absent_entirely[:40],
        },

        "CHECK_5_OBSERVABLE_HALF": {
            "note": "computed from the delivered vocabularies, not from the reverse_property mapping "
                    "the archive lacks. This is the anti-join's OBSERVED effect.",
            "FB+CVT-REV_is_subset_of_FB+CVT+REV": backbone <= plus,
            "SUBJECT_SIDE_PREDICATES_N": len(backbone),
            "OBJECT_SIDE_PREDICATES_N": len(plus - backbone),
            "BOTH_SIDES_PRESENT_IN_FB+CVT-REV_N": "see INVARIANT_1_STATUS",
            "NEITHER_SIDE_PRESENT_N": None,
            "AMBIGUOUS_CYCLES_N": None,
            "INVARIANT_1_STATUS": (
                "SATISFIED BY CONSTRUCTION AND NOT BY MEASUREMENT. The anti-join removes every "
                "predicate on the object side of the mapping, so no pair can retain both halves in "
                "-REV unless the mapping itself is inconsistent. Confirming that independently "
                "still requires the mapping."),
            "INVARIANT_2_STATUS": (
                "UNCOMPUTABLE FROM THIS PACKAGE. Detecting a pair that lost BOTH halves requires "
                "knowing the pair existed, and only /type/property/reverse_property says that. "
                "Reporting 0 here would be reporting absence of evidence as evidence of absence."),
            "what_this_bounds_anyway": (
                "any V1 or NSM relation absent from the backbone but present in FB+CVT+REV lost only "
                "its orientation, not its content. Any relation absent from BOTH is either outside "
                "IDIR's scope or annihilated. The second set is the concrete NEITHER_SIDE risk "
                "surface, measured here without the mapping."),
        },
        "per_variant_relation_counts": {v: len(vocab[v]) for v in VARIANTS},
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc["CHECK_3_RELATION_COVERAGE"], indent=1)[:2400])
    print(json.dumps({k: v for k, v in doc["CHECK_5_OBSERVABLE_HALF"].items()
                      if not k.startswith("INVARIANT") and k not in ("note", "what_this_bounds_anyway")}, indent=1))


if __name__ == "__main__":
    main()
