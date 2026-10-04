"""Check 3b: weight the relation coverage by triple mass, and compare against ALL four variants.

    python scratchpad/final_canonical_build/webqsp_v1/v3_check3b_mass.py

Two corrections to check 3 as first run:
  - it compared the missing relations against FB+CVT+REV only and labelled the residue "absent from
    every variant". FB-CVT+REV has 5,028 relations, MORE than FB+CVT+REV's 4,425, so that label was
    wrong. Compare against the union.
  - relation COUNT is not relation MASS. 4,895 missing relations is only alarming in proportion to
    the triples they carry. NSM's 8,558,342 MID-to-MID triples are the measure.
"""
import json, os, time
from collections import Counter
import pyarrow.parquet as pq

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK3B_RELATION_MASS.json"
VARIANTS = ["FB+CVT-REV", "FB+CVT+REV", "FB-CVT-REV", "FB-CVT+REV"]


def load_rel(v):
    out = set()
    with open(f"{IDIR}/{v}/relation2id.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                out.add(line.rsplit(",", 1)[0].lstrip("/").replace("/", "."))
    return out


def main():
    t0 = time.time()
    vocab = {v: load_rel(v) for v in VARIANTS}
    backbone = vocab["FB+CVT-REV"]
    union = set().union(*vocab.values())
    plus = vocab["FB+CVT+REV"]

    v1 = set(pq.read_table(f"{PROBE}/v1_relation_keys.parquet").column("relation_key").to_pylist())
    preds = pq.read_table(f"{PROBE}/nsm_mid_triples.parquet").column("predicate").to_pylist()
    total = len(preds)
    freq = Counter(preds)
    print(f"[load] {total:,} triples, {len(freq):,} distinct predicates t={time.time()-t0:.0f}s",
          flush=True)

    def mass(pred_set):
        return sum(c for p, c in freq.items() if p in pred_set)

    m_backbone = mass(backbone)
    m_plus = mass(plus)
    m_union = mass(union)

    missing = set(freq) - backbone
    only_in_plus = missing & plus                 # orientation lost, content survives
    only_in_union = (missing & union) - plus      # present in some other variant
    absent_all = missing - union                  # not in IDIR at all, in any variant

    top_absent = sorted(((freq[p], p) for p in absent_all), reverse=True)[:30]
    top_reversed = sorted(((freq[p], p) for p in only_in_plus), reverse=True)[:20]

    doc = {
        "schema": "V3_CHECK3B_RELATION_MASS/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instrument": "NSM MID-to-MID triples (8,558,342), the only WebQSP/CWQ structural oracle "
                      "that preserves MIDs on both ends. V1 cannot weight by mass because RoG "
                      "substituted names for MIDs on 57% of its endpoints.",

        "CORRECTION_TO_CHECK_3": {
            "the_label_absent_from_every_variant_was_wrong": "it compared only against FB+CVT+REV. "
                                                             "FB-CVT+REV carries 5,028 relations, "
                                                             "more than FB+CVT+REV's 4,425, because "
                                                             "collapsing CVTs mints composed "
                                                             "predicates that do not exist in the "
                                                             "+CVT variants.",
            "union_of_all_four_variants": len(union),
            "recomputed_against_the_union": {
                "v1_relations_in_union": len(v1 & union),
                "v1_relations_absent_from_all_four": len(v1 - union),
            },
        },

        "RELATION_COUNT_VIEW": {
            "v1_relations": len(v1),
            "in_backbone": len(v1 & backbone),
            "in_backbone_pct": round(100 * len(v1 & backbone) / len(v1), 3),
            "in_union_but_not_backbone": len((v1 & union) - backbone),
            "absent_from_all_four_variants": len(v1 - union),
        },

        "RELATION_MASS_VIEW": {
            "nsm_mid_to_mid_triples": total,
            "distinct_predicates_in_use": len(freq),
            "triples_whose_predicate_is_in_backbone": m_backbone,
            "triples_whose_predicate_is_in_backbone_pct": round(100 * m_backbone / total, 3),
            "triples_whose_predicate_is_in_FB+CVT+REV_pct": round(100 * m_plus / total, 3),
            "triples_whose_predicate_is_in_any_variant_pct": round(100 * m_union / total, 3),
            "triples_lost_to_orientation_only": mass(only_in_plus),
            "triples_lost_to_orientation_only_pct": round(100 * mass(only_in_plus) / total, 3),
            "triples_on_predicates_absent_from_all_four": mass(absent_all),
            "triples_on_predicates_absent_from_all_four_pct": round(100 * mass(absent_all) / total, 3),
        },

        "PREDICATES_ABSENT_FROM_EVERY_VARIANT": {
            "n": len(absent_all),
            "top_by_triple_count": [{"predicate": p, "triples": c} for c, p in top_absent],
        },
        "PREDICATES_REMOVED_AS_OBJECT_SIDE": {
            "n": len(only_in_plus),
            "note": "content survives in FB+CVT+REV under the reverse name; only the orientation "
                    "was dropped. Recoverable without the raw dump.",
            "top_by_triple_count": [{"predicate": p, "triples": c} for c, p in top_reversed],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("CORRECTION_TO_CHECK_3", "RELATION_COUNT_VIEW",
                                          "RELATION_MASS_VIEW")}, indent=1))
    print(json.dumps(doc["PREDICATES_ABSENT_FROM_EVERY_VARIANT"], indent=1)[:1800])


if __name__ == "__main__":
    main()
