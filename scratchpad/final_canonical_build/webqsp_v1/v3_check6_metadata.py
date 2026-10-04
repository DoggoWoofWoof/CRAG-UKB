"""Check 6: metadata coverage -- can IDIR render V3 without the raw dump?

    python scratchpad/final_canonical_build/webqsp_v1/v3_check6_metadata.py

V1's whole representation problem was that RoG substituted names for MIDs, destroying identity to
buy readability (WEBQSP_REPRESENTATION_AUDIT). V3 keeps MIDs as identity, which means the readable
surface has to come from somewhere else. IDIR ships object_names.csv (4.01 GB) and object_types.csv
(13.65 GB, recovered in FINDING_4). Whether those cover our nodes decides whether the 31.3 GB raw
mirror is needed for RENDERING, separately from whether it is needed for CONTENT.

Also measured, because it falls out of the same pass and matters more than the coverage number: are
V1's CVT_MEDIATOR nodes type-separable from its MID_NAMED_ENTITY nodes? V1 detects CVTs by absence
of a name, which is a heuristic over a lossy surface. If IDIR's types separate the two populations,
V3 gets a deterministic, source-level CVT classifier instead -- which is what section 3 wanted and
could not have from RoG.
"""
import json, os, time
from collections import Counter, defaultdict
import pyarrow.parquet as pq

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK6_METADATA_COVERAGE.json"


def main():
    t0 = time.time()
    t = pq.read_table(f"{PROBE}/v1_mid_node_kind.parquet")
    kind = dict(zip(t.column("mid").to_pylist(), t.column("node_kind").to_pylist()))
    nsm_mids = set(pq.read_table(f"{PROBE}/nsm_all_mids.parquet").column(0).to_pylist())
    ans = set(pq.read_table(f"{PROBE}/nsm_answer_mids.parquet").column(0).to_pylist())
    top = set(pq.read_table(f"{PROBE}/nsm_topic_mids.parquet").column(0).to_pylist())
    want = {"/" + m.replace(".", "/", 1): m for m in nsm_mids}
    print(f"[probe] {len(want):,} mids t={time.time()-t0:.0f}s", flush=True)

    # check 7 found 488,337 CVT nodes absent from the graph. Whether IDIR's METADATA still knows
    # them decides what dropped them: if a node is typed/named but has no edges, the MID-to-MID
    # filter removed its edges and the raw dump would restore them. If it is absent from the
    # metadata too, the object is outside IDIR's scope entirely and the raw dump is the only source.
    in_graph = set()
    with open(f"{IDIR}/FB+CVT-REV/entity2id.txt", encoding="utf-8") as fh:
        for line in fh:
            c = line.rfind(",")
            if c > 0:
                m = want.get(line[:c])
                if m is not None:
                    in_graph.add(m)
    print(f"[entity2id] {len(in_graph):,} probe mids in graph t={time.time()-t0:.0f}s", flush=True)

    named, typed, idpath = set(), set(), set()
    name_lang = Counter()
    types_by_kind = defaultdict(Counter)
    type_owner_kinds = defaultdict(set)

    n = 0
    with open(f"{IDIR}/Metadata/object_names.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            c = line.find(",")
            if c < 0:
                continue
            m = want.get(line[:c])
            if m is None:
                continue
            named.add(m)
            q = line.rfind("@")
            if q > 0:
                name_lang[line[q + 1:].strip()] += 1
    print(f"[names] {n:,} lines, {len(named):,} probe mids named t={time.time()-t0:.0f}s", flush=True)

    n = 0
    with open(f"{IDIR}/Metadata/object_types.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            c = line.find(",")
            if c < 0:
                continue
            m = want.get(line[:c])
            if m is None:
                continue
            typed.add(m)
            ty = line[line.rfind(",") + 1:].strip()
            k = kind.get(m, "NOT_IN_V1")
            types_by_kind[k][ty] += 1
            type_owner_kinds[ty].add(k)
    print(f"[types] {n:,} lines, {len(typed):,} probe mids typed t={time.time()-t0:.0f}s", flush=True)

    n = 0
    with open(f"{IDIR}/Metadata/object_ids.csv", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            c = line.find(",")
            if c > 0 and line[:c] in want:
                idpath.add(want[line[:c]])
    print(f"[ids] {n:,} lines, {len(idpath):,} probe mids with id path t={time.time()-t0:.0f}s",
          flush=True)

    def cov(sel, label):
        k = len(sel)
        return {"set": label, "n": k,
                "named": len(sel & named), "named_pct": round(100 * len(sel & named) / k, 3),
                "typed": len(sel & typed), "typed_pct": round(100 * len(sel & typed) / k, 3),
                "id_path": len(sel & idpath), "id_path_pct": round(100 * len(sel & idpath) / k, 3),
                "named_or_typed": len(sel & (named | typed)),
                "neither": len(sel - named - typed)}

    by_kind = {}
    for k in set(kind.values()):
        sel = {m for m, kk in kind.items() if kk == k}
        by_kind[k] = cov(sel, k)

    cvt = {m for m, kk in kind.items() if kk == "CVT_MEDIATOR"}
    ent = {m for m, kk in kind.items() if kk == "MID_NAMED_ENTITY"}
    cvt_types = set(types_by_kind["CVT_MEDIATOR"])
    ent_types = set(types_by_kind["MID_NAMED_ENTITY"])
    shared = cvt_types & ent_types
    cvt_only = cvt_types - ent_types
    ent_only = ent_types - cvt_types
    cvt_mass = sum(types_by_kind["CVT_MEDIATOR"].values())
    cvt_mass_on_shared = sum(c for ty, c in types_by_kind["CVT_MEDIATOR"].items() if ty in shared)

    doc = {
        "schema": "V3_CHECK6_METADATA_COVERAGE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CHECK_6_COVERAGE": {
            "probe_mids": len(nsm_mids),
            "overall": cov(nsm_mids, "all NSM mids"),
            "by_role": [cov(ans, "gold answer mids"), cov(top, "question topic mids")],
            "by_v1_node_kind": by_kind,
            "name_language_tags_top": name_lang.most_common(10),
        },
        "CVT_TYPE_SEPARABILITY": {
            "question": "V1 detects CVTs by absence of a name, a heuristic over a surface RoG "
                        "already damaged. Do IDIR's types separate the populations deterministically?",
            "cvt_nodes_with_a_type": len(cvt & typed),
            "entity_nodes_with_a_type": len(ent & typed),
            "distinct_types_on_cvt_nodes": len(cvt_types),
            "distinct_types_on_entity_nodes": len(ent_types),
            "types_exclusive_to_cvt_nodes": len(cvt_only),
            "types_exclusive_to_entity_nodes": len(ent_only),
            "types_shared_by_both": len(shared),
            "cvt_type_assertions": cvt_mass,
            "cvt_type_assertions_on_shared_types": cvt_mass_on_shared,
            "cvt_type_assertions_on_shared_types_pct":
                round(100 * cvt_mass_on_shared / cvt_mass, 3) if cvt_mass else None,
            "top_cvt_exclusive_types": [
                {"type": ty, "n": types_by_kind["CVT_MEDIATOR"][ty]}
                for ty in sorted(cvt_only, key=lambda x: -types_by_kind["CVT_MEDIATOR"][x])[:25]],
            "top_shared_types": [
                {"type": ty, "cvt_n": types_by_kind["CVT_MEDIATOR"][ty],
                 "entity_n": types_by_kind["MID_NAMED_ENTITY"][ty]}
                for ty in sorted(shared, key=lambda x: -types_by_kind["CVT_MEDIATOR"][x])[:20]],
            "INTERPRETATION_RULE_FIXED_BEFORE_LOOKING": (
                "separable if the share of CVT type assertions falling on types that entities also "
                "carry is under 5%. Above that, types alone cannot classify and V3 would still need "
                "a structural signal."),
        },
        "WHY_THE_MISSING_CVTS_ARE_MISSING": {
            "question": "check 7 found 488,337 V1 CVT_MEDIATOR nodes absent from FB+CVT-REV. Does "
                        "IDIR's metadata still know those objects?",
            "decision_rule_fixed_before_measurement": (
                "if the missing CVTs are largely PRESENT in the metadata, their edges were removed "
                "by the subject-and-object-must-be-/m/-or-/g/ filter -- literal-valued arguments -- "
                "and the raw dump restores them. If they are largely ABSENT from metadata too, they "
                "are outside IDIR's scope and only the raw dump has them. Either way the raw dump "
                "is implicated; the two cases differ in what else it would fix."),
            "by_kind": {
                k: {
                    "absent_from_graph": len(miss),
                    "of_those_named": len(miss & named),
                    "of_those_typed": len(miss & typed),
                    "of_those_with_id_path": len(miss & idpath),
                    "of_those_in_no_metadata_at_all": len(miss - named - typed - idpath),
                    "in_no_metadata_pct": round(100 * len(miss - named - typed - idpath) / len(miss), 3)
                    if miss else None,
                }
                for k, miss in (
                    (kk, {m for m, v in kind.items() if v == kk} - in_graph)
                    for kk in sorted(set(kind.values()))
                ) if miss
            },
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    sep = doc["CVT_TYPE_SEPARABILITY"]
    p = sep["cvt_type_assertions_on_shared_types_pct"]
    sep["VERDICT"] = ("SEPARABLE" if p is not None and p < 5.0 else
                      "NOT SEPARABLE BY TYPE ALONE") if p is not None else "NO DATA"

    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc["CHECK_6_COVERAGE"]["overall"], indent=1))
    print(json.dumps(doc["CHECK_6_COVERAGE"]["by_v1_node_kind"], indent=1))
    print(json.dumps(doc["CHECK_6_COVERAGE"]["by_role"], indent=1))
    print(json.dumps({k: v for k, v in sep.items() if not k.startswith("top_")}, indent=1))
    print(json.dumps(doc["WHY_THE_MISSING_CVTS_ARE_MISSING"]["by_kind"], indent=1))


if __name__ == "__main__":
    main()
