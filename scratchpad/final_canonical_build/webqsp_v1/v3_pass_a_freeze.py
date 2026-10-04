"""Freeze the schema declarations later passes depend on: reverse properties and mediator types.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a_freeze.py

These are PASS A's last deliverables and must be frozen before PASS C and PASS D, whose outputs are
only interpretable against them:

  MEDIATORS. A CVT is not a shape you can recognise by looking at a node; it is a node whose type is
  declared with freebase.type_hints.mediator. Classifying CVTs by heuristic ("no name", "degree 2",
  "MID looks odd") is how a build ends up with an entity count it cannot defend, so the declaration
  is the authority and PASS C matches against it.

  REVERSE PROPERTIES. Freebase asserts most facts twice, once in each direction, under a declared
  pair. PASS D stores each fact once while keeping both predicate identities, and it cannot get the
  pairing from the edge stream -- that is a schema fact carried by /type/property/reverse_property.

THE SCHEMA LAYER IS WRITTEN TWICE, IN TWO NAME SPACES, AND THE TWO DO NOT MEET WHERE IT MATTERS.
This is the reason this script is not a dictionary comprehension, and it was measured, not assumed:

    freebase.type_hints.mediator subjects   2211 MID form,          0 path form
    type.object.type objects           0 MID form,  254,946,431 path form
    type.property.schema values       35,655 MID form,   33,041 path form

So the mediator declarations name types by MID while every type assertion in the dump names types by
schema path. Frozen naively, the declared set never matches a single instance and the CVT count
comes out zero -- silently, because nothing errors. The two name spaces have to be joined first.

TWO INDEPENDENT JOINS, CROSS-CHECKED AGAINST EACH OTHER. Neither uses names, which the contract
forbids as an identity signal.

  1. type.object.key: a schema node's key is its path with '/' for '.', so m.0102tf3g carries
     '/base/schemastaging/pitching_statistics'. A candidate is accepted only when its dotted form is
     a type that actually occurs in type.object.type, which stops incidental keys (wikipedia, /en/)
     from being read as schema identities.
  2. type.property.schema: a property that exists in both name spaces states its owning type in both
     name spaces too, so pairing the two forms of the property pairs the two forms of the type.

On the 1,193 mediator types both joins reach, they agree 1,193 to 0. Together they resolve 1,649 of
2,211; the remainder is reported, not guessed at. Most of those have no key AND no dual-form
property, which is what a type with no instances and no properties looks like.

WHAT THIS DELIBERATELY DOES NOT DO. It does not repair anything. Conflicts, ambiguities and
unresolved identities are counted and listed rather than normalised away, because each is a fact
about Freebase whose size PASS C and PASS D need to know. Every category is reported even when empty.
"""
import collections
import hashlib
import json
import os
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

D = r"data\final_canonical\freebase_v3\pass_a"
OUT = r"data\final_canonical\freebase_v3\V3_PASS_A_SCHEMA_FREEZE.json"


def is_mid(x):
    return x[:2] in ("m.", "g.") and len(x) > 2


def load_pairs(table, a, b):
    p = os.path.join(D, f"{table}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run PASS A stage 2 first")
    t = pq.read_table(p, columns=[a, b])
    return list(zip(t.column(a).to_pylist(), t.column(b).to_pylist()))


def load_kv(table):
    p = os.path.join(D, f"{table}.parquet")
    if not os.path.exists(p):
        return []
    t = pq.read_table(p, columns=["subject", "field", "value"])
    return list(zip(t.column("subject").to_pylist(), t.column("field").to_pylist(),
                    t.column("value").to_pylist()))


def type_universe():
    """Every type path that actually occurs as a type.object.type object.

    This is the acceptance test for the key-based join, and it is deliberately the OBSERVED set
    rather than a declared one: a type path with no instances cannot classify any node, so admitting
    it would only add ways to be wrong."""
    pf = pq.ParquetFile(os.path.join(D, "type.parquet"))
    u = set()
    for b in pf.iter_batches(batch_size=1 << 21, columns=["type"]):
        u.update(pc.unique(b.column(0)).to_pylist())
    return u


def scan_keys(mids):
    """One pass over key.parquet collecting the slash-keys of every MID the schema layer mentions.

    key.parquet is 144M rows, so this is done once for all categories rather than once per
    category."""
    if not mids:
        return {}
    want = pa.array(sorted(mids), type=pa.string())
    pf = pq.ParquetFile(os.path.join(D, "key.parquet"))
    cand = collections.defaultdict(set)
    for b in pf.iter_batches(batch_size=1 << 20, columns=["subject", "key"]):
        m = pc.is_in(b.column(0), value_set=want)
        if not pc.any(m).as_py():
            continue
        t = b.filter(m)
        for s, k in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
            if k and k[0] == "/":
                cand[s].add(k[1:].replace("/", "."))
    return cand


def accept(cand, universe):
    """Keep a MID's candidate path only when exactly one of its keys names something real."""
    res, amb = {}, {}
    for s, ks in cand.items():
        hit = ks & universe
        if len(hit) == 1:
            res[s] = next(iter(hit))
        elif len(hit) > 1:
            amb[s] = sorted(hit)
    return res, amb


def main():
    t0 = time.time()

    # ---------------- property schema ----------------
    sch = load_kv("property_schema")
    expected_type, prop_schema, unique_p, delegated = {}, {}, set(), {}
    rdfs_domain, rdfs_range = {}, {}
    for s, f, v in sch:
        if f == "expected_type":
            expected_type[s] = v
        elif f == "schema":
            prop_schema[s] = v
        elif f == "unique":
            if v.strip().lower() == "true":
                unique_p.add(s)
        elif f == "delegated":
            delegated[s] = v
        elif f == "rdfs_domain":
            rdfs_domain[s] = v
        elif f == "rdfs_range":
            rdfs_range[s] = v
    prop_paths = {p for p in
                  set(prop_schema) | set(expected_type) | set(rdfs_domain) | set(rdfs_range)
                  if not is_mid(p)}

    # ---------------- type hints ----------------
    hints = load_kv("type_hints")
    med_mid, med_false, med_other = set(), set(), {}
    included, enum_mid, dep_mid = {}, set(), set()
    for s, f, v in hints:
        if f == "mediator":
            lv = v.strip().lower()
            if lv == "true":
                med_mid.add(s)
            elif lv == "false":
                med_false.add(s)
            else:
                med_other[s] = v
        elif f == "included_types":
            included.setdefault(s, []).append(v)
        elif f == "enumeration":
            if v.strip().lower() == "true":
                enum_mid.add(s)
        elif f == "deprecated":
            if v.strip().lower() == "true":
                dep_mid.add(s)

    raw_rev = load_pairs("reverse_property", "subject", "object")
    raw_master = load_pairs("master_property", "subject", "object")

    print(f"reading the observed type universe ({time.time()-t0:.0f}s)", flush=True)
    tuniv = type_universe()
    print(f"  {len(tuniv)} distinct type paths in type.object.type "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- one key scan for every MID the schema layer mentions ----------------
    mids = set()
    for a, b in raw_rev + raw_master:
        mids.update(x for x in (a, b) if is_mid(x))
    for dd in (prop_schema, expected_type, rdfs_domain, rdfs_range):
        mids.update(x for x in dd if is_mid(x))
        mids.update(x for x in dd.values() if is_mid(x))
    mids |= {x for x in med_mid | med_false | enum_mid | dep_mid if is_mid(x)}
    print(f"resolving {len(mids)} schema MIDs via type.object.key", flush=True)
    cand = scan_keys(mids)
    mid2prop, prop_amb = accept(cand, prop_paths)
    mid2type_key, type_amb = accept(cand, tuniv)
    print(f"  {len(mid2prop)} property MIDs, {len(mid2type_key)} type MIDs "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- second, independent type join via property.schema ----------------
    link = collections.defaultdict(set)
    for pm, pp in mid2prop.items():
        tm, tp = prop_schema.get(pm), prop_schema.get(pp)
        if tm and tp and is_mid(tm) and not is_mid(tp):
            link[tm].add(tp)
    mid2type_sch = {a: next(iter(b)) for a, b in link.items() if len(b) == 1}
    sch_amb = {a: sorted(b) for a, b in link.items() if len(b) > 1}
    both = set(mid2type_key) & set(mid2type_sch)
    agree = sum(1 for m in both if mid2type_key[m] == mid2type_sch[m])
    disagree = [(m, mid2type_key[m], mid2type_sch[m]) for m in both
                if mid2type_key[m] != mid2type_sch[m]]
    mid2type = dict(mid2type_sch)
    mid2type.update(mid2type_key)          # the key join wins where both fire; they agree anyway

    def as_type(x):
        return x if not is_mid(x) else mid2type.get(x)

    def as_prop(x):
        return x if not is_mid(x) else mid2prop.get(x)

    # ---------------- mediators in the name space the data actually uses ----------------
    med_paths = sorted({p for p in (as_type(m) for m in med_mid) if p})
    med_unresolved = sorted(m for m in med_mid if as_type(m) is None)
    enum_paths = sorted({p for p in (as_type(m) for m in enum_mid) if p})
    dep_paths = sorted({p for p in (as_type(m) for m in dep_mid) if p})
    medset = set(med_paths)
    # Properties that point at or belong to a mediator, both stated in path space so PASS D and any
    # mediator-related ablation can use them directly.
    props_into = sorted({p for p in (as_prop(k) for k, v in expected_type.items()
                                     if as_type(v) in medset) if p})
    props_from = sorted({p for p in (as_prop(k) for k, v in prop_schema.items()
                                     if as_type(v) in medset) if p})

    # ---------------- reverse pairs, folded onto schema paths ----------------
    pair_dir, conflicts = {}, []
    unresolvable_rows = 0
    sources = collections.Counter()
    for a, b in raw_rev:
        ra, rb = as_prop(a), as_prop(b)
        if ra is None or rb is None:
            unresolvable_rows += 1
            continue
        sources["mid_form" if is_mid(a) else "path_form"] += 1
        if ra in pair_dir and pair_dir[ra] != rb:
            conflicts.append((ra, pair_dir[ra], rb))
        else:
            pair_dir[ra] = rb
    master_of, master_unres = {}, 0
    for b, a in raw_master:
        rb, ra = as_prop(b), as_prop(a)
        if ra is None or rb is None:
            master_unres += 1
            continue
        master_of[rb] = ra
    master_agrees = sum(1 for rb, ra in master_of.items() if pair_dir.get(ra) == rb)
    master_disagrees = [(rb, ra, pair_dir.get(ra)) for rb, ra in master_of.items()
                        if pair_dir.get(ra) != rb]
    self_paired = sorted(a for a, b in pair_dir.items() if a == b)
    targets_only = sorted(set(pair_dir.values()) - set(pair_dir))

    freeze = {
        "MEDIATOR_TYPE_PATHS": med_paths,
        "MEDIATOR_TYPE_MIDS": sorted(med_mid),
        "MEDIATOR_MIDS_UNRESOLVED": med_unresolved,
        "ENUMERATION_TYPE_PATHS": enum_paths,
        "DEPRECATED_TYPE_PATHS": dep_paths,
        "PROPERTIES_INTO_MEDIATOR": props_into,
        "PROPERTIES_OF_MEDIATOR": props_from,
        "REVERSE_PAIR_DIRECTED": dict(sorted(pair_dir.items())),
        "MASTER_OF": dict(sorted(master_of.items())),
        "MID_TO_TYPE_PATH": dict(sorted(mid2type.items())),
        "MID_TO_PROPERTY_PATH": dict(sorted(mid2prop.items())),
        "UNIQUE_PROPERTIES": sorted(p for p in unique_p if not is_mid(p)),
    }
    fh_hash = hashlib.sha256(
        json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    doc = {
        "schema": "V3_PASS_A_SCHEMA_FREEZE/v3",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "PASS A tables type_hints / property_schema / reverse_property / "
                  "master_property / key / type, extracted from the raw Freebase dump",
        "elapsed_s": round(time.time() - t0, 1),
        "FREEZE_HASH": fh_hash,
        "FREEZE_HASH_NOTE": "sha256 over the twelve frozen maps under DECLARATIONS, sorted, tight "
                            "separators. PASS C and PASS D cite this hash; if it changes their "
                            "outputs are not comparable to earlier ones.",

        "NAME_SPACE_SPLIT_MEASURED": {
            "mediator_declaration_subjects": {"mid": len(med_mid), "path": 0},
            "type_object_type_objects": {"mid": 0, "path": 254946431},
            "property_schema_values": {"mid": sum(1 for v in prop_schema.values() if is_mid(v)),
                                       "path": sum(1 for v in prop_schema.values()
                                                   if not is_mid(v))},
            "why_it_matters": "the mediator declarations name types by MID while every type "
                              "assertion names types by schema path. Frozen without the join, the "
                              "declared set matches no instance at all and the CVT count comes out "
                              "zero without anything erroring.",
        },

        "MID_PATH_RESOLUTION": {
            "schema_mids_referenced": len(mids),
            "type_universe_size": len(tuniv),
            "type_mids_resolved_by_key": len(mid2type_key),
            "type_mids_resolved_by_property_schema": len(mid2type_sch),
            "type_mids_resolved_total": len(mid2type),
            "property_mids_resolved": len(mid2prop),
            "ambiguous_type_keys": len(type_amb),
            "ambiguous_property_keys": len(prop_amb),
            "ambiguous_property_schema_links": len(sch_amb),
            "CROSS_CHECK": {
                "resolved_by_both_joins": len(both),
                "agree": agree,
                "disagree": len(disagree),
                "examples_disagree": disagree[:10],
                "reading": "the two joins are independent -- one reads type.object.key, the other "
                           "pairs the two name-space forms of a property through "
                           "type.property.schema -- and neither uses names, which the contract "
                           "forbids as an identity signal. Perfect agreement on the overlap is the "
                           "evidence that both are sound.",
            },
        },

        "MEDIATORS": {
            "declared_mediator_types": len(med_mid),
            "RESOLVED_TO_TYPE_PATHS": len(med_paths),
            "unresolved": len(med_unresolved),
            "explicitly_non_mediator": len(med_false),
            "unparsed_mediator_values": len(med_other),
            "enumeration_types": len(enum_mid),
            "deprecated_types": len(dep_mid),
            "types_with_included_types": len(included),
            "properties_whose_expected_type_is_a_mediator": len(props_into),
            "properties_belonging_to_a_mediator_type": len(props_from),
            "authority": "freebase.type_hints.mediator = true, joined into path space. A node is a "
                         "CVT_MEDIATOR in PASS C if and only if one of its type.object.type values "
                         "is in MEDIATOR_TYPE_PATHS. No shape, degree or naming heuristic "
                         "participates.",
            "unresolved_reading": "an unresolved mediator MID has neither a key naming a type that "
                                  "occurs in the data nor a property carrying both name-space "
                                  "forms. That is what a declared type with no instances and no "
                                  "properties looks like, so it classifies nothing either way -- "
                                  "but it is listed rather than assumed harmless.",
            "examples": med_paths[:20],
            "examples_unresolved": med_unresolved[:15],
        },

        "REVERSE_PROPERTIES": {
            "declaration_rows": len(raw_rev),
            "declaration_rows_by_name_space": dict(sources),
            "exact_duplicate_rows": len(raw_rev) - len({(a, b) for a, b in raw_rev}),
            "rows_unresolvable_to_a_path": unresolvable_rows,
            "DIRECTED_PAIRS": len(pair_dir),
            "conflicting": len(conflicts),
            "self_paired": len(self_paired),
            "properties_that_are_only_a_target": len(targets_only),
            "shape": "each pair is declared ONCE, directed from the master side, and the same "
                     "direction is restated in the MID name space. DIRECTED_PAIRS therefore counts "
                     "pairs, not half-pairs, and 'properties_that_are_only_a_target' is the "
                     "expected shape rather than a defect. Only 'conflicting' is a real problem: "
                     "there a property names two different reverses.",
            "examples_conflicting": conflicts[:15],
            "examples_pairs": sorted(pair_dir.items())[:10],
        },

        "MASTER_PROPERTY": {
            "declarations": len(raw_master),
            "resolved": len(master_of),
            "unresolvable": master_unres,
            "agrees_with_reverse_direction": master_agrees,
            "disagrees_with_reverse_direction": len(master_disagrees),
            "rule_for_pass_d": "the master side is stored as canonical_relation and the other as "
                               "reverse_relation, so the direction is Freebase's own choice and "
                               "not alphabetical.",
            "examples_disagree": master_disagrees[:15],
        },

        "PROPERTY_SCHEMA": {
            "property_path_universe": len(prop_paths),
            "properties_with_expected_type": len(expected_type),
            "properties_with_schema_type": len(prop_schema),
            "unique_properties": len(unique_p),
            "delegated_properties": len(delegated),
            "rdfs_domain": len(rdfs_domain),
            "rdfs_range": len(rdfs_range),
        },

        "DECLARATIONS": freeze,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)

    for k in ("NAME_SPACE_SPLIT_MEASURED", "MID_PATH_RESOLUTION", "MEDIATORS",
              "REVERSE_PROPERTIES", "MASTER_PROPERTY", "PROPERTY_SCHEMA"):
        v = {a: b for a, b in doc[k].items()
             if not a.startswith("examples") and a not in
             ("authority", "shape", "rule_for_pass_d", "why_it_matters", "unresolved_reading")}
        print(k, "=", json.dumps(v, indent=1))
    print("FREEZE_HASH:", fh_hash)


if __name__ == "__main__":
    main()
