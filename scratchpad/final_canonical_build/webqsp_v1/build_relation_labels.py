"""V1 section 7 -- adaptive minimally-qualified relation labels (user decision, 2026-09-06).

    python scratchpad/final_canonical_build/webqsp_v1/build_relation_labels.py

Three columns, one identity:

    relation_key             UNTOUCHED canonical identity.  edges.parquet references relation_uid,
                             whose mapping to relation_key is preserved exactly by this rewrite.
    relation_label_short     last segment, naturalized.  Human inspection / GRAPH_SAMPLES only.
    relation_label_qualified type + property; ONLY the relations that collide there escalate to
                             domain + type + property.  This is what SPLADE / CVT rendering consumes.

Minimum qualification necessary for uniqueness, rather than paying the domain prefix on all 7,058:
measured, type+property already separates 7,009 of 7,058.

Segmentation (Freebase convention, uniform for every key with >= 3 segments):
    people . person . place_of_birth
    domain   type     property                      -> domain is EVERY leading segment, so
    base.mediaasset . recorded_work . representations  has domain "base mediaasset".

Naturalization is deterministic and derived-view only: '_' and '#' become spaces, whitespace is
collapsed.  Case is NOT folded -- 'inverseOf' stays as released.  Section 2's lesson applies:
normalization never touches graph identity, only derived surfaces.

The ladder terminates in a rung that is unique BY CONSTRUCTION (the naturalized full key), and
uniqueness is verified over the WHOLE label set after escalation -- an escalated label could
otherwise collide with some other relation's un-escalated one.
"""
import json, os, time

import pyarrow as pa
import pyarrow.parquet as pq

RELS = "data/final_canonical/webqsp/v1/relations.parquet"
OUT = "data/final_canonical/webqsp/V1_RELATION_LABELS.json"
RULE_VERSION = "relation_label/v2_adaptive_minimal_qualification"


def natural(s):
    return " ".join(s.replace("_", " ").replace("#", " ").split())


def split_key(k):
    """-> (domain_segments, type_segment_or_None, property_segment)"""
    seg = k.split(".")
    if len(seg) == 1:
        return [], None, seg[0]
    if len(seg) == 2:
        return [], seg[0], seg[1]
    return seg[:-2], seg[-2], seg[-1]


def main():
    t0 = time.time()
    t = pq.read_table(RELS)
    uids = t.column("relation_uid").to_pylist()
    keys = t.column("relation_key").to_pylist()

    short, q_tp, q_dtp, q_full = [], [], [], []
    for k in keys:
        dom, typ, prop = split_key(k)
        short.append(natural(prop))
        q_tp.append(natural(f"{typ} {prop}") if typ else natural(prop))
        q_dtp.append(natural(f"{' '.join(dom)} {typ} {prop}") if typ else natural(k))
        q_full.append(natural(k))

    def dupes(labels):
        seen, bad = {}, set()
        for i, s in enumerate(labels):
            if s in seen:
                bad.add(s)
            seen[s] = i
        return bad

    # rung 1
    bad_tp = dupes(q_tp)
    escalate = [i for i, s in enumerate(q_tp) if s in bad_tp]
    label = list(q_tp)
    level = ["TYPE_PROPERTY"] * len(keys)
    for i in escalate:
        label[i] = q_dtp[i]
        level[i] = "DOMAIN_TYPE_PROPERTY"

    # rung 2 -- verify over the WHOLE set, not only the escalated slice
    bad_after = dupes(label)
    escalate2 = [i for i, s in enumerate(label) if s in bad_after]
    for i in escalate2:
        label[i] = q_full[i]
        level[i] = "FULL_KEY"
    bad_final = dupes(label)

    out = t.append_column("relation_label_qualified", pa.array(label, pa.string())) \
           .append_column("qualification_level", pa.array(level, pa.string()))
    out = out.rename_columns(["relation_uid", "relation_key", "relation_label_short",
                              "relation_label_qualified", "qualification_level"])
    tmp = RELS + ".tmp"
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, RELS)

    # relation_uid <-> relation_key mapping must be byte-identical to what edges.parquet references
    chk = pq.read_table(RELS)
    identity_preserved = (chk.column("relation_uid").to_pylist() == uids
                          and chk.column("relation_key").to_pylist() == keys)

    from collections import Counter
    lv = Counter(level)
    doc = {
        "schema": "V1_RELATION_LABELS/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule_version": RULE_VERSION,
        "decision": "user, 2026-09-06 -- minimum qualification necessary for uniqueness",
        "relations": len(keys),
        "LADDER": {
            "rung_1_TYPE_PROPERTY": {"unique_labels": len(set(q_tp)),
                                     "colliding_relations": len(escalate)},
            "rung_2_DOMAIN_TYPE_PROPERTY": {"applied_to": len(escalate),
                                            "still_colliding_after": len(escalate2)},
            "rung_3_FULL_KEY": {"applied_to": len(escalate2),
                                "note": "unique by construction -- the naturalized canonical key"},
        },
        "qualification_level_counts": dict(lv),
        "QUALIFIED_LABEL_COLLISIONS": len(bad_final),
        "QUALIFIED_LABELS_UNIQUE": len(bad_final) == 0,
        "SHORT_LABEL_COLLISIONS": len(dupes(short)),
        "short_label_note": "relation_label_short is deliberately NOT unique. It is for human "
                            "inspection and GRAPH_SAMPLES only and is never consumed by retrieval "
                            "text. 34% of the schema collides there, which is exactly why the "
                            "qualified column exists.",
        "RELATION_IDENTITY_PRESERVED": bool(identity_preserved),
        "identity_note": "relation_uid -> relation_key is byte-identical to the pre-rewrite table, "
                         "so every edges.parquet reference still resolves to the same relation. "
                         "Directive section 7: never change edge semantics.",
        "naturalization_rule": "'_' and '#' -> space; whitespace collapsed; case NOT folded. "
                               "Derived view only -- graph identity is never normalized.",
        "escalated_relations": [{"relation_key": keys[i], "collided_as": q_tp[i],
                                 "resolved_as": label[i]} for i in escalate[:20]],
        "consumers": {"SPLADE_and_CVT_rendering": "relation_label_qualified",
                      "GRAPH_SAMPLES_and_UI": "relation_label_short",
                      "identity": "relation_key"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("LADDER", "qualification_level_counts",
                                          "QUALIFIED_LABEL_COLLISIONS", "QUALIFIED_LABELS_UNIQUE",
                                          "SHORT_LABEL_COLLISIONS", "RELATION_IDENTITY_PRESERVED",
                                          "escalated_relations")}, indent=1)[:2400])


if __name__ == "__main__":
    main()
