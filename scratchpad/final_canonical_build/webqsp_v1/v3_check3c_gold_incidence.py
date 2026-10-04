"""Check 3c: does the mass IDIR is missing actually touch gold answers?

    python scratchpad/final_canonical_build/webqsp_v1/v3_check3c_gold_incidence.py

23.8% of NSM's MID-to-MID triple mass sits on predicates absent from all four IDIR variants. Whether
that matters to CRAG depends entirely on WHAT that mass is. Two hypotheses, and the data decides:

  H_JUNK      the absent mass is web/media/annotation plumbing -- the "admin/schema junk as
              structural edges" the V3 design already says to remove. Then IDIR's filter agrees
              with our design and the deficit is cosmetic.
  H_SEMANTIC  the absent mass carries content on paths to answers. Then IDIR is not a usable
              backbone without augmentation, whatever else it does well.

Measured, not assumed: family breakdown of the absent predicates by mass, and the share of
ANSWER-INCIDENT triples (either endpoint a gold answer MID) that IDIR would drop.
"""
import json, os, re, time
from collections import Counter, defaultdict
import pyarrow.parquet as pq

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK3C_GOLD_INCIDENCE.json"
VARIANTS = ["FB+CVT-REV", "FB+CVT+REV", "FB-CVT-REV", "FB-CVT+REV"]

# families declared BEFORE looking at the per-family numbers, so the classification cannot be
# tuned to produce a comfortable answer. Order matters: first match wins.
FAMILIES = [
    ("valuenotation_schema_metadata", r"^freebase\.valuenotation\."),
    ("freebase_admin",                r"^freebase\."),
    ("web_media_plumbing",            r"^common\.(webpage|resource|image|document)\.|"
                                      r"^common\.topic\.(webpage|article|image|topical_webpage|"
                                      r"official_website|social_media_presence)$"),
    ("type_assertion",                r"^common\.topic\.notable_types$|^type\.|^kg\."),
    ("notable_for",                   r"^common\.topic\.notable_for$"),
    ("text_annotation_base",          r"^base\.(kwebbase|descriptive_names|schemastaging\."
                                      r"context_name)\."),
    ("user_contributed_base",         r"^base\."),
    ("user_namespace",                r"^user\."),
    ("ordinary_domain_content",       r".*"),
]


def fam(p):
    for name, rx in FAMILIES:
        if re.match(rx, p):
            return name
    return "ordinary_domain_content"


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
    union = set().union(*(load_rel(v) for v in VARIANTS))
    backbone = load_rel("FB+CVT-REV")

    t = pq.read_table(f"{PROBE}/nsm_mid_triples.parquet")
    s = t.column("subject_mid").to_pylist()
    p = t.column("predicate").to_pylist()
    o = t.column("object_mid").to_pylist()
    ans = set(pq.read_table(f"{PROBE}/nsm_answer_mids.parquet").column(0).to_pylist())
    top = set(pq.read_table(f"{PROBE}/nsm_topic_mids.parquet").column(0).to_pylist())
    total = len(p)
    print(f"[load] {total:,} triples, {len(ans):,} answer mids t={time.time()-t0:.0f}s", flush=True)

    freq = Counter(p)
    absent = set(freq) - union

    # family breakdown of the absent mass
    by_fam = defaultdict(lambda: {"predicates": 0, "triples": 0})
    for pred in absent:
        f = by_fam[fam(pred)]
        f["predicates"] += 1
        f["triples"] += freq[pred]
    absent_mass = sum(freq[x] for x in absent)

    # answer- and topic-incident slices
    inc_tot = inc_absent = inc_backbone = 0
    topic_tot = topic_absent = 0
    absent_answer_pred = Counter()
    for i in range(total):
        a = s[i] in ans or o[i] in ans
        if a:
            inc_tot += 1
            if p[i] in absent:
                inc_absent += 1
                absent_answer_pred[p[i]] += 1
            if p[i] in backbone:
                inc_backbone += 1
        if s[i] in top or o[i] in top:
            topic_tot += 1
            if p[i] in absent:
                topic_absent += 1

    fam_sorted = dict(sorted(
        ((k, {**v, "pct_of_absent_mass": round(100 * v["triples"] / absent_mass, 2)})
         for k, v in by_fam.items()), key=lambda kv: -kv[1]["triples"]))
    junk = {"valuenotation_schema_metadata", "freebase_admin", "web_media_plumbing",
            "text_annotation_base"}
    junk_mass = sum(v["triples"] for k, v in by_fam.items() if k in junk)

    doc = {
        "schema": "V3_CHECK3C_GOLD_INCIDENCE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "23.8% of NSM triple mass sits on predicates absent from every IDIR variant. "
                    "Is that junk the V3 design already discards, or content on paths to answers?",
        "families_declared_before_measurement": [f[0] for f in FAMILIES],

        "ABSENT_MASS_BY_FAMILY": {
            "absent_predicates": len(absent),
            "absent_triples": absent_mass,
            "absent_pct_of_all_triples": round(100 * absent_mass / total, 3),
            "by_family": fam_sorted,
            "clearly_discardable_families_mass": junk_mass,
            "clearly_discardable_pct_of_absent": round(100 * junk_mass / absent_mass, 2),
            "clearly_discardable_definition": sorted(junk),
        },

        "ANSWER_INCIDENT_SLICE": {
            "definition": "triples with a gold ANSWER mid on either endpoint -- the edges that "
                          "actually carry answers, as opposed to the corpus at large.",
            "answer_incident_triples": inc_tot,
            "on_predicates_absent_from_all_variants": inc_absent,
            "absent_pct_of_answer_incident": round(100 * inc_absent / inc_tot, 3) if inc_tot else None,
            "on_predicates_in_the_backbone": inc_backbone,
            "backbone_pct_of_answer_incident": round(100 * inc_backbone / inc_tot, 3) if inc_tot else None,
            "compare_corpus_wide_absent_pct": round(100 * absent_mass / total, 3),
            "top_absent_predicates_on_answer_edges": [
                {"predicate": k, "answer_incident_triples": v}
                for k, v in absent_answer_pred.most_common(25)],
        },

        "TOPIC_INCIDENT_SLICE": {
            "definition": "triples touching a question's topic entity -- where retrieval enters the "
                          "graph.",
            "topic_incident_triples": topic_tot,
            "on_predicates_absent_from_all_variants": topic_absent,
            "absent_pct_of_topic_incident": round(100 * topic_absent / topic_tot, 3) if topic_tot else None,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc["ABSENT_MASS_BY_FAMILY"], indent=1))
    print(json.dumps({k: v for k, v in doc["ANSWER_INCIDENT_SLICE"].items()
                      if k != "top_absent_predicates_on_answer_edges"}, indent=1))
    print(json.dumps(doc["TOPIC_INCIDENT_SLICE"], indent=1))
    print(json.dumps(doc["ANSWER_INCIDENT_SLICE"]["top_absent_predicates_on_answer_edges"][:12], indent=1))


if __name__ == "__main__":
    main()
