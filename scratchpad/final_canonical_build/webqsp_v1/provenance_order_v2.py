"""Lock the 2026-09-08 ruling as NAME_PROVENANCE_ORDER/v2.

    python .../provenance_order_v2.py

A NEW RECORD WITH ITS OWN HASH.  V1 (record_sha256 39f0e46d...) is not edited, not deleted and not
contradicted after the fact; it stands as what was locked before this ruling.  That is the
contract-amendment rule this campaign runs under.

WHAT CHANGED, AND WHY IT IS NOT A REORDERING
  V1 ranked FREEBASE_DELETED_NAME above FREEBASE_HISTORICAL_PAGE_NAME and left
  FREEBASE_HISTORICAL_DUMP_NAME provisional between them.  The ruling rejects that framing outright:
  all three are Freebase's OWN assertion of /type/object/name, differing only in the artifact that
  preserved them -- a published snapshot, a deletion log, a rendered page.  Ranking one artifact as
  intrinsically more authoritative than another is not scientifically justified, so they collapse
  into ONE class and the choice between them is made by TIME.

  The worked case the ruling gives:
      2010 dump    "Foo Corporation"
      2012 deleted "Foo Corp."
      2013 page    "Foo Holdings"
  A rigid DELETED > DUMP > PAGE picks "Foo Corp." purely because of where it was found.  The
  chronological rule picks "Foo Holdings", the later canonical identity.  Source establishes
  AUTHENTICITY; time establishes WHICH historical name to display.
"""
import io, json, time, hashlib

V3 = "data/final_canonical/freebase_v3"

rec = {
    "schema": "NAME_PROVENANCE_ORDER/v2",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "STATUS": "LOCKED by user ruling 2026-09-08",
    "SUPERSEDES": {"schema": "NAME_PROVENANCE_ORDER/v1",
                   "record_sha256": "39f0e46d87abf8c5fb4ca7ef02eecca2c95f47c11e5f75fc2900ee579f69"
                                    "d2cb",
                   "note": "V1 is retained unedited. This is an amendment record, not a rewrite."},
    "APPEND_ONLY": "new record; CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 and node_kind untouched. "
                   "Amendments require a new record with its own hash, never an edit.",
    "PURPOSE": "total order over identity tiers, with tier 2 resolved internally by chronology "
               "rather than by source. Every tier is retained as evidence regardless of display.",

    "ORDER": [
        "FREEBASE_CURRENT_NAME",
        "FREEBASE_HISTORICAL_ASSERTION",
        "FREEBASE_ALIAS",
        "WORDNET_SENSE_LEMMA",
        "IDIR_FREEBASE_NAME",
        "IDIR_CANONICAL_LABEL",
        "SAMSUNG_WIKIDATA_EXACT",
        "CURRENT_WIKIDATA_EXACT",
        "WIKIPEDIA_TITLE",
        "EXTERNAL_AUTHORITY_EXACT",
        "DBPEDIA_EXACT",
        "FACC1_ATTESTED_SURFACE",
        "EXTERNAL_URL_LIVE_EXACT",
        "EXTERNAL_URL_ARCHIVE_EXACT",
        "STRUCTURAL_"],

    "RULED_EXPLICITLY": ["FREEBASE_CURRENT_NAME", "FREEBASE_HISTORICAL_ASSERTION", "FREEBASE_ALIAS",
                         "WORDNET_SENSE_LEMMA", "IDIR_FREEBASE_NAME", "SAMSUNG_WIKIDATA_EXACT",
                         "CURRENT_WIKIDATA_EXACT", "EXTERNAL_AUTHORITY_EXACT",
                         "FACC1_ATTESTED_SURFACE", "EXTERNAL_URL_ARCHIVE_EXACT"],
    "PLACED_BY_INFERENCE_NOT_RULED": {
        "tiers": ["IDIR_CANONICAL_LABEL", "WIKIPEDIA_TITLE", "DBPEDIA_EXACT",
                  "EXTERNAL_URL_LIVE_EXACT"],
        "rule_used": "the ruling listed ten tiers and closed with '...', so tiers it did not name "
                     "keep their V1 relative positions among the unnamed remainder.",
        "STATUS": "OPEN -- flagged rather than assumed settled. None of the four currently changes "
                  "any display name: WIKIPEDIA_TITLE is refuted and empty, DBPEDIA_EXACT has no "
                  "source, IDIR_CANONICAL_LABEL contributes 0 new, EXTERNAL_URL_LIVE_EXACT 91."},

    "HISTORICAL_ASSERTION_CLASS": {
        "tier": 2,
        "DEFINITION": "one class holding every ORIGINAL Freebase assertion of /type/object/name "
                      "that is not in the current snapshot. Membership is by authenticity; "
                      "position within the class is by time.",
        "source_kinds": {
            "HISTORICAL_DUMP": {
                "tiers_merged": "FREEBASE_HISTORICAL_DUMP_NAME",
                "artifacts": ["freebase-data-dump-2010-07-16 simple topic dump",
                              "freebase-data-dump-2010-07-16 per-type TSV",
                              "freebase-data-dump-2010-07-16 quadruples",
                              "freebase-datadump-quadruples 2008-03-28",
                              "freebase-wex-data-dump-2010-07-05"],
                "time_field": "snapshot_ts (the dump's release date)"},
            "DELETED_TRIPLES": {
                "tiers_merged": "FREEBASE_DELETED_NAME",
                "artifacts": ["deleted_freebase.tar.gz (md5 3a2b903862ea9d7d79f106a3821c3b02)"],
                "time_field": "valid_from = created_ts, valid_until = deleted_ts",
                "assertion_status": "DELETED"},
            "HISTORICAL_PAGE": {
                "tiers_merged": "FREEBASE_HISTORICAL_PAGE_NAME",
                "artifacts": ["web.archive.org captures of www.freebase.com topic pages"],
                "time_field": "capture_ts"}},
        "WHY_ONE_CLASS": "all three are Freebase itself asserting the name; they differ only in the "
                         "artifact that preserved the assertion. Declaring one artifact "
                         "intrinsically more authoritative than another is not justified."},

    "EVIDENCE_SCHEMA": {
        "required_per_attested_label": [
            "node_uid", "label", "language",
            "source_kind  (HISTORICAL_DUMP | DELETED_TRIPLES | HISTORICAL_PAGE)",
            "valid_from / created_ts   if known",
            "valid_until / deleted_ts  if known",
            "snapshot_ts               for dump",
            "capture_ts                for page",
            "is_original_name = true"],
        "assertion_status": {
            "values": ["ACTIVE_AT_SOURCE_TIME", "DELETED"],
            "RULE": "a DELETED name is NOT penalised as non-original -- it absolutely was an "
                    "original Freebase name. The flag exists only so that a later surviving "
                    "Freebase assertion wins the display slot."},
        "last_attested_ts": "the single key the display rule compares across source kinds: "
                            "valid_until for DELETED_TRIPLES, snapshot_ts for HISTORICAL_DUMP, "
                            "capture_ts for HISTORICAL_PAGE. It is the last instant at which that "
                            "label is known to have been the asserted name.",
        "NOTHING_IS_DISCARDED": "every attested label in every language is kept, per the "
                                "multi-label rule; display selection chooses among them without "
                                "deleting any."},

    "DISPLAY_SELECTION": {
        "algorithm": [
            "1. if a current-snapshot assertion exists -> use it (FREEBASE_CURRENT_NAME)",
            "2. else gather EVERY original Freebase historical assertion for the node",
            "3. prefer English if any English label exists",
            "4. among equivalent-language candidates, choose the LATEST last_attested_ts",
            "5. if still tied, fall back to the deterministic language ordering "
            "(lexicographically first language code), then to a stable node/label ordering",
            "6. if no historical assertion exists, continue down the ORDER above"],
        "WHAT_THIS_PREVENTS": "picking a name because of the artifact it survived in rather than "
                             "because it was the object's later identity.",
        "DETERMINISM": "steps 3-5 are total and seed-independent, so the chosen display name is a "
                       "function of the evidence table alone."},

    "ZERO_JOIN_INVARIANT": {
        "STATUS": "PERMANENT VALIDATION INVARIANT, added 2026-09-08",
        "RULE": "a zero-hit result from a large external or exact source is NOT acceptable evidence "
                "on its own. It requires a positive-control join before it can be believed.",
        "ORIGIN": "the deleted-triples dump returned exactly 0 of 5,863,639 rows because subjects "
                  "are slash-form /m/040_1l9 against dot-form node_id m.040_1l9. Normalised, the "
                  "same data yields 43,033 residue nodes. A second instance followed within the "
                  "hour: the timestamp pass split a COMMA-delimited file on TAB, matched 0 of "
                  "63,036,271 lines and reported provenance_attached = 0.",
        "REQUIRED_FIELDS_PER_SOURCE": {
            "SOURCE_ROWS": "rows read from the artifact",
            "VALID_ID_ROWS": "rows whose identifier parsed into the expected shape",
            "IN_GRAPH_POSITIVE_CONTROL_N": "rows joining ANYWHERE in the 302M-node graph, not only "
                                           "the residue. This is the control: it must be > 0 "
                                           "before a residue zero means anything.",
            "RESIDUE_HIT_N": "rows landing on the unnamed residue",
            "JOIN_NORMALIZATION": "the exact transform applied to the identifier, stated in full"},
        "ENFORCEMENT": "a join reporting RESIDUE_HIT_N = 0 with IN_GRAPH_POSITIVE_CONTROL_N = 0 is "
                       "a suspected bug and must not be recorded as a finding."},

    "UNTYPED_CLASSIFICATION_DEFERRED": {
        "RULING": "do NOT terminally classify the untyped population yet, and do no further "
                  "nameless grading, until all exact historical passes are complete.",
        "WHY": "the 2010 simple-topic dump alone recovered 167,487 residue names, overwhelmingly "
               "UNTYPED_CANDIDATE, and showed those nodes formerly carried real /common/topic "
               "types (person, album, author, location). The per-type TSV added 58,276 marginal, "
               "notably /music/track. The class is demonstrably not terminal.",
        "PASSES": {"2010 simple-topic": "COMPLETE",
                   "2010 per-type TSV": "COMPLETE",
                   "2010 quadruples": "RUNNING",
                   "2008 quadruples": "COMPLETE -- integrate fully",
                   "deleted triples": "COMPLETE",
                   "WEX": "COMPLETE",
                   "historical pages": "RUNNING"},
        "THEN": "rebuild the historical-assertion union ONCE, from all passes together."},

    "PRIORITY": "finish the exact-source sweep before spending further effort on generic web "
                "recovery.",

    "UNCHANGED_FROM_V1": {
        "NAMELESS_EVIDENCE_GRADES": "SOURCE_DECLARED_NAMELESS / EMPIRICALLY_NAMELESS / "
                                    "INFERRED_NAMELESS remain three separate grades.",
        "NEAR_ZERO_IS_NOT_A_GRADE": "a near-zero named rate can never be promoted to "
                                    "source-declared namelessness.",
        "OBSERVED_ANOMALY_MID_AS_NAME": "a MID-shaped string is never accepted as a recovered name "
                                        "in any tier.",
        "TERMINAL_BUCKETS": ["A ACTUAL_NAME_RECOVERED", "B SOURCE_DECLARED_NAMELESS",
                             "C EMPIRICALLY_NAMELESS", "D INFORMATION_DESTROYED"],
        "BAND_SPEND_ORDER": ["HIGH", "MID", "LOW", "UNTYPED", "NEAR_ZERO"]},
}

body = json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True)
rec["record_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
P = f"{V3}/V3_NAME_PROVENANCE_ORDER_V2.json"
with io.open(P, "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"wrote {P}")
print("record_sha256 =", rec["record_sha256"])
print("ORDER:")
for i, t in enumerate(rec["ORDER"], 1):
    mark = "   <- class of 3, resolved by time" if t == "FREEBASE_HISTORICAL_ASSERTION" else ""
    print(f"  {i:>2}. {t}{mark}")
