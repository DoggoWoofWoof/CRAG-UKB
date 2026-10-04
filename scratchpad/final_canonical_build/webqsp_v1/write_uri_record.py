"""Record what the EXTERNAL_URI renderer achieves and, explicitly, what it refuses to do."""
import io, json, time
V3 = "data/final_canonical/freebase_v3"
rec = {
 "schema": "URI_RENDER_MEASURE/v1",
 "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "population": {"external_uri_nodes": 74454117,
                "sampled": 9308829, "shards_sampled": 33, "shards_total": 262, "stride": 8},
 "OUTCOME_PCT": {"HOST_PATH": 46.149, "WIKIPEDIA_TITLE": 30.285, "WIKIPEDIA_CURID": 21.750,
                 "WIKIPEDIA_UNDECODABLE": 0.962, "HOST_ONLY": 0.853, "RAW": 0.0},
 "ENCODINGS_HANDLED": {
   "utf8_percent": "the ordinary case",
   "double_percent": "%25D9%25BE is %D9%BE encoded twice. A 'contains %25' trigger was WRONG and was removed: %25 is also a legitimate UTF-16 low byte for Cyrillic Ha, so the trigger hijacked Cyrillic titles.",
   "legacy_codepage": "%E9 on an en wiki is latin-1 e-acute. Codepage is chosen by wiki language, because trying cp1251 first rendered Rub%E9n as Rubjn.",
   "utf16_low_byte_fixed_offset": "the high byte was dropped. For scripts confined to one 256-block (Cyrillic, Thai, Greek, Arabic, Hebrew, Indic) it is a constant recoverable from the wiki language.",
   "utf16_low_byte_measured": "for Latin-script languages the high byte VARIES within one title (Romanian needs 0x01 for a-breve and 0x02 for t-comma; English needs 0x20 for an en dash), so it is measured per language by build_uri_charmap.py from the frozen metadata -- see V3_URI_CHARMAP.json."},
 "METHOD": "every encoding is decoded and the candidates are SCORED by how many of their letters fall in the script block the wiki language implies; the best wins. No trigger heuristic decides which encoding applies, because both trigger rules tried first were wrong on real data.",
 "WHAT_IS_DELIBERATELY_NOT_DECODED": {
   "residual_undecodable_pct": 0.962,
   "share_of_residue_that_is_zh_ja_ko": 94.6,
   "reason": "in the UTF-16 low-byte form the high byte is destroyed. For Cyrillic or Thai it is a constant and so recoverable; for hanzi, kanji and hangul it ranges over thousands of values and is NOT. Adding a fixed CJK offset produces characters that are individually valid and collectively meaningless -- a diagnostic confirmed this by making +0x3000 and +0x3040 BOTH score 200/200 on the same Japanese sample while yielding contradictory text. These URIs keep a language-only rendering ('Chinese Wikipedia article') instead of invented titles.",
   "also_abstains_when": "a language's measured map does not resolve every escaped byte in the title, e.g. a Polish place name on the English wiki. Filling the unresolved bytes from non-dominant characters would be invention."},
 "IMPROVEMENT": {"undecodable_before_charmap_pct": 1.486, "after_pct": 0.962,
                 "titles_recovered_per_sample": 48770,
                 "note": "the 1.486% baseline is itself higher than an earlier 1.22% reading because widening the URL pattern to /zh-cn/ style paths RECLASSIFIED ~60k CJK URLs out of HOST_PATH into the wiki family. That reclassification recognises more URLs correctly; it did not decode fewer."},
 "NOT_AN_IDENTITY_CLAIM": "URI_DERIVED text is a rendering of the identifier, which was always readable. It is never evidence about what the linked entity was called, and carries is_original_name = false."}
with io.open(f"{V3}/V3_URI_RENDER_MEASURE.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print("wrote V3_URI_RENDER_MEASURE.json")

amend = {
 "schema": "RESOLUTION_OVERLAY_CONTRACT/v2",
 "name": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1",
 "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "status": "AMENDMENT",
 "amends": "V3_RESOLUTION_OVERLAY_CONTRACT.json (v1)",
 "amendment_rule": "v1 is NOT edited. Contract changes are recorded as a new document so the superseded text stays readable, per the standing rule that an amendment never overwrites the record it amends.",
 "WHAT_CHANGED": {
  "COLUMNS": "v1 named the text column display_text and its provenance resolution_source. The finished node schema uses display_name and display_name_source (same enum, renamed), and adds is_original_name so a reader can separate a real Freebase name from a generated description without consulting the enum.",
  "columns_v2": {
    "node_uid": "int64, joins to canonical/nodes/*.parquet",
    "display_name": "string, NEVER null or empty and NEVER a bare MID",
    "display_name_source": "string, one of RESOLUTION_SOURCE in v1",
    "is_original_name": "bool. TRUE only for LITERAL_SELF, FREEBASE_CURRENT_EXACT and FREEBASE_DELETED_EXACT -- the tiers where the string is an actual Freebase name (or the literal itself). A readable key is an identifier, not a name, so FREEBASE_KEY_EXACT is FALSE.",
    "resolution_confidence": "float, the v1 scale unchanged",
    "source_reference": "string, the v1 meaning unchanged"},
  "deferred_to_the_semantic_pass": ["types", "semantic_text"],
  "why_deferred": "display_name completeness must not depend on the 2.06B-edge pass. Every tier in the build is a lookup or a pure function of the identifier, so the zero-opaque invariant is reached first and an interrupted edge pass still leaves a clean graph."},
 "TIER_PRECEDENCE": ["FREEBASE_CURRENT_EXACT / LITERAL_SELF (already in the frozen node table)",
                     "URI_DERIVED (EXTERNAL_URI only)", "FREEBASE_DELETED_EXACT",
                     "FREEBASE_KEY_EXACT", "STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"],
 "PRECEDENCE_NOTE": "FREEBASE_DELETED_EXACT (confidence 0.9) is preferred over FREEBASE_KEY_EXACT (1.0). The confidence scale measures TEMPORAL certainty, while this column is a display name: an actual historical type.object.name is a better name than a current non-name identifier. Both confidences are still reported at their v1 values, so nothing is relabelled to justify the ordering.",
 "CORRECTION_TO_TIER_SIZING": {
  "v1_said": "FREEBASE_KEY_EXACT recovers 3,064,237 of the 19,387,189 unnamed entities (15.8%).",
  "measured": "that counted nodes that HAVE a key, not nodes whose key is a NAME. 75% of them are /dataworld/freeq/job_<uuid> load artifacts and 6.9% are WordNet sense keys; only 369,104 (12.05% of key-bearing nodes, 1.90% of the unnamed population) yield readable text.",
  "record": "V3_KEY_READABILITY.json",
  "consequence": "the AUTHORITATIVE_SOURCE_RESOLUTION figure is materially lower than the v1 sizing implied, and the structural share correspondingly higher."}}
with io.open(f"{V3}/V3_RESOLUTION_OVERLAY_CONTRACT_V2.json", "w", encoding="utf-8") as f:
    json.dump(amend, f, indent=1, ensure_ascii=False)
print("wrote V3_RESOLUTION_OVERLAY_CONTRACT_V2.json")
