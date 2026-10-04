"""Strengthen the FreeQ terminal proposal's falsification check, and correct its var-shape claim.

TWO CORRECTIONS TO V3_KEY_EVIDENCE.json, both from counts taken after it was written.

1. THE TEST WAS WEAKER THAN IT LOOKED.  key_evidence.py ran its check against the 207,792 names
   that existed when it STARTED.  Re-tested against the 579,712 rows now available across seven
   sources (including the tier-2 deleted names and the 2008 dump, neither of which existed then),
   the answer is still 0 of 92,350.  Same verdict, roughly triple the evidence.

2. "dummy_cvt" WAS A SAMPLING ARTEFACT AND THE RECORD SHOULD NOT IMPLY OTHERWISE.  The namespace
   census sampled the first four /dataworld/freeq keys it saw and they read
   job_<uuid>_var_dummy_cvt_..., which invited the reading that these nodes are junk.  Across the
   whole residue dummy_cvt is 6,997 rows, and on HUNT nodes it is exactly ZERO.  The 92,350 hunt
   nodes carry 9,110 distinct var shapes and the large ones are meaningful bulk-load variables:
   google_nces_university_domestic_tuition (23,395), authority_usda_nutrient (12,975),
   organization_irs_headquarters (12,804), business_business_operation_{income,asset,revenue}id
   (~11,325), mb_ngs (1,301 in hunt, 2,931,623 across the residue).

   That is a stronger basis for a terminal bucket than "junk", not a weaker one: each key names the
   measurement schema the row was loaded under, so these are compound-value rows from bulk imports
   (MusicBrainz NGS, USDA nutrients, NCES tuition, IRS filings) that never carried
   /type/object/name because CVTs do not.  It also says the recovery_class overlay binned them as
   entity-like when they are mediators.  STILL NOT APPLIED: machine-origin evidence is not Freebase
   declaring an object nameless, and the three nameless grades stay separate.
"""
import io, json

P = "data/final_canonical/freebase_v3/V3_KEY_EVIDENCE.json"
r = json.load(io.open(P, encoding="utf-8"))
prop = r["FREEQ_TERMINAL_PROPOSAL"]
prop["FALSIFICATION_CHECK_RERUN"] = {
    "why": "the first run tested against only the names that existed when it started.",
    "tested_against": ["fb2010_names", "fb2010_quad_names", "fb2010_quad_names_2008",
                       "fb2010_tsv_names", "wex_names", "fb_page_names", "deleted_names_join"],
    "names_available_for_the_test": 579712,
    "freeq_hunt_nodes_that_ARE_named_by_a_recovered_source": 0,
    "verdict": "proposal survives, on roughly triple the evidence."}
prop["VAR_SHAPE_CORRECTION"] = {
    "what_the_census_sample_suggested": "job_<uuid>_var_dummy_cvt_... , i.e. junk rows",
    "what_the_full_count_shows": "dummy_cvt is 6,997 rows across the residue and ZERO on hunt "
                                 "nodes. The 92,350 hunt nodes carry 9,110 distinct var shapes.",
    "largest_hunt_var_shapes": {"google_nces_university_domestic_tuition": 23395,
                                "authority_usda_nutrient": 12975,
                                "organization_irs_headquarters": 12804,
                                "business_business_operation_incomeid": 3966,
                                "business_business_operation_assetid": 3963,
                                "business_business_operation_revenueid": 3396,
                                "mb_ngs": 1301},
    "reading": "compound-value rows from bulk imports (MusicBrainz NGS, USDA nutrients, NCES "
               "tuition, IRS filings). They never carried /type/object/name because CVTs do not. "
               "This is a STRONGER basis for a terminal bucket than 'junk', and it also says the "
               "recovery_class overlay binned mediators as entity-like.",
    "STILL_NOT_APPLIED": "machine-origin evidence is not Freebase declaring an object nameless."}
io.open(P, "w", encoding="utf-8").write(json.dumps(r, indent=1, ensure_ascii=False))
print("V3_KEY_EVIDENCE.json updated: rerun check + var-shape correction")
