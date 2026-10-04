"""Amendment 3 to the CRAG_FREEBASE_CANONICAL contract.

    python scratchpad/final_canonical_build/webqsp_v1/v3_amend_contract_v3.py

V1 froze with the rule: "Any later change to the contract must appear as a new record with its own
hash, never as an edit that leaves this one looking prescient." So this is a separate record citing
v2's hash. Neither v1 nor v2 is touched.

WHAT PROMPTED IT. v2 held SOURCE_SUBJECT_CONTIGUOUS as PROVISIONAL on a 3 GB prefix and named the
three numbers that had to be measured over the whole source before PASS B could rely on it. They
have now been measured over all 31,305,093,084 verified bytes. The gate is satisfied, so the clause
moves from PROVISIONAL to ESTABLISHED and the subject-side external sort leaves the budget.

A SECOND, UNANTICIPATED CHANGE. The source is not one gzip stream but 200 independent members, so
every full pass is parallel rather than serial. v1 and v2 both budgeted passes as serial
decompression. That is now a property of the source worth recording, because it changes what the
remaining passes cost and because the seam it introduces is a correctness hazard, not just a
performance note.
"""
import hashlib
import json
import os
import time

D = "data/final_canonical/freebase_v3"
OUT = f"{D}/V3_CANONICAL_CONTRACT_V3.json"
V1 = json.load(open(f"{D}/V3_CANONICAL_CONTRACT.json", encoding="utf-8"))
V2 = json.load(open(f"{D}/V3_CANONICAL_CONTRACT_V2.json", encoding="utf-8"))
CONTIG = json.load(open(f"{D}/V3_SUBJECT_CONTIGUITY.json", encoding="utf-8"))
IDX = json.load(open(f"{D}/V3_GZIP_MEMBER_INDEX.json", encoding="utf-8"))
ACQ = json.load(open(f"{D}/V3_RAW_MIRROR_ACQUISITION.json", encoding="utf-8"))
META = json.load(open(f"{D}/pass_a/shards/_all_meta.json", encoding="utf-8"))

lines = sum(m["lines"] for m in META)
rows = sum(m["rows"] for m in META)
dec = sum(m["decompressed_bytes"] for m in META)

body = {
    "schema": "V3_CANONICAL_CONTRACT/v3",
    "name": "CRAG_FREEBASE_CANONICAL",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "AMENDMENT. Resolves the SUBJECT_CONTIGUITY_IS_PROVISIONAL clause of v2 and adds a "
              "source-structure clause. Every other v1 and v2 clause stands.",
    "amends": {
        "record": "V3_CANONICAL_CONTRACT_V2.json",
        "v2_contract_hash": V2["CONTRACT_HASH"],
        "v1_contract_hash": V1["CONTRACT_HASH"],
        "v1_rule_being_honoured": V1["CONTRACT_HASH_NOTE"],
        "v1_and_v2_are_unmodified": True,
    },

    "SUBJECT_CONTIGUITY_RESOLVED": {
        "supersedes": "V3_CANONICAL_CONTRACT_V2.json SUBJECT_CONTIGUITY_IS_PROVISIONAL",
        "SOURCE_SUBJECT_CONTIGUOUS": "ESTABLISHED",
        "SUBJECT_BLOCKS_N": CONTIG["SUBJECT_BLOCKS_N"],
        "DISTINCT_SUBJECTS_N": CONTIG["DISTINCT_SUBJECTS_N"],
        "REOPENED_SUBJECT_BLOCKS_N": CONTIG["REOPENED_SUBJECT_BLOCKS_N"],
        "measured_over": "the whole verified source, all 200 members, not a prefix",
        "gate_from_v2": "only if REOPENED_SUBJECT_BLOCKS_N == 0 over the whole verified source may "
                        "PASS B skip subject-side external sorting.",
        "gate_result": "SATISFIED. PASS B does not sort the subject side. The object side still "
                       "needs the external join, which is what PASS C exists for.",
        "why_the_result_is_exact_not_probabilistic":
            "openers were hashed to 64 bits so ~1.2e8 of them fit in memory, and ZERO duplicate "
            "hashes were found. A real reopening necessarily produces a duplicate hash; a hash can "
            "manufacture a false duplicate but cannot conceal a true one. Zero duplicate hashes "
            "therefore proves zero reopenings outright, and the exact-resolution second pass had "
            "nothing to resolve. Had any duplicate appeared it would have been resolved against "
            "the real strings before being reported.",
        "why_it_held": "the export partitions by SUBJECT, not by triple. Every subject lives in "
                       "exactly one member and none straddles any of the 199 member seams "
                       "(blocks_merged_across_seams = 0), so no subject is split by the way the "
                       "file was written OR by the way it is read.",
        "sortedness_is_still_not_the_test": "the source is 200 internally sorted members and is "
                                            "globally unsorted by construction. Contiguity was "
                                            "decided by duplicate openers alone, as v2 required.",
    },

    "SOURCE_STRUCTURE_MEASURED": {
        "not_a_single_gzip_stream": True,
        "MEMBERS_N": IDX["MEMBERS_N"],
        "member_mean_compressed_bytes": IDX["member_size_compressed"]["mean"],
        "source_bytes": ACQ["bytes"] if "bytes" in ACQ else IDX["source_bytes"],
        "decompressed_bytes": dec,
        "expansion_ratio": round(dec / IDX["source_bytes"], 3),
        "LINES_N": lines,
        "PASS_A_ROWS_N": rows,
        "line_alignment": "every member begins at a line boundary and ends with a newline; "
                          "verified by decompressing one interior member in full during indexing, "
                          "then confirmed across all 200 in PASS A "
                          "(members_not_newline_terminated = 0, members_with_unused_data = 0).",
        "consequence": "each member is a self-contained deflate stream, so every full pass over "
                       "the source is member-parallel rather than serial. Measured: 409 GB "
                       "decompressed in 26.1 min at 8 workers, against ~1.3 h for the fastest "
                       "serial reader. v1 and v2 budgeted passes as serial; PASSES B, C and D "
                       "should be built member-parallel.",
        "correction_to_earlier_estimates": "earlier notes carried ~250 GB decompressed and ~1.9e9 "
                                           "lines, extrapolated from a 40-second window. The "
                                           "measured figures are 409 GB and 3.008e9 lines; the "
                                           "expansion ratio is 13.1x, not ~8x.",
        "SEAM_HAZARD": "a member boundary is a correctness hazard for anything that reasons about "
                       "consecutive lines, not merely a performance detail. A subject block "
                       "straddling a seam is opened once by each of two workers and would be "
                       "counted as a reopening. Any member-parallel pass MUST report the first and "
                       "last subject of each member so seams can be stitched. Measured here: 0 "
                       "straddling subjects, but the check must remain in place because that is a "
                       "property of this source, not a guarantee about it.",
    },

    "ENGINEERING_CONSTRAINT_RECORDED": {
        "finding": "on the build machine, Git Bash (MSYS) pipe emulation throttles bulk streams by "
                   "roughly 20x. Measured on the same file with the same gzip binary: 4.4 MB/s "
                   "compressed through 'gzip -dc | grep | python' versus 99 MB/s decompressed "
                   "reading a native subprocess pipe from Python.",
        "rule": "no production pass may route the decompressed stream through a shell pipeline. "
                "Passes read the source with in-process zlib over a member byte range, or via a "
                "native subprocess pipe.",
        "why_recorded_in_the_contract": "it is not a preference. A pass built the shell way takes "
                                        "hours and looks like a slow parser, which is exactly the "
                                        "wrong diagnosis and cost a rebuild once already.",
        "diagnostic_lesson": "the first reading of this blamed per-line Python cost, by comparing "
                             "Python's matched-BYTE rate against the C chain's INPUT rate. Compare "
                             "like units before concluding which stage is slow. Windows Defender "
                             "and disk contention were also hypothesised and both measured false.",
    },

    "PASS_A_STATUS": {
        "stage_1": "COMPLETE. 200/200 members, 3,008,314,716 lines, 815,908,863 rows, 26.1 min. "
                   "short_lines = 0, literal_parse_failed = 0, subject_not_ns = 0.",
        "stage_2": "converts shards to Parquet and makes the two redundancy decisions as per-member "
                   "exact set comparisons, permitted only because contiguity is now ESTABLISHED.",
        "redundancy_rule_unchanged_from_v2": "rdfs:label is dropped only where it equals a "
                                             "type.object.name row of the SAME subject; everything "
                                             "else lands in label_residue. rdf:type is NOT "
                                             "collapsed into type.object.type wholesale -- only "
                                             "per-subject covered rows drop, the rest land in "
                                             "rdf_type_residue, and out-of-namespace rdf:type is "
                                             "the RDF/OWL vocabulary layer and is always kept.",
    },

    "NEXT_CHECKPOINT": {
        "was": V2.get("NEXT_CHECKPOINT"),
        "now": "PASS A stage 2 complete, then PASS B built member-parallel with literal_uid minting, "
               "validated against the IDIR oracle at the pre-registered thresholds in v2 "
               "(PASS >= 99.9% / INVESTIGATE 99.0-99.9% / STOP < 99.0%).",
    },
}

payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
body["CONTRACT_HASH"] = hashlib.sha256(payload).hexdigest()
body["CONTRACT_HASH_NOTE"] = ("sha256 over this amendment with this field absent. The v2 hash is "
                              "cited above; v1 and v2 remain byte-identical to when they were "
                              "frozen. Any later change is a new record, never an edit.")

os.makedirs(D, exist_ok=True)
tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(body, fh, indent=2)
os.replace(tmp, OUT)

print("V1 hash:", V1["CONTRACT_HASH"])
print("V2 hash:", V2["CONTRACT_HASH"])
print("V3 hash:", body["CONTRACT_HASH"])
print(json.dumps(body["SUBJECT_CONTIGUITY_RESOLVED"], indent=1)[:700])
