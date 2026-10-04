"""Correct the VERDICT field of V3_CURRENT_NAME_GATE.json.

The measurements were right; the verdict LABEL derived from them was not.  current_name_gate.py
computed `verdict = EMPTY_BY_CONSTRUCTION if name_hits == 0 else EMPTY_BY_OMISSION`, and name_hits
came back as 1 -- so the record asserted "FREE NAMES FOUND".  The one hit is node m.011q6t0v whose
lexical value is a single SPACE.  A whitespace-only /type/object/name is not a name, so the honest
verdict is the opposite of the one stored.

Numbers are not touched.  The verdict, and the test that produces it, are.
"""
import io, json

V3 = "data/final_canonical/freebase_v3"
P = f"{V3}/V3_CURRENT_NAME_GATE.json"
r = json.load(io.open(P, encoding="utf-8"))
nm = r["results"]["name"]
al = r["results"]["alias"]
blank = [s for s in nm["samples"] if not s["lexical"].strip()]

r["VERDICT"] = "EMPTY_BY_CONSTRUCTION"
r["VERDICT_CORRECTED"] = {
    "previous_value": "EMPTY_BY_OMISSION -- FREE NAMES FOUND",
    "why_it_was_wrong": "the verdict tested name_hits == 0 and name_hits is 1. That single row is "
                        "node m.011q6t0v whose /type/object/name literal is a single space. A "
                        "whitespace-only name is not a name, so the correct verdict is the "
                        "opposite of the one first stored.",
    "test_should_be": "hits whose lexical value is non-empty after stripping whitespace",
    "non_blank_name_hits": nm["rows_whose_subject_is_an_UNNAMED_RESIDUE_node"] - len(blank)}
r["WHAT_THIS_ESTABLISHES"] = {
    "FREEBASE_CURRENT_NAME": "empty BY CONSTRUCTION, not by omission. Exactly 1 of 68,362,456 "
                             "name rows across 240 languages lands on a residue node and its value "
                             "is a space. The residue was computed against ALL languages, not "
                             "English only, so no original non-English name is sitting unclaimed.",
    "FREEBASE_ALIAS": f"complete. {al['rows_whose_subject_is_an_UNNAMED_RESIDUE_node']} alias rows "
                      f"on {al['distinct_such_nodes']} residue nodes, 0 in the hunt, and the "
                      f"cascade already carries exactly 45 FREEBASE_ALIAS source rows -- the same "
                      f"nodes. Independently confirms that tier is fully drained.",
    "m.011q6t0v": "a node whose only /type/object/name is whitespace. Not promoted, not graded; "
                  "recorded as a data-quality curiosity for the terminal-bucket discussion."}
io.open(P, "w", encoding="utf-8").write(json.dumps(r, indent=1, ensure_ascii=False))
print("corrected VERDICT ->", r["VERDICT"])
print("blank-name hits:", len(blank), " non-blank name hits:",
      r["VERDICT_CORRECTED"]["non_blank_name_hits"])

# and fix the test in the script itself so a re-run cannot reproduce the wrong label
p = "scratchpad/final_canonical_build/webqsp_v1/current_name_gate.py"
s = io.open(p, encoding="utf-8").read()
old = '''nh = report["name"]["rows_whose_subject_is_an_UNNAMED_RESIDUE_node"]
verdict = ("EMPTY_BY_CONSTRUCTION" if nh == 0 else "EMPTY_BY_OMISSION -- FREE NAMES FOUND")'''
new = '''# A whitespace-only /type/object/name is not a name. Counting one as a hit once flipped this
# verdict to "FREE NAMES FOUND" on the strength of a single space, so the test strips first.
nh = report["name"]["rows_whose_subject_is_an_UNNAMED_RESIDUE_node"]
nblank = sum(1 for smp in report["name"]["samples"] if not smp["lexical"].strip())
verdict = ("EMPTY_BY_CONSTRUCTION" if nh - nblank <= 0 else "EMPTY_BY_OMISSION -- NAMES FOUND")'''
assert old in s, "anchor: verdict"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched current_name_gate.py: verdict now ignores whitespace-only names")
