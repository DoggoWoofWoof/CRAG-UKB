import io

p = "scratchpad/final_canonical_build/webqsp_v1/cascade_union.py"
s = io.open(p, encoding="utf-8").read()

# A tier can be empty for two very different reasons and the record must not conflate them:
# "no source acquired yet" is pending work; "REFUTED" is a measured dead end nobody should spend a
# job on again.  Anchors below deliberately avoid backslash escapes -- the shell heredoc that first
# tried this patch silently turned a literal \n into a newline and the anchor missed.
old1 = '''    mark = "  (no source yet)" if tier in absent else ""'''
new1 = '''    mark = ("  " + CLOSED[tier]) if tier in CLOSED else (
        "  (no source yet)" if tier in absent else "")'''
assert old1 in s, "anchor 1"
s = s.replace(old1, new1, 1)

old2 = '''for tier, _, _, _, _ in CASCADE:
    mark = ('''
new2 = '''# WIKIPEDIA_TITLE is the REFUTED case: V3_KEY_NAMESPACE_CENSUS censused all 143,981,520
# /type/object/key rows against the unnamed residue and /wikipedia/en reaches 1 node,
# /wikipedia/en_title 1, every Wikipedia TITLE namespace together about 15.  A node holding a
# Wikipedia title key kept its /type/object/name, so the unnamed residue does not carry them.
CLOSED = {"WIKIPEDIA_TITLE": "(REFUTED: ~15 residue nodes hold any Wikipedia title key"
                             " -- V3_KEY_NAMESPACE_CENSUS)"}
for tier, _, _, _, _ in CASCADE:
    mark = ('''
assert old2 in s, "anchor 2"
s = s.replace(old2, new2, 1)

old3 = '''       "WHAT_THIS_IS_NOT": ('''
new3 = '''       "EMPTY_TIERS_ARE_NOT_ALL_ALIKE": {
           "WIKIPEDIA_TITLE": "REFUTED, not pending. V3_KEY_NAMESPACE_CENSUS censused all "
                              "143,981,520 /type/object/key rows against the unnamed residue: "
                              "/wikipedia/en reaches 1 node, /wikipedia/en_title 1, every Wikipedia "
                              "TITLE namespace together about 15. Do not commission a decoder. "
                              "Unrelated and still live: the 56,285 wikipedia_en_page_id values "
                              "captured from ARCHIVED PAGES, resolvable by ?curid=.",
           "FREEBASE_CURRENT_NAME": "genuinely pending, not refuted.",
           "FREEBASE_DELETED_NAME": "genuinely pending, not refuted.",
           "DBPEDIA_EXACT": "genuinely pending, not refuted."},
       "WHAT_THIS_IS_NOT": ('''
assert old3 in s, "anchor 3"
s = s.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("patched cascade_union.py: REFUTED tiers now distinguished from PENDING tiers")
