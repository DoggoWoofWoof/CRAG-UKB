import io

p = "scratchpad/final_canonical_build/webqsp_v1/cascade_union.py"
s = io.open(p, encoding="utf-8").read()

old = '''    ("FREEBASE_DELETED_NAME",     [],                                "node_id", None, True),'''
new = '''    # Freebase's own deleted-triples dump (deleted_freebase.tar.gz, md5 3a2b9038..., 63,036,271
    # triples through 2013-03). 5,863,639 /type/object/name rows; 44,404 of them fall on unnamed
    # residue nodes, naming 43,033. It sat on disk unused because V3_TIER2_DELETED_EXTRACT measured
    # it against unresolved MEDIATORS only and this list was empty.
    # The subject column is SLASH form (/m/040_1l9) against dot-form node_id (m.040_1l9); joined
    # raw it returns exactly 0 of 5,863,639, which is why deleted_name_join.py normalises first.
    ("FREEBASE_DELETED_NAME",     ["deleted_names_join/*.parquet"],
     "node_id", "display_name", True),'''
assert old in s, "anchor: deleted tier"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched cascade_union.py: FREEBASE_DELETED_NAME now sourced")
