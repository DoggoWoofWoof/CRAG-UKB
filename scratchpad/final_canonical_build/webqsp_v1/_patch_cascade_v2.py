import io

p = "scratchpad/final_canonical_build/webqsp_v1/cascade_union.py"
s = io.open(p, encoding="utf-8").read()

old_hdr = '''    # ORDER IS LOCKED by V3_NAME_PROVENANCE_ORDER_V1.json (user ruling 2026-09-08).
    # Earlier tier wins the display name; every tier is still retained as evidence.'''
new_hdr = '''    # ORDER IS LOCKED by V3_NAME_PROVENANCE_ORDER_V2.json (user ruling 2026-09-08),
    # sha c4a2a85e461084a459650cfe084293211429cba7217e9590cff7dd8604eec0e7.
    # V1 is superseded, not edited, and is retained as the record of what was locked before.
    # Earlier tier wins the display name; every tier is still retained as evidence.'''
assert old_hdr in s, "anchor: header"
s = s.replace(old_hdr, new_hdr, 1)

# the three historical tiers collapse into ONE class, resolved internally by time
i0 = s.index('''    # Freebase's own deleted-triples dump''')
i1 = s.index('''    ("FREEBASE_ALIAS",''')
new_block = '''    # V2 RULING: FREEBASE_HISTORICAL_DUMP_NAME, FREEBASE_DELETED_NAME and
    # FREEBASE_HISTORICAL_PAGE_NAME are NOT three ranks. They are one class of ORIGINAL Freebase
    # assertions of /type/object/name, differing only in the artifact that preserved them -- a
    # published snapshot, a deletion log, a rendered page. Ranking one artifact as intrinsically
    # more authoritative than another is not justified, so they collapse here and the choice
    # between them is made by TIME, in historical_assertion_union.py:
    #     english if present -> else lexicographically first language -> then latest
    #     last_attested_ts (valid_until for deleted, snapshot_ts for a dump, capture_ts for a page)
    # That matters: on the current evidence 53,804 nodes get a DIFFERENT display name under
    # chronology than under any rigid source order. Source establishes authenticity; time decides
    # which historical name is shown.
    # A deleted name carries assertion_status = DELETED. It is NOT penalised as non-original -- it
    # was an original Freebase name -- the flag exists only so a later surviving assertion wins.
    ("FREEBASE_HISTORICAL_ASSERTION", ["historical_assertion_display/*.parquet"],
     "node_id", "display_name", True),
'''
s = s[:i0] + new_block + s[i1:]

io.open(p, "w", encoding="utf-8").write(s)
print("patched cascade_union.py to V2: three historical tiers -> FREEBASE_HISTORICAL_ASSERTION")
for line in s.splitlines():
    t = line.strip()
    if t.startswith('("'):
        print("   ", t.split('"')[1])
