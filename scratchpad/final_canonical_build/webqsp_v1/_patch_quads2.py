import io

p = "scratchpad/final_canonical_build/webqsp_v1/fb2010_quads.py"
s = io.open(p, encoding="utf-8").read()
n = 0


def rep(a, b):
    global s, n
    if a not in s:
        raise SystemExit(f"anchor {n} not found:\n{a[:240]}")
    s = s.replace(a, b, 1)
    n += 1


rep('''"""Names (all languages) and aliases from the 2010-07-16 Freebase QUADRUPLES dump.

    PYTHONHASHSEED=0 python .../fb2010_quads.py
''', '''"""Names (all languages) and aliases from a Freebase QUADRUPLES dump.

    PYTHONHASHSEED=0 python .../fb2010_quads.py [2010|2008]

TWO RELEASES, ONE PARSER
  2010-07-16 (4.23 GB) and 2008-03-28 (0.56 GB) share the format exactly:
      subject <TAB> predicate <TAB> object <TAB> literal
  They differ only in what the subject can be.  The 2008 file mixes SCHEMA objects, whose subject is
  a path like /american_football/football_coach, with INSTANCES, whose subject is /guid/<32hex>;
  measured on 600,000 lines, name quads split 40,465 path to 61,457 guid.  Path subjects are Freebase
  schema, not graph nodes, and the guid-prefix test already drops them without coercion.
''')

rep('''SRC = f"{ACQ}/raw/quads_freebase-datadump-quadruples.tsv.bz2"
LBL = f"{ACQ}/fb2010_quad_labels"
OUT = f"{ACQ}/fb2010_quad_names"
STR = f"{ACQ}/fb2010_quad_struct"''', '''WHICH = sys.argv[1] if len(sys.argv) > 1 else "2010"
if WHICH not in ("2010", "2008"):
    sys.exit("usage: fb2010_quads.py [2010|2008]")
SRCF = {"2010": "quads_freebase-datadump-quadruples.tsv.bz2",
        "2008": "quads2008_freebase-datadump-quadruples.tsv.bz2"}[WHICH]
SFX = "" if WHICH == "2010" else "_2008"
SRC = f"{ACQ}/raw/{SRCF}"
LBL = f"{ACQ}/fb2010_quad_labels{SFX}"
OUT = f"{ACQ}/fb2010_quad_names{SFX}"
STR = f"{ACQ}/fb2010_quad_struct{SFX}"''')

rep('''TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = "2010-07-16"''', '''TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = {"2010": "2010-07-16", "2008": "2008-03-28"}[WHICH]''')

rep('''    print(f"  {r[1]:<15} {r[2][:50]!r:<52} [{r[3]}] labels={r[5]} hunt={r[6]}")''',
    '''    print(f"  {r[1]:<15} {r[2][:50]!r:<52} [{r[3]}] labels={r[5]} hunt={r[6]}")''')

s = s.replace('f"{V3}/V3_FB2010_QUADS.json"', 'f"{V3}/V3_FB2010_QUADS{SFX.upper()}.json"')
s = s.replace('"source": "freebase-data-dump-2010-07-16 / freebase-datadump-quadruples.tsv.bz2 "',
              '"source": f"{SRCF} at release {RELEASE} "')
io.open(p, "w", encoding="utf-8").write(s)
print(f"patched {n} anchors")
