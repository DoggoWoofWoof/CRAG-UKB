import io

p = "scratchpad/final_canonical_build/webqsp_v1/fb2010_quads.py"
s = io.open(p, encoding="utf-8").read()
n = 0


def rep(a, b):
    global s, n
    if a not in s:
        raise SystemExit(f"anchor {n} not found:\n{a[:200]}")
    s = s.replace(a, b, 1)
    n += 1


rep('''OUTPUT (append-only)
  _acquisition/fb2010_quad_labels/part_*.parquet   every attested label: node_uid, lang, label, kind
  _acquisition/fb2010_quad_names/part_00000.parquet the display name per node (the tier table)
  V3_FB2010_QUADS.json
"""''', '''TYPES AND KEYS COME FREE IN THE SAME STREAM
  Measured on a 2,500,000-line sample of the real file, the predicate mix is
      /type/object/type 582,406   /type/object/key 381,436   /type/object/name 230,370
  so the quads dump also states, for every 2010 object, the types it carried and the keys it was
  minted under.  That is the source-attested answer to the question "what ARE the 5,832,113
  UNTYPED_CANDIDATE nodes", for the subset the 2010 dump covers -- the frozen 2015 graph states no
  type for them at all.  Capturing it costs nothing here and would cost a second 4.23 GB stream
  later, so it is captured.  It goes to its own evidence table and is NOT applied: the frozen
  node_kind is not touched and no band is recomputed by this script.

OUTPUT (append-only)
  _acquisition/fb2010_quad_labels/part_*.parquet   every attested label: node_uid, lang, label, kind
  _acquisition/fb2010_quad_names/part_00000.parquet the display name per node (the tier table)
  _acquisition/fb2010_quad_struct/part_*.parquet   2010 types and keys for residue nodes (evidence)
  V3_FB2010_QUADS.json
"""''')

rep('''LBL = f"{ACQ}/fb2010_quad_labels"
OUT = f"{ACQ}/fb2010_quad_names"
os.makedirs(LBL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)''', '''LBL = f"{ACQ}/fb2010_quad_labels"
OUT = f"{ACQ}/fb2010_quad_names"
STR = f"{ACQ}/fb2010_quad_struct"
for _d in (LBL, OUT, STR):
    os.makedirs(_d, exist_ok=True)''')

rep('''NAME_P = "/type/object/name"
ALIAS_P = "/common/topic/alias"''', '''NAME_P = "/type/object/name"
ALIAS_P = "/common/topic/alias"
TYPE_P = "/type/object/type"
KEY_P = "/type/object/key"
KIND = {NAME_P: "name", ALIAS_P: "alias", TYPE_P: "type", KEY_P: "key"}''')

rep('''# ---------------------------------------------------------------- pass 1
part = len(glob.glob(f"{LBL}/*.parquet"))
b_uid, b_mid, b_lang, b_lab, b_kind = [], [], [], [], []
k_uid, k_mid, k_lang, k_lab, k_kind, k_hunt = [], [], [], [], [], []
n_lines = n_name = n_alias = n_kept = 0


def flush():''', '''# ---------------------------------------------------------------- pass 1
part = len(glob.glob(f"{LBL}/*.parquet"))
spart = len(glob.glob(f"{STR}/*.parquet"))
b_uid, b_mid, b_lang, b_lab, b_kind = [], [], [], [], []
k_uid, k_mid, k_lang, k_lab, k_kind, k_hunt = [], [], [], [], [], []
s_uid, s_mid, s_kind, s_val, s_hunt = [], [], [], [], []
n_lines = n_kept = n_struct = 0
seen_p = collections.Counter()


def sflush():
    global spart, s_uid, s_mid, s_kind, s_val, s_hunt
    if not s_uid:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array(s_uid, pa.int64()), "node_id": pa.array(s_mid),
        "fact_kind": pa.array(s_kind), "value_2010": pa.array(s_val),
        "in_hunt": pa.array(s_hunt, pa.bool_())}),
        f"{STR}/part_{spart:05d}.parquet", compression="zstd")
    spart += 1
    s_uid, s_mid, s_kind, s_val, s_hunt = [], [], [], [], []


def flush():''')

rep('''def drain():
    global n_kept
    if not b_uid:
        return
    arr = np.array(b_uid, np.int64)
    p = np.clip(np.searchsorted(U, arr), 0, len(U) - 1)
    inres = U[p] == arr
    q = np.clip(np.searchsorted(H, arr), 0, len(H) - 1)
    inhunt = (H[q] == arr) & inres
    for i in np.flatnonzero(inres).tolist():
        k_uid.append(b_uid[i]); k_mid.append(b_mid[i]); k_lang.append(b_lang[i])
        k_lab.append(b_lab[i]); k_kind.append(b_kind[i]); k_hunt.append(bool(inhunt[i]))
    n_kept += int(inres.sum())
    b_uid.clear(); b_mid.clear(); b_lang.clear(); b_lab.clear(); b_kind.clear()
    if len(k_uid) >= 500000:
        flush()''', '''def drain():
    global n_kept, n_struct
    if not b_uid:
        return
    arr = np.array(b_uid, np.int64)
    p = np.clip(np.searchsorted(U, arr), 0, len(U) - 1)
    inres = U[p] == arr
    q = np.clip(np.searchsorted(H, arr), 0, len(H) - 1)
    inhunt = (H[q] == arr) & inres
    for i in np.flatnonzero(inres).tolist():
        kd = b_kind[i]
        if kd == "name" or kd == "alias":
            k_uid.append(b_uid[i]); k_mid.append(b_mid[i]); k_lang.append(b_lang[i])
            k_lab.append(b_lab[i]); k_kind.append(kd); k_hunt.append(bool(inhunt[i]))
            n_kept += 1
        else:
            s_uid.append(b_uid[i]); s_mid.append(b_mid[i]); s_kind.append(kd)
            s_val.append(b_lab[i]); s_hunt.append(bool(inhunt[i]))
            n_struct += 1
    b_uid.clear(); b_mid.clear(); b_lang.clear(); b_lab.clear(); b_kind.clear()
    if len(k_uid) >= 500000:
        flush()
    if len(s_uid) >= 500000:
        sflush()''')

rep('''        p_ = f[1]
        if p_ == NAME_P:
            n_name += 1
            kind = "name"
        elif p_ == ALIAS_P:
            n_alias += 1
            kind = "alias"
        else:
            continue
        lab = f[3]
        if not lab:
            continue''', '''        kind = KIND.get(f[1])
        if kind is None:
            continue
        seen_p[kind] += 1
        # name/alias carry the string in col 4; type/key name their object in col 3
        lab = f[3] if (kind == "name" or kind == "alias") else (f[2] or f[3])
        if not lab:
            continue''')

rep('''        if n_lines % 20000000 == 0:
            print(f"  {n_lines:,} quads  names {n_name:,} aliases {n_alias:,}  "
                  f"residue labels {n_kept:,}  ({time.time()-t0:.0f}s)", flush=True)
drain()
flush()
print(f"pass 1 done: {n_lines:,} quads, {n_kept:,} residue labels ({time.time()-t0:.0f}s)", flush=True)''',
    '''        if n_lines % 20000000 == 0:
            print(f"  {n_lines:,} quads  name {seen_p['name']:,} alias {seen_p['alias']:,} "
                  f"type {seen_p['type']:,} key {seen_p['key']:,}  residue labels {n_kept:,} "
                  f"struct {n_struct:,}  ({time.time()-t0:.0f}s)", flush=True)
drain()
flush()
sflush()
print(f"pass 1 done: {n_lines:,} quads, {n_kept:,} residue labels, {n_struct:,} residue "
      f"type/key facts ({time.time()-t0:.0f}s)", flush=True)''')

rep('''       "quads_read": n_lines, "name_quads": n_name, "alias_quads": n_alias,
       "residue_label_rows": n_kept,''',
    '''       "quads_read": n_lines,
       "quads_by_predicate": {k: int(v) for k, v in seen_p.items()},
       "residue_label_rows": n_kept,
       "residue_type_and_key_facts": n_struct,
       "TYPES_AND_KEYS_ARE_EVIDENCE_ONLY": "fb2010_quad_struct/ records the types and keys the 2010 "
                                           "dump states for residue nodes. The frozen node_kind is "
                                           "NOT touched and no band is recomputed here; applying it "
                                           "would be a new overlay record with its own hash.",''')

io.open(p, "w", encoding="utf-8").write(s)
print(f"patched {n} anchors")
