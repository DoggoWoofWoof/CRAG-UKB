import io

p = "scratchpad/final_canonical_build/webqsp_v1/key_namespace_census.py"
s = io.open(p, encoding="utf-8").read()
old = '''for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["subject", "key"])
    subj = t["subject"].to_numpy()
    pos = np.clip(np.searchsorted(U, subj), 0, len(U) - 1)
    hit = U[pos] == subj
    if not hit.any():
        continue
    n_hit += int(hit.sum())
    keys = pc.filter(t["key"], pa.array(hit))
    keys = pc.cast(keys, pa.string()) if pa.types.is_dictionary(keys.type) else keys
    hp = pos[hit]
    for k, i in zip(keys.to_pylist(), hp.tolist()):'''
new = '''# key.parquet is keyed on the node_id STRING, not on node_uid, so each row group's subjects are
# hashed here. node_uid = hash(node_id) under PYTHONHASHSEED=0, which is why this file refuses to
# run without it: a different seed would silently match nothing.
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["subject", "key"])
    sub = t["subject"]
    sub = pc.cast(sub, pa.string()) if pa.types.is_dictionary(sub.type) else sub
    subj = np.fromiter((hash(x) for x in sub.to_pylist()), np.int64, t.num_rows)
    pos = np.clip(np.searchsorted(U, subj), 0, len(U) - 1)
    hit = U[pos] == subj
    if not hit.any():
        continue
    n_hit += int(hit.sum())
    keys = pc.filter(t["key"], pa.array(hit))
    keys = pc.cast(keys, pa.string()) if pa.types.is_dictionary(keys.type) else keys
    hp = pos[hit]
    for k, i in zip(keys.to_pylist(), hp.tolist()):'''
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched")
