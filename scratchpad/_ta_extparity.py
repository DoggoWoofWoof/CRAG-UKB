"""EXTERNAL parity check for the corrected Track-A BASE.

The in-script M=0 gate only proves A1a/A2 collapse to *my* BASE. This checks the stronger thing: that my
re-implemented canonical router reproduces the FROZEN L1 partition selection that the existing L2 corpus
was actually built from -- data/l2_corpus/{ds}/val/query_meta.json records ANY_GOLD_PRESENT /
ALL_GOLD_PRESENT / N_SCOPE per query, produced by build_l2_corpus.py from the frozen top-50 partitions.

  python scratchpad/_ta_extparity.py <ds>
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA

DS = sys.argv[1] if len(sys.argv) > 1 else "2wiki_clean"
BASE = f"data/ukb_storage/{DS}/gte_qwen/"
NQ = int(os.environ.get("EXT_NQ", "400"))

hard, mem, npart, _adj, _deg, id2row = TA.load_topology(DS)
part_sizes = np.bincount(hard[hard >= 0], minlength=npart)
qm = json.load(open(f"data/l2_corpus/{DS}/val/query_meta.json"))
j = json.load(open(BASE + "query_ids_all.json"))
golds = j["golds"]

qm = qm[:NQ]
rows = [int(m["row_all"]) for m in qm]
dense_all = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
splade_all = np.load(BASE + "splade_top200_all.npy", mmap_mode="r")
dK = [np.asarray(dense_all[i][:TA.K_LOCK], np.int64) for i in rows]
sK = [np.asarray(splade_all[i][:TA.K_LOCK], np.int64) for i in rows]

rank = TA.rrf_partitions([TA.partition_ranking(dK, mem, npart),
                          TA.partition_ranking(sK, mem, npart)], npart)

ok_any = ok_all = ok_scope = 0; n = 0
d_any = d_all = 0
for qi, m in enumerate(qm):
    sel = set(int(x) for x in rank[qi][:TA.P_MAIN])
    g = [id2row[x] for x in golds[rows[qi]] if x in id2row]
    if not g:
        continue
    n += 1
    ret = sum(1 for x in g if int(hard[x]) in sel)
    mine_any = ret > 0
    mine_all = (ret == m["N_GOLD_EXPECTED_INCORP"]) and m["N_GOLD_EXPECTED_INCORP"] > 0
    mine_scope = int(part_sizes[list(sel)].sum())
    ok_any += int(mine_any == m["ANY_GOLD_PRESENT"])
    ok_all += int(mine_all == m["ALL_GOLD_PRESENT"])
    ok_scope += int(mine_scope == m["N_SCOPE"])
    d_any += int(mine_any != m["ANY_GOLD_PRESENT"]); d_all += int(mine_all != m["ALL_GOLD_PRESENT"])

print(f"=== EXTERNAL PARITY {DS} (val, n={n}) : my BASE vs frozen build_l2_corpus query_meta ===")
print(f"  ANY_GOLD_PRESENT agree : {ok_any}/{n} = {ok_any/max(n,1):.4f}   (disagreements {d_any})")
print(f"  ALL_GOLD_PRESENT agree : {ok_all}/{n} = {ok_all/max(n,1):.4f}   (disagreements {d_all})")
print(f"  N_SCOPE exact match    : {ok_scope}/{n} = {ok_scope/max(n,1):.4f}")
print(f"  VERDICT = {'EXACT' if ok_any==n and ok_all==n and ok_scope==n else 'MISMATCH'}")
