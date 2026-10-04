import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _ta_prepartition as TA
DS = sys.argv[1]
hard, mem, npart, _a, _d, id2row = TA.load_topology(DS)
d = f"data/l2_corpus/{DS}/val"
off = np.load(f"{d}/query_offsets.npy"); cand = np.load(f"{d}/cand_ids.npy")
pid = np.load(f"{d}/part_id.npy"); qm = json.load(open(f"{d}/query_meta.json"))
B = f"data/ukb_storage/{DS}/gte_qwen/"
dA = np.load(B+"dense_top200_all.npy", mmap_mode="r"); sA = np.load(B+"splade_top200_all.npy", mmap_mode="r")
N = 200; rows = [int(m["row_all"]) for m in qm[:N]]
rank = TA.rrf_partitions([TA.partition_ranking([np.asarray(dA[i][:100],np.int64) for i in rows], mem, npart),
                          TA.partition_ranking([np.asarray(sA[i][:100],np.int64) for i in rows], mem, npart)], npart)
agree_pid = 0; ov = []
for q in range(N):
    s, e = int(off[q]), int(off[q+1])
    c = cand[s:e]
    agree_pid += int(np.array_equal(pid[s:e].astype(np.int64), hard[c]))   # is the STORED part_id still right?
    stored_sel = set(int(x) for x in np.unique(pid[s:e]))
    ov.append(len(stored_sel & set(int(x) for x in rank[q][:50])) / 50)
print(f"{DS}: stored part_id still equals current hard[cand] : {agree_pid}/{N}")
print(f"{DS}: stored selected-partition set vs recomputed top50 overlap = {np.mean(ov):.4f}")
print(f"{DS}: stored N_SCOPE mean {np.mean([m['N_SCOPE'] for m in qm[:N]]):.0f} vs stored len(cand) mean "
      f"{np.mean([int(off[q+1])-int(off[q]) for q in range(N)]):.0f}")
