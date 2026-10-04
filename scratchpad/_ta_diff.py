import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
from l1_eval_phase1 import load_partition_topology
from ac_scope_analysis import _partition_ranking as REF_PR, _fuse as REF_FUSE

DS = sys.argv[1]
NQ = 200
hardR, memR, npartR, docsR, _ = load_partition_topology(DS, "C")
hardM, memM, npartM, _a, _d, id2row = TA.load_topology(DS)
mem_ptr, mem_flat = memM
print(f"npart ref={npartR} mine={npartM}   hard equal={np.array_equal(hardR, hardM)}")
mine = [list(mem_flat[mem_ptr[i]:mem_ptr[i+1]]) for i in range(len(hardR))]
diff = [i for i in range(len(hardR)) if list(memR[i]) != [int(x) for x in mine[i]]]
print(f"mem_idx rows differing: {len(diff)} / {len(hardR)}")
if diff:
    i = diff[0]
    print(f"  e.g. row {i}: ref={memR[i][:12]} mine={[int(x) for x in mine[i]][:12]}")
    print(f"  ref sizes mean={np.mean([len(memR[k]) for k in range(len(hardR))]):.2f} "
          f"mine mean={np.mean([len(x) for x in mine]):.2f}")
qm = json.load(open(f"data/l2_corpus/{DS}/val/query_meta.json"))[:NQ]
rows = [int(m["row_all"]) for m in qm]
B = f"data/ukb_storage/{DS}/gte_qwen/"
dA = np.load(B+"dense_top200_all.npy", mmap_mode="r"); sA = np.load(B+"splade_top200_all.npy", mmap_mode="r")
dt = np.array([dA[i] for i in rows]); st = np.array([sA[i] for i in rows])
ref = REF_FUSE(REF_PR(dt[:, :100], memR, npartR, 100), REF_PR(st[:, :100], memR, npartR, 100), npartR)
myr = TA.rrf_partitions([TA.partition_ranking([dt[q][:100] for q in range(len(rows))], memM, npartM),
                         TA.partition_ranking([st[q][:100] for q in range(len(rows))], memM, npartM)], npartM)
eq = np.array_equal(ref[:, :50], myr[:, :50])
ov = np.mean([len(set(ref[q][:50]) & set(myr[q][:50]))/50 for q in range(len(rows))])
print(f"top50 partition sets identical: {eq}   mean overlap {ov:.4f}")
ps = np.bincount(hardR[hardR>=0], minlength=npartR)
sr = [int(ps[list(ref[q][:50])].sum()) for q in range(len(rows))]
print(f"N_SCOPE(ref recompute) == query_meta : {sum(sr[q]==qm[q]['N_SCOPE'] for q in range(len(rows)))}/{len(rows)}")
