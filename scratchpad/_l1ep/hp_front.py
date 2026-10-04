"""front-run the decisive hotpot partitionings while the main chain grinds through P2_METIS_STRONG.
build() is cached on the .npy, so the main chain will skip whatever finishes here first.
Writes NO metadata json -- the chain owns that file, and this must not race it."""
import os, sys, time
sys.path.insert(0, os.path.join(os.getcwd(), "scratchpad"))
sys.path.insert(0, os.getcwd())
import _l1ep_part as PP

DS = "hotpotqa_clean"
ORDER = ["P4_CE_LOCAL_ONLY", "PM3_TOPOLOGY_C_seed1", "PM3_TOPOLOGY_C_seed2",
         "PM3_TOPOLOGY_C_seed3", "PM3_TOPOLOGY_C_seed4", "P0_RANDOM_BALANCED_s0",
         "P4_CE_UNWEIGHTED", "P4_CE_NER_ONLY", "PM4_TOPOLOGY_C_NERW"]
for w in ORDER:
    fp = f"{PP.CACHE}/{DS}__{w}.npy"
    if os.path.exists(fp):
        print(f"SKIP {w} (already built)", flush=True)
        continue
    t = time.time()
    PP.build(DS, w)
    print(f"FRONT_DONE {w} ({time.time()-t:.0f}s)", flush=True)
print("FRONT_ALL_DONE", flush=True)
