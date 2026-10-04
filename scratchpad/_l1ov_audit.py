"""PHASE 8 -- the explosion audit, run BEFORE any retrieval evaluation.

The program is explicit that this must show the actual explosion rather than guess it:
unrestricted 1-hop overlap is only usable if the membership replication factor is small.

    R = sum_j |C_j u H_j| / |V|

R = 1 is a hard partition.  R = 1.4 would be excellent.  R = 17 would make unrestricted
overlap unusable and send us to the bounded halos of Phase 13.

  python scratchpad/_l1ov_audit.py [ds ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ov_core as OV
import _l1ep_pu as PU

OUT, log = OV.OUT, OV.log


def main(dss):
    os.makedirs(f"{OUT}/overlap", exist_ok=True)
    os.makedirs(f"{OUT}/exposure", exist_ok=True)
    fp = f"{OUT}/overlap/EXPLOSION_AUDIT.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dss:
        hard, npart = PU.load_assignment(ds, "CURRENT")
        hard = np.asarray(hard, np.int64)
        log(f"{ds}: N={len(hard):,} npart={npart:,}")
        r = OV.explosion(ds, hard, npart, "CURRENT", log=log)
        rec[ds] = r
        json.dump(rec, open(fp, "w"), indent=1)
        for fam in ["O0_CORE"] + OV.FAMS:
            v = r[fam]
            log(f"  {fam:12s} R={v['REPLICATION_FACTOR']:8.3f}  halo/core={v['HALO_CORE_RATIO']:8.3f}"
                f"  mult mean {v['NODE_MULTIPLICITY']['mean']:8.3f} p99 "
                f"{v['NODE_MULTIPLICITY']['p99']:9.1f} max {v['NODE_MULTIPLICITY']['max']:8d}"
                f"  block mean {v['BLOCK_SIZE']['mean']:10.1f} p99 {v['BLOCK_SIZE'].get('p99', 0):10.1f}")
        if "PROVENANCE" in r:
            log(f"  provenance {json.dumps(r['PROVENANCE'])}")
    log("wrote", fp)


if __name__ == "__main__":
    main(sys.argv[1:] or list(OV.DS))
