"""L1X -- the development population for HotpotQA and 2Wiki (write-once record; development only, nothing sealed is touched).
(User 2026-10-02 goal: "try to get 90+ for all datasets"; HotpotQA / 2Wiki had no development population -- their 969 / 996 rows are the SEALED transfer sets of L1_ROUTING_CARRY_FORWARD__v1.)

Rule (pre-declared here, before any L1 number is computed on these rows): eligible = rows of the TRAIN split (split_code 0; the evaluation split and the TEST split are never read) whose gold nodes are all resolved
(every dataset row passes: gold_pos >= 0), minus the rows of the LEGACY_CONTINUITY lane.  One fixed seed 20261002: RandomState(seed).permutation of the sorted eligible row list; the first 2000 rows, sorted ascending.
These rows are 'historically exposed' (train rows were used to train earlier heads); they are freely reusable development rows, like MetaQA / SQuAD / MuSiQue in loc_population.json, and never a confirmatory population.

  python scratchpad/_l1x_txt_pop.py        -> results/L1_X/TXT_DEV_POPULATION__v1.json   (refuses to overwrite)
"""
import io
import json
import os
import time

import numpy as np

import _l1d_lib as D

SEED = 20261002
N_ROWS = 2000
DATASETS = ("hotpotqa", "2wiki")
OUT = os.path.join(D.REPO, "results", "L1_X", "TXT_DEV_POPULATION__v1.json")


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    t0 = time.time()
    pops = {}
    for ds in DATASETS:
        cd = D.AD.CanonicalDataset(ds)
        zq = D.qindex(cd)
        sc = np.asarray(zq["split_code"])
        gp, gpos = np.asarray(zq["gold_ptr"]), np.asarray(zq["gold_pos"])
        assert cd.split_ranges["train"][0] == 0 and (sc[:cd.split_ranges["train"][1]] == 0).all() and (sc[cd.split_ranges["train"][1]:] != 0).all()
        ids = cd.query_ids
        lane = os.path.join(D.REPO, "data", "final_canonical", ds, "queries", "lanes", "LEGACY_CONTINUITY.jsonl")
        lq = {json.loads(l)["query_id"] for l in io.open(lane, encoding="utf-8")}
        tr = np.flatnonzero(sc == 0)
        resolved = np.array([(gpos[gp[r]:gp[r + 1]] >= 0).all() and gp[r + 1] > gp[r] for r in tr])
        excl = np.array([ids[int(r)] in lq for r in tr])
        elig = tr[resolved & ~excl]
        perm = np.random.RandomState(SEED).permutation(len(elig))
        rows = np.sort(elig[perm[:N_ROWS]])
        qids = [ids[int(r)] for r in rows]
        ng = np.diff(gp)[rows]
        pops[ds] = {"n": int(len(rows)), "rows": [int(r) for r in rows], "query_ids": qids, "query_ids_sha256": D.sha_text(",".join(qids)),
                    "rows_sha256": D.sha_text(",".join(str(int(r)) for r in rows)),
                    "composition": {"train_rows": int(len(tr)), "train_rows_with_all_golds_resolved": int(resolved.sum()), "excluded_legacy_continuity_lane": int(excl.sum()), "eligible": int(len(elig)),
                                    "n_gold_nodes_per_row": {str(k): int((ng == k).sum()) for k in sorted(set(ng.tolist()))}},
                    "dataset_pins": {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": D.sha_file(cd._query_index_path())}}
        print(ds, pops[ds]["composition"], "rows", len(rows), flush=True)
    rec = {"mode": "TXT_DEV_POPULATION", "label": "development population (train rows; historically exposed; not held-out; freely reusable)", "rule": __doc__, "seed": SEED, "populations": pops,
           "guards": {"evaluation_split_rows_read": 0, "TEST_rows_read": 0, "sealed_transfer_sets_touched": 0},
           "code": {"path": "scratchpad/_l1x_txt_pop.py", "sha256": D.sha_file(os.path.abspath(__file__))}, "pinned_repo": D.PINNED_REPO, "seconds": round(time.time() - t0, 1)}
    D.G.S.wj(OUT, rec)
    print("wrote", OUT, D.sha_file(OUT))


if __name__ == "__main__":
    main()
