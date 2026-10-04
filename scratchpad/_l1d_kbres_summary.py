"""L1 DEVELOPMENT -- KBRES L3 split: the kbres v1 lever L3, split by where a second hop would have to be computed (reads
results/L1_DEV/kbres_<ds>__v1.{json,npz} only; nothing is recomputed from the corpus).  Development rows only; descriptive; no rule
is selected and L1 is not changed.

The v1 levers L2, LOCAL and ROUTE are kept exactly; the v1 lever L3 is split (an intermediate = a STRUCT neighbour of a hit and
of the gold, i.e. a one-hop STRUCT node of V_q; "contacted" = its shard is contacted by the cell's ES fan-out):
  L3_PUSH2  kbres REACH_2 with an intermediate in a contacted shard: the second hop can be computed from contacted rows, but it
            emits a gold whose own shard is not contacted (one more shard per such gold)
  L3_X2     kbres CROWD_2X: the gold's shard is contacted, every intermediate's shard is not (the second hop needs uncontacted rows)
  L3_FAR2   kbres REACH_2 with no intermediate in a contacted shard (both the second hop and the gold outside the contacted shards)
  L3_HOP3   kbres CROWD_3 u REACH_3 (dS >= 3: at least a third hop)
UNROUTED contacts every shard: its L3 is L3_HOP3 only.  The "ALL if levers X recover every missed gold" counts are ORACLE UPPER
BOUNDS (perfect recovery at no exposure cost), not mechanisms.

Identities asserted (per cell and B_N): the split reproduces the v1 classes (L3_PUSH2 u L3_FAR2 == REACH_2, L3_X2 == CROWD_2X);
CROWD_2L == contacted & not V_q & dS 2 & an intermediate contacted & not served; CROWD_2X the same with no intermediate contacted;
the ALL counts and the v1 lever-set counts equal the v1 record.

Usage: python scratchpad/_l1d_kbres_summary.py <dataset> -> results/L1_DEV/kbres_<dataset>__v1__L3SPLIT.json (write-once)
"""
import decimal
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "L1_DEV")
V1 = ("L2", "LOCAL", "ROUTE", "L3")
SUB = ("L2", "LOCAL", "ROUTE", "L3_PUSH2", "L3_X2", "L3_FAR2", "L3_HOP3")
UPPER = (("L2",), ("LOCAL",), ("L2", "LOCAL"), ("L2", "LOCAL", "L3_PUSH2"), ("L2", "LOCAL", "ROUTE"),
         ("L2", "LOCAL", "ROUTE", "L3_PUSH2"), ("L2", "LOCAL", "ROUTE", "L3_PUSH2", "L3_X2"),
         ("L2", "LOCAL", "ROUTE", "L3_PUSH2", "L3_X2", "L3_FAR2"), SUB)
BN_SHOW = (500, 1000, 2000, 5000)
NG_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))
ACT = 200
KT = "tables (per cell, B_N, stratum: query level and gold level)"
KU = "ALL_if_levers_recover_every_missed_gold (oracle upper bound)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def share(k, n):
    if n == 0:
        return None
    return str((decimal.Decimal(int(k)) / decimal.Decimal(int(n))).quantize(decimal.Decimal("0.001"), rounding=decimal.ROUND_HALF_UP))


def main():
    ds = sys.argv[1]
    fj = os.path.join(OUT, "kbres_%s__v1.json" % ds)
    fo = os.path.join(OUT, "kbres_%s__v1__L3SPLIT.json" % ds)
    assert not os.path.exists(fo), "write-once: %s exists" % fo
    rec = json.load(open(fj, encoding="utf-8"))
    fz = os.path.join(OUT, rec["npz"]["path"])
    assert sha_file(fz) == rec["npz"]["sha256"], "kbres npz changed"
    hp = rec["code"]["harness"]
    assert sha_file(os.path.join(REPO, hp["path"])) == hp["sha256"], "_l1d_kbres.py changed"
    z = np.load(fz)
    CI = {str(c): i for i, c in enumerate(z["classes"])}
    cells = [str(c) for c in z["cells"]]
    assert [str(x) for x in z["levers"]] == list(V1)
    gptr = z["gptr"].astype(np.int64)
    nq, ng = len(gptr) - 1, int(gptr[-1])
    ngold = np.diff(gptr)
    rog = np.repeat(np.arange(nq), ngold)
    hops = z["HOPS"].astype(np.int64)
    ev = (z["pos_FLAT"] < ACT) | (z["G_LPOS"] >= 0)
    assert (z["G_LPOS"] == z["lpos__IR_L1"]).all()
    ds2 = z["G_DS"] == 2
    QS = {"all": np.ones(nq, bool)}
    for h in sorted(set(int(x) for x in hops)):
        QS["hop%d" % h] = hops == h
    for nm, lo, hi in NG_BUCKETS:
        QS["ng_" + nm] = (ngold >= lo) & (ngold <= hi)
    for h in sorted(set(int(x) for x in hops)):
        for nm, lo, hi in NG_BUCKETS:
            m = (hops == h) & (ngold >= lo) & (ngold <= hi)
            if m.any():
                QS["hop%d|ng_%s" % (h, nm)] = m

    def split(c, M):
        cl = z["CL__%s__%d" % (c, M)]
        served = (cl == CI["SERVED_EV"]) | (cl == CI["SERVED_FILL"])
        lv = np.full(ng, -1, np.int64)
        lv[cl == CI["CROWD_EV"]] = SUB.index("L2")
        lv[cl == CI["CROWD_2L"]] = SUB.index("LOCAL")
        lv[cl == CI["REACH_EV"]] = SUB.index("ROUTE")
        lv[cl == CI["CROWD_2X"]] = SUB.index("L3_X2")
        lv[(cl == CI["CROWD_3"]) | (cl == CI["REACH_3"])] = SUB.index("L3_HOP3")
        if c == "UNROUTED":
            for k in ("CROWD_2X", "REACH_EV", "REACH_2", "REACH_3"):
                assert not (cl == CI[k]).any()
            assert ((cl == CI["CROWD_2L"]) == (~ev & ds2 & ~served)).all()
        else:
            cont, mc = z["cont__" + c], z["mid_cont__" + c]
            r2 = cl == CI["REACH_2"]
            lv[r2 & mc] = SUB.index("L3_PUSH2")
            lv[r2 & ~mc] = SUB.index("L3_FAR2")
            assert (r2 == (~cont & ~ev & ds2)).all()
            assert ((cl == CI["CROWD_2L"]) == (cont & ~ev & ds2 & mc & ~served)).all()
            assert ((cl == CI["CROWD_2X"]) == (cont & ~ev & ds2 & ~mc & ~served)).all()
            assert ((cl == CI["REACH_EV"]) == (~cont & ev)).all()
        assert ((lv == -1) == served).all(), "a missed gold without a lever (%s %d)" % (c, M)
        return lv, served

    out = {"RECORD": "KBRES_L3SPLIT", "dataset": ds, "version": 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "STATUS": "DEVELOPMENT DIAGNOSTIC (descriptive; no rule selected; development rows only)", "definitions": __doc__,
           "source": {"json": "results/L1_DEV/kbres_%s__v1.json" % ds, "json_sha256": sha_file(fj),
                      "npz": "results/L1_DEV/" + rec["npz"]["path"], "npz_sha256": rec["npz"]["sha256"]},
           "n_rows": nq, "n_gold_nodes": ng, "sublevers": list(SUB), "tables": {}}
    for c in ["UNROUTED"] + cells:
        out["tables"][c] = {}
        for M in BN_SHOW:
            lv, served = split(c, M)
            qa = np.add.reduceat(served.astype(np.int64), gptr[:-1]) == ngold
            bits = np.zeros(ng, np.int64)
            bits[lv >= 0] = np.left_shift(1, lv[lv >= 0])
            qb = np.bitwise_or.reduceat(bits, gptr[:-1])
            assert ((qb == 0) == qa).all()
            # the v1 lever sets are the sub-lever sets with the four L3 parts merged
            l3 = sum(1 << SUB.index(k) for k in SUB if k.startswith("L3_"))
            v1b = (qb & 0b111) | np.where((qb & l3) != 0, 0b1000, 0)
            e = {}
            for sn, qm in QS.items():
                gm = qm[rog]
                v1q = rec["diagnostics"][KT][c][str(M)][sn]["queries"]
                assert int(qa[qm].sum()) == v1q["ALL"], (c, M, sn)
                v1sets = {}
                for b in np.unique(v1b[qm & ~qa]):
                    v1sets["+".join(V1[i] for i in range(4) if int(b) >> i & 1)] = int((qm & ~qa & (v1b == b)).sum())
                assert v1sets == v1q["lever_sets_of_the_not_ALL_rows"], (c, M, sn)
                up = {}
                for X in UPPER:
                    xb = sum(1 << SUB.index(k) for k in X)
                    k = int((qm & ((qb & ~xb) == 0)).sum())
                    up["+".join(X)] = {"n": k, "share": share(k, qm.sum())}
                sets = {}
                for b in np.unique(qb[qm & ~qa]):
                    sets["+".join(SUB[i] for i in range(len(SUB)) if int(b) >> i & 1)] = int((qm & ~qa & (qb == b)).sum())
                e[sn] = {"n": int(qm.sum()), "ALL": int(qa[qm].sum()), "ALL_share": share(qa[qm].sum(), qm.sum()),
                         "golds": int(gm.sum()), "golds_served": int((served & gm).sum()),
                         "missed_golds_by_sublever": {L: int(((lv == i) & gm).sum()) for i, L in enumerate(SUB)},
                         "not_ALL_rows_by_sublever_set": dict(sorted(sets.items(), key=lambda kv: (-kv[1], kv[0]))),
                         KU: up}
            out["tables"][c][str(M)] = e
    out["code"] = {"path": "scratchpad/_l1d_kbres_summary.py", "sha256": sha_file(os.path.abspath(__file__))}
    out["guards"] = {"held_out_rows_read": 0, "split_B_rows_read": 0, "TEST_rows_read": 0, "data_written": False,
                     "corpus_read": False}
    with open(fo, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s sha256 %s" % (fo, sha_file(fo)))
    for c in ["UNROUTED"] + cells:
        for sn in ("all", "hop2", "hop3", "ng_5-10", "ng_11+"):
            e = out["tables"][c]["1000"][sn]
            print("%-9s B_N 1000 %-8s n %4d ALL %4d | missed %s" % (c, sn, e["n"], e["ALL"], e["missed_golds_by_sublever"]))
            print("%-9s          upper %s" % ("", {k: v["n"] for k, v in e[KU].items()}))


if __name__ == "__main__":
    main()
