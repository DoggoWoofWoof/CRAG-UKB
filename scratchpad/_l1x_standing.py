"""L1X -- six-dataset STANDING of the node-level L1 (descriptive; reads only stored records; nothing is chosen, nothing is scored, no gold label is read here).

ALL-gold at node budgets B_N = 100 / 250 / 500 / 1000 / 2000 / 5000, from exact counts:
  hotpotqa, 2wiki  results/L1_X/txtscore_<ds>__dev_v1.{json,npz}   (2,000 train-row dev population; unrouted served order O and routed K500)
  musique, squad, metaqa  results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json  (unrouted_n; the shipped IR_L1)
  metaqa typed candidate v2 / webqsp typed candidate v1  results/L1_X/L1_TYPED_SELECT_CANDIDATE__v{2,1}.json  (half B only; DEV candidate, adoption = a contract ruling)

  python -u scratchpad/_l1x_standing.py            -> prints the table
"""
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X = os.path.join(REPO, "results", "L1_X")
MC = [100, 250, 500, 1000, 2000, 5000]


def fmt(k, n):
    return "%.3f (%d/%d)" % (k / n, k, n)


def txt(ds):
    j = json.load(open(os.path.join(X, "txtscore_%s__dev_v1.json" % ds), encoding="utf-8"))
    z = np.load(os.path.join(X, "txtscore_%s__dev_v1.npz" % ds))
    n = int(j["n_rows"])
    nb = len(MC)
    unr = np.unpackbits(z["UNR"], axis=1)[:, :nb].sum(0)
    flt = np.unpackbits(z["FLT"], axis=1)[:, :nb].sum(0)
    r500 = np.unpackbits(z["ALL__500"], axis=1)[:, :nb].sum(0)
    return n, [int(v) for v in flt], [int(v) for v in unr], [int(v) for v in r500]


def main():
    out = {}
    print("%-22s %s" % ("dataset / system", "  ".join("%-14d" % b for b in MC)))
    for ds in ("hotpotqa", "2wiki"):
        n, flt, unr, r500 = txt(ds)
        out[ds] = {"n": n, "FLAT": flt, "unrouted_O": unr, "routed_K500": r500}
        for nm, v in (("FLAT only", flt), ("unrouted O", unr), ("routed K500", r500)):
            print("%-22s %s" % ("%s %s" % (ds, nm), "  ".join("%-14s" % fmt(k, n) for k in v)))
    cf = json.load(open(os.path.join(REPO, "results", "L1_DEV", "L1_ROUTING_CARRY_FORWARD__v1.json"), encoding="utf-8"))["evidence"]
    for ds in ("squad", "musique", "metaqa"):
        n = int(cf[ds]["n_rows"])
        v = [int(cf[ds]["unrouted_n"][str(b)]) for b in MC]
        out[ds] = {"n": n, "unrouted_O": v}
        print("%-22s %s" % ("%s unrouted O" % ds, "  ".join("%-14s" % fmt(k, n) for k in v)))
    c2 = json.load(open(os.path.join(X, "L1_TYPED_SELECT_CANDIDATE__v2.json"), encoding="utf-8"))
    pk = c2["evidence_half_B_metaqa"]["per_K"]["432"]
    nB = c2["evidence_half_B_metaqa"]["n_half_B"]
    print("metaqa typed v2 K432 half B (n=%d) shipped / v1 / v2:" % nB)
    for k in ("shipped_ALL_B", "candidate_v1_ALL_B", "candidate_v2_ALL_B"):
        print("   %-22s %s" % (k, "  ".join("%-14.4f" % r[k] for r in pk)))
    out["metaqa_typed_v2_halfB"] = {k: [r[k] for r in pk] for k in ("shipped_ALL_B", "candidate_v1_ALL_B", "candidate_v2_ALL_B")}
    c1 = json.load(open(os.path.join(X, "L1_TYPED_SELECT_CANDIDATE__v1.json"), encoding="utf-8"))
    for k, v in c1.items():
        if "webqsp" in k.lower():
            print("v1 key", k, str(v)[:600])
    print(json.dumps(out)[:200])


if __name__ == "__main__":
    main()
