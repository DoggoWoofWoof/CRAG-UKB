"""Read-only: the (router x served rule) maximin over MetaQA (routed2_metaqa__v2, alpha .75) and WebQSP (rrtroute_webqsp__<tag>) -- half A selects, half B reports.
Objective per cell = mean ALL over B_N 100/250/500/1000; gate cells MetaQA K 100/250/432/500/1000, WebQSP K 100/250/500; delta vs (ES, SHIPPED).
  python _l1x_rrt_route_combine.py <wq tag> [--tol=0.003]"""
import json
import os
import sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "L1_X")
ROUT = ["ES", "ES_a", "ES_b", "ES_c"]
RULE = ["SHIPPED", "Sro+Sri+Tin", "S_out+Sro+Sri+Tin"]
MQ = ["100", "250", "432", "500", "1000"]
WQ = ["100", "250", "500"]


def main():
    tag = sys.argv[1]
    tol = next((float(x[6:]) for x in sys.argv if x.startswith("--tol=")), 0.003)
    m = json.load(open(os.path.join(OUT, "routed2_metaqa__v2.json"), encoding="utf-8"))["table"]
    w = json.load(open(os.path.join(OUT, "rrtroute_webqsp__%s.json" % tag), encoding="utf-8"))["table"]

    def mq(rv, rule, half):
        return float(np.mean([np.mean(m[("PHG_k%s" % K) if rv == "ES" else ("PHG_k%s@%s" % (K, rv))][rule][half][:4]) for K in MQ]))

    def wq(rv, rule, half):
        return float(np.mean([np.mean(w["%s|%s" % (K, rv)][rule][half][:4]) for K in WQ]))
    base = {h: (mq("ES", "SHIPPED", h), wq("ES", "SHIPPED", h)) for h in "AB"}
    print("baseline (ES, SHIPPED): A metaqa %.4f webqsp %.4f | B metaqa %.4f webqsp %.4f" % (*base["A"], *base["B"]))
    rows = []
    for rv in ROUT:
        for ru in RULE:
            dA = (mq(rv, ru, "A") - base["A"][0], wq(rv, ru, "A") - base["A"][1])
            dB = (mq(rv, ru, "B") - base["B"][0], wq(rv, ru, "B") - base["B"][1])
            rows.append((rv, ru, dA, dB))
    rows.sort(key=lambda r: -min(r[2]))
    print("%-5s %-20s | dA metaqa  webqsp  min    | dB metaqa  webqsp" % ("router", "rule"))
    for rv, ru, dA, dB in rows:
        print("%-5s %-20s | %+.4f %+.4f %+.4f | %+.4f %+.4f" % (rv, ru, dA[0], dA[1], min(dA), dB[0], dB[1]))
    ok = [r for r in rows if min(r[2]) >= -tol]
    print("maximin by A (tol %.4f): %s + %s" % (tol, ok[0][0], ok[0][1]))


if __name__ == "__main__":
    main()
