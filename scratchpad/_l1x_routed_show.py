"""print the routed table (half B and half A) of results/L1_X/routed_<ds>__<tag>.json, deltas vs SHIPPED, objective = mean ALL over B_N 100..1000"""
import json, sys, os
import numpy as np
ds, tag = sys.argv[1], sys.argv[2]
stem = sys.argv[3] if len(sys.argv) > 3 else "routed"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "L1_X")
j = json.load(open(os.path.join(R, "%s_%s__%s.json" % (stem, ds, tag)), encoding="utf-8"))
print("mean B_P", j["mean_B_P"], "pool share in contacted", j["pool_share_in_contacted"])
for c in j["cells"]:
    t = j["table"][c]
    sh = t["SHIPPED"]
    print("== %s   (B_N 100/250/500/1000; objB = mean; dB vs shipped)" % c)
    for nm, e in t.items():
        oA, oB = np.mean(e["A"][:4]), np.mean(e["B"][:4])
        dA, dB = oA - np.mean(sh["A"][:4]), oB - np.mean(sh["B"][:4])
        print("  %-26s objA %.4f (%+.4f) | objB %.4f (%+.4f) | B %s" % (nm, oA, dA, oB, dB, " ".join("%.3f" % v for v in e["B"][:4])))
