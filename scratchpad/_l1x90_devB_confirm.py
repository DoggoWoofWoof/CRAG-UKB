"""ONE-SHOT DEV_B confirmation of the preregistered configuration (PREREGISTRATION_DEV_B.json must exist and
its module hashes must match the live modules).  Writes CONFIRMATION_DEV_B.json.  Run once."""
import datetime
import json
import os
import sys

import numpy as np

import _l1x90_core as X
import _l1x90_universal as UN

prereg_path = os.path.join(X.OUT, "PREREGISTRATION_DEV_B.json")
out_path = os.path.join(X.OUT, "CONFIRMATION_DEV_B.json")
if os.path.exists(out_path):
    sys.exit("CONFIRMATION_DEV_B.json already exists; the confirmation is one-shot")
with open(prereg_path, encoding="utf-8") as f:
    prereg = json.load(f)
here = os.path.dirname(os.path.abspath(__file__))
for m, h in prereg["configuration_frozen"]["modules_sha256"].items():
    live = X.sha_file(os.path.join(here, m))
    if live != h:
        sys.exit("module %s changed since preregistration (%s != %s)" % (m, live[:12], h[:12]))
res = {"record": "L1_P90_EXPLOIT_CONFIRMATION_DEV_B", "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "preregistration_sha256": X.sha_file(prereg_path), "results": {}}
for name in ("metaqa", "metaqa_phg", "squad", "musique"):
    C = X.Cache(name)
    B = C.split == "B"
    if prereg["caches"][name]["cache_sha256"] != X.sha_file(C.path):
        sys.exit("cache %s changed since preregistration" % name)
    T_rank = C.base_rank.astype(np.int64)
    base_sel = X.fuse(C, T_rank, T_rank, "T")
    r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
    entry = {"n_B": int(B.sum()), "BASE_B": r0["ALL_split"]["B"], "BASE_B_ANY": float(base_any[B].mean())}
    for gate in ("G2", "G1"):
        sel, mode, info = UN.universal_select(C, gate, cache_tag=name)
        r, allv, anyv = X.evaluate(C, sel, gate)
        g, l, p = X.mcnemar(base_all[B], allv[B])
        entry[gate] = {"B": r["ALL_split"]["B"], "B_ANY": float(anyv[B].mean()), "gained": g, "lost": l, "p_mcnemar": p,
                       "identical_to_BASE": bool(np.array_equal(sel, base_sel)), "info": info,
                       "typed_rate_B": float(mode[B].mean()), "scope_nodes": r["scope_nodes"]}
    res["results"][name] = entry
    a = entry["G2"]["B"]
    hop = "".join(" h%d %.3f" % (h, a["hop%d" % h]["ALL"]) for h in (1, 2, 3) if ("hop%d" % h) in a)
    print("%-11s DEV_B n=%d  BASE ALL %.4f | G2 ALL %.4f (+%d/-%d p=%.3g)%s | G1 ALL %.4f | identical_to_BASE=%s" % (
        name, entry["n_B"], entry["BASE_B"]["ALL"], a["ALL"], entry["G2"]["gained"], entry["G2"]["lost"], entry["G2"]["p_mcnemar"], hop,
        entry["G1"]["B"]["ALL"], entry["G2"]["identical_to_BASE"]), flush=True)
m = res["results"]["metaqa"]["G2"]
ok = m["B"]["ALL"] >= 0.90 and m["p_mcnemar"] < 0.01 and m["gained"] > m["lost"]
res["label"] = "P90_TYPED_PATH_CONFIRMED" if ok else "P90_TYPED_PATH_NOT_CONFIRMED"
res["decision_rule_applied"] = prereg["decision_rule"]["primary"]
X.wj(out_path, res)
print("LABEL:", res["label"], "->", out_path)
