"""One-shot DEV_B confirmation of the pre-registered L1_STATIC candidate (results/L1_STATIC/PREREGISTRATION_DEV_B.json).
Refuses to run twice; verifies the code and cache hashes recorded in the pre-registration; aborts unless the recorded
DEV_A numbers are reproduced exactly; then reads DEV_B once, applies the pre-stated rule, and records everything."""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

import _l1s_core as S
import _l1s_candidate as CAND
import _l1x90_core as X

OUT = S.OUT
PRE = os.path.join(OUT, "PREREGISTRATION_DEV_B.json")
CONF = os.path.join(OUT, "DEV_B_CONFIRMATION.json")
if os.path.exists(CONF):
    sys.exit("REFUSED: %s exists (one-shot confirmation; supersede by ruling, never re-run)" % CONF)
pre = json.load(io.open(PRE, encoding="utf-8"))
HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for fn, h in pre["code_sha256"].items():
    cur = sha(os.path.join(HERE, fn))
    if cur != h:
        sys.exit("ABORT: %s changed since pre-registration (%s != %s)" % (fn, cur[:12], h[:12]))
for name, c in pre["populations"]["caches"].items():
    cur = sha(X.CACHES[name])
    if cur != c["sha256"]:
        sys.exit("ABORT: cache %s changed since pre-registration" % name)
pre_sha = sha(PRE)
S.log("pre-registration %s verified (code + caches); reading DEV_B once" % pre_sha[:12])
rec = {"record": "DEV_B_CONFIRMATION", "lane": "L1_STATIC", "preregistration_sha256": pre_sha,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "caches": {}, "decision": {}}
for name in ("metaqa", "squad", "musique"):
    D = S.Data(name, dense_fp32=False)
    ranks, T = CAND.all_arm_ranks(D)
    entry = {"n_A": int(D.A.sum()), "n_B": int(D.B.sum()), "n_full": int(D.nq),
             "table_span_mean": {k: round(float(np.diff(v[0]).mean()), 3) for k, v in T.items()},
             "BASE": {"A": D.r_base["ALL_split"]["A"]["ALL"], "B": D.r_base["ALL_split"]["B"]["ALL"], "full": D.r_base["ALL_split"]["ALL"]["ALL"],
                      "per_hop_B": {k: v["ALL"] for k, v in D.r_base["ALL_split"]["B"].items() if k.startswith("hop")}},
             "arms": {}}
    for arm, rank in ranks.items():
        out, allv = D.eval_rank(rank, arm, quiet=True, splits=("A", "B", "full"))
        want = pre["recorded_DEV_A"][name][arm]
        got = out["A"]
        if not (abs(got["ALL"] - want["ALL"]) < 1e-9 and got["gained"] == want["gained"] and got["lost"] == want["lost"]):
            sys.exit("ABORT: DEV_A not reproduced for %s %s: got %s, pre-registered %s" % (name, arm, got, want))
        entry["arms"][arm] = {"DEV_A_reproduced": True, "A": got, "B": out["B"], "full": out["full"], "scope_nodes": out["scope_nodes"]}
        b = out["B"]
        hops = " ".join("%s %.3f" % (k[3:], v) for k, v in sorted(b["per_hop"].items()))
        S.log("  %-8s %-9s DEV_B n=%d BASE %.4f -> %.4f (+%d/-%d p=%.1e) | hops %s || full %.4f (+%d/-%d p=%.1e)" % (
            name, arm, entry["n_B"], entry["BASE"]["B"], b["ALL"], b["gained"], b["lost"], b["p"], hops,
            out["full"]["ALL"], out["full"]["gained"], out["full"]["lost"], out["full"]["p"]))
    rec["caches"][name] = entry
    del D, ranks, T


def verdict(arm):
    losses = []
    gains = []
    for name, e in rec["caches"].items():
        b = e["arms"][arm]["B"]
        if b["lost"] > b["gained"] and b["p"] < 0.05:
            losses.append(name)
        if b["gained"] > b["lost"] and b["p"] < 0.01:
            gains.append(name)
    return {"significant_losses": losses, "significant_gains": gains, "PASS": (not losses) and bool(gains)}


rec["decision"]["PRIMARY"] = verdict("PRIMARY")
rec["decision"]["SECONDARY"] = verdict("SECONDARY")
rec["decision"]["rule"] = pre["decision_rule"]["primary_passes_iff"]
if rec["decision"]["PRIMARY"]["PASS"]:
    rec["decision"]["outcome"] = "PRIMARY_CONFIRMED"
elif rec["decision"]["SECONDARY"]["PASS"]:
    rec["decision"]["outcome"] = "PRIMARY_FAILED_SECONDARY_CONFIRMED"
else:
    rec["decision"]["outcome"] = "NOT_CONFIRMED"
S.wj(CONF, rec)
S.log("outcome %s  primary %s  secondary %s  -> %s (sha256 %s)" % (
    rec["decision"]["outcome"], rec["decision"]["PRIMARY"], rec["decision"]["SECONDARY"], os.path.relpath(CONF, S.REPO), sha(CONF)[:12]))
