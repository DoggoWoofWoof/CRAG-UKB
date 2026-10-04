"""One-shot DEV_B confirmation of the pre-registered L1_GEOM candidate (results/L1_GEOM/PREREGISTRATION_DEV_B.json).
Refuses to run twice; verifies the code and cache hashes recorded in the pre-registration; aborts unless the recorded
DEV_A numbers are reproduced exactly; then reads DEV_B once, applies the pre-stated rule (no significant loss on any of
the five populations at p < 0.05; a significant gain on at least one at p < 0.005), and records everything."""
import hashlib
import io
import json
import os
import sys
import time

import _l1g_candidate as CAND
import _l1g_core as G
import _l1x90_core as X

OUT = G.OUT
PRE = os.path.join(OUT, "PREREGISTRATION_DEV_B.json")
CONF = os.path.join(OUT, "DEV_B_CONFIRMATION.json")
if os.path.exists(CONF):
    sys.exit("REFUSED: %s exists (one-shot confirmation; supersede by ruling, never re-run)" % CONF)
pre = json.load(io.open(PRE, encoding="utf-8"))
HERE = os.path.dirname(os.path.abspath(__file__))
P_LOSS, P_GAIN = 0.05, 0.005


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
G.log("pre-registration %s verified (code + caches); reading DEV_B once" % pre_sha[:12])
rec = {"record": "DEV_B_CONFIRMATION", "lane": "L1_GEOM", "preregistration_sha256": pre_sha,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "thresholds": {"loss_p": P_LOSS, "gain_p": P_GAIN},
       "DEV_B_reuse_disclosure": pre["populations"]["DEV_B_reuse_disclosure"], "caches": {}, "decision": {}}
for name in pre["populations"]["caches"]:
    D = G.Data(name, dense_fp32=True)
    ranks, ch, gdiag = CAND.candidate_ranks(D)
    entry = {"partition": G.PARTITION_OF[name], "n_A": int(D.A.sum()), "n_B": int(D.B.sum()), "n_full": int(D.nq),
             "BASE": {"A": D.r_base["ALL_split"]["A"]["ALL"], "B": D.r_base["ALL_split"]["B"]["ALL"], "full": D.r_base["ALL_split"]["ALL"]["ALL"],
                      "per_hop_B": {k: v["ALL"] for k, v in D.r_base["ALL_split"]["B"].items() if k.startswith("hop")}},
             "arms": {}}
    for arm, key in CAND.ARMS.items():
        out, allv = D.eval_rank(ranks[key], key, quiet=True, splits=("A", "B", "full"))
        want = pre["recorded_DEV_A"][name][arm]
        got = out["A"]
        if not (abs(got["ALL"] - want["ALL"]) < 1e-9 and got["gained"] == want["gained"] and got["lost"] == want["lost"]):
            sys.exit("ABORT: DEV_A not reproduced for %s %s: got %s, pre-registered %s" % (name, arm, got, want))
        entry["arms"][arm] = {"key": key, "DEV_A_reproduced": True, "A": got, "B": out["B"], "full": out["full"], "scope_nodes": out["scope_nodes"]}
        b = out["B"]
        hops = " ".join("%s %.3f" % (k[3:], v) for k, v in sorted(b["per_hop"].items()))
        G.log("  %-10s %-9s %-6s DEV_B n=%d BASE %.4f -> %.4f (+%d/-%d p=%.1e) | hops %s || full %.4f (+%d/-%d p=%.1e)" % (
            name, arm, key, entry["n_B"], entry["BASE"]["B"], b["ALL"], b["gained"], b["lost"], b["p"], hops,
            out["full"]["ALL"], out["full"]["gained"], out["full"]["lost"], out["full"]["p"]))
    rec["caches"][name] = entry
    del D, ranks, ch


def verdict(arm):
    losses, gains = [], []
    for name, e in rec["caches"].items():
        b = e["arms"][arm]["B"]
        if b["lost"] > b["gained"] and b["p"] < P_LOSS:
            losses.append(name)
        if b["gained"] > b["lost"] and b["p"] < P_GAIN:
            gains.append(name)
    return {"significant_losses": losses, "significant_gains": gains, "PASS": (not losses) and bool(gains)}


for arm in CAND.ARMS:
    rec["decision"][arm] = verdict(arm)
rec["decision"]["rule"] = pre["decision_rule"]["primary_passes_iff"]
rec["decision"]["decided_on"] = ["PRIMARY", "SECONDARY"]
if rec["decision"]["PRIMARY"]["PASS"]:
    rec["decision"]["outcome"] = "PRIMARY_CONFIRMED"
elif rec["decision"]["SECONDARY"]["PASS"]:
    rec["decision"]["outcome"] = "PRIMARY_FAILED_SECONDARY_CONFIRMED"
else:
    rec["decision"]["outcome"] = "NOT_CONFIRMED"
G.S.wj(CONF, rec)
G.log("outcome %s  primary %s  secondary %s  -> %s (sha256 %s)" % (
    rec["decision"]["outcome"], rec["decision"]["PRIMARY"], rec["decision"]["SECONDARY"], os.path.relpath(CONF, X.REPO), sha(CONF)[:12]))
