"""Assemble results/L1_GEOM/REPORT.md = narrative (this file) + tables generated from the records (_l1g_tables.py output)
+ the confirmation section generated from DEV_B_CONFIRMATION.json (if present)."""
import io
import json
import os
import subprocess
import sys

import _l1g_core as G

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
tab = subprocess.run([sys.executable, "-u", os.path.join(HERE, "_l1g_tables.py")], capture_output=True, text=True, env=env, encoding="utf-8").stdout
tab = "\n".join(l for l in tab.splitlines() if "nq=" not in l)
blocks = [b.strip() for b in tab.split("\n\n") if b.strip()]
T = {}
for b in blocks:
    head = b.splitlines()[0]
    key = ("factorial" if head.startswith("| arm |") else "headroom" if head.startswith("| DEV_A |") else "orientation" if "gold node in top-100" in head
           else "newevidence" if "new evidence" in head else "partition" if "paired DEV_A" in head else "controls" if "directional controls" in head
           else "combos" if "combinations" in head else head)
    T[key] = b
assert all(k in T for k in ("factorial", "headroom", "orientation", "newevidence", "partition", "controls", "combos")), list(T)

CONF = os.path.join(G.OUT, "DEV_B_CONFIRMATION.json")
PRE = os.path.join(G.OUT, "PREREGISTRATION_DEV_B.json")
conf_md = "_(pending — the pre-registered one-shot DEV_B run has not completed)_"
if os.path.exists(CONF):
    c = json.load(io.open(CONF, encoding="utf-8"))
    p = json.load(io.open(PRE, encoding="utf-8"))
    lab = {"metaqa": "metaqa MtK", "metaqa_phg": "metaqa PHG", "squad": "squad MtK", "squad_phg": "squad PHG", "musique": "musique PHG"}
    arms = list(p["candidate"]["arms"].keys())
    lines = ["| DEV_B (held out; second look, gain threshold p < 0.005) | " + " | ".join(lab[n] for n in c["caches"]) + " |", "|---|" + "---|" * len(c["caches"])]
    lines.append("| n | " + " | ".join(str(e["n_B"]) for e in c["caches"].values()) + " |")
    lines.append("| BASE | " + " | ".join("%.4f" % e["BASE"]["B"] for e in c["caches"].values()) + " |")
    for arm in arms:
        key = p["candidate"]["arms"][arm]
        row = []
        for e in c["caches"].values():
            b = e["arms"][arm]["B"]
            star = "**" if b["p"] < 0.05 else ""
            row.append("%s%.4f%s (+%d/−%d p=%.0e)" % (star, b["ALL"], star, b["gained"], b["lost"], b["p"]))
        lines.append("| %s `%s` | %s |" % (arm, key, " | ".join(row)))
    lines.append("| | | | | | |")
    lines.append("| full dev (A ∪ B), for the record | " + " | ".join(lab[n] for n in c["caches"]) + " |")
    lines.append("| BASE | " + " | ".join("%.4f" % e["BASE"]["full"] for e in c["caches"].values()) + " |")
    for arm in arms:
        row = []
        for e in c["caches"].values():
            f = e["arms"][arm]["full"]
            row.append("%.4f (+%d/−%d p=%.0e)" % (f["ALL"], f["gained"], f["lost"], f["p"]))
        lines.append("| %s | %s |" % (arm, " | ".join(row)))
    hop = []
    for n, e in c["caches"].items():
        if e["BASE"]["per_hop_B"]:
            hop.append("%s BASE %s → PRIMARY %s" % (lab[n], " / ".join("%s %.3f" % (k[3:], v) for k, v in sorted(e["BASE"]["per_hop_B"].items())),
                                                    " / ".join("%s %.3f" % (k[3:], v) for k, v in sorted(e["arms"]["PRIMARY"]["B"]["per_hop"].items()))))
    d = c["decision"]
    conf_md = "\n".join(lines) + "\n\nPer hop on DEV_B (metaqa): " + "; ".join(hop) + "\n\n" + \
        "Decision (rule pre-registered in `PREREGISTRATION_DEV_B.json`, sha256 %s…): **%s** — PRIMARY significant losses %s, significant gains %s; SECONDARY losses %s, gains %s. Record `DEV_B_CONFIRMATION.json` (%s)." % (
            c["preregistration_sha256"][:12], d["outcome"], d["PRIMARY"]["significant_losses"] or "none", d["PRIMARY"]["significant_gains"] or "none",
            d["SECONDARY"]["significant_losses"] or "none", d["SECONDARY"]["significant_gains"] or "none", c["utc"])
narr = io.open(os.path.join(HERE, "_l1g_report_narrative.md"), encoding="utf-8").read()
for k, v in T.items():
    narr = narr.replace("{{%s}}" % k, v)
narr = narr.replace("{{confirmation}}", conf_md)
assert "{{" not in narr, [l for l in narr.splitlines() if "{{" in l]
out = os.path.join(G.OUT, "REPORT.md")
io.open(out, "w", encoding="utf-8", newline="\n").write(narr)
print("wrote", out, len(narr), "chars")
