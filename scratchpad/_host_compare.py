"""Host housekeeping (laptop side, READ-ONLY): compare the host manifest with the laptop tree.  For every host file: on the laptop at the same relative path with the same size (and sha when the manifest has one)?
Prints per-group byte totals: SAFE (identical copy on the laptop), HOSTONLY (not on the laptop), DIFF (same path, different size/sha).   python scratchpad/_host_compare.py"""
import collections
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(REPO, "work", "HOST_HOUSEKEEPING", "manifest.jsonl")
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


rows = [json.loads(x) for x in open(MAN, encoding="utf-8")]
grp = collections.defaultdict(lambda: collections.Counter())
cnt = collections.defaultdict(lambda: collections.Counter())
hostonly = []
diff = []
for r in rows:
    key = r["root"] + ":" + "/".join(r["rel"].split("/")[:DEPTH])
    if r["root"] != "ws":
        st = "OTHERROOT"
    else:
        lp = os.path.join(REPO, r["rel"])
        if not os.path.exists(lp):
            st = "HOSTONLY"
            hostonly.append(r)
        else:
            ls = os.path.getsize(lp)
            if ls != r["size"]:
                st = "DIFF"
                diff.append((r, ls))
            elif "sha" in r and sha(lp) != r["sha"]:
                st = "DIFF"
                diff.append((r, ls))
            else:
                st = "SAFE"
    grp[key][st] += r["size"]
    cnt[key][st] += 1
tot = collections.Counter()
print("%-62s %9s %9s %9s %9s" % ("group", "SAFE GB", "HOSTONLY", "DIFF", "OTHER"))
for k in sorted(grp, key=lambda k: -sum(grp[k].values())):
    s = sum(grp[k].values())
    if s < 0.2e9:
        for st, v in grp[k].items():
            tot[st] += v
        continue
    print("%-62s %9.2f %9.2f %9.2f %9.2f   (%d files)" % (k, grp[k]["SAFE"] / 1e9, grp[k]["HOSTONLY"] / 1e9, grp[k]["DIFF"] / 1e9, grp[k]["OTHERROOT"] / 1e9, sum(cnt[k].values())))
    for st, v in grp[k].items():
        tot[st] += v
print("small groups (<0.2 GB) folded into totals")
print("TOTAL  SAFE %.2f  HOSTONLY %.2f  DIFF %.2f  OTHERROOT %.2f GB" % tuple(tot[x] / 1e9 for x in ("SAFE", "HOSTONLY", "DIFF", "OTHERROOT")))
json.dump({"hostonly": hostonly, "diff": [(r, ls) for r, ls in diff]}, open(os.path.join(REPO, "work", "HOST_HOUSEKEEPING", "compare.json"), "w"))
print("\nlargest HOSTONLY files:")
for r in sorted(hostonly, key=lambda r: -r["size"])[:25]:
    print("  %8.3f GB  %s" % (r["size"] / 1e9, r["rel"]))
print("\nDIFF files (largest 15):")
for r, ls in sorted(diff, key=lambda t: -t[0]["size"])[:15]:
    print("  host %.3f GB laptop %.3f GB  %s" % (r["size"] / 1e9, ls / 1e9, r["rel"]))
print("\nnon-ws roots:")
for k in sorted(grp):
    if not k.startswith("ws:"):
        print("  %-60s %8.2f GB %d files" % (k, sum(grp[k].values()) / 1e9, sum(cnt[k].values())))
nl = collections.Counter(r["nlink"] for r in rows if r["size"] > 1 << 30)
print("\nnlink of files > 1 GB:", dict(nl))
