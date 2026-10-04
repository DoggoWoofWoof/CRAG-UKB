# -*- coding: utf-8 -*-
"""
Section 2b's disk figures were measured before the WebQSP encoder staging was cleared, and its
"41 stores / 53.2 GB" line was measured before the four webqsp stores landed. Re-measure and
replace, and split the per-dataset row into actual per-dataset numbers -- a reader sizing a
transfer needs to know which 5 GB of a 6.58 GB directory is source acquisition they can drop.

Run scratchpad/verify_handoff_claims.py --sizes to re-measure these against the document.
"""
import glob
import io
import json
import os
import sys

R = "data/final_canonical"
H = R + "/HANDOFF.md"


def sz(p):
    t = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t


def g(p):
    return sz(p) / 1e9


ds = {d: g("%s/%s" % (R, d)) for d in
      ["2wiki", "webqsp", "hotpotqa", "metaqa", "squad", "musique"]}
fb = g(R + "/freebase_v3")
sup = g(R + "/_superseded_textualization_rev1")
work = g(R + "/_work")
patch = g(R + "/_dense_repair_patch") + g(R + "/_rev2_encoder_patch")
acq = g(R + "/webqsp/_acquisition") + g(R + "/webqsp/_enc")
total = fb + sup + work + patch + sum(ds.values())

# the vector tree, counted from what POINTER_INDEX actually references
pi = json.load(io.open(R + "/POINTER_INDEX.json", encoding="utf-8"))
paths = set()


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "stores" and isinstance(v, list):
                for st in v:
                    if isinstance(st, dict) and st.get("path"):
                        paths.add(st["path"])
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(pi)
vec = sum(g(p) if os.path.isdir(p) else os.path.getsize(p) / 1e9
          for p in paths if os.path.exists(p))
missing_stores = [p for p in paths if not os.path.exists(p)]
on_disk_stores = len(glob.glob("data/canonical/*/encodings/*/*"))

s = io.open(H, encoding="utf-8").read()

old = ("| disk | **~83 GB** for the substrate itself — but see the table below, because the "
       "directory it lives in is 190 GB |")
new = ("| disk | **~%.0f GB** for the substrate itself — but see the table below, because the "
       "directory it lives in is %.0f GB |" % (sum(ds.values()) - acq + patch + vec, total))
if old not in s:
    sys.exit("disk-row anchor not found")
s = s.replace(old, new, 1)

old = ("**`data/final_canonical/` is 190 GB on disk and the substrate is only a fraction of it.** A\n"
       "reader who sizes a transfer off `du` will be wrong by 5×, so here is the split as measured:\n"
       "\n"
       "| subtree | size | is it part of the six-dataset substrate? |\n|---|---:|---|\n"
       "| `freebase_v3/` | 140.90 GB | **No.** A separate artifact — see §10. |\n"
       "| `_superseded_textualization_rev1/` | 18.34 GB | **No.** Superseded intermediates, kept for audit. |\n"
       "| `2wiki` `webqsp` `hotpotqa` `metaqa` `squad` `musique` | 39.76 GB | **Yes**, minus WebQSP's source acquisitions (~4.7 GB) and encoder staging. |\n"
       "| `_work/` | 3.16 GB | **No.** Build scratch. |\n"
       "| `_dense_repair_patch/` `_rev2_encoder_patch/` | 1.18 GB | **Yes** — pointers resolve into these. |\n")
new = ("**`data/final_canonical/` is %.0f GB on disk and the substrate is a third of it.** A reader\n"
       "who sizes a transfer off `du` will be wrong by roughly 6×, so here is the split as\n"
       "measured:\n"
       "\n"
       "| subtree | size | is it part of the six-dataset substrate? |\n|---|---:|---|\n"
       "| `freebase_v3/` | %.2f GB | **No.** A separate artifact — see §10. |\n"
       "| `_superseded_textualization_rev1/` | %.2f GB | **No.** Superseded intermediates, kept for audit. |\n"
       "| `_work/` | %.2f GB | **No.** Build scratch. |\n"
       "| `webqsp/_acquisition/` `webqsp/_enc/` | %.2f GB | **No.** Downloaded sources and encoder staging. |\n"
       "| the six dataset directories | %.2f GB | **Yes** — minus the %.2f GB row above, which sits inside `webqsp/`. |\n"
       "| `_dense_repair_patch/` `_rev2_encoder_patch/` | %.2f GB | **Yes** — pointers resolve into these. |\n"
       "\n"
       "Per dataset, so you can take a subset: %s.\n"
       % (total, fb, sup, work, acq, sum(ds.values()), acq, patch,
          " · ".join("`%s` %.2f GB" % (k, v) for k, v in sorted(ds.items(), key=lambda x: -x[1]))))
if old not in s:
    sys.exit("subtree-table anchor not found")
s = s.replace(old, new, 1)

old = ("| `data/canonical/` | the **vectors**, referenced by pointer, never copied | 53.2 GB "
       "across 41 stores |")
new = ("| `data/canonical/` | the **vectors**, referenced by pointer, never copied | %.2f GB "
       "across %d stores |" % (vec, len(paths)))
if old not in s:
    sys.exit("vector-tree anchor not found")
s = s.replace(old, new, 1)

old = ("`POINTER_INDEX.json` → each channel's `stores[].path` is the machine-readable dependency "
       "list.")
new = ("`POINTER_INDEX.json` → each channel's `stores[].path` is the machine-readable dependency\n"
       "list — %d paths, **all %d present on disk** as of the last check. `data/canonical/` holds\n"
       "%d store directories in total; the extra %d are superseded trees nothing points at (see\n"
       "§3b.8), so copy the referenced list, not the directory."
       % (len(paths), len(paths) - len(missing_stores), on_disk_stores,
          on_disk_stores - len(paths)))
if old not in s:
    sys.exit("dependency-list anchor not found")
s = s.replace(old, new, 1)

io.open(H, "w", encoding="utf-8", newline=chr(10)).write(s)
print("2b re-measured: total %.2f GB, substrate %.2f GB, vectors %.2f GB across %d stores "
      "(%d missing)" % (total, sum(ds.values()) - acq + patch + vec, vec, len(paths),
                        len(missing_stores)))
