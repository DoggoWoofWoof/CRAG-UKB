"""Validate sharded NER == established build_ner_edges on squad (orientation-agnostic edge set + weights)."""
import sys
EST = "data/canonical/squad/graph_ner.tsv"
NEW = "scratchpad/ner_validate/canonical/squad/graph_ner.tsv"

def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        a, b, w = ln.rstrip("\n").split("\t")
        d[frozenset((a, b))] = float(w)
    return d

est, new = load(EST), load(NEW)
ek, nk = set(est), set(new)
only_est = ek - nk; only_new = nk - ek
common = ek & nk
wdiffs = [(k, est[k], new[k]) for k in common if abs(est[k] - new[k]) > 2e-6]
print(f"established edges : {len(est)}")
print(f"sharded edges     : {len(new)}")
print(f"edges only in EST : {len(only_est)}")
print(f"edges only in NEW : {len(only_new)}")
print(f"common edges      : {len(common)}")
print(f"weight diffs >2e-6: {len(wdiffs)}")
if only_est: print("  sample only_est:", [tuple(sorted(x)) for x in list(only_est)[:3]])
if only_new: print("  sample only_new:", [tuple(sorted(x)) for x in list(only_new)[:3]])
if wdiffs: print("  sample wdiff:", wdiffs[:3])
ok = (len(only_est) == 0 and len(only_new) == 0 and len(wdiffs) == 0)
print("VALIDATION:", "PASS — sharded == established (freeze methodology)" if ok else "FAIL — investigate")
sys.exit(0 if ok else 1)
