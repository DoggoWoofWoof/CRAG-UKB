"""Render the MuSiQue hop-stratified diagnostic table from results/L2/kg_relsig_musique_clean.json.

Rows: dense, offset, edge_raw(relation_raw), path_raw, oracle[old], oracle[+edge], oracle[+edge+path]
Cols: overall, 2-hop, 3-hop, 4-hop   (hop = #gold docs = intrinsic chain depth)
Plus per-hop deltas: d_edge = +edge - old ; d_path = (+edge+path) - (+edge).
Prediction under test: d_path(4hop) > d_path(3hop) > d_path(2hop)  (need not be perfectly monotonic).
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/L2/kg_relsig_musique_clean.json"
d = json.load(open(path))
HOPS = ["2hop", "3hop", "4hop"]
psh = d.get("per_signal_R@5_by_hop", {})
och = d.get("oracle_R@5_by_hop", {})
ps = d.get("per_signal_R@5", {})
oc = d.get("oracle_R@5", {})


def cell(v):
    return "  -  " if v is None else f"{v:5.1f}"


def prow(label, overall, byhop):
    print(f"  {label:<26} {cell(overall):>6} | " + " | ".join(cell(byhop.get(h)) for h in HOPS))


print("=" * 68)
print(f"MuSiQue hop-stratified R@5   (n_by_hop={d.get('n_by_hop')})")
print("=" * 68)
print(f"  {'signal / oracle':<26} {'overall':>6} | " + " | ".join(f"{h:>5}" for h in HOPS))
print("  " + "-" * 60)
# per-signal rows
for s, lab in [("dense", "dense"), ("offset", "offset"),
               ("relation_raw", "edge_raw (connect-sent)"), ("path_raw", "path_raw (k-hop)")]:
    prow(lab, ps.get(s), {h: psh.get(h, {}).get(s) for h in HOPS})
print("  " + "-" * 60)
# oracle rows
for name, lab in [("old", "oracle: old(5 sig)"), ("+relation", "oracle: +edge"),
                  ("+relation+path", "oracle: +edge+path")]:
    prow(lab, oc.get(name), {h: och.get(h, {}).get(name) for h in HOPS})
print("  " + "-" * 60)
# deltas by hop
print("  DELTAS by hop:")
for h in HOPS:
    old = och.get(h, {}).get("old"); ed = och.get(h, {}).get("+relation"); pa = och.get(h, {}).get("+relation+path")
    de = (ed - old) if (old is not None and ed is not None) else None
    dp = (pa - ed) if (ed is not None and pa is not None) else None
    print(f"    {h}:  d_edge={cell(de)}   d_path={cell(dp)}")
print("=" * 68)
# prediction check
dps = {}
for h in HOPS:
    ed = och.get(h, {}).get("+relation"); pa = och.get(h, {}).get("+relation+path")
    dps[h] = (pa - ed) if (ed is not None and pa is not None) else None
vals = [dps[h] for h in HOPS if dps[h] is not None]
if len(vals) >= 2:
    mono = all(dps["4hop"] is not None and dps["2hop"] is not None and dps["4hop"] >= dps["2hop"] for _ in [0])
    print(f"Prediction d_path(4)>=d_path(2): {dps.get('4hop')} vs {dps.get('2hop')} -> "
          f"{'HOLDS' if (dps.get('4hop') is not None and dps.get('2hop') is not None and dps['4hop'] >= dps['2hop']) else 'does NOT hold'}")
print("reranker_R@5:", d.get("reranker_R@5", {}).get("none"), "->",
      d.get("reranker_R@5", {}).get("+relation"), "->", d.get("reranker_R@5", {}).get("+relation+path"))
