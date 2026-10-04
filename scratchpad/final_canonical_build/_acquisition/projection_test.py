"""Projection diagnostic -- how destructive is name-keying?  Run on WEBQSP_NSM_MID ONLY.

    python scratchpad/final_canonical_build/_acquisition/projection_test.py

The +2.41% triple gap between the NSM and RoG CWQ unions mixes two causes: RoG's name collapse AND
genuine extraction differences (RoG re-ran extraction with an unpublished seed list and a
max_ent=2000 PPR cap).  Those are not separable from two counts of two different graphs.

This test removes the extraction confound entirely by applying the MID->name projection to the SAME
graph and measuring it before and after.  Whatever shrinks is collapse, by construction.

HARD RULE (user, 2026-09-06): the result is evidence about how destructive name-keying is.  It is
NOT a recipe.  Nothing here modifies V1 identities, and V1 does not read this file.

Projection rule models what RoG did: an endpoint with a known name becomes that name; an endpoint
without one stays itself (RoG left unresolved MIDs visible).  So uncovered MIDs never merge, and
every merge measured here is a real name collision.

Everything is split by m. / g. / other because entities_names.json contains ZERO g. entries -- without
the split, "did not resolve" would silently mix "absent from this resource" with a real failure.
"""
import json, os, re, time
from collections import Counter

import numpy as np

DST = "data/final_canonical/webqsp/_acquisition/nsm/extracted"
NAMES = "data/final_canonical/webqsp/_acquisition/nsm/entities_names.json"
OUT = "data/final_canonical/webqsp/NSM_PROJECTION_DIAGNOSTIC.json"
DS = {"webqsp": f"{DST}/webqsp/webqsp", "CWQ": f"{DST}/CWQ/CWQ"}
SPLITS = ("train", "dev", "test")

EP_BITS, REL_BITS = 23, 14
EP_MASK, REL_MASK = (1 << EP_BITS) - 1, (1 << REL_BITS) - 1
CHUNK_Q, CONSOLIDATE_EVERY = 2000, 12

MID_RE = re.compile(r"^[mg]\.")


def klass(s):
    if s.startswith("m."):
        return "m"
    if s.startswith("g."):
        return "g"
    return "other"


def build_union():
    """Rebuild the NSM MID union, returning (codes, ep_list, rel_list)."""
    gep, grel, ep_list, rel_list, maps = {}, {}, [], [], {}
    for ds, d in DS.items():
        with open(f"{d}/entities.txt", encoding="utf-8") as fh:
            ents = [ln.rstrip("\n") for ln in fh]
        with open(f"{d}/relations.txt", encoding="utf-8") as fh:
            rels = [ln.rstrip("\n") for ln in fh]
        e_map = np.empty(len(ents), dtype=np.int64)
        for i, s in enumerate(ents):
            g = gep.get(s)
            if g is None:
                g = gep[s] = len(ep_list)
                ep_list.append(s)
            e_map[i] = g
        r_map = np.empty(len(rels), dtype=np.int64)
        for i, s in enumerate(rels):
            g = grel.get(s)
            if g is None:
                g = grel[s] = len(rel_list)
                rel_list.append(s)
            r_map[i] = g
        maps[ds] = (e_map, r_map)

    parts = []

    def consolidate():
        if len(parts) > 1:
            u = np.unique(np.concatenate(parts))
            parts.clear()
            parts.append(u)

    for ds, d in DS.items():
        e_map, r_map = maps[ds]
        nq = 0
        for sp in SPLITS:
            p = f"{d}/{sp}_simple.json"
            if not os.path.exists(p):
                continue
            buf = []
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    tp = (json.loads(line).get("subgraph") or {}).get("tuples") or []
                    nq += 1
                    if tp:
                        buf.append(np.asarray(tp, dtype=np.int64))
                    if nq % CHUNK_Q == 0 and buf:
                        a = np.concatenate(buf)
                        buf.clear()
                        parts.append(np.unique((e_map[a[:, 0]] << (REL_BITS + EP_BITS))
                                               | (r_map[a[:, 1]] << EP_BITS) | e_map[a[:, 2]]))
                        if len(parts) >= CONSOLIDATE_EVERY:
                            consolidate()
            if buf:
                a = np.concatenate(buf)
                parts.append(np.unique((e_map[a[:, 0]] << (REL_BITS + EP_BITS))
                                       | (r_map[a[:, 1]] << EP_BITS) | e_map[a[:, 2]]))
            consolidate()
        print(f"[union] {ds} q={nq:,}", flush=True)
    return np.unique(np.concatenate(parts)), ep_list, rel_list


def main():
    t0 = time.time()
    codes, ep_list, rel_list = build_union()
    heads = (codes >> (REL_BITS + EP_BITS)) & EP_MASK
    rels = (codes >> EP_BITS) & REL_MASK
    tails = codes & EP_MASK
    used = np.union1d(heads, tails)                       # endpoints actually referenced by a triple
    print(f"[union] triples={codes.size:,} used_endpoints={used.size:,} t={time.time()-t0:.0f}s",
          flush=True)

    names = json.load(open(NAMES, encoding="utf-8"))
    cls_of = np.zeros(len(ep_list), dtype=np.int8)        # 0=m, 1=g, 2=other
    CODE = {"m": 0, "g": 1, "other": 2}
    proj_id = np.full(len(ep_list), -1, dtype=np.int64)
    surf, surf_list = {}, []
    covered = np.zeros(len(ep_list), dtype=bool)

    for g in used:
        s = ep_list[g]
        cls_of[g] = CODE[klass(s)]
        nm = names.get(s)
        if nm is not None:
            covered[g] = True
        p = nm if nm is not None else s                   # uncovered endpoints stay themselves
        pid = surf.get(p)
        if pid is None:
            pid = surf[p] = len(surf_list)
            surf_list.append(p)
        proj_id[g] = pid

    # collision groups: projected surfaces reached by >=2 distinct source endpoints
    grp = Counter(proj_id[used].tolist())
    coll = {p: c for p, c in grp.items() if c > 1}
    max_grp = max(coll.values()) if coll else 0
    biggest = sorted(coll.items(), key=lambda kv: -kv[1])[:8]

    ph, pt = proj_id[heads], proj_id[tails]
    after = np.unique(((ph.astype(np.int64) << (REL_BITS + EP_BITS))
                       | (rels << EP_BITS) | pt.astype(np.int64)))
    ep_after = int(np.union1d(ph, pt).size)
    ep_before = int(used.size)
    tri_before, tri_after = int(codes.size), int(after.size)

    per_class = {}
    for name, c in CODE.items():
        sel = used[cls_of[used] == c]
        cov = int(covered[sel].sum())
        # collapse attributable to this class: endpoints of this class minus distinct surfaces they reach
        distinct_surf = int(np.unique(proj_id[sel]).size) if sel.size else 0
        per_class[name] = {
            "MID_ENDPOINTS_BEFORE": int(sel.size),
            "NAME_COVERED_MID_N": cov,
            "NAME_UNCOVERED_MID_N": int(sel.size) - cov,
            "COVERAGE_PCT": round(100.0 * cov / sel.size, 4) if sel.size else None,
            "DISTINCT_SURFACES_REACHED": distinct_surf,
            "ENDPOINT_SHRINK_N": int(sel.size) - distinct_surf,
            "ENDPOINT_SHRINK_PCT": round(100.0 * (int(sel.size) - distinct_surf) / sel.size, 4)
                                   if sel.size else None,
        }

    cov_total = int(covered[used].sum())
    doc = {
        "schema": "NSM_PROJECTION_DIAGNOSTIC/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graph": "WEBQSP_NSM_MID (sibling extraction; NOT the RoG corpus, NOT a V1 input)",
        "naming_resource": {"file": "entities_names.json", "entries": len(names),
                            "sha256": "93ce094acccb06b1906db8a4534175481efe55bb9fa8ed711fedd1f8b37d2f98"},
        "projection_rule": "endpoint with a known name -> that name; endpoint without one -> itself. "
                           "Models RoG leaving unresolved MIDs visible, so every merge measured is a "
                           "real name collision and never an artefact of dropping nodes.",
        "HARD_RULE": "Evidence about how destructive name-keying is. NOT a recipe. V1 identities are "
                     "not modified and V1 does not read this file.",
        "totals": {
            "MID_ENDPOINTS_BEFORE": ep_before,
            "NAME_COVERED_MID_N": cov_total,
            "NAME_UNCOVERED_MID_N": ep_before - cov_total,
            "COVERAGE_PCT": round(100.0 * cov_total / ep_before, 4),
            # distinct NAME strings reached by covered endpoints -- not the same as the post-projection
            # surface count, which also contains every uncovered endpoint passed through unchanged
            "DISTINCT_NAMES": len({names[ep_list[g]] for g in used.tolist() if covered[g]}),
            "COLLISION_GROUP_N": len(coll),
            "MIDS_IN_COLLISION_GROUPS": int(sum(coll.values())),
            "MAX_COLLISION_GROUP": max_grp,
            "ENDPOINTS_AFTER_PROJECTION": ep_after,
            "ENDPOINT_SHRINK_N": ep_before - ep_after,
            "ENDPOINT_SHRINK_PCT": round(100.0 * (ep_before - ep_after) / ep_before, 4),
            "TRIPLES_BEFORE": tri_before,
            "TRIPLES_AFTER_PROJECTION": tri_after,
            "TRIPLE_COLLAPSE_N": tri_before - tri_after,
            "TRIPLE_COLLAPSE_PCT": round(100.0 * (tri_before - tri_after) / tri_before, 4),
        },
        "by_endpoint_class": per_class,
        "class_note": "entities_names.json contains zero g. entries, so g. coverage is 0 BY "
                      "CONSTRUCTION. That is a property of the resource, not evidence that g. "
                      "objects lack Freebase names.",
        "largest_collision_groups": [{"surface": surf_list[p], "endpoints_merged": c}
                                     for p, c in biggest],
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({"totals": doc["totals"], "by_endpoint_class": doc["by_endpoint_class"],
                      "largest_collision_groups": doc["largest_collision_groups"],
                      "elapsed_s": doc["elapsed_s"]}, indent=1))


if __name__ == "__main__":
    main()
