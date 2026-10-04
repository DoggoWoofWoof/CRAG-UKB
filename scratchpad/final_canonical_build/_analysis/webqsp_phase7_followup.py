"""Phase 7 follow-ups: (a) the exact Phase-C node-id rule, (b) why the legacy titles only
match a re-run of deduce() 89% of the time.

(b) is the interesting one.  loader_webqsp.py builds its adjacency by iterating a Python SET of
triples, and deduce() returns "the first non-MID neighbour in adjacency order".  Python set
iteration order over str-containing tuples depends on PYTHONHASHSEED, which is randomised per
process, so the *arbitrary* neighbour chosen is not even stable across runs.  Test: for every MID
node whose stored title differs from this run's deduce(), is the STORED title nonetheless SOME
named neighbour of that MID?  If yes, the bug is confirmed AND shown to be non-deterministic.
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = "C:/Users/Swastik/Desktop/CRAG"
A = f"{ROOT}/scratchpad/final_canonical_build/_audit"
MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")


def main():
    out = {}

    # ---- (a) Phase-C node id rule ------------------------------------------
    import hashlib
    rows = []
    with open(f"{ROOT}/data/canonical/webqsp/documents.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rows.append(json.loads(line))
            if i >= 4: break
    out["phaseC_sample_keys"] = sorted(rows[0].keys())
    out["phaseC_sample"] = {k: (str(v)[:80]) for k, v in rows[0].items()}
    cand = {}
    for r in rows:
        did = r.get("id") or r.get("doc_id") or r.get("node_id")
        for fld in ("text", "original_source_id", "source_id", "title"):
            v = r.get(fld)
            if not isinstance(v, str):
                continue
            h = hashlib.sha256(v.encode("utf-8")).hexdigest()
            for n in (16, 24, 32):
                for pre in ("webqsp_ent_", "webqsp:", ""):
                    if did == pre + h[:n]:
                        cand[f"{pre}sha256({fld})[:{n}]"] = cand.get(f"{pre}sha256({fld})[:{n}]", 0) + 1
    out["phaseC_id_rule_matches_on_5_rows"] = cand

    # ---- (b) legacy fabricated-title non-determinism ------------------------
    import pyarrow.parquet as pq
    triples = set()
    for p in (f"{ROOT}/data/raw/full/kb/webqsp_test0.parquet",
              f"{ROOT}/data/raw/full/kb/webqsp_test1.parquet"):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=200, columns=["graph"]):
            for g in b.column("graph").to_pylist():
                for t in g:
                    triples.add((str(t[0]), str(t[1]), str(t[2])))
    adj = defaultdict(list)
    ents = set()
    for h, rel, tl in triples:
        adj[h].append((rel, tl)); adj[tl].append((rel, h)); ents.add(h); ents.add(tl)
    entities = sorted(ents)
    is_mid = lambda x: bool(MID_RE.match(x))

    named_nb = {}          # MID -> set of its NAMED neighbours (order-independent)
    for e in entities:
        if is_mid(e):
            named_nb[e] = {o for _, o in adj.get(e, []) if not is_mid(o)}

    with open(f"{ROOT}/data/processed/master_nodes_webqsp.json", encoding="utf-8") as fh:
        legacy = json.load(fh)
    if isinstance(legacy, dict):
        legacy = legacy.get("nodes", legacy.get("data", list(legacy.values())))
    stored = {}
    for nd in legacy:
        nid = nd.get("node_id") or nd.get("id")
        stored[nid] = (nd.get("metadata") or {}).get("title", "")
    del legacy

    n_mid = n_match_thisrun = n_stored_is_some_named_neighbour = 0
    n_stored_not_a_neighbour = 0
    n_blank = 0
    examples = []
    for i, e in enumerate(entities):
        if not is_mid(e):
            continue
        n_mid += 1
        st = stored.get(f"webqsp_doc_{i}")
        nb = named_nb.get(e, set())
        this_run = next((o for _, o in adj.get(e, []) if not is_mid(o)), None)
        if st == (this_run or ""):
            n_match_thisrun += 1
        if not nb:
            n_blank += 1
            continue
        if st in nb:
            n_stored_is_some_named_neighbour += 1
        else:
            n_stored_not_a_neighbour += 1
            if len(examples) < 6:
                examples.append({"mid": e, "stored_title": st,
                                 "named_neighbours": sorted(nb)[:5], "n_named_neighbours": len(nb)})

    out["legacy_fabrication"] = {
        "MID_nodes": n_mid,
        "MID_with_no_named_neighbour": n_blank,
        "stored_title_equals_THIS_RUN_deduce": n_match_thisrun,
        "stored_title_is_SOME_named_neighbour_of_the_MID": n_stored_is_some_named_neighbour,
        "stored_title_is_NOT_any_named_neighbour": n_stored_not_a_neighbour,
        "pct_stored_title_is_a_named_neighbour": round(
            100.0 * n_stored_is_some_named_neighbour / max(n_mid - n_blank, 1), 4),
        "BUG_VERIFIED": bool(n_stored_not_a_neighbour == 0 and n_stored_is_some_named_neighbour > 0),
        "NON_DETERMINISTIC": bool(n_match_thisrun < n_stored_is_some_named_neighbour),
        "_reading": ("Every fabricated title is SOME named neighbour of the MID (the bug), but only "
                     f"{n_match_thisrun:,} of {n_stored_is_some_named_neighbour:,} match the neighbour "
                     "THIS run picks: loader_webqsp.py iterates a Python set, so which neighbour's "
                     "name a mediator node steals is not reproducible across processes."),
        "counterexamples_if_any": examples,
    }
    with open(f"{A}/phase7_followup.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
