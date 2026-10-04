"""PHASE 4/5 pass 2 -- resolve the out-degree-zero bare MIDs via RELATION RANGE.

Pass 1 classified bare MIDs from the Freebase types they assert as a HEAD.  636,098 bare MIDs
have no outgoing edge at all in the released union (they are leaves of the 2-hop extraction),
so pass 1 could say nothing about them.

Pass 2 uses the other observable: the RANGE of the relation that points AT them.  For every
relation R we measure `obj_named_rate(R)` = fraction of R's distinct tail endpoints that RoG
rendered as a human-readable name.  `people.person.spouse_s` ranges over marriage CVTs and sits
at ~0; `film.film.directed_by` ranges over people and sits at ~1.

  all in-relations have obj_named_rate <= TAU_LOW   -> NAMELESS_MEDIATOR (range is a mediator type)
  any in-relation has obj_named_rate >= TAU_HIGH    -> NAME_RESOLVABLE (its range-peers ARE named
                                                      here, so Freebase metadata would probably
                                                      name it too -- this is the population that
                                                      would justify a dump)
  otherwise                                         -> MIXED / UNKNOWN

The classifier is VALIDATED against pass 1: every bare MID that pass 1 could label from its own
outgoing types is re-labelled here from its incoming relations only, and the agreement is reported.

Reads only _audit/ temp artifacts; writes only _audit/.
"""
import collections
import json
import os
import sys
import time

import numpy as np

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit")
OUT = os.path.join(A, "phase45_mid_pass2_inrelation.json")

ENT_BITS, REL_BITS = 23, 14
REL_SHIFT, HEAD_SHIFT = ENT_BITS, ENT_BITS + REL_BITS
ENT_MASK, REL_MASK = (1 << ENT_BITS) - 1, (1 << REL_BITS) - 1
TAU_LOW, TAU_HIGH = 0.02, 0.50

LAB = {0: "NOT_A_BARE_MID", 1: "NAMELESS_MEDIATOR", 2: "NAME_RESOLVABLE",
       3: "MIXED", 4: "UNKNOWN_no_outgoing_edges"}


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    with open(os.path.join(A, "union_entities.jsonl"), encoding="utf-8") as fh:
        ents = [json.loads(l) for l in fh]
    u_e = np.load(os.path.join(A, "union_entity_ids.npy"))
    with open(os.path.join(A, "union_relations.txt"), encoding="utf-8") as fh:
        rels = [l.rstrip("\n") for l in fh]
    with open(os.path.join(A, "union_entity_class.txt"), encoding="utf-8") as fh:
        cls = [l.rstrip("\n") for l in fh]
    lab1 = np.load(os.path.join(A, "union_mid_label.npy"))
    packed = np.load(os.path.join(A, "union_packed_triples.npy"))
    N, NR = len(ents), len(rels)

    h_v = (packed >> HEAD_SHIFT) & ENT_MASK
    r_v = (packed >> REL_SHIFT) & REL_MASK
    t_v = packed & ENT_MASK
    u_r = np.unique(r_v)
    h = np.searchsorted(u_e, h_v).astype(np.int32)
    t = np.searchsorted(u_e, t_v).astype(np.int32)
    r = np.searchsorted(u_r, r_v).astype(np.int32)
    del h_v, r_v, t_v, packed
    log(f"[load] N={N:,} E={r.size:,} R={NR:,}  {time.time()-t0:.0f}s")

    is_mid = np.fromiter((c == "FREEBASE_MID" for c in cls), bool, N)
    is_named = np.fromiter((c == "HUMAN_READABLE_SURFACE" for c in cls), bool, N)

    # ---- obj_named_rate(R) over DISTINCT (relation, tail) pairs --------------
    rt_pairs = np.unique(r.astype(np.int64) * N + t)
    p_rel = (rt_pairs // N).astype(np.int32)
    p_tail = (rt_pairs % N).astype(np.int32)
    tot_r = np.bincount(p_rel, minlength=NR).astype(np.int64)
    nam_r = np.bincount(p_rel, weights=is_named[p_tail], minlength=NR).astype(np.int64)
    obj_rate = np.divide(nam_r, np.maximum(tot_r, 1), dtype=np.float64)
    log(f"[range] distinct (rel,tail) pairs = {rt_pairs.size:,}")

    bins = [0.0, 0.001, 0.01, 0.02, 0.05, 0.25, 0.5, 0.9, 0.99, 1.0001]
    hist = [{"bin": f"[{a},{b})", "n_relations": int(((obj_rate >= a) & (obj_rate < b)).sum()),
             "n_rel_tail_pairs": int(tot_r[(obj_rate >= a) & (obj_rate < b)].sum())}
            for a, b in zip(bins[:-1], bins[1:])]

    med_rel = obj_rate <= TAU_LOW
    ord_rel = obj_rate >= TAU_HIGH

    # ---- per-node aggregation over distinct in-relations ---------------------
    order = np.argsort(p_tail, kind="stable")
    tn, tr = p_tail[order], p_rel[order]
    st = np.searchsorted(tn, np.arange(N), side="left")
    en = np.searchsorted(tn, np.arange(N), side="right")
    indeg_rels = en - st
    has_in = indeg_rels > 0

    cm = np.concatenate(([0], np.cumsum(med_rel[tr])))
    co = np.concatenate(([0], np.cumsum(ord_rel[tr])))
    nmed = cm[en] - cm[st]
    nord = co[en] - co[st]
    all_med_in = np.zeros(N, bool)
    any_ord_in = np.zeros(N, bool)
    all_med_in[has_in] = nmed[has_in] == indeg_rels[has_in]
    any_ord_in[has_in] = nord[has_in] > 0

    def label_from_in(idx):
        lb = np.full(idx.size, 4, dtype=np.int8)
        hi = has_in[idx]
        am, ao = all_med_in[idx], any_ord_in[idx]
        lb[hi & am] = 1
        lb[hi & ~am & ao] = 2
        lb[hi & ~am & ~ao] = 3
        return lb

    # ---- VALIDATION: re-label pass-1-labelled MIDs from in-relations only ----
    val_idx = np.flatnonzero(is_mid & ((lab1 == 1) | (lab1 == 2)))
    lb2 = label_from_in(val_idx)
    l1 = lab1[val_idx]
    conf = collections.Counter(zip(l1.tolist(), lb2.tolist()))
    both = (lb2 != 4)
    agree = int(((l1 == lb2) & both).sum())
    comparable = int(both.sum())
    validation = {
        "pass1_labelled_bare_mids_compared": int(val_idx.size),
        "with_an_in_relation_signal": comparable,
        "agreement": agree,
        "agreement_pct": round(100.0 * agree / max(comparable, 1), 2),
        "confusion_pass1_x_pass2": {f"{LAB[a]} -> {LAB[b]}": c
                                    for (a, b), c in sorted(conf.items(), key=lambda kv: -kv[1])},
    }
    log("[validate] " + json.dumps({k: validation[k] for k in
                                    ("pass1_labelled_bare_mids_compared",
                                     "with_an_in_relation_signal", "agreement_pct")}))

    # ---- apply to the out-degree-zero population ----------------------------
    unk = np.flatnonzero(is_mid & (lab1 == 4))
    lbu = label_from_in(unk)
    cnt = collections.Counter(lbu.tolist())
    resolved = {LAB[k]: int(v) for k, v in sorted(cnt.items())}

    # what relations bring in the NAME_RESOLVABLE leaves? (this decides the dump)
    nr_nodes = unk[lbu == 2]
    nr_set = np.zeros(N, bool)
    nr_set[nr_nodes] = True
    m = nr_set[p_tail] & ord_rel[p_rel]
    top_rel = collections.Counter(p_rel[m].tolist()).most_common(20)
    nr_examples = [ents[i] for i in nr_nodes[:10].tolist()]

    # ---- MERGED final table --------------------------------------------------
    final = lab1.copy()
    final[unk] = lbu
    bare = np.flatnonzero(is_mid)
    fc = collections.Counter(final[bare].tolist())
    TOTAL = int(bare.size)
    MERGED = {
        "TOTAL_BARE_MIDS": TOTAL,
        "NAMELESS_MEDIATOR_MIDS": int(fc.get(1, 0)),
        "NAME_RESOLVABLE_MIDS": int(fc.get(2, 0)),
        "MIXED_MIDS": int(fc.get(3, 0)),
        "UNKNOWN_MIDS": int(fc.get(4, 0)),
    }
    MERGED["pct"] = {k: round(100.0 * v / TOTAL, 2)
                     for k, v in MERGED.items() if k != "TOTAL_BARE_MIDS"}
    np.save(os.path.join(A, "union_mid_label_final.npy"), final)

    # ---- ordinary-topic vs CVT population for the whole union ---------------
    POP = {
        "ORDINARY_TOPIC_named_by_RoG": int(is_named.sum()),
        "ORDINARY_TOPIC_bare_mid_name_resolvable": MERGED["NAME_RESOLVABLE_MIDS"],
        "CVT_MEDIATOR": MERGED["NAMELESS_MEDIATOR_MIDS"],
        "MIXED": MERGED["MIXED_MIDS"],
        "UNDECIDABLE_from_graph_alone": MERGED["UNKNOWN_MIDS"],
        "LITERAL_OTHER": int(N - is_named.sum() - is_mid.sum()),
        "TOTAL": N,
    }

    res = {
        "_what": "Phase 4/5 pass 2 -- relation-RANGE classification of the out-degree-zero bare MIDs, "
                 "plus the merged union-wide MID classification table.",
        "_method": "obj_named_rate(R) = fraction of R's distinct tail endpoints RoG rendered as a "
                   "human-readable name. Query-independent; derived from the released triples only.",
        "_thresholds": {"TAU_LOW": TAU_LOW, "TAU_HIGH": TAU_HIGH},
        "relation_range_named_rate_histogram": hist,
        "VALIDATION_pass2_against_pass1": validation,
        "out_degree_zero_population_resolved": resolved,
        "name_resolvable_leaf_top_incoming_relations":
            [{"relation": rels[i], "n_leaves": c, "obj_named_rate": round(float(obj_rate[i]), 4)}
             for i, c in top_rel],
        "name_resolvable_leaf_examples": nr_examples,
        "MERGED_MID_CLASSIFICATION": MERGED,
        "UNION_POPULATION_TABLE": POP,
        "_elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log(json.dumps(MERGED, indent=2))
    log(json.dumps(POP, indent=2))
    log("WROTE " + OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
