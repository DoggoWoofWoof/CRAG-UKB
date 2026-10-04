"""PHASES 4/5 -- MID_CLASSIFICATION_REQUIRED, done from the released graph alone.

The acceptance gate demands TOTAL_BARE_MIDS / NAME_RESOLVABLE_MIDS / NAMELESS_MEDIATOR_MIDS /
UNKNOWN_MIDS and says: "VERIFY THAT QUANTITATIVELY; do not accept it as a blanket statement."

We have no Freebase dump.  But the released RoG graph is itself a *labelled* observation of
Freebase's `type.object.name` coverage: RoG projected each endpoint to its English name where
Freebase had one and left the bare MID where it did not.  So for every Freebase TYPE
(= the relation id minus its last component, e.g. `people.marriage` for `people.marriage.spouse`)
we can MEASURE, over the ~2.59M endpoints, what fraction of the nodes asserting that type were
given a name by RoG.  A mediator/CVT type should sit at ~0; an ordinary topic type at ~1.

That per-type named-rate is then used to classify each of the 1,652,618 bare MIDs:

  NAMELESS_MEDIATOR_MID  every outgoing type of the node is a near-zero-named-rate type
                         -> a Freebase dump would NOT give it a name either
  NAME_RESOLVABLE_MID    at least one outgoing type is a high-named-rate type, i.e. its own
                         type-peers ARE named in this same graph -> a dump WOULD likely name it
  MIXED_MID              neither
  UNKNOWN_MID            no outgoing edges at all -> nothing structural to go on

Everything here is derived from the released triple set.  Query-independent: no question text,
no answers, no gold paths, no topic annotations.  Reads only _audit/ temp artifacts; writes only
_audit/.  Nothing canonical is written.
"""
import collections
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_audit")
OUT = os.path.join(A, "phase45_mid_cvt_classification.json")

ENT_BITS, REL_BITS = 23, 14
REL_SHIFT, HEAD_SHIFT = ENT_BITS, ENT_BITS + REL_BITS
ENT_MASK, REL_MASK = (1 << ENT_BITS) - 1, (1 << REL_BITS) - 1

MID_RE = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")

TAU_LOW = 0.02      # <= this named-rate  => mediator-like type
TAU_HIGH = 0.50     # >= this named-rate  => ordinary named type


def log(*a):
    print(*a, flush=True)


def load_union():
    with open(os.path.join(A, "union_entities.jsonl"), encoding="utf-8") as fh:
        ents = [json.loads(l) for l in fh]
    u_e = np.load(os.path.join(A, "union_entity_ids.npy"))
    with open(os.path.join(A, "union_relations.txt"), encoding="utf-8") as fh:
        rels = [l.rstrip("\n") for l in fh]
    u_r_n = len(rels)
    with open(os.path.join(A, "union_entity_class.txt"), encoding="utf-8") as fh:
        cls = [l.rstrip("\n") for l in fh]
    packed = np.load(os.path.join(A, "union_packed_triples.npy"))
    assert len(ents) == u_e.size == len(cls), (len(ents), u_e.size, len(cls))
    return ents, u_e, rels, u_r_n, cls, packed


def main():
    t0 = time.time()
    ents, u_e, rels, n_rel, cls, packed = load_union()
    N, E = len(ents), packed.size
    log(f"[load] N={N:,} E={E:,} R={n_rel:,}  {time.time()-t0:.0f}s")

    # ---- decode packed triples into POSITIONS in the union arrays -------------
    h_v = ((packed >> HEAD_SHIFT) & ENT_MASK)
    r_v = ((packed >> REL_SHIFT) & REL_MASK)
    t_v = (packed & ENT_MASK)
    u_r = np.unique(r_v)
    assert u_r.size == n_rel, (u_r.size, n_rel)
    h = np.searchsorted(u_e, h_v).astype(np.int32)
    t = np.searchsorted(u_e, t_v).astype(np.int32)
    r = np.searchsorted(u_r, r_v).astype(np.int32)
    assert (u_e[h] == h_v).all() and (u_e[t] == t_v).all() and (u_r[r] == r_v).all()
    del h_v, t_v, r_v, packed
    log(f"[decode] ok  {time.time()-t0:.0f}s")

    # ---- Freebase TYPE = relation id minus its last component -----------------
    def type_of(rel):
        p = rel.rsplit(".", 1)
        return p[0] if len(p) == 2 else rel

    rel_type = [type_of(x) for x in rels]
    types = sorted(set(rel_type))
    tid = {x: i for i, x in enumerate(types)}
    rel2type = np.array([tid[x] for x in rel_type], dtype=np.int64)
    NT = len(types)
    log(f"[types] distinct Freebase types over {n_rel:,} relations = {NT:,}")

    is_mid = np.fromiter((cls[i] == "FREEBASE_MID" for i in range(N)), bool, N)
    is_named = np.fromiter((cls[i] == "HUMAN_READABLE_SURFACE" for i in range(N)), bool, N)
    is_other = ~(is_mid | is_named)
    log(f"[class] MID={is_mid.sum():,}  SURFACE={is_named.sum():,}  OTHER={is_other.sum():,}")

    # ---- distinct (node, out-type) and (node, in-type) pairs ------------------
    rt = rel2type[r]
    out_pairs = np.unique(h.astype(np.int64) * NT + rt)
    in_pairs = np.unique(t.astype(np.int64) * NT + rt)
    out_node, out_type = (out_pairs // NT).astype(np.int32), (out_pairs % NT).astype(np.int32)
    in_node, in_type = (in_pairs // NT).astype(np.int32), (in_pairs % NT).astype(np.int32)
    log(f"[pairs] out={out_pairs.size:,} in={in_pairs.size:,}  {time.time()-t0:.0f}s")

    # ---- per-type named rate over nodes ASSERTING that type (outgoing) --------
    tot_by_type = np.bincount(out_type, minlength=NT).astype(np.int64)
    named_by_type = np.bincount(out_type, weights=is_named[out_node], minlength=NT).astype(np.int64)
    mid_by_type = np.bincount(out_type, weights=is_mid[out_node], minlength=NT).astype(np.int64)
    rate = np.divide(named_by_type, np.maximum(tot_by_type, 1), dtype=np.float64)

    # bimodality evidence
    bins = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0001]
    hist_types, hist_nodes = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rate >= lo) & (rate < hi)
        hist_types.append(int(m.sum()))
        hist_nodes.append(int(tot_by_type[m].sum()))

    mediator_type = rate <= TAU_LOW
    ordinary_type = rate >= TAU_HIGH

    # ---- classify every bare MID ---------------------------------------------
    order = np.argsort(out_node, kind="stable")
    on, ot = out_node[order], out_type[order]
    starts = np.searchsorted(on, np.arange(N), side="left")
    ends = np.searchsorted(on, np.arange(N), side="right")
    outdeg_types = ends - starts

    has_out = outdeg_types > 0
    # per-node: does EVERY out-type look mediator-like / does ANY look ordinary?
    all_med = np.zeros(N, bool)
    any_ord = np.zeros(N, bool)
    med_flag = mediator_type[ot]
    ord_flag = ordinary_type[ot]
    cum_med = np.concatenate(([0], np.cumsum(med_flag)))
    cum_ord = np.concatenate(([0], np.cumsum(ord_flag)))
    nmed = cum_med[ends] - cum_med[starts]
    nord = cum_ord[ends] - cum_ord[starts]
    all_med[has_out] = (nmed[has_out] == outdeg_types[has_out])
    any_ord[has_out] = nord[has_out] > 0

    bare = np.flatnonzero(is_mid)
    TOTAL_BARE_MIDS = int(bare.size)
    b_has_out = has_out[bare]
    b_all_med = all_med[bare]
    b_any_ord = any_ord[bare]

    NAMELESS_MEDIATOR = int((b_has_out & b_all_med).sum())
    NAME_RESOLVABLE = int((b_has_out & (~b_all_med) & b_any_ord).sum())
    MIXED = int((b_has_out & (~b_all_med) & (~b_any_ord)).sum())
    UNKNOWN = int((~b_has_out).sum())
    assert NAMELESS_MEDIATOR + NAME_RESOLVABLE + MIXED + UNKNOWN == TOTAL_BARE_MIDS

    # sensitivity of the split to the thresholds
    sens = {}
    for tl in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        mt = rate <= tl
        mf = mt[ot]
        cm = np.concatenate(([0], np.cumsum(mf)))
        nm = cm[ends] - cm[starts]
        am = np.zeros(N, bool)
        am[has_out] = nm[has_out] == outdeg_types[has_out]
        sens[f"TAU_LOW={tl}"] = int((b_has_out & am[bare]).sum())

    # ---- how many named (ordinary-topic) nodes look mediator-like? (control) --
    named_idx = np.flatnonzero(is_named)
    ctrl = {
        "named_nodes_with_outgoing_edges": int(has_out[named_idx].sum()),
        "named_nodes_all_out_types_mediator_like": int((has_out[named_idx] & all_med[named_idx]).sum()),
        "_meaning": "false-positive rate of the mediator test on nodes KNOWN to have a Freebase name",
    }

    # ---- CVT / mediator TEXT recipe (Phase 5): deterministic, schema only -----
    # text = "CVT | types: <sorted distinct out-types> | relations: <sorted distinct out-relations>"
    cvt_nodes = bare[b_has_out & b_all_med]
    cvt_set = np.zeros(N, bool)
    cvt_set[cvt_nodes] = True

    rel_order = np.argsort(h, kind="stable")
    hs, rs = h[rel_order], r[rel_order]
    rstarts = np.searchsorted(hs, np.arange(N), side="left")
    rends = np.searchsorted(hs, np.arange(N), side="right")

    text_counter = collections.Counter()
    samples = []
    for k, nid in enumerate(cvt_nodes.tolist()):
        rr = np.unique(rs[rstarts[nid]:rends[nid]])
        rl = sorted(rels[i] for i in rr.tolist())
        tl_ = sorted({type_of(x) for x in rl})
        txt = "CVT | types: " + ", ".join(tl_) + " | relations: " + ", ".join(rl)
        text_counter[txt] += 1
        if len(samples) < 12 and k % 977 == 0:
            samples.append({"node_id": ents[nid], "text": txt})

    distinct_cvt_texts = len(text_counter)
    top_cvt = text_counter.most_common(15)

    # type-level headline table
    types_tbl = sorted(range(NT), key=lambda i: -tot_by_type[i])[:40]
    type_table = [{"type": types[i], "nodes_asserting": int(tot_by_type[i]),
                   "named": int(named_by_type[i]), "bare_mid": int(mid_by_type[i]),
                   "named_rate": round(float(rate[i]), 6)} for i in types_tbl]
    med_top = sorted(np.flatnonzero(mediator_type).tolist(), key=lambda i: -tot_by_type[i])[:25]
    ord_top = sorted(np.flatnonzero(ordinary_type).tolist(), key=lambda i: -tot_by_type[i])[:25]

    # ---- ordinary-topic / CVT population split for Phase 3+4 reporting -------
    res = {
        "_what": "Phase 4/5 MID + CVT classification, derived ONLY from the released RoG union "
                 "(WebQSP u CWQ, 2,592,894 endpoints / 8,309,195 triples). No Freebase dump used.",
        "_method": "Freebase TYPE = relation id minus last component. Per type, measure the fraction "
                   "of endpoints asserting it (as head) that RoG rendered as a human-readable name. "
                   "RoG names an endpoint iff Freebase had type.object.name for it, so this rate is a "
                   "direct empirical estimate of name coverage per Freebase type.",
        "_query_independent": True,
        "_thresholds": {"TAU_LOW_mediator": TAU_LOW, "TAU_HIGH_ordinary": TAU_HIGH},
        "graph": {"N": N, "E": E, "R": n_rel, "distinct_freebase_types": NT},
        "endpoint_classes": {"FREEBASE_MID": int(is_mid.sum()),
                             "HUMAN_READABLE_SURFACE": int(is_named.sum()),
                             "OTHER": int(is_other.sum())},
        "per_type_named_rate_histogram": {
            "bins": [f"[{a},{b})" for a, b in zip(bins[:-1], bins[1:])],
            "n_types": hist_types, "n_node_type_assertions": hist_nodes,
            "_reading": "strong bimodality = the mediator/ordinary split is real, not an artefact "
                        "of the threshold",
        },
        "MID_CLASSIFICATION": {
            "TOTAL_BARE_MIDS": TOTAL_BARE_MIDS,
            "NAMELESS_MEDIATOR_MIDS": NAMELESS_MEDIATOR,
            "NAME_RESOLVABLE_MIDS": NAME_RESOLVABLE,
            "MIXED_MIDS": MIXED,
            "UNKNOWN_MIDS": UNKNOWN,
            "pct": {k: round(100.0 * v / TOTAL_BARE_MIDS, 2) for k, v in
                    (("NAMELESS_MEDIATOR_MIDS", NAMELESS_MEDIATOR),
                     ("NAME_RESOLVABLE_MIDS", NAME_RESOLVABLE),
                     ("MIXED_MIDS", MIXED), ("UNKNOWN_MIDS", UNKNOWN))},
        },
        "threshold_sensitivity_NAMELESS_MEDIATOR_MIDS": sens,
        "control_false_positive_on_named_nodes": ctrl,
        "type_table_top40_by_population": type_table,
        "mediator_like_types_top25": [{"type": types[i], "nodes": int(tot_by_type[i]),
                                       "named_rate": round(float(rate[i]), 6)} for i in med_top],
        "ordinary_types_top25": [{"type": types[i], "nodes": int(tot_by_type[i]),
                                  "named_rate": round(float(rate[i]), 6)} for i in ord_top],
        "PHASE5_CVT_TEXT": {
            "cvt_nodes": int(cvt_nodes.size),
            "distinct_deterministic_texts": distinct_cvt_texts,
            "compression_ratio": round(cvt_nodes.size / max(distinct_cvt_texts, 1), 1),
            "_meaning": "encoder cost for CVT text is bounded by DISTINCT texts, not by node count",
            "top_texts": [{"n_nodes": n, "text": t[:300]} for t, n in top_cvt],
            "samples": samples,
        },
        "_elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    log(json.dumps(res["MID_CLASSIFICATION"], indent=2))
    log(json.dumps(res["PHASE5_CVT_TEXT"]["cvt_nodes"], indent=2),
        "distinct texts:", distinct_cvt_texts)
    log("WROTE " + OUT)

    # ---- persist per-node classification for Phase 8 (temp) ------------------
    lab = np.full(N, 0, dtype=np.int8)   # 0 named/other, 1 mediator, 2 name-resolvable, 3 mixed, 4 unknown
    lab[bare[b_has_out & b_all_med]] = 1
    lab[bare[b_has_out & (~b_all_med) & b_any_ord]] = 2
    lab[bare[b_has_out & (~b_all_med) & (~b_any_ord)]] = 3
    lab[bare[~b_has_out]] = 4
    np.save(os.path.join(A, "union_mid_label.npy"), lab)
    log("WROTE " + os.path.join(A, "union_mid_label.npy"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
