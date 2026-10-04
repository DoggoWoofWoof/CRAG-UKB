"""V1 section 3 -- classify every node, per the definition frozen in
data/final_canonical/webqsp/V1_SECTION3_CLASSIFIER_FREEZE.json.

    python scratchpad/final_canonical_build/webqsp_v1/classify_nodes.py

Runs the inherited tau=0.02 / tau=0.50 structural classifier UNCHANGED against the V1 tables,
rather than importing the audit's saved labels, so the result is self-contained and reproducible
from the frozen graph.  The audit's union_mid_label_final.npy is then used as an INDEPENDENT
reproduction check -- it was computed on a different endpoint ordering, so agreement is meaningful.

Writes node_kind, classification_source, classification_score, classification_rule_version,
source_mid, retrieval_role.  Does NOT write display/canonical_name fields: the user's order is
classify -> audit -> freeze node_kind -> render, so rendering cannot feed back into classification.
"""
import hashlib, json, os, re, time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
FREEZE = "data/final_canonical/webqsp/V1_SECTION3_CLASSIFIER_FREEZE.json"
AUDIT_LABELS = "scratchpad/final_canonical_build/_audit/union_mid_label_final.npy"
AUDIT_ENTS = "scratchpad/final_canonical_build/_audit/union_entities.txt"
OUT = "data/final_canonical/webqsp/V1_CLASSIFICATION_REPORT.json"

RULE_VERSION = "webqsp_v1_section3/v1"
TAU_LOW, TAU_HIGH = 0.02, 0.50

MID_RE = re.compile(r"^[mg]\.[0-9A-Za-z_]+$")
SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
DATEISH_RE = re.compile(r"^-?\d{1,4}(-\d{2}){0,2}(T[\d:]+Z?)?$")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

MID, SURFACE, OTHER = 0, 1, 2


def surface_class(s):
    if s is None:
        return OTHER, "null"
    t = s.strip()
    if t == "":
        return OTHER, "empty"
    if MID_RE.match(t):
        return MID, "mid"
    if DATEISH_RE.match(t):
        return OTHER, "date_or_year_literal"
    if NUMERIC_RE.match(t):
        return OTHER, "numeric_literal"
    if SCHEMA_RE.match(t) and " " not in t:
        return OTHER, "schema_path_string"
    return SURFACE, "surface"


def main():
    t0 = time.time()
    frz = json.load(open(FREEZE, encoding="utf-8"))
    assert frz["rule_version"] == RULE_VERSION, "rule_version drift"
    assert frz["inherited_classifier"]["TAU_LOW_mediator"] == TAU_LOW
    assert frz["inherited_classifier"]["TAU_HIGH_ordinary"] == TAU_HIGH

    nodes = pq.read_table(f"{D}/nodes.parquet")
    eps = nodes.column("source_rog_endpoint").to_pylist()
    N = len(eps)

    rels = pq.read_table(f"{D}/relations.parquet")
    rkeys = rels.column("relation_key").to_pylist()
    # Freebase TYPE = relation id minus its last dot-segment
    types, tindex = [], {}
    rel2type = np.empty(len(rkeys), dtype=np.int32)
    for i, k in enumerate(rkeys):
        ty = k.rsplit(".", 1)[0] if "." in k else k
        j = tindex.get(ty)
        if j is None:
            j = tindex[ty] = len(types)
            types.append(ty)
        rel2type[i] = j
    NT = len(types)

    e = pq.read_table(f"{D}/edges.parquet")
    h = e.column("src_uid").to_numpy()
    r = e.column("relation_uid").to_numpy()
    t = e.column("dst_uid").to_numpy()
    del e

    sc = np.empty(N, dtype=np.int8)
    subclass = []
    for i, s in enumerate(eps):
        c, sub = surface_class(s)
        sc[i] = c
        subclass.append(sub)
    is_mid = sc == MID
    is_named = sc == SURFACE
    print(f"[class] MID={is_mid.sum():,} SURFACE={is_named.sum():,} OTHER={(sc==OTHER).sum():,}",
          flush=True)

    # ---------- pass 1: per-type named rate over distinct (head, type) pairs ----------
    rt = rel2type[r]
    out_pairs = np.unique(h.astype(np.int64) * NT + rt)
    out_node = (out_pairs // NT).astype(np.int64)
    out_type = (out_pairs % NT).astype(np.int64)
    tot_by_type = np.bincount(out_type, minlength=NT).astype(np.int64)
    named_by_type = np.bincount(out_type, weights=is_named[out_node], minlength=NT).astype(np.int64)
    rate = np.divide(named_by_type, np.maximum(tot_by_type, 1), dtype=np.float64)
    mediator_type = rate <= TAU_LOW
    ordinary_type = rate >= TAU_HIGH

    order = np.argsort(out_node, kind="stable")
    on, ot = out_node[order], out_type[order]
    starts = np.searchsorted(on, np.arange(N), side="left")
    ends = np.searchsorted(on, np.arange(N), side="right")
    ndeg = ends - starts
    has_out = ndeg > 0
    cum_med = np.concatenate(([0], np.cumsum(mediator_type[ot])))
    cum_ord = np.concatenate(([0], np.cumsum(ordinary_type[ot])))
    nmed = cum_med[ends] - cum_med[starts]
    nord = cum_ord[ends] - cum_ord[starts]
    all_med = np.zeros(N, bool)
    any_ord = np.zeros(N, bool)
    all_med[has_out] = nmed[has_out] == ndeg[has_out]
    any_ord[has_out] = nord[has_out] > 0

    # score = max out-type named rate  (running max over the same sorted runs)
    score1 = np.full(N, np.nan)
    rmax = np.maximum.accumulate(np.concatenate(([-1.0], rate[ot])))
    # accumulate-based max is only valid within a run, so do it segment-wise via reduceat
    seg = starts[has_out]
    score1[has_out] = np.maximum.reduceat(rate[ot], seg) if seg.size else np.nan
    del rmax

    # ---------- pass 2: relation-range named rate, for the pass-1 UNKNOWN set ----------
    in_pairs = np.unique(t.astype(np.int64) * len(rkeys) + r)
    in_node = (in_pairs // len(rkeys)).astype(np.int64)
    in_rel = (in_pairs % len(rkeys)).astype(np.int64)
    tail_pairs = np.unique(r.astype(np.int64) * N + t)
    tr = (tail_pairs // N).astype(np.int64)
    tt = (tail_pairs % N).astype(np.int64)
    tot_by_rel = np.bincount(tr, minlength=len(rkeys)).astype(np.int64)
    named_by_rel = np.bincount(tr, weights=is_named[tt], minlength=len(rkeys)).astype(np.int64)
    obj_rate = np.divide(named_by_rel, np.maximum(tot_by_rel, 1), dtype=np.float64)
    med_rel = obj_rate <= TAU_LOW
    ord_rel = obj_rate >= TAU_HIGH

    iorder = np.argsort(in_node, kind="stable")
    inn, inr = in_node[iorder], in_rel[iorder]
    istarts = np.searchsorted(inn, np.arange(N), side="left")
    iends = np.searchsorted(inn, np.arange(N), side="right")
    ideg = iends - istarts
    has_in = ideg > 0
    icum_med = np.concatenate(([0], np.cumsum(med_rel[inr])))
    icum_ord = np.concatenate(([0], np.cumsum(ord_rel[inr])))
    inmed = icum_med[iends] - icum_med[istarts]
    inord = icum_ord[iends] - icum_ord[istarts]
    all_med_in = np.zeros(N, bool)
    any_ord_in = np.zeros(N, bool)
    all_med_in[has_in] = inmed[has_in] == ideg[has_in]
    any_ord_in[has_in] = inord[has_in] > 0
    score2 = np.full(N, np.nan)
    iseg = istarts[has_in]
    score2[has_in] = np.maximum.reduceat(obj_rate[inr], iseg) if iseg.size else np.nan

    # ---------- merge: pass 1 where it decided, pass 2 on the pass-1 UNKNOWN set ----------
    kind = np.empty(N, dtype=object)
    csrc = np.empty(N, dtype=object)
    score = np.full(N, np.nan)

    kind[is_named] = "READABLE_ENTITY"
    kind[sc == OTHER] = "VALUE_LITERAL"
    csrc[is_named] = "SURFACE_FORM"
    csrc[sc == OTHER] = "SURFACE_FORM"

    bare = np.flatnonzero(is_mid)
    b_out = has_out[bare]
    p1_med = bare[b_out & all_med[bare]]
    p1_ord = bare[b_out & ~all_med[bare] & any_ord[bare]]
    p1_mix = bare[b_out & ~all_med[bare] & ~any_ord[bare]]
    p1_unk = bare[~b_out]

    for idx, k in ((p1_med, "CVT_MEDIATOR"), (p1_ord, "MID_NAMED_ENTITY"),
                   (p1_mix, "UNRESOLVED_OTHER")):
        kind[idx] = k
        csrc[idx] = "OUT_TYPE_NAMED_RATE_PASS1"
        score[idx] = score1[idx]

    u_in = has_in[p1_unk]
    u2_med = p1_unk[u_in & all_med_in[p1_unk]]
    u2_ord = p1_unk[u_in & ~all_med_in[p1_unk] & any_ord_in[p1_unk]]
    u2_mix = p1_unk[u_in & ~all_med_in[p1_unk] & ~any_ord_in[p1_unk]]
    u2_none = p1_unk[~u_in]
    for idx, k in ((u2_med, "CVT_MEDIATOR"), (u2_ord, "MID_NAMED_ENTITY"),
                   (u2_mix, "UNRESOLVED_OTHER")):
        kind[idx] = k
        csrc[idx] = "IN_RELATION_RANGE_PASS2"
        score[idx] = score2[idx]
    kind[u2_none] = "UNRESOLVED_OTHER"
    csrc[u2_none] = "NO_SIGNAL"

    assert all(k is not None for k in kind), "unclassified node"

    # ---------- independent reproduction check against the audit labels ----------
    repro = {"available": False}
    if os.path.exists(AUDIT_LABELS) and os.path.exists(AUDIT_ENTS):
        lab = np.load(AUDIT_LABELS)
        with open(AUDIT_ENTS, encoding="utf-8") as fh:
            aents = [ln.rstrip("\n") for ln in fh]
        if len(aents) == len(lab):
            AUD = {1: "CVT_MEDIATOR", 2: "MID_NAMED_ENTITY", 3: "UNRESOLVED_OTHER",
                   4: "UNRESOLVED_OTHER"}
            pos = {s: i for i, s in enumerate(eps)}
            agree = disagree = compared = 0
            conf = Counter()
            for s, lv in zip(aents, lab.tolist()):
                if lv == 0:
                    continue
                i = pos.get(s)
                if i is None:
                    continue
                compared += 1
                want = AUD[lv]
                if kind[i] == want:
                    agree += 1
                else:
                    disagree += 1
                    conf[f"{want} -> {kind[i]}"] += 1
            repro = {"available": True, "compared": compared, "agree": agree,
                     "disagree": disagree,
                     "agreement_pct": round(100.0 * agree / compared, 4) if compared else None,
                     "confusion": dict(conf.most_common(10)),
                     "note": "The audit union used a DIFFERENT endpoint ordering, so this is a "
                             "genuine re-derivation check, not a tautology. Labels 3 (MIXED) and 4 "
                             "(UNKNOWN) both map to UNRESOLVED_OTHER by the frozen mapping."}

    # ---------- write ----------
    role = np.where(kind == "CVT_MEDIATOR", "UNDECIDED_PENDING_CVT_GOLD_GATE",
                    "UNDECIDED_PENDING_RETRIEVAL_POLICY")
    src_mid = np.where(is_mid, np.array(eps, dtype=object), None)

    tbl = nodes.drop_columns(["node_kind", "retrieval_role", "source_mid"])
    tbl = tbl.append_column("node_kind", pa.array(kind.tolist(), pa.string()))
    tbl = tbl.append_column("classification_source", pa.array(csrc.tolist(), pa.string()))
    tbl = tbl.append_column("classification_score",
                            pa.array([None if np.isnan(v) else float(v) for v in score],
                                     pa.float64()))
    tbl = tbl.append_column("classification_rule_version",
                            pa.array([RULE_VERSION] * N, pa.string()))
    tbl = tbl.append_column("source_mid", pa.array(src_mid.tolist(), pa.string()))
    tbl = tbl.append_column("retrieval_role", pa.array(role.tolist(), pa.string()))
    tbl = tbl.set_column(tbl.schema.get_field_index("resolution_status"), "resolution_status",
                         pa.array(["CLASSIFIED_PENDING_RENDER"] * N, pa.string()))
    tmp = f"{D}/nodes.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, f"{D}/nodes.parquet")

    kc = Counter(kind.tolist())
    sc_c = Counter(csrc.tolist())
    sub_c = Counter(subclass)
    mid_kind = Counter(kind[i] for i in bare.tolist())
    h_kind = hashlib.sha256()
    for k in kind:
        h_kind.update(k.encode())
        h_kind.update(b"\n")

    doc = {
        "schema": "V1_CLASSIFICATION_REPORT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule_version": RULE_VERSION,
        "freeze_document": FREEZE,
        "thresholds_used": {"TAU_LOW": TAU_LOW, "TAU_HIGH": TAU_HIGH},
        "thresholds_changed_after_seeing_results": False,
        "nodes": N,
        "NODE_KIND_COUNTS": dict(kc.most_common()),
        "NODE_KIND_PCT": {k: round(100.0 * v / N, 4) for k, v in kc.most_common()},
        "CLASSIFICATION_SOURCE_COUNTS": dict(sc_c.most_common()),
        "surface_subclass_counts": dict(sub_c.most_common()),
        "bare_mid_population": {"total": int(is_mid.sum()), "by_kind": dict(mid_kind.most_common())},
        "EVERY_NODE_CLASSIFIED": True,
        "KINDS_DISJOINT_AND_EXHAUSTIVE": int(sum(kc.values())) == N,
        "NODE_KIND_HASH": h_kind.hexdigest(),
        "independent_reproduction_vs_2026_09_05_audit": repro,
        "distinct_freebase_types": NT,
        "type_named_rate_bimodality": {
            "types_below_0.01": int((rate < 0.01).sum()),
            "types_above_0.9": int((rate > 0.9).sum()),
            "types_in_between": int(((rate >= 0.01) & (rate <= 0.9)).sum()),
        },
        "retrieval_role_counts": dict(Counter(role.tolist()).most_common()),
        "retrieval_role_note": "No indexing policy is ratified for any class. CVT waits on the "
                               "pre-registered gold gate; everything else waits on retrieval policy.",
        "fields_still_null": ["canonical_name", "display_name", "display_name_disambiguated",
                              "display_source"],
        "resolution_status": "CLASSIFIED_PENDING_RENDER",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in
                      ("NODE_KIND_COUNTS", "NODE_KIND_PCT", "CLASSIFICATION_SOURCE_COUNTS",
                       "surface_subclass_counts", "bare_mid_population",
                       "independent_reproduction_vs_2026_09_05_audit",
                       "type_named_rate_bimodality", "NODE_KIND_HASH", "elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
