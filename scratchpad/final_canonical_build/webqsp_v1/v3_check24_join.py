"""Checks 2 (WebQSP MID coverage) and 4 (structural agreement), by real join against the backbone.

    python scratchpad/final_canonical_build/webqsp_v1/v3_check24_join.py

Check 3b INFERRED that the 28.98% of NSM triple mass sitting on object-side predicates is "lost to
orientation only" -- the same facts stored under the reverse name. That inference is the single
assumption holding up the claim that IDIR's effective content coverage is 76.2% rather than 47.2%.
It has not been measured. This checks it against the delivered 134,213,735 triples.

Per NSM triple (s, p, o), four tests against the backbone:
    EXACT        (s, p, o) present            -- p must be in the backbone vocabulary
    FORWARD_ANY  (s, *, o) present            -- same direction, any predicate
    REVERSE_ANY  (o, *, s) present            -- the orientation-flip hypothesis
    ABSENT       neither pair present         -- the fact is not in IDIR in any form

Predicate-agnostic on the reverse side because the reverse_property mapping is not in the archive
(FINDING_1); we can ask whether the FACT is present, not whether it carries the name we would
predict. Restricting the backbone to probe-set endpoints is a measurement scope, not a corpus
subset: nothing is being built here.
"""
import json, os, time
import numpy as np
import pyarrow.parquet as pq
from pyarrow import csv as pacsv

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
BB = f"{IDIR}/FB+CVT-REV"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK2_CHECK4_JOIN.json"


def norm(m):
    """m.010016 -> /m/010016 ; g.11abc -> /g/11abc"""
    return "/" + m.replace(".", "/", 1)


def main():
    t0 = time.time()
    nsm_mids = set(pq.read_table(f"{PROBE}/nsm_all_mids.parquet").column(0).to_pylist())
    v1_mids = set(pq.read_table(f"{PROBE}/v1_mid_node_kind.parquet").column("mid").to_pylist())
    ans_mids = set(pq.read_table(f"{PROBE}/nsm_answer_mids.parquet").column(0).to_pylist())
    top_mids = set(pq.read_table(f"{PROBE}/nsm_topic_mids.parquet").column(0).to_pylist())
    probe = nsm_mids | v1_mids
    want = {norm(m): m for m in probe}
    print(f"[probe] nsm={len(nsm_mids):,} v1={len(v1_mids):,} union={len(probe):,}", flush=True)

    # ---- check 2: stream entity2id, record which probe MIDs IDIR knows ----
    found = {}
    n_lines = 0
    max_id = 0
    with open(f"{BB}/entity2id.txt", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            c = line.rfind(",")
            if c < 0:
                continue
            i = int(line[c + 1:])
            if i > max_id:
                max_id = i
            m = want.get(line[:c])
            if m is not None:
                found[m] = i
    print(f"[entity2id] {n_lines:,} lines, max_id={max_id:,}, probe hits={len(found):,} "
          f"t={time.time()-t0:.0f}s", flush=True)

    check2 = {
        "idir_entity2id_lines": n_lines,
        "idir_max_entity_id": max_id,
        "published_entity_count_for_this_variant": 59894890,
        "nsm_mids": len(nsm_mids),
        "nsm_mids_present_in_idir": sum(1 for m in nsm_mids if m in found),
        "v1_mid_labelled_nodes": len(v1_mids),
        "v1_mids_present_in_idir": sum(1 for m in v1_mids if m in found),
        "answer_mids": len(ans_mids),
        "answer_mids_present_in_idir": sum(1 for m in ans_mids if m in found),
        "topic_mids": len(top_mids),
        "topic_mids_present_in_idir": sum(1 for m in top_mids if m in found),
    }
    for a, b in (("nsm_mids", "nsm_mids_present_in_idir"),
                 ("v1_mid_labelled_nodes", "v1_mids_present_in_idir"),
                 ("answer_mids", "answer_mids_present_in_idir"),
                 ("topic_mids", "topic_mids_present_in_idir")):
        check2[b + "_pct"] = round(100 * check2[b] / check2[a], 3)
    print(json.dumps(check2, indent=1), flush=True)

    # ---- dense remap and LUT ----
    dense = {}
    lut = np.full(max_id + 1, -1, dtype=np.int32)
    for m, i in found.items():
        d = len(dense)
        dense[m] = d
        lut[i] = d
    nd = len(dense)
    assert nd < (1 << 23), nd
    print(f"[dense] {nd:,} entities t={time.time()-t0:.0f}s", flush=True)

    rel2id = {}
    with open(f"{BB}/relation2id.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                p, i = line.rsplit(",", 1)
                rel2id[p.lstrip("/").replace("/", ".")] = int(i)

    # ---- stream the backbone, keep triples with BOTH endpoints in the probe set ----
    trip_keys, pair_fwd = [], []
    kept = seen = 0
    for split in ("train", "test", "valid"):
        t = pacsv.read_csv(f"{BB}/{split}.txt",
                           read_options=pacsv.ReadOptions(autogenerate_column_names=True),
                           convert_options=pacsv.ConvertOptions(
                               column_types={"f0": "int32", "f1": "int32", "f2": "int32"}))
        s = t.column(0).to_numpy()
        r = t.column(1).to_numpy()
        o = t.column(2).to_numpy()
        seen += len(s)
        ds, do = lut[s], lut[o]
        m = (ds >= 0) & (do >= 0)
        ds = ds[m].astype(np.int64)
        do = do[m].astype(np.int64)
        rr = r[m].astype(np.int64)
        kept += len(ds)
        trip_keys.append((ds << 35) | (rr << 23) | do)
        pair_fwd.append((ds << 23) | do)
        del t, s, r, o, ds, do, rr, m
        print(f"[{split}] kept {kept:,} of {seen:,} t={time.time()-t0:.0f}s", flush=True)

    trip = np.sort(np.concatenate(trip_keys))
    del trip_keys
    pair = np.sort(np.concatenate(pair_fwd))
    del pair_fwd
    print(f"[index] {len(trip):,} triples, {len(pair):,} pairs t={time.time()-t0:.0f}s", flush=True)

    def has(sorted_arr, keys):
        i = np.searchsorted(sorted_arr, keys)
        i[i >= len(sorted_arr)] = len(sorted_arr) - 1
        return sorted_arr[i] == keys

    # ---- NSM side ----
    t = pq.read_table(f"{PROBE}/nsm_mid_triples.parquet")
    ns = t.column("subject_mid").to_pylist()
    npd = t.column("predicate").to_pylist()
    no = t.column("object_mid").to_pylist()
    N = len(ns)
    S = np.fromiter((dense.get(x, -1) for x in ns), dtype=np.int64, count=N)
    O = np.fromiter((dense.get(x, -1) for x in no), dtype=np.int64, count=N)
    R = np.fromiter((rel2id.get(x, -1) for x in npd), dtype=np.int64, count=N)
    joinable = (S >= 0) & (O >= 0)
    print(f"[nsm] joinable {int(joinable.sum()):,} of {N:,} t={time.time()-t0:.0f}s", flush=True)

    fwd_pair = np.where(joinable, (S << 23) | O, -1)
    rev_pair = np.where(joinable, (O << 23) | S, -1)
    exact_k = np.where(joinable & (R >= 0), (S << 35) | (R << 23) | O, -1)

    has_fwd = has(pair, fwd_pair) & joinable
    has_rev = has(pair, rev_pair) & joinable
    has_exact = has(trip, exact_k) & joinable & (R >= 0)
    present = has_fwd | has_rev

    def pct(x):
        return round(100 * float(x) / N, 3)

    check4 = {
        "nsm_triples": N,
        "joinable_both_endpoints_known_to_idir": int(joinable.sum()),
        "joinable_pct": pct(joinable.sum()),
        "EXACT_same_predicate_same_direction": int(has_exact.sum()),
        "EXACT_pct": pct(has_exact.sum()),
        "FORWARD_ANY_predicate": int(has_fwd.sum()),
        "FORWARD_ANY_pct": pct(has_fwd.sum()),
        "REVERSE_ANY_predicate": int(has_rev.sum()),
        "REVERSE_ANY_pct": pct(has_rev.sum()),
        "PRESENT_in_either_orientation": int(present.sum()),
        "PRESENT_pct": pct(present.sum()),
        "ABSENT_from_backbone_entirely": int(N - present.sum()),
        "ABSENT_pct": pct(N - present.sum()),
        "reverse_only_no_forward": int((has_rev & ~has_fwd).sum()),
        "reverse_only_pct": pct((has_rev & ~has_fwd).sum()),
    }

    # the orientation hypothesis, tested on exactly the predicates it was made about
    vocab_bb = set(rel2id)
    in_bb = np.fromiter((x in vocab_bb for x in npd), dtype=bool, count=N)
    for label, sel in (("predicate_in_backbone", in_bb), ("predicate_not_in_backbone", ~in_bb)):
        n = int(sel.sum())
        if not n:
            continue
        check4[label] = {
            "n": n,
            "present_either_orientation": int((present & sel).sum()),
            "present_pct": round(100 * float((present & sel).sum()) / n, 3),
            "reverse_only": int((has_rev & ~has_fwd & sel).sum()),
            "reverse_only_pct": round(100 * float((has_rev & ~has_fwd & sel).sum()) / n, 3),
        }

    ans_np = np.fromiter((a in ans_mids or b in ans_mids for a, b in zip(ns, no)),
                         dtype=bool, count=N)
    na = int(ans_np.sum())
    check4["ANSWER_INCIDENT"] = {
        "n": na,
        "present_either_orientation": int((present & ans_np).sum()),
        "present_pct": round(100 * float((present & ans_np).sum()) / na, 3),
        "absent": int((~present & ans_np).sum()),
        "absent_pct": round(100 * float((~present & ans_np).sum()) / na, 3),
    }

    doc = {
        "schema": "V3_CHECK2_CHECK4_JOIN/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CHECK_2_MID_COVERAGE": check2,
        "NOTE_ON_ENTITY_COUNT": "entity2id.txt line count is recorded as delivered. Triple files "
                                "reference ids above the published 59,894,890, so that figure "
                                "cannot be the size of this variant's id space. Recorded as an "
                                "observation; not reconciled here and not used downstream.",
        "CHECK_4_STRUCTURAL_AGREEMENT": check4,
        "backbone_triples_seen": seen,
        "backbone_triples_with_both_endpoints_in_probe_set": kept,
        "scope_note": "restricting the backbone to probe-set endpoints measures coverage of a probe "
                      "set. It builds nothing and subsets no corpus.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(check4, indent=1))


if __name__ == "__main__":
    main()
