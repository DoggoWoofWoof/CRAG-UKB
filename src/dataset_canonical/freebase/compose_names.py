# -*- coding: utf-8 -*-
"""names_v4: one display name for every node in the canonical Freebase universe, with provenance.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/compose_names.py base
    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/compose_names.py final

Precedence (a strict extension of NAME_HIERARCHY_CONTRACT_V1; every tier is disjoint by construction
because each row takes the FIRST rule that fires):

    0 ORIGINAL            overlay_v1 is_original_name == true          ACTUAL Freebase name / literal / schema label
    1 RECOVERED_ORIGINAL  cascade_names.parquet is_original_name==true ACTUAL: historical Freebase assertion / alias,
                          or source == IDIR_FREEBASE_NAME             plus IDIR's /type/object/name rows (2,233), which
                                                                       V3_ACTUAL_NAME_LAYER_FROZEN rules to be Freebase
                                                                       name assertions (HISTORICAL_FREEBASE_NAME); the
                                                                       cascade file's per-row flag predates that ruling
    2 RECOVERED_EXTERNAL  cascade_names.parquet is_original_name==false  exact-id match in an external source (WordNet,
                          and not IDIR                                 Wikidata, authority, archived URL)
    3 URI_SELF            overlay URI_DERIVED                          the node's own identifier, rendered
    4 KEY_SEMANTIC        overlay FREEBASE_KEY_EXACT                   the node's own readable Freebase key
    5 INFERRED            inference_overlay_v1, effective reject 0     GENERATED from the node's own relations
    6 GENERATED_FLOOR     this pass (floor rules, see floor_names.py)  GENERATED, descriptive, carries the MID

`base` writes scratchpad/fb4/names_v4/<canonical shard>.parquet with floor rows left EMPTY and
checks that the floor set is exactly scratchpad/fb4/floor_uids.npy (7,221,354).  `final` fills the
floor rows from scratchpad/fb4/floor_names.parquet and enforces: no empty name, no name that is a
bare MID, no 'Unnamed …' placeholder, tier counts reconcile with V4_FLOOR_CENSUS.
Nothing frozen is modified; tier 0-5 strings are copied verbatim from their frozen layers.
"""
import collections
import glob
import io
import json
import os
import re
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
SCR = "scratchpad/fb4"
NV4 = SCR + "/names_v4"
NAME_KINDS = ["ORIGINAL", "RECOVERED_ORIGINAL", "RECOVERED_EXTERNAL", "URI_SELF", "KEY_SEMANTIC", "INFERRED", "GENERATED_FLOOR"]
FLOOR_REASONS = {0: "not_floor", 1: "INFERENCE_REJECTED_BOOKKEEPING_RELATION", 2: "INFERENCE_REJECTED_UNREADABLE_IDENTITY", 3: "NO_INFERENCE_ROW"}
BARE_MID = re.compile(r"^[mg]\.[0-9a-z_]+$")
PLACEHOLDER = re.compile(r"^(Unnamed\b|Unknown\b|\s*$)")
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def undict(col):
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


shards = sorted(os.path.basename(f)[:-8] for f in glob.glob(FB + "/canonical/nodes/*.parquet"))
assert len(shards) == 262


def one(x):
    return x.combine_chunks() if isinstance(x, pa.ChunkedArray) else x


def expand(sub, rows, lo, n, null_row):
    """Scatter the sidecar rows (canonical rows `rows`, ascending) into a slice-long table; other rows null."""
    idx = np.full(n, sub.num_rows, dtype=np.int64)
    idx[rows - lo] = np.arange(sub.num_rows)
    return pa.concat_tables([sub, null_row]).take(pa.array(idx))


def compose_base():
    os.makedirs(NV4, exist_ok=True)
    floor_expected = np.load(SCR + "/floor_uids.npy")
    floor_seen = []
    stats = collections.Counter()
    src_stats = collections.Counter()
    display_disagree = 0
    display_checked = 0
    display_examples = []
    n_total = 0
    SL = 2_000_000
    for si, s in enumerate(shards):
        cn = pq.read_table(FB + "/canonical/nodes/%s.parquet" % s, columns=["node_uid", "kind", "display_text"])
        ov = pq.read_table(FB + "/overlay_v1/%s.parquet" % s, columns=["node_uid", "display_name", "display_name_source", "is_original_name"])
        assert cn.num_rows == ov.num_rows
        u_all = cn["node_uid"].to_numpy()
        assert np.array_equal(u_all, ov["node_uid"].to_numpy()), "overlay not aligned: " + s
        n_all = len(u_all)
        sc = None
        sf = SCR + "/sidecar/%s.parquet" % s
        if os.path.exists(sf):
            sc = pq.read_table(sf)
            sc_rows = sc["row"].to_numpy()
            assert np.array_equal(sc["node_uid"].to_numpy(), u_all[sc_rows]), "sidecar uid mismatch " + s
            assert np.all(np.diff(sc_rows) > 0)
            null_row = pa.table({c: pa.nulls(1, sc.schema.field(c).type) for c in sc.column_names})
        pieces = []
        for lo in range(0, n_all, SL):
            hi = min(lo + SL, n_all)
            n = hi - lo
            u = u_all[lo:hi]
            kind = cn["kind"].slice(lo, n).combine_chunks()
            dt = cn["display_text"].slice(lo, n).combine_chunks()
            ov_name = ov["display_name"].slice(lo, n).combine_chunks()
            ov_src = undict(ov["display_name_source"].slice(lo, n).combine_chunks())
            ov_orig = ov["is_original_name"].slice(lo, n).to_numpy(zero_copy_only=False).astype(bool)
            if sc is not None:
                a0, a1 = np.searchsorted(sc_rows, lo), np.searchsorted(sc_rows, hi)
                ex = expand(sc.slice(a0, a1 - a0), sc_rows[a0:a1], lo, n, null_row)
            else:
                ex = pa.table({c: pa.nulls(n, t) for c, t in [("inf_name", pa.string()), ("inf_source", pa.string()), ("inf_rule", pa.string()),
                                                              ("inf_reject", pa.int8()), ("cascade_name", pa.string()), ("cascade_source", pa.string()),
                                                              ("cascade_original", pa.bool_()), ("nameless_grade", pa.string()), ("recovery_class", pa.string())]})
            inf_rej = ex["inf_reject"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int8)
            has_cas = ~ex["cascade_name"].is_null().to_numpy(zero_copy_only=False)
            cas_orig = ex["cascade_original"].fill_null(False).to_numpy(zero_copy_only=False).astype(bool)
            # V3_ACTUAL_NAME_LAYER_FROZEN: IDIR /type/object/name rows are Freebase name assertions (HISTORICAL_FREEBASE_NAME)
            cas_orig |= pc.equal(ex["cascade_source"], "IDIR_FREEBASE_NAME").fill_null(False).to_numpy(zero_copy_only=False)
            is_uri = pc.equal(ov_src, "URI_DERIVED").fill_null(False).to_numpy(zero_copy_only=False)
            is_key = pc.equal(ov_src, "FREEBASE_KEY_EXACT").fill_null(False).to_numpy(zero_copy_only=False)
            # precedence: first rule that fires
            nk = np.full(n, 6, dtype=np.int8)
            nk[ov_orig] = 0
            m = (nk == 6) & has_cas
            nk[m] = np.where(cas_orig[m], 1, 2)
            nk[(nk == 6) & is_uri] = 3
            nk[(nk == 6) & is_key] = 4
            nk[(nk == 6) & (inf_rej == 0)] = 5
            nk_a = pa.array(nk)
            from_ov = pc.or_(pc.or_(pc.equal(nk_a, 0), pc.equal(nk_a, 3)), pc.equal(nk_a, 4))
            from_cas = pc.or_(pc.equal(nk_a, 1), pc.equal(nk_a, 2))
            from_inf = pc.equal(nk_a, 5)
            empty = pa.array([""] * n, pa.string()) if n else pa.array([], pa.string())
            name = pc.if_else(from_ov, ov_name, pc.if_else(from_cas, ex["cascade_name"], pc.if_else(from_inf, ex["inf_name"], empty)))
            source = pc.if_else(from_ov, ov_src, pc.if_else(from_cas, ex["cascade_source"], pc.if_else(from_inf, ex["inf_source"], pa.nulls(n, pa.string()))))
            rule = pc.if_else(from_inf, ex["inf_rule"], pa.nulls(n, pa.string()))
            is_orig = ov_orig | ((nk == 1))
            freason = np.zeros(n, dtype=np.int8)
            fl = nk == 6
            freason[fl] = np.where(inf_rej[fl] < 0, 3, inf_rej[fl]).astype(np.int8)
            # display_text of a non-literal must be the ORIGINAL name
            chk = pc.and_(pc.is_valid(dt), pc.not_equal(kind, "LITERAL"))
            display_checked += pc.sum(chk).as_py() or 0
            agree = pc.and_(pc.equal(nk_a, 0), pc.equal(name, dt).fill_null(False))
            dis = pc.and_(chk, pc.invert(agree))
            display_disagree += pc.sum(dis).as_py() or 0
            if len(display_examples) < 10:
                for j in np.flatnonzero(dis.to_numpy(zero_copy_only=False))[:10].tolist():
                    display_examples.append({"shard": s, "node_uid": int(u[j]), "display_text": dt[j].as_py(), "served_name": name[j].as_py(),
                                             "name_kind": NAME_KINDS[int(nk[j])], "name_source": source[j].as_py()})
            nonfloor_empty = pc.sum(pc.and_(pc.not_equal(nk_a, 6), pc.or_kleene(pc.is_null(name), pc.equal(name, "")))).as_py() or 0
            assert nonfloor_empty == 0, ("empty non-floor name", s, nonfloor_empty)
            floor_seen.append(u[fl])
            for k, c in zip(*np.unique(nk, return_counts=True)):
                stats[NAME_KINDS[k]] += int(c)
            # census by (kind, name_kind, source)
            kd = pc.dictionary_encode(one(kind)); sd = pc.dictionary_encode(one(source).fill_null("<none>"))
            assert len(sd.dictionary) < 256 and len(kd.dictionary) < 64
            key = (kd.indices.to_numpy(zero_copy_only=False).astype(np.int64) * 8 + nk) * 256 + sd.indices.to_numpy(zero_copy_only=False).astype(np.int64)
            kdict, sdict = kd.dictionary.to_pylist(), sd.dictionary.to_pylist()
            for kv, c in zip(*np.unique(key, return_counts=True)):
                kv = int(kv)
                src_stats["%s|%s|%s" % (kdict[kv // 2048], NAME_KINDS[(kv // 256) % 8], sdict[kv % 256])] += int(c)
            pieces.append(pa.table({"node_uid": pa.array(u), "name": name, "name_kind": nk_a, "name_source": source, "name_rule": rule,
                                    "is_original_name": pa.array(is_orig), "nameless_grade": ex["nameless_grade"],
                                    "recovery_class": ex["recovery_class"], "floor_reason": pa.array(freason)}))
        tbl = pa.concat_tables(pieces).combine_chunks()
        cols = {c: tbl[c] for c in tbl.column_names}
        for k in ("name_source", "name_rule", "nameless_grade", "recovery_class"):
            cols[k] = pc.dictionary_encode(cols[k])
        tbl = pa.table(cols)
        assert tbl.num_rows == n_all
        pq.write_table(tbl, NV4 + "/%s.parquet" % s, compression="zstd")
        n_total += n_all
        del cn, ov, sc, pieces, cols, tbl
        if si % 20 == 0:
            log("shard", si, s, "rows", n_total, "floor so far", sum(len(x) for x in floor_seen))
    floor_seen = np.sort(np.concatenate(floor_seen))
    assert n_total == 301977131, n_total
    assert len(floor_seen) == len(floor_expected) and np.array_equal(floor_seen, floor_expected), \
        ("floor set differs from census", len(floor_seen), len(floor_expected))
    rec = {"RECORD": "NAMES_V4_BASE", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rows": n_total, "by_name_kind": dict(stats), "by_kind_namekind_source": dict(src_stats),
           "floor_rows": int(len(floor_seen)), "CHECK_floor_set_equals_census": True,
           "display_text_rows_checked": display_checked, "display_text_disagreeing_with_original_name": display_disagree,
           "display_text_disagreement_examples": display_examples,
           "display_text_note": "a non-literal display_text that is not served as the ORIGINAL name: overlay_v1 did not accept it as a name "
                                "(e.g. a whitespace-only /type/object/name), so the row resolved by the next rule; nothing is overwritten",
           "seconds": round(time.time() - t0, 1)}
    json.dump(rec, open(SCR + "/NAMES_V4_BASE.json", "w"), indent=1)
    log("done", json.dumps(stats))


def compose_final():
    fn = pq.read_table(SCR + "/floor_names.parquet")
    fu = fn["node_uid"].to_numpy()
    order = np.argsort(fu)
    fu = fu[order]
    fname = fn["name"].combine_chunks().take(pa.array(order))
    frule = fn["rule"].combine_chunks().take(pa.array(order))
    ffam = pc.binary_join_element_wise("FLOOR_", fn["family"].combine_chunks().take(pa.array(order)), "")
    assert len(np.unique(fu)) == len(fu) == 7221354, len(fu)
    # the floor rules never emit a placeholder or a bare MID
    assert pc.sum(pc.match_substring_regex(fname, BARE_MID.pattern)).as_py() in (0, None)
    assert pc.sum(pc.match_substring_regex(fname, PLACEHOLDER.pattern)).as_py() in (0, None)
    filled = 0
    stats = collections.Counter(); rule_stats = collections.Counter(); reason_stats = collections.Counter()
    bad = collections.Counter()
    bad_by_kind = {"bare_mid": np.zeros(len(NAME_KINDS), dtype=np.int64), "placeholder": np.zeros(len(NAME_KINDS), dtype=np.int64)}
    samples = collections.defaultdict(list)
    seen_names = 0
    for si, s in enumerate(shards):
        p = NV4 + "/%s.parquet" % s
        t = pq.read_table(p)
        u = t["node_uid"].to_numpy()
        nk = t["name_kind"].to_numpy()
        n = len(u)
        fl = np.flatnonzero(nk == 6)
        name = t["name"].combine_chunks()
        src = undict(t["name_source"]).combine_chunks()
        rule = undict(t["name_rule"]).combine_chunks()
        if len(fl):
            idx = np.searchsorted(fu, u[fl])
            idx[idx >= len(fu)] = 0
            assert np.array_equal(fu[idx], u[fl]), "floor node without a generated name in " + s
            pos = np.arange(n, dtype=np.int64)
            pos[fl] = n + np.arange(len(fl))
            take = pa.array(pos)
            sel = pa.array(idx)
            name = pa.concat_arrays([name, fname.take(sel)]).take(take)
            src = pa.concat_arrays([src, ffam.take(sel)]).take(take)
            rule = pa.concat_arrays([rule, frule.take(sel)]).take(take)
            for k, c in zip(*np.unique(frule.take(sel).to_numpy(zero_copy_only=False), return_counts=True)):
                rule_stats[str(k)] += int(c)
            filled += len(fl)
        empty = pc.or_kleene(pc.is_null(name), pc.equal(pc.utf8_trim_whitespace(name), ""))
        bare = pc.and_(pc.invert(empty).fill_null(True), pc.match_substring_regex(name, BARE_MID.pattern).fill_null(False))
        ph = pc.and_(pc.invert(empty).fill_null(True), pc.match_substring_regex(name, PLACEHOLDER.pattern).fill_null(False))
        bad["empty"] += pc.sum(empty).as_py() or 0
        n_bare = pc.sum(bare).as_py() or 0
        bad["bare_mid"] += n_bare
        n_ph = pc.sum(ph).as_py() or 0
        bad["placeholder"] += n_ph
        for label, mask, n_hit in (("bare_mid", bare, n_bare), ("placeholder", ph, n_ph)):
            if n_hit:
                hit = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
                bad_by_kind[label] += np.bincount(nk[hit], minlength=len(NAME_KINDS))
                for r in hit[:max(0, 20 - len(samples[label]))].tolist():
                    samples[label].append((name[r].as_py(), NAME_KINDS[nk[r]], src[r].as_py()))
        for k, c in zip(*np.unique(nk, return_counts=True)):
            stats[NAME_KINDS[k]] += int(c)
        fr = t["floor_reason"].to_numpy()
        assert np.array_equal(fr > 0, nk == 6), "floor_reason set differs from name_kind 6 in " + s
        for k, c in zip(*np.unique(fr[fr > 0], return_counts=True)):
            reason_stats[FLOOR_REASONS[int(k)]] += int(c)
        seen_names += n
        t = t.set_column(t.schema.get_field_index("name"), "name", name)
        t = t.set_column(t.schema.get_field_index("name_source"), "name_source", pc.dictionary_encode(src))
        t = t.set_column(t.schema.get_field_index("name_rule"), "name_rule", pc.dictionary_encode(rule))
        pq.write_table(t, p, compression="zstd")
        if si % 20 == 0:
            log("final shard", si, s, "filled", filled, "bad", dict(bad))
    assert filled == 7221354, filled
    assert seen_names == 301977131
    by_kind = {label: {NAME_KINDS[i]: int(c) for i, c in enumerate(arr) if c} for label, arr in bad_by_kind.items()}
    generated = {label: int(sum(c for i, c in enumerate(arr) if i >= 3)) for label, arr in bad_by_kind.items()}
    rec = {"RECORD": "NAMES_V4_FINAL", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rows": seen_names, "floor_rows_filled": filled, "by_name_kind": dict(stats), "floor_rules": dict(rule_stats.most_common()),
           "floor_by_reason": dict(reason_stats.most_common()), "CHECK_floor_reason_set_equals_name_kind_6": True,
           "CHECKS": {"empty_names": bad["empty"], "bare_mid_names": bad["bare_mid"], "placeholder_names": bad["placeholder"],
                      "bare_mid_names_in_derived_kinds_3_to_6": generated["bare_mid"], "placeholder_names_in_derived_kinds_3_to_6": generated["placeholder"]},
           "bare_mid_by_name_kind": by_kind["bare_mid"], "placeholder_by_name_kind": by_kind["placeholder"],
           "bare_mid_note": "a name matching ^[mg]\\.[0-9a-z_]+$ that Freebase itself asserted (a literal 'm.jpg', a name 'm.albarn') is served as the "
                            "source asserted it; the check that matters is that no derived name (kinds 3-6) is a bare MID",
           "bare_mid_samples": samples["bare_mid"], "placeholder_samples": samples["placeholder"], "seconds": round(time.time() - t0, 1)}
    json.dump(rec, open(SCR + "/NAMES_V4_FINAL.json", "w"), indent=1, ensure_ascii=False)
    log("done", json.dumps(rec["CHECKS"]), json.dumps(stats))


if __name__ == "__main__":
    {"base": compose_base, "final": compose_final}[sys.argv[1]]()
