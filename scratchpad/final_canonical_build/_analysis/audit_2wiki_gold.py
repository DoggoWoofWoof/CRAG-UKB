"""SECONDARY TASK -- gold-reference semantics for the five built canonical_v1 datasets.

Correct semantics (replaces build.py:344/564/570/580):
    gold ref -> {canonical node, ...}
    a ref RESOLVES iff it maps to >= 1 canonical node
    ALL_GOLD_REFS_RESOLVE  = every gold reference in every gold-bearing split resolves
    an AMBIGUOUS ref is DISJUNCTIVE: satisfied iff  retrieved & {alternatives} != {}
NEVER len(gold_node_ids) == len(gold_refs)  -- that test fails a question merely because one
of its references is ambiguous, which is what produced the stale ALL_EVAL_GOLDS_PRESENT=false.

2wiki is re-derived INDEPENDENTLY from the frozen corpus (title -> {node_id} multimap built by
one streaming scan of nodes.jsonl), and its single ambiguous title is disambiguated from
OFFICIAL SOURCE CONTEXT (each question's own context paragraph for that title).

Read-only.  Asserts 2wiki CORPUS_HASH before and after.
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = "C:/Users/Swastik/Desktop/CRAG"
FC = ROOT + "/data/final_canonical"
OUT = ROOT + "/scratchpad/final_canonical_build/_audit"
os.makedirs(OUT, exist_ok=True)

DATASETS = ["2wiki", "musique", "metaqa", "squad", "hotpotqa"]
# HISTORICAL CONSTANT -- do not "fix" it.  This script ran during the gold-semantics task, whose contract was that
# the fix must NOT move the corpus hash; asserting this literal is what proved that.  TEXTUALIZATION_REV 2
# (2026-09-06) then moved 2wiki's hash DELIBERATELY, by rewriting the text of 87,765 empty-body nodes, so re-running
# this script today will report a mismatch.  That mismatch is the expected outcome, not a regression: the live value
# is data/final_canonical/2wiki/build_info.json -> CORPUS_HASH, and the transition is recorded in
# data/final_canonical/_superseded_textualization_rev1/SUPERSEDED.md.
EXPECT_2WIKI_CORPUS_HASH = "81fa7d1a5d4bbb24b0925e8ec65a4b59cb2416e87d6c5b7c20f74ac41a6d76d4"   # rev1 (pre-REV2)
TITLE_KEY = '"title": "'
NID_KEY = '"node_id": "'


def corpus_hash(ds):
    return json.load(open(f"{FC}/{ds}/integrity_report.json", encoding="utf-8")).get("CORPUS_HASH")


def stream_json_array(path):
    dec = json.JSONDecoder()
    buf = ""
    with open(path, "r", encoding="utf-8") as fh:
        fh.read(1)
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            i = 0
            while True:
                j = buf.find("{", i)
                if j < 0:
                    break
                try:
                    obj, end = dec.raw_decode(buf, j)
                except ValueError:
                    break
                yield obj
                i = end
            buf = buf[i:]


def refs_of(rec):
    """Dataset-agnostic gold-reference key list, order-preserving and deduplicated."""
    out = []
    for r in rec.get("gold_refs") or []:
        k = json.dumps(r, sort_keys=True, ensure_ascii=False) if isinstance(r, (dict, list)) else str(r)
        if k not in out:
            out.append(k)
    return out


def build_title_index(nodes_path, needed):
    """title -> [node_id, ...] for the needed titles only. One streaming scan, no json.loads
    per line: 'title' is the last key in every record so it can be sliced directly."""
    idx = defaultdict(list)
    n = 0
    with open(nodes_path, "r", encoding="utf-8") as fh:
        for line in fh:
            n += 1
            ti = line.rfind(TITLE_KEY)
            if ti < 0:
                continue
            end = line.rindex('"}')
            raw = line[ti + len(TITLE_KEY):end]
            title = json.loads('"' + raw + '"') if "\\" in raw else raw
            if title not in needed:
                continue
            ni = line.find(NID_KEY)
            nid = line[ni + len(NID_KEY):line.index('"', ni + len(NID_KEY))]
            idx[title].append(nid)
    return idx, n


def rederive_2wiki():
    qdir = f"{FC}/2wiki/queries"
    per_split, needed = {}, set()
    recs_by_split = {}
    for fn in sorted(os.listdir(qdir)):
        if not fn.endswith(".jsonl"):
            continue
        rows = []
        for line in open(os.path.join(qdir, fn), encoding="utf-8"):
            r = json.loads(line)
            rr = refs_of(r)
            needed.update(rr)
            rows.append((r["query_id"], rr, list(r.get("gold_node_ids") or [])))
        recs_by_split[fn[:-6]] = rows
    print(f"  [2wiki] distinct gold titles needed = {len(needed):,}; scanning frozen corpus ...", flush=True)
    idx, n_nodes = build_title_index(f"{FC}/2wiki/nodes.jsonl", needed)
    print(f"  [2wiki] corpus lines scanned = {n_nodes:,}; titles matched = {len(idx):,}", flush=True)
    amb = {t: v for t, v in idx.items() if len(v) > 1}
    unmatched = sorted(needed - set(idx))

    for split, rows in recs_by_split.items():
        refs_tot = res = miss = ambn = 0
        q_all = q_any = q_none = q_with = 0
        mismatch = 0
        for qid, rr, nodes in rows:
            if not rr:
                continue
            q_with += 1
            got = 0
            for t in rr:
                cands = idx.get(t, [])
                refs_tot += 1
                if cands:
                    res += 1
                    got += 1
                    if len(cands) > 1:
                        ambn += 1
                else:
                    miss += 1
            # cross-check the frozen record against our independent re-resolution
            exp = sorted({c for t in rr for c in idx.get(t, [])})
            if exp != sorted(set(nodes)):
                mismatch += 1
            if got == len(rr):
                q_all += 1
            if got:
                q_any += 1
            else:
                q_none += 1
        per_split[split] = {"n_questions": len(rows), "n_with_gold_refs": q_with,
                            "GOLD_REFS_TOTAL": refs_tot, "GOLD_REFS_RESOLVED": res,
                            "MISSING_GOLD_REFS": miss, "AMBIGUOUS_GOLD_REFS": ambn,
                            "questions_all_refs_resolve": q_all,
                            "questions_any_ref_resolves": q_any,
                            "questions_no_ref_resolves": q_none,
                            "records_disagreeing_with_independent_rederivation": mismatch,
                            "ALL_GOLD_REFS_RESOLVE": refs_tot > 0 and miss == 0}
    return per_split, idx, amb, unmatched, len(needed)


def resolve_unconquered(idx):
    qids = {}
    for line in open(f"{FC}/2wiki/queries/train.jsonl", encoding="utf-8"):
        if "Unconquered" not in line:
            continue
        r = json.loads(line)
        if "Unconquered" in (r.get("gold_refs") or []):
            qids[r["query_id"]] = r["question"]
    cands = {}
    for line in open(f"{FC}/2wiki/nodes.jsonl", encoding="utf-8"):
        if "c62717110" in line or "c8073502" in line:
            r = json.loads(line)
            if r["node_id"] in ("2wiki:c62717110", "2wiki:c8073502"):
                cands[r["node_id"]] = r
                if len(cands) == 2:
                    break
    out = {}
    for rec in stream_json_array(f"{ROOT}/data/original/2wiki/v1.0_ids_april2021/train.json"):
        qid = rec.get("_id")
        if qid not in qids:
            continue
        para = None
        for e in rec.get("context") or []:
            if e and e[0] == "Unconquered":
                para = " ".join(e[1])
                break
        matches = []
        if para is not None:
            p = " ".join(para.split()).strip()
            for nid, nd in cands.items():
                t = " ".join((nd.get("text") or "").split()).strip()
                if t == p or t.startswith(p[:150]) or p.startswith(t[:150]):
                    matches.append(nid)
        out[qid] = {"question": qids[qid],
                    "official_context_paragraph": para,
                    "supporting_facts_for_title": [s for s in (rec.get("supporting_facts") or [])
                                                   if s and s[0] == "Unconquered"],
                    "evidences_for_title": [e for e in (rec.get("evidences") or [])
                                            if e and e[0] == "Unconquered"],
                    "matched_canonical_nodes": matches,
                    "candidates": sorted(idx.get("Unconquered", []))}
        if len(out) == len(qids):
            break
    return cands, out


def simple_stats(ds):
    """Per-reference stats for the datasets whose refs are structurally 1:1 (no title ambiguity
    exists: their integrity reports record ambiguous_titles=0 / no gold_title_ambiguous field)."""
    qdir = f"{FC}/{ds}/queries"
    per_split = {}
    for fn in sorted(os.listdir(qdir)):
        if not fn.endswith(".jsonl"):
            continue
        refs_tot = res = miss = ambn = 0
        q_all = q_any = q_none = q_with = n = 0
        for line in open(os.path.join(qdir, fn), encoding="utf-8"):
            r = json.loads(line)
            n += 1
            rr = refs_of(r)
            nodes = list(dict.fromkeys(r.get("gold_node_ids") or []))
            a = set(map(str, r.get("gold_title_ambiguous") or []))
            if not rr:
                continue
            q_with += 1
            refs_tot += len(rr)
            ambn += len(a & set(rr))
            # with no ambiguity recorded, |nodes| < |refs| is the only way a ref can be unresolved
            unres = max(0, len(rr) - len(nodes))
            miss += unres
            res += len(rr) - unres
            if unres == 0:
                q_all += 1
            if len(nodes):
                q_any += 1
            else:
                q_none += 1
        per_split[fn[:-6]] = {"n_questions": n, "n_with_gold_refs": q_with,
                              "GOLD_REFS_TOTAL": refs_tot, "GOLD_REFS_RESOLVED": res,
                              "MISSING_GOLD_REFS": miss, "AMBIGUOUS_GOLD_REFS": ambn,
                              "questions_all_refs_resolve": q_all,
                              "questions_any_ref_resolves": q_any,
                              "questions_no_ref_resolves": q_none,
                              "ALL_GOLD_REFS_RESOLVE": refs_tot > 0 and miss == 0}
    return per_split


def roll(per_split):
    g = [s for s, v in per_split.items() if v["GOLD_REFS_TOTAL"] > 0]
    return {"_SPLITS_WITH_GOLD": g,
            "ALL_GOLD_REFS_RESOLVE": bool(g) and all(per_split[s]["ALL_GOLD_REFS_RESOLVE"] for s in g),
            "MISSING_GOLD_REFS": sum(per_split[s]["MISSING_GOLD_REFS"] for s in g),
            "AMBIGUOUS_GOLD_REFS": sum(per_split[s]["AMBIGUOUS_GOLD_REFS"] for s in g),
            "GOLD_REFS_TOTAL": sum(per_split[s]["GOLD_REFS_TOTAL"] for s in g),
            "per_split": per_split}


def main():
    before = {ds: corpus_hash(ds) for ds in DATASETS}
    assert before["2wiki"] == EXPECT_2WIKI_CORPUS_HASH, ("2wiki CORPUS_HASH drift BEFORE", before["2wiki"])
    print(f"[assert] 2wiki CORPUS_HASH BEFORE = {before['2wiki'][:16]}... OK", flush=True)

    out = {"_semantics": "gold ref -> {canonical node,...}; a ref resolves iff it maps to >=1 node; "
                         "ALL_GOLD_REFS_RESOLVE = every ref in every gold-bearing split resolves; "
                         "ambiguous refs are DISJUNCTIVE (retrieved & {alternatives} != {}), never conjunctive.",
           "_supersedes": "build.py:344,564,570,580 -- ALL_EVAL_GOLDS_PRESENT used "
                          "len(gold_node_ids)==len(set(gold_refs)), which the directive forbids.",
           "_note": "This audit does NOT rewrite integrity_report.json; the corrected fields live here.",
           "CORPUS_HASH_BEFORE": before}

    print("[2wiki] independent re-derivation from the frozen corpus ...", flush=True)
    ps, idx, amb, unmatched, n_needed = rederive_2wiki()
    out["2wiki"] = roll(ps)
    out["2wiki"]["_independent_rederivation"] = {
        "titles_needed": n_needed, "titles_matched": len(idx),
        "titles_unmatched": len(unmatched), "unmatched_examples": unmatched[:10],
        "ambiguous_titles": len(amb), "ambiguous_map": {k: sorted(v) for k, v in amb.items()},
    }
    print(f"  [2wiki] ALL_GOLD_REFS_RESOLVE={out['2wiki']['ALL_GOLD_REFS_RESOLVE']} "
          f"MISSING={out['2wiki']['MISSING_GOLD_REFS']} AMBIGUOUS={out['2wiki']['AMBIGUOUS_GOLD_REFS']}", flush=True)

    print("[2wiki] disambiguating 'Unconquered' from official source context ...", flush=True)
    cands, unconq = resolve_unconquered(idx)
    n_uniq = sum(1 for v in unconq.values() if len(v["matched_canonical_nodes"]) == 1)
    out["2wiki_unconquered"] = {
        "candidates": {k: {"title": v.get("title"), "text_head": (v.get("text") or "")[:220]}
                       for k, v in cands.items()},
        "n_affected_questions": len(unconq),
        "AMBIGUITY_RESOLVED_BY_SOURCE": n_uniq,
        "AMBIGUITY_UNRESOLVED_BY_SOURCE": len(unconq) - n_uniq,
        "fallback_semantics": "disjunction over {2wiki:c62717110, 2wiki:c8073502} -- satisfied if "
                              "retrieved & alternatives != {}; NEVER require both",
        "per_question": unconq,
    }
    print(f"  [unconq] uniquely resolved from official source context: {n_uniq}/{len(unconq)}", flush=True)

    for ds in DATASETS:
        if ds == "2wiki":
            continue
        print(f"[gold] {ds} ...", flush=True)
        out[ds] = roll(simple_stats(ds))
        ir = json.load(open(f"{FC}/{ds}/integrity_report.json", encoding="utf-8"))
        out[ds]["_cross_check_build_report"] = {
            "unresolved_gold_refs": ir.get("unresolved_gold_refs"),
            "title_resolution": ir.get("query_stats", {}).get("_title_resolution"),
            "build_ALL_EVAL_GOLDS_PRESENT": ir.get("ALL_EVAL_GOLDS_PRESENT")}
        print(f"  {ds}: ALL_GOLD_REFS_RESOLVE={out[ds]['ALL_GOLD_REFS_RESOLVE']} "
              f"MISSING={out[ds]['MISSING_GOLD_REFS']} AMBIGUOUS={out[ds]['AMBIGUOUS_GOLD_REFS']}", flush=True)

    after = {ds: corpus_hash(ds) for ds in DATASETS}
    assert after == before, ("CORPUS_HASH drift AFTER", before, after)
    out["CORPUS_HASH_AFTER"] = after
    out["CORPUS_HASH_UNCHANGED"] = True
    print(f"[assert] 2wiki CORPUS_HASH AFTER = {after['2wiki'][:16]}... OK (unchanged)", flush=True)

    with open(OUT + "/gold_semantics.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("WROTE " + OUT + "/gold_semantics.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
