"""V1 section 3 step 6 -- the pre-registered CVT encoding gate.

    python scratchpad/final_canonical_build/webqsp_v1/gold_cvt_gate.py

Rules frozen BEFORE rendering in data/final_canonical/webqsp/V1_CVT_GATE_PREREGISTRATION.json.
Nothing here may be tuned; the four cells are all emitted and exactly one of them is the gate:

    GATE = ENTITY_TEXT NAME_ONLY x MATCHING NORMALIZED -> GOLD_ONLY_IN_CVT_TEXT_N

Matching is plain substring containment via Aho-Corasick over 33,655 distinct gold strings,
streamed across every rendered document.  No cap anywhere, so containment cannot false-negative
through truncation.

Gold strings are read ONLY to measure coverage.  They never touched corpus membership, node
identity, or any rendered text: corpus <- source graph only, gold coverage <- measured afterward.
"""
import json, os, time, unicodedata
from collections import Counter

import ahocorasick
import pyarrow.parquet as pq

D = "data/final_canonical/webqsp/v1"
SRC = {"webqsp": "data/original/webqsp/rog_webqsp", "cwq": "data/original/cwq/rog_cwq"}
PREREG = "data/final_canonical/webqsp/V1_CVT_GATE_PREREGISTRATION.json"
OUT = "data/final_canonical/webqsp/V1_CVT_GATE_RESULT.json"


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s).casefold().strip().split())


def build_automaton(golds):
    A = ahocorasick.Automaton()
    for i, g in enumerate(golds):
        if g:
            A.add_word(g, i)
    A.make_automaton()
    return A


def scan(A, texts, n_gold):
    """-> set of gold indices found in any text"""
    hit = bytearray(n_gold)
    for t in texts:
        if not t:
            continue
        for _, i in A.iter(t):
            hit[i] = 1
    return {i for i, v in enumerate(hit) if v}


def main():
    t0 = time.time()
    prereg = json.load(open(PREREG, encoding="utf-8"))
    assert prereg["THE_GATE"]["cell"] == "ENTITY_TEXT = NAME_ONLY x MATCHING = NORMALIZED"

    # ---- gold ----
    gold, occ = set(), Counter()
    for d in SRC.values():
        for sh in sorted(f for f in os.listdir(d) if f.endswith(".parquet")):
            for a in pq.read_table(f"{d}/{sh}", columns=["answer"]).column("answer").to_pylist():
                for s in a:
                    gold.add(s)
                    occ[s] += 1
    gold = sorted(gold)
    G = len(gold)
    print(f"[gold] distinct={G:,} occurrences={sum(occ.values()):,} t={time.time()-t0:.0f}s",
          flush=True)

    ent = pq.read_table(f"{D}/entity_text.parquet",
                        columns=["text_name_only", "text_name_plus_facts"])
    name_only = ent.column("text_name_only").to_pylist()
    name_facts = ent.column("text_name_plus_facts").to_pylist()
    del ent
    cvt = pq.read_table(f"{D}/cvt_text.parquet", columns=["text_cvt_record"])
    cvt_text = cvt.column("text_cvt_record").to_pylist()
    del cvt
    print(f"[text] entity={len(name_only):,} cvt={len(cvt_text):,} t={time.time()-t0:.0f}s",
          flush=True)

    cells = {}
    for match in ("EXACT", "NORMALIZED"):
        g = gold if match == "EXACT" else [norm(x) for x in gold]
        # collapse duplicates created by normalization, keeping the mapping back to originals
        idx_of, uniq = {}, []
        back = []
        for x in g:
            j = idx_of.get(x)
            if j is None:
                j = idx_of[x] = len(uniq)
                uniq.append(x)
            back.append(j)
        A = build_automaton(uniq)
        prep = (lambda t: t) if match == "EXACT" else norm
        cvt_hit = scan(A, (prep(t) for t in cvt_text), len(uniq))
        print(f"[{match}] cvt scanned t={time.time()-t0:.0f}s", flush=True)

        for entity_def, texts in (("NAME_ONLY", name_only), ("NAME_PLUS_FACTS", name_facts)):
            ent_hit = scan(A, (prep(t) for t in texts), len(uniq))
            # map back to ORIGINAL distinct gold strings so GOLD_TOTAL_N is stable across cells
            in_e = {i for i, j in enumerate(back) if j in ent_hit}
            in_c = {i for i, j in enumerate(back) if j in cvt_hit}
            both = in_e & in_c
            only_e = in_e - in_c
            only_c = in_c - in_e
            neither = G - len(in_e | in_c)
            cells[f"{entity_def}__{match}"] = {
                "ENTITY_TEXT": entity_def, "MATCHING": match,
                "GOLD_TOTAL_N": G,
                "GOLD_IN_ENTITY_TEXT_N": len(in_e),
                "GOLD_IN_CVT_TEXT_N": len(in_c),
                "GOLD_IN_BOTH_N": len(both),
                "GOLD_ONLY_IN_ENTITY_TEXT_N": len(only_e),
                "GOLD_ONLY_IN_CVT_TEXT_N": len(only_c),
                "GOLD_IN_NEITHER_N": neither,
                "GOLD_ONLY_IN_CVT_TEXT_PCT": round(100.0 * len(only_c) / G, 4),
                "occurrence_weighted": {
                    "GOLD_ONLY_IN_CVT_TEXT_OCC": sum(occ[gold[i]] for i in only_c),
                    "GOLD_IN_NEITHER_OCC": sum(occ[gold[i]] for i in range(G)
                                               if i not in in_e and i not in in_c),
                    "TOTAL_OCC": sum(occ.values()),
                },
                "examples_only_in_cvt": [gold[i] for i in sorted(only_c)[:15]],
            }
            print(f"[{match}/{entity_def}] only_cvt={len(only_c):,} "
                  f"neither={neither:,} t={time.time()-t0:.0f}s", flush=True)

    gate = cells["NAME_ONLY__NORMALIZED"]
    doc = {
        "schema": "V1_CVT_GATE_RESULT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": PREREG,
        "preregistration_frozen_before_rendering": True,
        "GATE_CELL": "NAME_ONLY x NORMALIZED",
        "GATE_STATISTIC": "GOLD_ONLY_IN_CVT_TEXT_N",
        "GATE_VALUE": gate["GOLD_ONLY_IN_CVT_TEXT_N"],
        "GATE_PCT_OF_GOLD": gate["GOLD_ONLY_IN_CVT_TEXT_PCT"],
        "cells": cells,
        "INTERPRETATION_MATRIX_INPUTS": {
            "name_only_cvt_only": cells["NAME_ONLY__NORMALIZED"]["GOLD_ONLY_IN_CVT_TEXT_N"],
            "name_plus_facts_cvt_only":
                cells["NAME_PLUS_FACTS__NORMALIZED"]["GOLD_ONLY_IN_CVT_TEXT_N"],
        },
        "matching_rule": "plain substring containment; NORMALIZED = NFC -> casefold -> strip -> "
                         "collapse whitespace, applied identically to both sides",
        "no_caps_applied": True,
        "gold_never_entered_corpus_or_text": True,
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("GATE_CELL", "GATE_VALUE", "GATE_PCT_OF_GOLD",
                                          "INTERPRETATION_MATRIX_INPUTS", "cells")}, indent=1))


if __name__ == "__main__":
    main()
