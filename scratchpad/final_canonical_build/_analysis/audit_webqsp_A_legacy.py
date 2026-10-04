"""SECTION A -- legacy WebQSP forensics.

Measures the two legacy WebQSP representations that exist on disk:
  L1  data/processed/master_nodes_webqsp.json   ("legacy 781k" substrate, commit ba6bd71 text)
  L2  data/canonical/webqsp/documents.jsonl     (Phase-C, 1,316,466 rows)

For each: LEGACY_N / MID_ONLY_N / READABLE_TEXT_N / MIXED_OR_OTHER_N, classified at BOTH
the identity level (what the node id / source endpoint is) and the text level (what the
frozen encoder actually saw).

Read-only.  Writes one JSON under scratchpad/final_canonical_build/_audit/.
"""
import glob
import json
import os
import re
import sys
from collections import Counter

OUT = "C:/Users/Swastik/Desktop/CRAG/scratchpad/final_canonical_build/_audit"
os.makedirs(OUT, exist_ok=True)

MID_FULL = re.compile(r"^[mg]\.[0-9a-z_]+$")          # loader_webqsp.py:34 verbatim
MID_ANY = re.compile(r"(?<![0-9A-Za-z_.])[mg]\.[0-9a-z_]{2,}")  # a MID token anywhere in text


def classify_text(t):
    if MID_FULL.match(t):
        return "MID_ONLY"
    return "MIXED_OR_OTHER" if MID_ANY.search(t) else "READABLE_TEXT"


def stream_master_nodes(path):
    """Stream the pretty-printed StandardNode array without loading 373 MB into objects."""
    dec = json.JSONDecoder()
    buf, depth, start = "", 0, None
    with open(path, "r", encoding="utf-8") as fh:
        fh.read(1)  # leading '['
        chunk = fh.read(1 << 20)
        while chunk:
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
            chunk = fh.read(1 << 20)


def rebuild_legacy_endpoints():
    """Reproduce loader_webqsp.py's node-id rule to recover each legacy node's source endpoint.

    loader_webqsp.py:118-123 defaults --paths to the two RoG *TEST* parquets, and
    :37-47 does: triples=set over df['graph']; ents = heads+tails; entities=sorted(ents);
    ent2nid[e] = f'{source}_doc_{i}'.  So the endpoint of webqsp_doc_i is sorted(ents)[i].
    """
    import pandas as pd
    paths = sorted(glob.glob("C:/Users/Swastik/Desktop/CRAG/data/original/webqsp/rog_webqsp/test-*.parquet"))
    ents, triples = set(), set()
    for p in paths:
        df = pd.read_parquet(p, columns=["graph"])
        for g in df["graph"]:
            for t in g:
                triples.add((str(t[0]), str(t[1]), str(t[2])))
        del df
    for h, r, tl in triples:
        ents.add(h)
        ents.add(tl)
    return sorted(ents), len(triples), paths


def main():
    res = {}

    # ---------- L1: legacy 781k substrate ----------
    print("[A] rebuilding legacy endpoint order from RoG TEST parquets ...", flush=True)
    entities, n_tr, paths = rebuild_legacy_endpoints()
    print(f"    test-only entities={len(entities)} triples={n_tr}", flush=True)

    p1 = "C:/Users/Swastik/Desktop/CRAG/data/processed/master_nodes_webqsp.json"
    id_cls, txt_cls, title_cls = Counter(), Counter(), Counter()
    n_doc = n_q = 0
    endpoint_match = endpoint_checked = 0
    deduced_examples = []
    empty_name = 0
    for nd in stream_master_nodes(p1):
        md = nd.get("metadata", {})
        if md.get("type") == "question":
            n_q += 1
            continue
        n_doc += 1
        nid = nd["node_id"]
        idx = int(nid.rsplit("_", 1)[1])
        ep = entities[idx] if idx < len(entities) else None
        content = nd.get("content", "")
        title = md.get("title", "")
        if ep is not None:
            endpoint_checked += 1
            # non-MID endpoints keep the endpoint verbatim as their title
            if not MID_FULL.match(ep):
                endpoint_match += int(title == ep)
            id_cls["FREEBASE_MID" if MID_FULL.match(ep) else "HUMAN_READABLE_SURFACE"] += 1
            if MID_FULL.match(ep) and len(deduced_examples) < 8:
                deduced_examples.append({"endpoint": ep, "legacy_title": title,
                                         "legacy_text_head": content[:160]})
        txt_cls[classify_text(content)] += 1
        title_cls[classify_text(title) if title else "EMPTY"] += 1
        if title == "":
            empty_name += 1
    res["L1_legacy_781k"] = {
        "path": p1,
        "source_parquets": paths,
        "source_scope": "RoG-webqsp TEST split only (loader_webqsp.py:118-123 default --paths)",
        "LEGACY_N": n_doc,
        "question_nodes": n_q,
        "endpoint_recovery": {"checked": endpoint_checked, "title==endpoint for non-MID": endpoint_match},
        "identity_level": dict(id_cls),
        "text_level": {"MID_ONLY_N": txt_cls["MID_ONLY"],
                       "READABLE_TEXT_N": txt_cls["READABLE_TEXT"],
                       "MIXED_OR_OTHER_N": txt_cls["MIXED_OR_OTHER"]},
        "title_level": dict(title_cls),
        "empty_title_n": empty_name,
        "deduced_examples": deduced_examples,
    }
    print(f"    L1 docs={n_doc} q={n_q} text={dict(txt_cls)} id={dict(id_cls)}", flush=True)

    # ---------- L2: Phase-C 1,316,466 ----------
    p2 = "C:/Users/Swastik/Desktop/CRAG/data/canonical/webqsp/documents.jsonl"
    id_cls2, txt_cls2 = Counter(), Counter()
    n2 = 0
    text_eq_id = 0
    with open(p2, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            n2 += 1
            ep = d["original_source_id"]
            id_cls2["FREEBASE_MID" if MID_FULL.match(ep) else "HUMAN_READABLE_SURFACE"] += 1
            txt_cls2[classify_text(d["text"])] += 1
            text_eq_id += int(d["text"] == ep)
    res["L2_phaseC_1316466"] = {
        "path": p2,
        "source_scope": "RoG-webqsp ALL splits (c1c2_webqsp.py:13 glob) + injected a_entity/q_entity",
        "LEGACY_N": n2,
        "identity_level": dict(id_cls2),
        "text_level": {"MID_ONLY_N": txt_cls2["MID_ONLY"],
                       "READABLE_TEXT_N": txt_cls2["READABLE_TEXT"],
                       "MIXED_OR_OTHER_N": txt_cls2["MIXED_OR_OTHER"]},
        "text_identical_to_endpoint_n": text_eq_id,
    }
    print(f"    L2 docs={n2} text={dict(txt_cls2)} id={dict(id_cls2)} text==id:{text_eq_id}", flush=True)

    with open(OUT + "/A_legacy.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("WROTE " + OUT + "/A_legacy.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
