"""
Build a LABEL-FREE, deduplicated substrate for 2wiki/musique (leak fix).
========================================================================
The audit found that on 2wiki/musique the ONLY doc-doc edges in node.neighbors
are "bridge edges" linking the co-supporting GOLD docs of each question — built
over ALL questions before the split, so overlap/traversal read test labels.
Corpus is also per-question copies (150k docs, 64k unique titles on 2wiki).

This produces a clean source "{ds}_clean":
  - dedup docs by (title, content-hash) -> canonical passage set
  - doc.neighbors = LABEL-FREE title-mention links only (efficient first-token
    index; the indexer adds its own kNN edges on top). NO gold bridges, NO
    doc->question backedges (so degree-weighting is label-free too).
  - question.neighbors = remapped gold doc ids (the LABELS — used only for
    split/eval, never as doc-doc structure).
Writes data/processed/master_nodes_{ds}_clean.json. Then run the indexer:
  build_all(master_nodes_path=that file, target_datasets=['{ds}_clean'], skip_colbert=True)
"""
import os
import re
import json
import time
import hashlib
import logging
import argparse
from collections import defaultdict

log = logging.getLogger("pipeline.build_clean")

MASTER = "data/processed/master_nodes.json"


def _title_links_slow(docs):
    """Reference implementation (kept for the equivalence test). Undirected label-free
    edges: doc A -> doc B if A's content contains B's title. First-token index."""
    titled = [(d["metadata"].get("title", ""), d["node_id"]) for d in docs if d["metadata"].get("title")]
    first_tok = defaultdict(list)
    for t, nid in titled:
        tlo = t.lower()
        if tlo:
            first_tok[tlo.split(" ", 1)[0]].append((tlo, nid))
    id2doc = {d["node_id"]: d for d in docs}
    edges = 0
    for d in docs:
        cl = d["content"].lower()
        for tk in set(re.findall(r"[a-z0-9]+", cl)):
            for tlo, nid in first_tok.get(tk, ()):
                if nid != d["node_id"] and tlo in cl:
                    if nid not in d["neighbors"]:
                        d["neighbors"].append(nid); edges += 1
                    other = id2doc[nid]
                    if d["node_id"] not in other["neighbors"]:
                        other["neighbors"].append(d["node_id"])
    return edges


_TOK = re.compile(r"[a-z0-9]+")


def _title_links(docs):
    """Undirected label-free edges: doc A <-> doc B if A's content contains B's title.

    Produces the SAME edge set as `_title_links_slow` but scales to 500k+ docs: the
    reference version indexes titles by their FIRST word, so common Wikipedia first-words
    ("the", "list", "united"…) form huge candidate buckets that every doc rescans — O(hours)
    at 500k docs. This version indexes each title by its RAREST token (small buckets) and
    replicates the reference's exact reachability quirk (a title is only linkable when its
    first space-delimited word is itself a pure [a-z0-9]+ token), so the two are provably
    identical: an edge exists iff (first-word is a clean token) AND (title ⊆ content), and
    a title's rarest token is always present when the title is a substring. Verified by an
    exact edge-set comparison on squad + musique (scripts/verify_titlelinks)."""
    from collections import Counter
    df = Counter()
    entries = []                                              # (title_lower, frozenset(tokens), node_id)
    for d in docs:
        t = d["metadata"].get("title", "")
        if not t:
            continue
        tlo = t.lower()
        first = tlo.split(" ", 1)[0]
        if not _TOK.fullmatch(first):                        # reference quirk: unreachable -> never linked
            continue
        toks = _TOK.findall(tlo)
        if not toks:
            continue
        fs = frozenset(toks)
        entries.append((tlo, fs, d["node_id"]))
        for w in fs:
            df[w] += 1
    index = defaultdict(list)
    for tlo, fs, nid in entries:
        key = min(fs, key=lambda w: (df[w], w))              # rarest token (deterministic tiebreak) -> small buckets
        index[key].append((tlo, fs, nid))

    id2doc = {d["node_id"]: d for d in docs}
    edges = 0
    n = len(docs)
    for i, d in enumerate(docs):
        cl = d["content"].lower()
        ctoks = set(_TOK.findall(cl))
        matched = []
        seen = set()
        for tk in ctoks:
            for tlo, fs, nid in index.get(tk, ()):
                if nid == d["node_id"] or nid in seen:
                    continue
                if fs <= ctoks and tlo in cl:                # necessary-condition prefilter, then substring
                    seen.add(nid)
                    matched.append(nid)
        for nid in matched:
            if nid not in d["neighbors"]:
                d["neighbors"].append(nid); edges += 1
            other = id2doc[nid]
            if d["node_id"] not in other["neighbors"]:
                other["neighbors"].append(d["node_id"])
        if (i + 1) % 50000 == 0:
            log.info(f"  title-links: {i+1}/{n} docs, {edges} edges so far")
    return edges


def build_clean(ds, master=MASTER):
    t0 = time.time()
    all_nodes = json.load(open(master, encoding="utf-8"))
    src_nodes = [n for n in all_nodes if n.get("metadata", {}).get("source") == ds]
    docs = [n for n in src_nodes if n["metadata"].get("type") != "question"]
    qs = [n for n in src_nodes if n["metadata"].get("type") == "question"]
    log.info(f"{ds}: {len(docs)} docs, {len(qs)} questions loaded")

    # dedup docs by (title, content-hash) -> canonical clean node
    canon_key_to_id, id_remap, clean_docs = {}, {}, []
    for d in docs:
        h = hashlib.md5(d["content"].encode("utf-8")).hexdigest()
        key = (d["metadata"].get("title", ""), h)
        if key not in canon_key_to_id:
            cid = f"{ds}_clean_doc_{len(clean_docs)}"
            canon_key_to_id[key] = cid
            clean_docs.append({
                "node_id": cid, "content": d["content"],
                "metadata": {"source": f"{ds}_clean", "type": "document",
                             "title": d["metadata"].get("title", "")},
                "neighbors": [],
            })
        id_remap[d["node_id"]] = canon_key_to_id[key]
    log.info(f"  deduped {len(docs)} -> {len(clean_docs)} canonical docs")

    # label-free doc-doc edges (title mentions)
    n_edges = _title_links(clean_docs)
    deg = sum(len(d["neighbors"]) for d in clean_docs) / max(len(clean_docs), 1)
    log.info(f"  title-link edges: {n_edges} directed adds, avg doc degree {deg:.2f} "
             f"({sum(1 for d in clean_docs if d['neighbors'])/len(clean_docs)*100:.0f}% docs linked)")

    # questions: keep gold edges as LABELS only, remapped to canonical doc ids
    clean_qs, dropped = [], 0
    for q in qs:
        golds = sorted({id_remap[nb] for nb in q["neighbors"] if nb in id_remap})
        if not golds:
            dropped += 1; continue
        evaluation_metadata = {
            key: q["metadata"][key]
            for key in ("answer", "answers", "answer_aliases", "is_impossible", "split")
            if key in q["metadata"]
        }
        clean_qs.append({
            "node_id": q["node_id"].replace(f"{ds}_q", f"{ds}_clean_q", 1),
            "content": q["content"],
            "metadata": {
                "source": f"{ds}_clean",
                "type": "question",
                **evaluation_metadata,
            },
            "neighbors": golds,
        })
    ng = sum(len(q["neighbors"]) for q in clean_qs) / max(len(clean_qs), 1)
    log.info(f"  questions: {len(clean_qs)} kept ({dropped} dropped, no gold), avg golds/q {ng:.2f}")

    out = clean_docs + clean_qs
    out_path = f"data/processed/master_nodes_{ds}_clean.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    log.info(f"  saved {len(out)} nodes -> {out_path}  ({time.time()-t0:.1f}s)")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Build label-free deduped substrate for leaked datasets.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    a = p.parse_args(argv)
    for ds in a.datasets:
        build_clean(ds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
