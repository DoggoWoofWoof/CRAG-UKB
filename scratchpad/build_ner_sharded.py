"""Sharded/resumable EXACT reproduction of the established CRAG NER 'shares-entity' graph, for large corpora
(2wiki_universe ~5.99M, hotpotqa ~5.23M) whose postings don't fit the in-RAM src/pipeline/ner_edges path.

Bit-for-bit faithful to src.pipeline.ner_edges.build_ner_edges:
  spaCy en_core_web_sm (disable parser,lemmatizer) · ENT_LABELS (imported, same 9) · full text ·
  per-doc dedup (set) · key = e.text.lower().strip() with len(e.text)>2 · global df filter 2<=df<=25 ·
  weight w[(a,b)] += 1/df summed over every shared qualifying entity · undirected.
Only difference vs the in-RAM builder: pair orientation is canonicalized by docid string order (undirected —
edge SET and WEIGHTS identical); endpoints are canonical_doc_id (same as the small-corpus canonical tsvs).

Stages (resumable):
  extract  : per _src/docs shard -> work/postings/shard_NNNNN.tsv  (entity\x01docid, sorted)   [expensive, spaCy]
  merge    : k-way entity merge (hub cutoff <=26 docs in RAM) -> bucketed pair partials -> per-bucket sum
             -> graph_ner.tsv (docid\tdocid\t%.6f) + ner_manifest.json
Usage:
  python build_ner_sharded.py --dataset 2wiki_universe --stage extract --shards 0-40
  python build_ner_sharded.py --dataset 2wiki_universe --stage merge
"""
import os, sys, json, glob, time, heapq, hashlib, argparse, itertools
from collections import defaultdict
sys.path.insert(0, os.path.abspath("."))
from src.pipeline.ner_edges import ENT_LABELS          # SAME 9-label whitelist, imported (no drift)

SEP = "\x01"; MAXDF = 25; BUCKETS = 64
CANON_ROOT = "data/canonical"                          # override with --canon-root for non-destructive validation
def canon_dir(ds): return f"{CANON_ROOT}/{ds}"
def src_dir(ds):   return f"{canon_dir(ds)}/encodings/_src/docs"
def work_dir(ds):  return f"{canon_dir(ds)}/_ner_work"
def post_dir(ds):  return f"{work_dir(ds)}/postings"
def bkt_dir(ds):   return f"{work_dir(ds)}/buckets"

_NLP = None
def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])   # identical to ner_edges._nlp
    return _NLP

def n_shards(ds):
    m = json.load(open(f"{src_dir(ds)}/_srcmeta.json")); return m["n_shards"]

def read_src_shard(ds, sid):
    ids, texts = [], []
    for line in open(f"{src_dir(ds)}/shard_{sid:05d}.jsonl", encoding="utf-8"):
        r = json.loads(line); ids.append(r["id"]); texts.append(r["text"])
    return ids, texts

# ── extract: spaCy NER over one shard -> sorted postings (resumable, atomic) ──
def extract_shard(ds, sid):
    outp = f"{post_dir(ds)}/shard_{sid:05d}.tsv"
    donef = f"{post_dir(ds)}/shard_{sid:05d}.done"
    if os.path.exists(donef) and os.path.exists(outp):
        return "skip"
    ids, texts = read_src_shard(ds, sid)
    rows = []
    for i, doc in enumerate(nlp().pipe([str(t) for t in texts], batch_size=256)):
        seen = set()                                              # per-doc dedup (== set() in ner_edges)
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) > 2:
                k = e.text.lower().strip()
                if k in seen: continue
                seen.add(k)
                # json-encode the entity key: bijective + single-line (entities may contain \n/\t);
                # preserves exact identity (never merges distinct entities). docid is a clean token.
                rows.append(json.dumps(k, ensure_ascii=False) + "\t" + ids[i])
    rows.sort()                                                   # sort by full line (entity-major)
    tmp = outp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
        if rows: f.write("\n")
    os.replace(tmp, outp)
    open(donef, "w").write(str(len(rows)))
    return f"{len(ids)} docs, {len(rows)} postings"

def extract(ds, shards):
    os.makedirs(post_dir(ds), exist_ok=True)
    t0 = time.time(); tot = 0
    for sid in shards:
        r = extract_shard(ds, sid); tot += 1
        print(f"[ner-extract] {ds} shard {sid}: {r}  ({time.time()-t0:.0f}s, {tot}/{len(shards)})", flush=True)

# ── merge: entity k-way -> pair buckets -> per-bucket sum -> graph_ner.tsv ──
def _line_iter(p):
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln: yield ln

def merge(ds):
    ns = n_shards(ds)
    posts = [f"{post_dir(ds)}/shard_{s:05d}.tsv" for s in range(ns)]
    miss = [p for p in posts if not os.path.exists(p)]
    if miss:
        print(f"[ner-merge] ABORT: {len(miss)} posting shards missing (run extract first): {miss[:3]}...", flush=True)
        return 1
    os.makedirs(bkt_dir(ds), exist_ok=True)
    for p in glob.glob(f"{bkt_dir(ds)}/bkt_*.tsv"): os.remove(p)
    bfh = [open(f"{bkt_dir(ds)}/bkt_{b:03d}.tsv", "w", encoding="utf-8") for b in range(BUCKETS)]

    t0 = time.time(); merged = heapq.merge(*[_line_iter(p) for p in posts])
    n_ent = 0; n_qual = 0; n_pairs = 0; n_hub = 0; n_single = 0
    for key, grp in itertools.groupby(merged, key=lambda ln: ln.split("\t", 1)[0]):
        n_ent += 1
        docs = set(); hub = False
        for ln in grp:                                            # exhaust group; stop STORING past maxdf
            if hub: continue
            d = ln.split("\t", 1)[1]; docs.add(d)
            if len(docs) > MAXDF: hub = True
        if hub: n_hub += 1; continue
        df = len(docs)
        if df < 2: n_single += 1; continue
        n_qual += 1
        wt = 1.0 / df
        dl = sorted(docs)
        for a in range(len(dl)):
            for b in range(a + 1, len(dl)):
                bfh[hash(dl[a]) % BUCKETS].write(f"{dl[a]}\t{dl[b]}\t{wt}\n"); n_pairs += 1
        if n_ent % 2000000 == 0:
            print(f"[ner-merge] {n_ent} entities, {n_qual} qual, {n_pairs} pair-emits, {time.time()-t0:.0f}s", flush=True)
    for f in bfh: f.close()
    print(f"[ner-merge] entities={n_ent} qual={n_qual} hub(df>25)={n_hub} single(df<2)={n_single} pair_emits={n_pairs} {time.time()-t0:.0f}s", flush=True)

    # per-bucket sum -> final tsv (sorted within bucket for determinism)
    outp = f"{canon_dir(ds)}/graph_ner.tsv"; tmp = outp + ".tmp"
    n_edges = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for b in range(BUCKETS):
            acc = defaultdict(float)
            for ln in _line_iter(f"{bkt_dir(ds)}/bkt_{b:03d}.tsv"):
                a, c, w = ln.split("\t"); acc[(a, c)] += float(w)
            for (a, c) in sorted(acc):
                out.write(f"{a}\t{c}\t{acc[(a,c)]:.6f}\n"); n_edges += 1
    os.replace(tmp, outp)
    h = hashlib.sha256()
    with open(outp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    man = {"dataset": ds, "edge_family": "ner", "edge_subtype": "shares_entity", "ner_available": True,
           "graph_scope": "full_universe" if ds.endswith("_universe") else "full_corpus",
           "algorithm": "src/pipeline/ner_edges.build_ner_edges (established CRAG NER) — sharded/resumable exact reproduction",
           "ner_model": "spacy en_core_web_sm (disable parser,lemmatizer)",
           "spacy_version": _ver("spacy"), "model_version": _ver("en_core_web_sm"),
           "ent_labels": sorted(ENT_LABELS), "df_min": 2, "df_max": MAXDF,
           "weight": "sum 1/df over shared entities", "min_ent_char": ">2 (len(e.text)>2)",
           "entity_norm": "e.text.lower().strip()", "full_text": True, "n_edges": n_edges,
           "edge_schema": "src_id\\tdst_id\\tweight", "graph_ner_tsv_sha256": h.hexdigest(),
           "directed": False, "pair_orientation": "canonicalized by docid string order (undirected; edge set+weights identical to in-RAM builder)",
           "provenance": "spaCy en_core_web_sm NER over canonical documents.jsonl; shared named-entity links, df-weighted 1/df (established CRAG NER, no LLM)",
           "secs": round(time.time() - t0, 1)}
    json.dump(man, open(f"{canon_dir(ds)}/ner_manifest.json", "w"), indent=2)
    print(f"[ner-merge] DONE {ds}: n_edges={n_edges} sha={h.hexdigest()[:16]}", flush=True)
    return 0

def _ver(mod):
    try:
        import importlib; return getattr(importlib.import_module(mod), "__version__", "?")
    except Exception: return "?"

def parse_shards(spec, ns):
    if spec in (None, "all"): return list(range(ns))
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return [s for s in out if 0 <= s < ns]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", choices=["extract", "merge"], required=True)
    ap.add_argument("--shards", default="all")
    ap.add_argument("--canon-root", default="data/canonical")
    a = ap.parse_args()
    CANON_ROOT = a.canon_root
    ns = n_shards(a.dataset)
    if a.stage == "extract":
        extract(a.dataset, parse_shards(a.shards, ns))
    else:
        sys.exit(merge(a.dataset))
