"""Local dry-run of the NER shares-entity edge + two-sentence connecting-context builder that was added
to kg_relsig.py (--ner-edges). Mirrors the _run NER block EXACTLY (spaCy en_core_web_sm, 2<=df<=25, weight
1/df, rarest shared entity per edge, first sentence in each doc mentioning that entity). Prints degree stats
and sample edge contexts so we can eyeball quality BEFORE spending Modal. Runs on a doc subset for speed.
"""
import json, re, collections, sys
import spacy

MFILE = "data/processed/master_nodes_musique_clean.json"
NDOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
NER_CAP = 24

nodes = json.load(open(MFILE, encoding="utf-8"))
docs = [n for n in nodes if n.get("metadata", {}).get("type") == "document"]
docs = docs[:NDOCS]
content_by_row = {i: n.get("content", "") for i, n in enumerate(docs)}
idx2title = {i: n.get("metadata", {}).get("title", "").strip() for i, n in enumerate(docs)}
N = len(docs)
print(f"docs={N}")

_ENTL = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}
_splitre = re.compile(r"(?<=[.!?])\s+")
def _ssplit(t): return [s for s in _splitre.split(t) if s.strip()]
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

rows = sorted(content_by_row); ent_docs = collections.defaultdict(set); doc_ent_sent = {}
for r, doc in zip(rows, nlp.pipe([str(content_by_row[r]) for r in rows], batch_size=256)):
    ss = _ssplit(content_by_row[r]); esent = {}
    for e in doc.ents:
        if e.label_ in _ENTL and len(e.text) > 2:
            el = e.text.lower().strip(); ent_docs[el].add(r)
            if el not in esent:
                et = e.text.strip(); esent[el] = next((s for s in ss if et.lower() in s.lower()), et)
    doc_ent_sent[r] = esent

best = {}
for el, ds in ent_docs.items():
    df = len(ds)
    if 2 <= df <= 25:
        w = 1.0 / df; dl = sorted(ds)
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                key = (dl[i], dl[j]); cur = best.get(key)
                if cur is None or w > cur[0]:
                    best[key] = (w, el)

adjd = collections.defaultdict(list)
for (u, v), (w, el) in best.items():
    adjd[u].append((v, w, el)); adjd[v].append((u, w, el))
ner_nb = {u: sorted(adjd[u], key=lambda t: -t[1])[:NER_CAP] for u in adjd}

degs = [len(v) for v in ner_nb.values()]
n_iso = N - len(ner_nb)
print(f"NER undirected edges={len(best)}  entities-in-vocab={sum(1 for e,ds in ent_docs.items() if 2<=len(ds)<=25)}")
print(f"docs with >=1 NER edge={len(ner_nb)}  isolated={n_iso} ({100*n_iso/N:.1f}%)")
if degs:
    degs.sort()
    print(f"degree (post-cap {NER_CAP}): min={degs[0]} median={degs[len(degs)//2]} max={degs[-1]} mean={sum(degs)/len(degs):.1f}")

print("\n=== sample edge contexts (both connecting sentences) ===")
shown = 0
for (u, v), (w, el) in sorted(best.items(), key=lambda kv: -kv[1][0]):   # rarest-entity (highest weight) first
    sa = doc_ent_sent.get(u, {}).get(el, ""); sb = doc_ent_sent.get(v, {}).get(el, "")
    if not (sa and sb):
        continue
    print(f"\n[{shown}] shared_entity='{el}' (w={w:.3f})  {idx2title.get(u,'?')}  <->  {idx2title.get(v,'?')}")
    print(f"    A: {sa[:180]}")
    print(f"    B: {sb[:180]}")
    print(f"    ctx: {(sa + ' [SEP] ' + sb)[:240]}")
    shown += 1
    if shown >= 8:
        break

# how many edges have BOTH sentences resolvable (context non-empty)?
nboth = sum(1 for (u, v), (w, el) in best.items()
            if doc_ent_sent.get(u, {}).get(el) and doc_ent_sent.get(v, {}).get(el))
print(f"\nedges with both-sentence context resolvable: {nboth}/{len(best)} ({100*nboth/max(len(best),1):.1f}%)")
