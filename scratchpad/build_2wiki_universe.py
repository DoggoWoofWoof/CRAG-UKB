"""Full 2Wiki hyperlink UNIVERSE substrate (~5.99M articles) — permanent reusable, SEPARATE from the 398k retrieval view.
Doc unit = one Wikipedia article (para_with_hyperlink record), stable id = curid -> canonical_doc_id '2wu:<curid>'.
  C1: data/canonical/2wiki_universe/documents.jsonl  (+ document_manifest.json)
  C3: data/canonical/2wiki_universe/graph_structural.tsv  (native hyperlinks on curids, +graph_manifest.json)
  crosswalk_to_retrieval_view.json : title-based map universe<->398k canonical retrieval view (subset later, no re-encode)
Two passes over the zip. Curid is unique per article -> per-record target dedup suffices (no global edge set)."""
import zipfile, json, hashlib, os, time
ZIP="data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"
OUTD="data/canonical/2wiki_universe"; os.makedirs(OUTD,exist_ok=True)
DOCS=f"{OUTD}/documents.jsonl"; GRAPH=f"{OUTD}/graph_structural.tsv"
VIEW398="data/canonical/2wiki"

def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

# 398k retrieval view: title -> canonical_398k_id  (for crosswalk)
view_title2id={}
for line in open(f"{VIEW398}/documents.jsonl",encoding="utf-8"):
    d=json.loads(line); view_title2id[d["title"]]=d["canonical_doc_id"]
print(f"398k retrieval-view titles: {len(view_title2id)}",flush=True)

# ---------- PASS 1: documents.jsonl + curid set + crosswalk ----------
t0=time.time()
z=zipfile.ZipFile(ZIP)
curids=set(); dup_curids=0; n=0; empty=0
title_seen={}; title_collisions=0
xwalk={}   # title -> {"universe_doc_id":..., "retrieval_view_id":...}  (only titles present in the 398k view)
uni_titles_in_view=set()
fout=open(DOCS,"w",encoding="utf-8")
with z.open("para_with_hyperlink.jsonl") as f:
    for raw in f:
        n+=1
        try: r=json.loads(raw)
        except Exception: continue
        cid=int(r["id"]); title=r.get("title","")
        if cid in curids: dup_curids+=1;
        curids.add(cid)
        docid=f"2wu:{cid}"
        text=" ".join(r.get("sentences",[])).strip()
        if not text: empty+=1
        rec={"dataset":"2wiki_universe","canonical_doc_id":docid,"curid":cid,
             "title":title,"text":text,
             "text_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "source_file":"official 2wiki para_with_hyperlink.jsonl (full universe)",
             "provenance":"2wiki full hyperlink universe article (intro paragraph)"}
        fout.write(json.dumps(rec,ensure_ascii=False)+"\n")
        if title in title_seen: title_collisions+=1
        else: title_seen[title]=docid
        # crosswalk: this universe article's title also in 398k retrieval view?
        if title in view_title2id and title not in uni_titles_in_view:
            uni_titles_in_view.add(title)
            xwalk[title]={"universe_doc_id":docid,"retrieval_view_id":view_title2id[title]}
        if n%1000000==0: print(f"  P1 {n} recs, {len(curids)} curids, {time.time()-t0:.0f}s",flush=True)
fout.close()
print(f"P1 done: {n} recs, curids={len(curids)}, dup_curids={dup_curids}, empty_text={empty}, title_collisions={title_collisions}, {time.time()-t0:.0f}s",flush=True)

# ---------- PASS 2: native hyperlink graph on curids ----------
t1=time.time()
n2=0; n_edges=0; raw_ment=0; tgt_in=0; tgt_out=0; self_loops=0; no_refids=0
fg=open(GRAPH,"w",encoding="utf-8")
with z.open("para_with_hyperlink.jsonl") as f:
    for raw in f:
        n2+=1
        try: r=json.loads(raw)
        except Exception: continue
        cid=int(r["id"]); src=f"2wu:{cid}"
        local=set()
        for m in r.get("mentions",[]):
            rids=m.get("ref_ids") or []
            if not rids: no_refids+=1; continue
            for tid in rids:
                raw_ment+=1
                try: t=int(tid)
                except Exception: continue
                if t not in curids: tgt_out+=1; continue
                tgt_in+=1
                if t==cid: self_loops+=1; continue
                if t in local: continue
                local.add(t)
                fg.write(f"{src}\t2wu:{t}\thyperlink\n"); n_edges+=1
        if n2%1000000==0: print(f"  P2 {n2} recs, {n_edges} edges, {time.time()-t1:.0f}s",flush=True)
fg.close()
print(f"P2 done: edges={n_edges} tgt_in={tgt_in} tgt_out={tgt_out} self_loops={self_loops} no_refids={no_refids} {time.time()-t1:.0f}s",flush=True)

# ---------- manifests + crosswalk ----------
docs_sha=sha_file(DOCS); graph_sha=sha_file(GRAPH)
json.dump({"dataset":"2wiki_universe","doc_unit":"wikipedia article (para_with_hyperlink intro), curid-keyed",
   "n_docs":n,"unique_curids":len(curids),"dup_curids":dup_curids,"empty_text":empty,
   "title_collisions_difftext_possible":title_collisions,
   "id_scheme":"canonical_doc_id='2wu:<curid>'","documents_jsonl_sha256":docs_sha,
   "source_file":ZIP,"provenance":"full official 2wiki hyperlink universe (NOT the retrieval view)",
   "concept":"FULL_ENCODED_UNIVERSE","separate_from":"data/canonical/2wiki (398k EXPERIMENT_RETRIEVAL_VIEW)"},
   open(f"{OUTD}/document_manifest.json","w"),indent=2)
json.dump({"dataset":"2wiki_universe","edge_family":"structural_native","edge_subtype":"hyperlink",
   "graph_scope":"full_universe","provenance":"official 2wiki para_with_hyperlink ref_ids (curid->curid), internal to universe",
   "n_edges":n_edges,"n_relations":1,"raw_mentions":raw_ment,"targets_in_universe":tgt_in,
   "targets_out_of_universe":tgt_out,"self_loops_dropped":self_loops,"mentions_without_ref_ids":no_refids,
   "directed":True,"edge_schema":"src_docid\\tdst_docid\\thyperlink","graph_tsv_sha256":graph_sha,
   "source_file":ZIP},open(f"{OUTD}/graph_manifest.json","w"),indent=2)
json.dump({"dataset":"2wiki_universe","description":"title-based crosswalk between FULL universe and the 398k EXPERIMENT_RETRIEVAL_VIEW; subset the universe into the retrieval view without re-encoding",
   "retrieval_view":"data/canonical/2wiki (398354 title-dedup context paragraphs)",
   "n_retrieval_view_titles":len(view_title2id),
   "n_view_titles_found_in_universe":len(uni_titles_in_view),
   "n_view_titles_missing_from_universe":len(view_title2id)-len(uni_titles_in_view),
   "view_coverage_rate":round(len(uni_titles_in_view)/len(view_title2id),4),
   "note":"universe text = article intro (para_with_hyperlink); retrieval-view text = official context paragraph. Same identity (title), possibly different text surface. Map is by title.",
   "map":xwalk},open(f"{OUTD}/crosswalk_to_retrieval_view.json","w"),indent=2)
print(f"CROSSWALK: {len(uni_titles_in_view)}/{len(view_title2id)} retrieval-view titles found in universe")
print("DONE_2WIKI_UNIVERSE")
