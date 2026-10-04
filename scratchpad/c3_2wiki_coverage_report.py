"""Freeze-guard #1 — explicit 2Wiki 5.99M hyperlink-universe -> 398k canonical-corpus mapping/edge coverage.
Re-scans para_with_hyperlink.jsonl once using the SAME normalization as c3_2wiki_graph.py and reports:
  (A) universe->corpus record mapping  (B) edge target-in vs target-out (dangling)  (C) node coverage of the corpus.
Does NOT rebuild the graph (read-only report). Writes data/canonical/2wiki/graph_coverage_report.json."""
import json, zipfile, time
OUTD="data/canonical/2wiki"
ZIP="data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"

title2id={}
for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
    d=json.loads(line); title2id[d["title"]]=d["canonical_doc_id"]
corpus=set(title2id.values())
print(f"corpus titles: {len(title2id)}",flush=True)

t0=time.time()
n_rec=0; src_in=0; src_out=0
raw_mentions=0; tgt_in=0; tgt_out=0; self_loops=0
uniq_edges=set(); danglers=set()
src_nodes=set(); dst_nodes=set()
z=zipfile.ZipFile(ZIP)
with z.open("para_with_hyperlink.jsonl") as f:
    for raw in f:
        n_rec+=1
        try: r=json.loads(raw)
        except Exception: continue
        src=title2id.get(r.get("title"))
        if src is None:
            src_out+=1
            if n_rec%1000000==0: print(f"  ...{n_rec} recs {time.time()-t0:.0f}s",flush=True)
            continue
        src_in+=1
        for m in r.get("mentions",[]):
            ref=m.get("ref_url")
            if not ref: continue
            raw_mentions+=1
            tgt=title2id.get(ref.replace("_"," "))
            if tgt is None:
                tgt_out+=1; danglers.add(ref.replace("_"," ")); continue
            tgt_in+=1
            if tgt==src: self_loops+=1; continue
            uniq_edges.add((src,tgt)); src_nodes.add(src); dst_nodes.add(tgt)
        if n_rec%1000000==0: print(f"  ...{n_rec} recs {time.time()-t0:.0f}s",flush=True)

nodes_touched=src_nodes|dst_nodes
rep={
 "dataset":"2wiki",
 "purpose":"explicit hyperlink-universe -> canonical-corpus mapping & edge coverage (freeze-guard)",
 "normalization":"src=title2id[title]; dst=title2id[ref_url.replace('_',' ')]; keep both-in-corpus, drop self-loops, dedup",
 "A_universe_record_mapping":{
   "total_records_scanned":n_rec,
   "source_records_in_corpus":src_in,
   "source_records_out_of_corpus":src_out,
   "note":"out-of-corpus source records are hyperlink-universe paragraphs NOT in the canonical retrieval corpus; correctly excluded (universe is NOT the retrieval corpus)"},
 "B_edge_target_coverage":{
   "raw_hyperlink_mentions_on_incorpus_sources":raw_mentions,
   "mentions_target_in_corpus":tgt_in,
   "mentions_target_out_of_corpus_dangling":tgt_out,
   "unique_dangling_target_titles":len(danglers),
   "self_loops_dropped":self_loops,
   "unique_directed_edges_kept":len(uniq_edges),
   "target_in_corpus_rate":round(tgt_in/max(raw_mentions,1),4)},
 "C_corpus_node_coverage":{
   "corpus_titles":len(title2id),
   "titles_as_edge_source":len(src_nodes),
   "titles_as_edge_target":len(dst_nodes),
   "titles_touched_by_any_edge":len(nodes_touched),
   "isolated_titles_degree0":len(title2id)-len(nodes_touched),
   "node_coverage_rate":round(len(nodes_touched)/len(title2id),4)},
 "secs":round(time.time()-t0,1)}
json.dump(rep,open(f"{OUTD}/graph_coverage_report.json","w"),indent=2)
print(json.dumps(rep,indent=2))
print("DONE_2WIKI_COVERAGE")
