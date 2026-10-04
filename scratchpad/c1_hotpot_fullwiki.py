"""Phase C1 — canonical document manifest for HotpotQA FULLWIKI (official abstracts corpus).
Doc unit = one Wikipedia article abstract. id = official wiki curid. text = plain abstract sentences.
Streams tar->bz2->json; writes documents.jsonl incrementally (memory-safe: dedup on int curid only).
Writes data/canonical/hotpotqa/{documents.jsonl,document_manifest.json}. Read-only over official raw."""
import tarfile, bz2, json, os, hashlib, time
TB="data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
OUTD="data/canonical/hotpotqa"; os.makedirs(OUTD, exist_ok=True)
OUT=f"{OUTD}/documents.jsonl"

def flat_text(t):
    if isinstance(t,str): return t
    if isinstance(t,list):
        out=[]
        for x in t: out.append(flat_text(x))
        return " ".join(s for s in out if s)
    return ""

def run():
    t0=time.time()
    n=0; empty=0; dup_ids=0; seen=set(); title_sample=set(); dup_title_sample=0; sampled=0
    fout=open(OUT,"w",encoding="utf-8")
    tar=tarfile.open(TB,"r:bz2"); nm=0
    for member in tar:
        if not member.isfile(): continue
        nm+=1
        try: text=bz2.decompress(tar.extractfile(member).read()).decode("utf-8")
        except Exception: continue
        for line in text.splitlines():
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            cid=r.get("id"); title=r.get("title","")
            try: cidi=int(cid)
            except Exception: cidi=None
            if cidi is not None:
                if cidi in seen: dup_ids+=1; continue
                seen.add(cidi)
            body=flat_text(r.get("text") or r.get("text_with_links") or "")
            body=body.strip()
            if not body: empty+=1
            if sampled<2000000:
                if title in title_sample: dup_title_sample+=1
                else: title_sample.add(title)
                sampled+=1
            did=f"hotpot_{cid}"
            d={"dataset":"hotpotqa","canonical_doc_id":did,"original_source_id":str(cid),
               "title":title,"text":body,"text_sha256":hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "source_file":TB,"url":r.get("url"),
               "provenance":"official enwiki-20171001 withlinks-abstracts article"}
            fout.write(json.dumps(d,ensure_ascii=False)+"\n"); n+=1
        if nm % 5000 == 0:
            print(f"  ...{nm} members, {n} docs, {time.time()-t0:.0f}s", flush=True)
    fout.close()
    fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
    man={"dataset":"hotpotqa","setting":"fullwiki","doc_unit":"wiki abstract article",
         "n_docs":n,"unique_curids":len(seen),"duplicate_curids_skipped":dup_ids,
         "empty_docs":empty,"title_dup_in_sample":dup_title_sample,"title_sample":sampled,
         "expected_articles":5233329,"source_file":TB,
         "documents_jsonl_sha256":fsha,"text_policy":"plain abstract sentences (no <a> tags)",
         "provenance":"official enwiki-20171001 withlinks-abstracts (structural_native hyperlinks in raw)",
         "secs":round(time.time()-t0,1)}
    json.dump(man, open(f"{OUTD}/document_manifest.json","w"), indent=2)
    print(json.dumps({k:man[k] for k in ("n_docs","unique_curids","duplicate_curids_skipped","empty_docs","title_dup_in_sample","documents_jsonl_sha256")},indent=2))
    print("DONE_HOTPOT_C1")

if __name__=="__main__": run()
