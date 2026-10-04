"""Verify the official HotpotQA FullWiki abstracts corpus (withlinks-abstracts.tar.bz2).
Streams tar->per-member bz2->json lines. Counts articles, unique-title sample, hyperlinks (from
text_with_links <a href> tags), self-loops, duplicate links. edge_family = structural_native.
Memory: link uniqueness via int64-hash set (compact); counters otherwise. NO graph build, NO embeddings."""
import tarfile, bz2, json, re, time, os, hashlib
TB="data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
AHREF=re.compile(r'<a href="([^"]*)">')

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def norm(s):
    # hrefs are URL-encoded titles; compare on the raw decoded-ish token
    from urllib.parse import unquote
    return unquote(s).replace("_"," ").strip()

def run():
    t0=time.time()
    n_art=n_links=self_loops=0
    sampled=0; prev=set(); dupe_titles=0
    pair_hashes=set()   # int64 hashes of (title\ttarget) for duplicate detection
    n_members=0
    tar=tarfile.open(TB,"r:bz2")
    for member in tar:
        if not member.isfile(): continue
        n_members+=1
        data=tar.extractfile(member).read()
        try: text=bz2.decompress(data).decode("utf-8")
        except Exception: continue
        for line in text.splitlines():
            line=line.strip()
            if not line: continue
            r=json.loads(line); n_art+=1
            title=r.get("title","")
            if sampled<2000000:
                if title in prev: dupe_titles+=1
                else: prev.add(title)
                sampled+=1
            twl=r.get("text_with_links") or []
            # text_with_links: list of paragraphs, each a list of sentence strings with <a> tags
            for para in twl:
                seq=para if isinstance(para,list) else [para]
                for sent in seq:
                    if not isinstance(sent,str): continue
                    for m in AHREF.finditer(sent):
                        tgt=norm(m.group(1)); n_links+=1
                        if tgt==title: self_loops+=1
                        hv=hash(title+"\t"+tgt) & 0xFFFFFFFFFFFFFFFF
                        pair_hashes.add(hv)
        if n_members % 2000 == 0:
            print(f"  ...{n_members} members, {n_art} articles, {n_links} links, {time.time()-t0:.0f}s", flush=True)
    uniq_pairs=len(pair_hashes)
    out={"compressed_bytes":os.path.getsize(TB),"sha256":sha256(TB),"tar_members":n_members,
         "articles":n_art,"sampled":sampled,"dupe_titles_in_sample":dupe_titles,
         "total_hyperlinks":n_links,"approx_unique_directed_pairs":uniq_pairs,
         "duplicate_links":n_links-uniq_pairs,"self_loops":self_loops,"secs":round(time.time()-t0,1)}
    os.makedirs("results/data_audit",exist_ok=True)
    json.dump(out,open("results/data_audit/hotpot_fullwiki_corpus_stats.json","w"),indent=2)
    print(json.dumps(out)); print("DONE_HOTPOT_CORPUS")

if __name__=="__main__":
    run()
