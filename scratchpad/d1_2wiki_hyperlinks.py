"""Memory-safe streaming count of official 2Wiki para_with_hyperlink native hyperlink graph (7GB).
Pure counters (O(1) memory) — exact unique-pair dedup deferred to the D5 graph build (out-of-core)."""
import zipfile, json, time, os
Z="data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"
t0=time.time()
n_para=n_ment=n_ref=self_loops=title_repeat_check=0
prev_titles=set()  # small bounded sample to test if titles are unique per record
sample_dupe=0; sampled=0
z=zipfile.ZipFile(Z)
with z.open("para_with_hyperlink.jsonl") as f:
    for raw in f:
        r=json.loads(raw); n_para+=1
        title=r.get("title")
        if sampled < 2000000:   # bounded uniqueness sample (~memory-safe)
            if title in prev_titles: sample_dupe+=1
            else: prev_titles.add(title)
            sampled+=1
        for m in r.get("mentions",[]):
            n_ment+=1
            tgt=m.get("ref_url")
            if tgt is not None:
                n_ref+=1
                if tgt==title: self_loops+=1
        if n_para % 1000000 == 0:
            print(f"  ...{n_para} paras, {n_ment} mentions, {time.time()-t0:.0f}s", flush=True)
out={"paragraphs":n_para,"total_mentions":n_ment,"mentions_with_ref":n_ref,"self_loops":self_loops,
     "sampled_records":sampled,"dupe_titles_in_sample":sample_dupe,"secs":round(time.time()-t0,1)}
os.makedirs("results/data_audit",exist_ok=True)
json.dump(out,open("results/data_audit/2wiki_hyperlink_stats.json","w"),indent=2)
print(json.dumps(out)); print("DONE_2WIKI_HYPERLINKS")
