"""Phase D1/D2 — dataset_manifest for the acquired OFFICIAL raw layer under data/original/.
Computes SHA256 of every official file + records source/version/counts/status. Read-only over data."""
import json, hashlib, os, glob, datetime

OUT = "results/data_audit"; os.makedirs(OUT, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

files = sorted(glob.glob("data/original/**/*", recursive=True))
hashes = {}
for p in files:
    if os.path.isfile(p) and not p.endswith(".zip"):   # hash extracted official files (skip the archive blobs)
        hashes[p.replace("\\","/")] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}

DATE = "2026-08-23"
DS = {
 "metaqa": {"source":"https://github.com/yuyuz/MetaQA (Zhang et al. 2018, AAAI)","version":"vanilla",
    "counts":{"1hop":{"train":96106,"dev":9992,"test":9947},"2hop":{"train":118980,"dev":14872,"test":14872},
              "3hop":{"train":114196,"dev":14274,"test":14274}},
    "official_test_labels":"public","native_graph":"kb.txt 134741 triples (subject|relation|object)",
    "match":True,"status":"OFFICIAL COMPLETE"},
 "webqsp": {"source":"Microsoft Download Center id=52763 (Yih et al. 2016) + rmanluo/RoG-webqsp (subgraphs)","version":"WebQSP 2016 + RoG",
    "counts":{"original":{"train":3098,"test":1639},"RoG":{"train":2826,"dev":246,"test":1628}},
    "answerable":{"original_train":3072,"original_test":1628},
    "official_test_labels":"public","native_graph":"Freebase relation triples via RoG per-question subgraph",
    "leakage_train_test":0,"match":True,"status":"OFFICIAL COMPLETE"},
 "2wiki": {"source":"https://github.com/Alab-NII/2wikimultihop (Ho et al. 2020); data_ids_april7.zip + para_with_hyperlink.zip","version":"v1.0 ids April 2021",
    "counts":{"train":167454,"dev":12576,"test":12576},
    "types":{"compositional":76481,"comparison":51963,"bridge_comparison":34631,"inference":4379},
    "official_test_labels":"HIDDEN (answers withheld)","native_graph":"para_with_hyperlink (Wikipedia hyperlinks) + evidences/supporting_facts",
    "leakage":{"train_dev":0,"train_test":0,"dev_test":0},"match":True,"status":"OFFICIAL COMPLETE"},
 "musique": {"source":"https://github.com/StonyBrookNLP/musique (Trivedi et al. 2022)","version":"MuSiQue-Ans v1.0",
    "counts":{"train":19938,"dev":2417,"test":2459},
    "hop_prefix_train":{"2hop":14376,"3hop":4387,"4hop":1175},
    "official_test_labels":"HIDDEN","native_graph":"none (question-specific paragraph sets; decomposition provided)",
    "leakage":{"train_dev":0,"train_test":0,"dev_test":0},"dup_ids":0,"match":True,
    "status":"OFFICIAL COMPLETE — CRAG derivation needed (global corpus)"},
 "hotpotqa": {"source":"authors' official HF release hotpotqa/hotpot_qa (CMU curtis host was DOWN; HF is the authors' org) (Yang et al. 2018)","version":"v1.1",
    "counts":{"distractor":{"train":90447,"dev":7405},"fullwiki":{"dev":7405,"test":7405}},
    "types":{"bridge":72991,"comparison":17456},"levels":{"easy":17972,"medium":56814,"hard":15661},
    "official_test_labels":"HIDDEN (fullwiki test: answers None, supporting_facts empty — verified)",
    "native_graph":"Wikipedia hyperlinks (in fullwiki corpus, NOT yet downloaded)",
    "leakage_train_dev":0,"match":True,
    "status":"OFFICIAL COMPLETE (distractor+fullwiki QA) — fullwiki Wikipedia corpus PENDING approval"},
 "squad": {"source":"https://rajpurkar.github.io/SQuAD-explorer/ (Rajpurkar et al. 2018)","version":"v2.0",
    "counts":{"train":130319,"dev":11873},
    "answerable":{"train":86821,"dev":5928},"unanswerable":{"train":43498,"dev":5945},
    "official_test_labels":"HIDDEN (no public labeled test; dev is the public eval set)","native_graph":"none",
    "leakage_train_dev":0,"match":True,"status":"OFFICIAL COMPLETE — CRAG derivation needed (answerable-retrieval layer)"},
}

manifest = {"_generated":DATE,"_policy":"official-fidelity; immutable originals under data/original/<dataset>/<version>/",
            "datasets":DS,"file_hashes":hashes,
            "pending_downloads":{"hotpotqa_fullwiki_wikipedia_corpus":
               {"why":"required only if fullwiki retrieval is the chosen setting",
                "file":"enwiki-20171001-pages-meta-current-withlinks-processed (~5 GB tar.bz2)",
                "host_note":"official CMU host curtis.ml.cmu.edu was UNREACHABLE this session; will need a mirror",
                "status":"HELD — awaiting user approval before pulling ~5 GB"}}}
json.dump(manifest, open(f"{OUT}/dataset_manifest.json","w"), indent=2)
print(f"hashed {len(hashes)} official files")
for ds,v in DS.items():
    print(f"{ds:10s} {v['status']}")
print(f"-> {OUT}/dataset_manifest.json")
