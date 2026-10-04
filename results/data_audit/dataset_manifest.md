# CRAG Official Raw Layer — Acquisition Manifest (Phase D1)

_Generated 2026-08-23. Immutable originals live under `data/original/<dataset>/<version>/`. Every count below was verified against the downloaded official files this session; SHA256 for all 58 extracted official files is in `dataset_manifest.json`._

## Acquisition table

| Dataset | Official source | Official counts (tr/dev/test) | Local verified | Exact? | Native structural graph | Test labels | Status |
|---|---|---|---|---|---|---|---|
| **MetaQA** | github.com/yuyuz/MetaQA | 1h 96106/9992/9947 · 2h 118980/14872/14872 · 3h 114196/14274/14274 | all hops exact | ✅ | **kb.txt** 134,741 triples (native) | public | **OFFICIAL COMPLETE** |
| **WebQSP** | MS Download id=52763 + RoG-webqsp | orig 3098/–/1639 · RoG 2826/246/1628 | 3098/1639 (3072/1628 answerable); RoG 2826/246/1628 | ✅ | Freebase triples via RoG subgraph | public | **OFFICIAL COMPLETE** |
| **2Wiki** | github.com/Alab-NII/2wikimultihop | 167454/12576/12576 | exact; 4 official types; 0 leakage | ✅ | **para_with_hyperlink** (Wikipedia hyperlinks) + evidences | HIDDEN (verified) | **OFFICIAL COMPLETE** |
| **MuSiQue-Ans** | github.com/StonyBrookNLP/musique | 19938/2417/2459 | exact; 0 dup; 0 leakage; hop-prefix derivable | ✅ | none native (paragraph sets + decomposition) | HIDDEN | **OFFICIAL COMPLETE — CRAG derivation needed** |
| **HotpotQA** | authors' HF `hotpotqa/hotpot_qa` | distractor 90447/7405 · fullwiki –/7405/7405 | exact; bridge/comparison; easy/med/hard; 0 leakage | ✅ | Wikipedia hyperlinks (in fullwiki corpus) | HIDDEN (fullwiki test verified empty) | **OFFICIAL COMPLETE (QA)** — fullwiki corpus pending |
| **SQuAD 2.0** | rajpurkar SQuAD-explorer | 130319/11873/(no public test) | exact; train 86821 ans+43498 unans; dev 5928/5945 | ✅ | none | HIDDEN | **OFFICIAL COMPLETE — CRAG derivation needed** |

## Notes / provenance decisions
- **HotpotQA source substitution (logged):** the original CMU host `curtis.ml.cmu.edu` was **unreachable** (port 80 refused) this session. Acquired instead from the **authors' own HuggingFace org** `hotpotqa/hotpot_qa` (priority-4 canonical mirror; same authors) — QA fields (`supporting_facts`, `context`, `type`, `level`) preserved. Train is shared between distractor and fullwiki settings; the two settings are kept in **separate directories** and never mixed.
- **WebQSP two-layer provenance:** original Microsoft WebQSP = the canonical **question/split** layer (official IDs `WebQTrn-*`/`WebQTest-*`); RoG-webqsp parquet = the **per-question Freebase subgraph** layer CRAG's KB retrieval consumes. Original test 1639 → 1628 answerable, which is exactly the RoG test size. No GraftNet 2848/250 dev is fabricated; if wanted it will be **derived from official train only**, later.
- **MuSiQue** hop labels come from the official `id` prefix (train: 2hop 14376 / 3hop 4387 / 4hop 1175) — no reconstruction.
- **2Wiki** ships `para_with_hyperlink.zip` (1.9 GB) → gives genuine `structural_native` hyperlink edges, distinct from derived title-mention.
- **Hidden test sets** (2Wiki, MuSiQue, HotpotQA, SQuAD-no-public-test) are preserved as hidden; **public dev** is the reproducible eval set. No official test was manufactured.

## Held for approval
- **HotpotQA fullwiki Wikipedia corpus** (processed enwiki, **~5 GB tar.bz2**). Needed only if **fullwiki** is our retrieval setting. The official CMU host is down, so a mirror will be required. **Not downloaded** — awaiting your go/no-go.

## Tightened acquisition (final)
- **WebQSP** `WebQSP.zip` SHA256 `95cb9cd2…b0ee` (4,383,010 B). Total train 3098 / test 1639; fully-parsed 3072 / 1628; partial-annotation files 680 / 393 = **1073** (matches paper). Canonical questions = **Microsoft zip only**; RoG parquet is the subgraph layer, not the question source.
- **2Wiki native hyperlink graph** (`para_with_hyperlink.jsonl`, 7.0 GB): **5,989,847 article nodes** (titles unique per record), **35,724,975 directed hyperlink edges** (`mentions[].ref_url/ref_ids`; ~98% unique pairs; 633 self-loops). **Replaces** heuristic title-mention as 2Wiki `structural_native`. Zip SHA256s recorded.
- **HotpotQA corpus (NOT downloaded)**: CMU host down; Stanford host up. Fullwiki retrieval corpus = `…withlinks-abstracts.tar.bz2` **1.55 GB** (~6–8 GB extracted, native hyperlinks); full-text superset `…withlinks-processed.tar.bz2` **7.4 GB** (~30–40 GB). Local 507,494-passage corpus is **distractor-derived**, not fullwiki. Held for approval.

## FINAL OFFICIAL RAW inventory (all six)

| Dataset | Off. questions complete? | Splits verified? | Off. corpus complete? | Native structural source? | NER source? | Hashes verified? | Canonical raw status |
|---|---|---|---|---|---|---|---|
| **MetaQA** | ✅ all hops | ✅ exact | ✅ kb.txt entities (40,151) | ✅ kb.txt 134,741 triples (native KB) | ✅ from entity docs | ✅ | **OFFICIAL COMPLETE** |
| **WebQSP** | ✅ 3098/1639 (+1073 partial recorded) | ✅ 0 leak | ✅ RoG Freebase subgraphs | ✅ Freebase relations (native KB) | ✅ | ✅ zip `95cb9cd2…` | **OFFICIAL COMPLETE** |
| **2Wiki** | ✅ 167454/12576/12576 | ✅ 0 leak | ✅ para_with_hyperlink 5,989,847 | ✅ hyperlinks 35,724,975 (native) | ✅ | ✅ both zips | **OFFICIAL COMPLETE** |
| **MuSiQue-Ans** | ✅ 19938/2417/2459 | ✅ 0 leak | ✅ question paragraphs (global corpus = derived) | ⚠️ none native → structural_derived | ✅ | ✅ | **OFFICIAL COMPLETE** |
| **HotpotQA** | ✅ distractor 90447/7405 + fullwiki 7405/7405 | ✅ 0 leak | ✅ fullwiki abstracts 5,233,329 | ✅ hyperlinks 23,438,500 (native) | ✅ | ✅ corpus `1acca1c5…` | **OFFICIAL COMPLETE** |
| **SQuAD 2.0** | ✅ 130319/11873 | ✅ 0 leak | ✅ contexts = corpus | ⚠️ none native → structural_derived | ✅ | ✅ | **OFFICIAL COMPLETE** |

Graph-policy lock recorded in `dataset_manifest.json.graph_policy_locked`: `G_main = structural_native + structural_derived(only where no native) + ner`; kNN ablation-only; edge_family ∈ {structural_native, structural_derived, ner, knn}; never collapse.

## Artifacts
- `results/data_audit/dataset_manifest.json` (SHA256 × 58 files, sources, counts)
- `results/data_audit/current_inventory.{json,md}` (Phase D0)
- `results/data_audit/verify_present.json`, `feat_inventory.json`
