# CRAG Data Foundation — Phase D0 Inventory (read-only)

_Generated 2026-08-23. No data artifact was modified. Official expected counts are user-supplied anchors, each **corroborated by an exact match to a local official raw file** where that file is present._

## 1. Headline finding

Every **canonical feature dump** (`results/L2/relsig_feats_{ds}.pt`) was built through a **silent cap** — `kg_relsig._run(... tr_cap=1104, te_cap=2000 ...)`. So the entire model-facing question layer is a ~1k-train / ~2k-test subset with **no dev split** and, for WebQSP, **train and test both sliced from the official TEST pool**. The underlying **corpora and embeddings are full-scale** (781k / 507k docs etc.) and largely reusable; it is the **question set + per-query features** that are capped and mis-provenanced.

## 2. Master mismatch table (questions)

| Dataset | Official (train/dev/test) | Local RAW present | Canonical feature dump (train/dev/test) | Verdict |
|---|---|---|---|---|
| **WebQSP** | 3098 / — / 1639 (RoG: 2826/246/1628) | **RoG TEST only** 1628 (814+814); no train/dev | 1015 / — / **147** | **MISMATCH** — train+test both carved from official TEST; test not official |
| **MetaQA** | 1h 96106/9992/9947 · 2h 118980/14872/14872 · 3h 114196/14274/14274 | **COMPLETE official** (all hops/splits exact) + kb.txt 134741 | **1091 / — / 1998** (single store, hop mix unverified) | **MISMATCH** — 99% of questions capped away; hops merged/unknown |
| **2Wiki** | 167454 / 12576 / 12576(hidden) | **DEV only** 12576; train ABSENT; hipporag 1000-subset | 1104 / — / **1500** | **MISMATCH + BLOCKED** — official train not downloaded; "test" is a train/dev slice |
| **MuSiQue-Ans** | 19938 / 2417 / 2459(hidden) | **COMPLETE official** (Ans + Full) | 1104 / — / 1994 | **MISMATCH** — capped, but full raw available → rebuildable |
| **HotpotQA** | 90447 / 7405 / 7405(hidden) | **COMPLETE distractor** (train 90447, dev 7405); fullwiki corpus absent | 1102 / — / 1985 | **MISMATCH** — capped; setting (distractor vs fullwiki) must be declared |
| **SQuAD 2.0** | 130319 / 11873 / (no public test) | **COMPLETE official** train+dev (exact) | 1088 / — / 1982 | **MISMATCH** — capped; "test" is a home-made slice, not official; unanswerables handling undocumented |

## 3. Corpus / graph scale (full — not capped)

| Dataset | Corpus docs (partition_map) | Retrieval unit |
|---|---|---|
| webqsp | 781,485 | Freebase entity (verbalized) |
| metaqa | 40,151 | WikiMovies entity |
| 2wiki_clean | 65,865 | passage |
| musique_clean | 13,672 | paragraph |
| hotpotqa_clean | 507,494 | Wikipedia passage |
| squad_clean | 19,029 | context paragraph |

## 4. Graph provenance defect (Phase D5)

`graph.pt` stores a **single unlabeled `node.neighbors` adjacency** — `build_clean.py`: _"doc.neighbors = LABEL-FREE title-mention links only; the indexer adds its own kNN edges on top."_ Three edge families are **collapsed with no `edge_type`**:

- **Official**: KB triples (webqsp Freebase, metaqa kb.txt).
- **Derived structural**: title/entity-mention links (`graph_builder.enrich_with_title_links`) for the text sets; NER `shares-entity` df-weighted edges stored separately (`ner_edges_w_df25.pkl`).
- **Synthetic**: gte-kNN similarity edges added on top for all datasets.

No per-edge provenance exists today. This must be rebuilt into a labeled multi-family graph.

## 5. Reuse decision (pre-compute)

- **Reusable (pending doc-hash verification in D4/D10):** all corpus-level embeddings — gte_qwen, SPLADE, BM25 — because the **corpus was never capped**. Re-embedding 781k/507k-doc corpora is the expensive step and appears **avoidable**.
- **Must rebuild:** the per-query feature dumps (uncap `tr_cap`/`te_cap` → full official question sets), the split definitions (official IDs + real dev), and the graph (into labeled families).
- **Must download:** official **2Wiki train** (167,454) — the only genuinely missing official release. Optionally HotpotQA **fullwiki** Wikipedia corpus if fullwiki retrieval is the intended setting.

## 6. Artifacts written
- `results/data_audit/current_inventory.json`
- `results/data_audit/feat_inventory.json`
- `results/data_audit/current_inventory.md` (this file)
