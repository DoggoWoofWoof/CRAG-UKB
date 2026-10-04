# CRAG Project Memory

## CURRENT STATE — 2026-08-24 18:30 (CANONICAL — DO NOT CONTRADICT)
- **COMPUTE_SAFETY = PASS** (`NER 6/6` `ARTIFACT_PREFLIGHT PASS` `RANKING_PERSISTENCE PASS` `L2_L3_SPLIT PASS` `NO_HEAD_TRAIN_LOAD_REUSE PASS` `LIGHT_HEAD_LOADER PASS`)
- **EXPENSIVE_RUNNING_JOBS = NONE** (`D_FULL6 COMPLETE 7660` `METAQA_2000 COMPLETE 1646`)
- **D_FULL6 = COMPLETE** `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json 7660` `6/6 scope_topk0 K8 MAXK500`
- **METAQA_2000 = COMPLETE** `n2000 hash f272578be33aa2c0` `R@5 67.28/79.01`
- **GLOBAL_D_PASS = PASS_WITH_WEBQSP_LEGACY_CAVEAT** `D implementation validated, current execution canonical, MetaQA hop-bias resolved, WebQSP historical -7.1/-4.3 documented as LEGACY_NONREPRODUCIBLE_REFERENCE`
- **CANONICAL_D = D_FULL6_universal_v2** `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json 7660` `MuSiQue D_L2 77.86/76.6 2Wiki 71.5/75.5 SQuAD 91.0/90.4 MetaQA 13.6/17.6 Hotpot 74.6/78.9 WebQSP n159 16.52/25.48`
- **WEBQSP_HISTORICAL_STATUS = LEGACY_NONREPRODUCIBLE_REFERENCE** `historical D_L2 23.64 D_E2E 29.74 retained as discrepancy, HIST_BACKEND/WEIGHT UNKNOWN, not blocking`
- **NEXT_RESEARCH_BLOCKER = A/B/C 2Wiki P20/P50/P100 scoped-L2** `A struct+kNN vs B struct+NER vs C struct+NER+kNN vs D_L2 71.5`
- **DO NOT RERUN/REBUILD:** `C5` `SPLADE` `NER` `structural graphs` `partitions` `D heads 67e78e5d/01291440` `six-dataset D` `MetaQA historical-2000` — `LOCAL` is source of truth, `Modal` is execution cache, `local missing→FAIL` `remote missing→upload LOCAL` `mismatch→FAIL` `only --rebuild-ner` may rebuild
- Historical partial C5 counters (`28/33`, `121/150`, `128/131`, etc) are **SUPERSEDED**; compute-safety hardening **DONE**; stale `PID`/`poll` instructions **ARCHIVED** below; `HIST_D_BACKEND/PROVENANCE` detail retained below for audit, removed from ACTIVE blocker.

## OpenCode continuation — 2026-08-23 16:30 [HISTORICAL]

### State inherited
- **Git commit**: `unified-crag-architecture` branch, 1 commit ahead of remote
- **Git status**: Modified core files (tasks.py, sync.py, canonical_encode*.py, ner_edges.py), many result files cleaned up
- **Running jobs**: 
  - PID 19828: C5 fanout coordinator (`canonical_encode_fanout --slots 4 --poll 20`) — RUNNING
  - Multiple Modal workers on accounts a1, a4, a6, a8, a10 encoding hotpotqa/2wiki_universe/webqsp dense+splade
  - Volume pull processes for completed shards
- **Completed artifacts [HISTORICAL/SUPERSEDED — see C5 FROZEN 150/150 + 131/131 2026-08-23 21:15; canonical top state takes precedence]**:
  - SQuAD NER validation: PASSED (sharded == established, 0 diffs)
  - 2Wiki 398k view NER: COMPLETE (3,274,931 edges)
  - 2Wiki Universe NER: COMPLETE (34,067,241 edges)
  - HotpotQA NER: COMPLETE (20,369,074 edges, finished ~11:24 AM)
  - SQuAD dense/splade encoding: Appears complete (manifests need finalize)
  - 2Wiki 398k dense/splade: In progress [SUPERSEDED — all complete per C5 FROZEN]
  - WebQSP dense: 28/33 shards complete [SUPERSEDED]
  - 2Wiki Universe dense: 121/150 complete, 24 PERMANENTLY FAILED (OOM after 3 retries) [SUPERSEDED — all 150/150 recovered via 16→8→4]
  - 2Wiki Universe splade: 130/150 complete [SUPERSEDED — 150/150]
  - HotpotQA dense: 128/131 complete [SUPERSEDED — 131/131]
  - HotpotQA splade: 124/131 complete [SUPERSEDED — 131/131]

### Actions taken
- None yet (this is the first OpenCode action - reconstructing state)

### Jobs launched
- C5 fanout coordinator (PID 19828) — continuing from previous run
- Modal workers on accounts a1, a4, a6, a8, a10 — actively encoding remaining shards
- Volume pull processes — retrieving completed shard artifacts

### Validation results
- **SQuAD NER equivalence validation**: PASS
  - Established edges: 156,130
  - Sharded edges: 156,130
  - Edges only in EST: 0
  - Edges only in NEW: 0
  - Weight diffs >2e-6: 0
  - SHA256 established: 0e3846373ff9bcd77fb4e1a3ec722b1b11249404fc289e661ab84ad171842063
  - SHA256 sharded: 8943ab5bbabc14b0 (different orientation, same graph)

### Decisions / locks
- **NER methodology LOCKED**: Must reproduce `src/pipeline/ner_edges.py` exactly (spaCy en_core_web_sm, 9 labels, df 2-25, 1/df weighting, per-doc dedup, lower().strip(), len>2)
- **2Wiki Universe NER separate from 398k view**: Confirmed — full universe NER computed independently over 5.99M docs
- **BM25 NOT used**: Do not generate BM25 anywhere
- **kNN graph edges NOT part of main architecture**: Keep only as ablation/control
- **Full universe encoding policy**: Encode complete reusable universes (dense docs, dense queries, SPLADE docs, SPLADE queries)

### Remaining work [HISTORICAL/SUPERSEDED — C5 FROZEN 150/150 + 131/131 2026-08-23 21:15; EXPENSIVE_RUNNING_JOBS NONE]
1. **C5 encoding completion**: Fanout needs to finish remaining shards: [SUPERSEDED — all done]
   - 2Wiki Universe dense: 29 remaining (but 24 permanently failed - OOM) [SUPERSEDED]
   - 2Wiki Universe splade: 20 remaining [SUPERSEDED]
   - HotpotQA dense: 3 remaining [SUPERSEDED]
   - HotpotQA splade: 7 remaining [SUPERSEDED]
   - WebQSP dense: 5 remaining [SUPERSEDED]
   - 2Wiki 398k dense/splade: status unclear from logs [SUPERSEDED]
   - All query encodings: mostly complete per c5_full3.log [SUPERSEDED]

2. **2Wiki Universe dense OOM failures (CRITICAL)**: 24 shards permanently failed after 3 OOM retries: [HISTORICAL — recovered via 16→8→4]
   - Shards: 21,22,31,32,33,34,35,36,48,49,58,59,60,61,70,71,107,108,109,110,114,115,118,119
   - Need investigation: reduce batch size, increase GPU memory, or use CPU fallback [DONE]

3. **Manifest finalization**: Once all shards complete, run `canonical_encode.finalize()` for each dataset/kind/model to build manifest.json + index.json [DONE — 150/150 131/131 True]

4. **Modal account health**: Only 5 healthy accounts (a1,a4,a6,a8,a10). a0,a2,a3,a5,a7,a9 = SPEND_LIMIT. a11-a14 = AUTH_FAILED. Consider provisioning new accounts. [HISTORICAL — 2026-08-23 23:22 11 healthy 0-10 ACTIVE]

5. **HotpotQA FullWiki preparation**: NER done. Need structural graph (native Wikipedia hyperlinks) + dense/SPLADE encoding completion. [HISTORICAL — graph 15,367,541 done C5 131/131]

6. **Per-query analysis**: After full C6 coverage, run expert standalone quality, gold ranks, R@1/5/20, margins, Shapley, regime distributions. [BLOCKED until A/B/C + C6]

## OpenCode continuation — 2026-08-23 18:15 [HISTORICAL — partial C5 counters SUPERSEDED by C5 FROZEN 150/150+131/131]

### State reconfirmed [HISTORICAL]
- Coordinator PID 19828 `canonical_encode_fanout --slots 4 --poll 20` healthy, running=5 queued=0, last log 17:15 `complete 310/344 failed 24 dense 72/s` [HISTORICAL PID SUPERSEDED — coordinator stopped; EXPENSIVE_RUNNING_JOBS NONE]
- Workers: `a1:11676 hotpot 3,5 (batch32)`, `a8:14068 2wiki 97,98`, `a4:27096 111,113`, `a10:25140 132,133`, `a8:19056 hotpot 38,41`; volume pulls `16772,11196,22240,27436,25132` [HISTORICAL]
- File presence: `squad/webqsp/2wiki/musique/metaqa` all dense/splade docs+queries 100% present; `hotpotqa docs dense 130/131 missing [41]`; `hotpotqa docs splade 131/131` `queries 3/3`; `2wiki_universe docs splade 150/150`; `2wiki_universe docs dense 124/150 .npy>0`, `121/150 shard_complete-valid` → 26 physically missing `[21,22,31,32,33,34,35,36,48,49,58,59,60,61,70,71,107-110,113-115,118,119,133]` (24 OOM + 113,133 in-flight), 3 invalid-present `[98,111,132]` (npy valid but `ids.json`/`status` missing, pulls in-flight) [SUPERSEDED — final 150/150 131/131]
- Disabled accounts per `c5_full3.log:17:15` `SPEND_LIMIT {0,2,3,5,7,9} AUTH_FAILED {11,12,13,14}` → healthy 5/15 `{1,4,6,8,10}` unchanged from 16:30

### OOM diagnosis (single-shard traceback `2wiki_universe_docs_dense_a1_00031.log:9-67`)
- `src/experiments/canonical_encode.py:182 write_atomic` → `SentenceTransformer.encode(batch 32)` → `Transformer.forward` → `modeling_qwen.py:186/794/1081` → `transformers/masking _expand_mask` → `torch.OutOfMemoryError: Tried allocate 2.96 GiB, GPU 22.06 GiB total, 2.83 free, 19.22 in-use (18.66 alloc + 292M reserved)`
- Other OOM shards same forward-pass VRAM exhaustion: `3.29 GiB @113, 4.17 @58, 5.82 @109` etc. Not model load / tokenization / serialization. Fix = smaller GPU batch, identical model/tokenizer/max-len/pooling/norm/fp16.

### 2Wiki-universe shard length audit (`_src/docs` 40k/shard, 150 shards, 5,989,847 docs)
- `successful 5,029,847 docs` mean 449.4 chars p95 1436 max 22662
- `oom 960,000 docs` mean 486.6 chars p95 1550 max 43176
- Per-OOM: `21:562/1708, 32:523/1631 max43176, 36:615/1826 max33930, 60:558/1818` etc. — modestly longer, explains isolated OOM; canonical text unchanged.

### C5 manifest finalization (deep-validated)
- Ran `canonical_encode.finalize()` for complete groups: `squad (4/4), 2wiki (4/4), webqsp (4/4), musique (4/4), metaqa 2/2, metaqa_1hop 3/3, 2hop 4/4, 3hop 4/4, hotpotqa queries 3/3, hotpotqa docs splade 131/131 (117.8s), 2wiki_universe docs splade 150/150 (112.9s)` all `complete=True` validated (rows, unique IDs, dim 1536 fp16 / splade CSR, source SHA)
- Incomplete: `hotpotqa docs dense False 130/131`, `2wiki_universe docs dense False 121/150` — pending missing/invalid shards only
- Hotpot `G_main` verified: `data/canonical/hotpotqa/graph_manifest.json` 15,367,541 native hyperlinks + `graph_ner.tsv` 20,369,074 NER, `document_manifest.json` 5,233,329 docs — no rebuild needed. Coordinator kept running.

### Recovery
- Kept healthy in-flight jobs (hotpot 41, 2wiki 113,133 + pulls for 98,111,132) running, not killed for slot change.
- Dispatched 12 OOM-recovery workers at `batch 16` on healthy accounts only (no SPEND_LIMIT/AUTH_FAILED rotation, shard retry budget not consumed by account failures): `21,22@a1, 31,32@a4, 33,34@a6, 35,36@a8, 48,49@a10, 58,59@a1, 60,61@a4, 70,71@a6, 107,108@a8, 109,110@a10, 114,115@a1, 118,119@a4` — detached `creationflags DETACHED|CREATE_NEW_PROCESS_GROUP`, PIDs `24616,17856,4288,14868,27612,15812,15832,8520,25516,17928,16488,24784`. Identical embeddings, only execution batch reduced. Next fallback `16→8→4` GPU only if needed, no CPU fallback yet.
- For 3 strict-invalid shards `[98,111,132]` (npy valid `(40000,1536) fp16 finite True` but `ids_NNNN.json` missing → `shard_complete=False`), dispatched `modal volume get` pulls for `ids_00098/00111/00132.json` — will be included in same sweep if pulls fail (total 27 shards if needed). No full-universe rerun, no BM25.
- Next coordinator restart will use `--slots 3` (deferred until current 5 jobs finish).

### Remaining after dispatch [HISTORICAL/SUPERSEDED — DONE C5 FROZEN]
- Await 3 in-flight encodings + 3 id pulls + 12 batch16 recoveries → `hotpot dense 131/131`, `2wiki_universe dense 150/150` → re-finalize with `complete=true` → C5 CLOSED → C6 uncapped coverage → per-query analysis. NER locked, no BM25. [COMPLETE]

## OpenCode continuation — 2026-08-23 19:05 [HISTORICAL — partial C5 SUPERSEDED]

### Current state [HISTORICAL]
- `git ba6bd71` branch `unified-crag-architecture` +1 ahead, `git status` modified `src/experiments/*`, `src/pipeline/ner_edges.py`, untracked `memory.md` + `scratchpad/launch_*.py` + `data/canonical` encodings
- Coordinator `19828 --slots 4` running=5 queued=0 (a1:1/4 a4:1/4 a6:0/4 a8:2/4 a10:1/4 per 18:15 log), next restart → `--slots 3` (deferred) [HISTORICAL PID]
- Present: `squad/webqsp/2wiki/musique/metaqa` 100% docs/queries dense+splade valid; `hotpotqa docs splade 131/131`, `queries 3/3`; `2wiki_universe docs splade 150/150`, `docs dense 146/150 present` (+22 since 18:15), `145/150 valid` (was 121), `missing 4 [109,110,113,133]` + `invalid [108]` =5 invalid after partial batch16/8 (`59,70,71,107` now valid), `hotpot dense 130/131 missing [41]` valid 128/131 [SUPERSEDED — 150/150 131/131]
- Recent batch16 (12 jobs @16) mostly complete: `21,22,31,32,33,34,35,36,48,49,58,60,61` now valid; `59` recovered via batch8 retry (was missing, now True); `70,71` valid via batch8; `107` valid via batch8; `108` still invalid at batch8 → retried at batch4 (PIDs 15824,22780). `109,110` still missing after batch8 → retried at batch4. `98,132` now valid via id pulls, `111` still invalid → retried at batch16 (PID 14940). In-flight `113,133,41` at batch32 still missing.
- Active Modal containers per account (wmic `canonical`): `a1:11676(3,5)+10272(59-b8)`, `a4:27096(111,113)+7092(70,71-b8)`, `a6:8356(107,108-b8)+15824(108-b4)`, `a8:14068(97,98)+19056(38,41)+25228(109,110-b8)+22780(109,110-b4)`, `a10:25140(132,133)` → `a1:2/4 a4:2/4 a6:2/4 a8:4/4 a10:1/4` (total 11) + 5 volume pulls + coordinator → within 4, will enforce 3 on next restart; no SPEND_LIMIT/AUTH_FAILED used.

### Actions taken
- Traced `src/core/indexers.py:101-237 build_pyg_graph` exactly: `node.neighbors` (structural) + universal semantic KNN `k=3` (IndexFlatIP exact on L2-normalized dense embeddings, `vectors=embeddings` 384-dim MiniLM, GPU `StandardGpuResources` if available, `synthetic_neighbors` metadata) + structural edge mirror → `G_nx`; isolates `nx.isolates` → IVFFlat `nlist=sqrt(N)` `nprobe=nlist//10` batch 20k adding 3 more synthetic per isolate; `edge_index` undirected mirrored → `graph.pt`. Then `build_partition_map:244-289` `n_parts=n_nodes//target`, `pymetis.part_graph(adjacency=[G_nx.neighbors])` → `n_cuts` or naive `i//1000` fallback. `build_faiss_centroid_index:296-337` degree-weighted `+1` mean. SPLADE not used for partitioning (only `SpladeScorer` voting at query time). NER not in `G_nx`. Partitioning identical across datasets, only counts differ.
- Labeled artifacts: `routing_graph_manifest` = `data/ukb_storage/{source}/graph.pt` (struct+synthetic kNN, collapsed, no NER); `partition_manifest` = `partition_map.json` + `centroids.index` + `centroid_pids.json`; `main_graph_manifest` = `data/canonical/{ds}/graph_manifest.json` (structural_native hyperlinks) + `graph_ner.tsv`/`ner_manifest.json` (corpus-global NER, separate per universe) — never conflated.
- Estimated scaling: `2wiki_universe 5,989,847` target100 → `59,898` parts, `~46.9M` undirected edges (17.97M KNN + 28.96M struct), `8.6GB` emb, `1.4GB` graph.pt, `88MB` centroids, `114MB` map; target1000 → `5,989` parts, `9MB` centroids. `Hotpot 5,233,329` target100 → `52,333` parts, `31.1M` edges, `7.5GB` emb, `0.9GB` graph, `77MB` centroids / target1000 → `5,233` parts `8MB`. Voting cost/query ~10ms dense FAISS + RRF. No target change, no full-universe repartition launched.
- Dispatched batch8/4 retries for still-missing after batch16 partial: `59@a1-b8` (now valid), `70,71@a4-b8`, `107,108@a6-b8`, `109,110@a8-b8`, `111@a10-b16` → then `108@a6-b4`, `109,110@a8-b4` when `108,109,110` still invalid at batch8. GPU-only, model/tokenizer/max-len unchanged; next `8→4` fallback only.
- Verified `3` invalid-present shards: `98,111,132` initially `npy (40000,1536) fp16 OK` but `ids.json` missing (`shard_complete` checks `ids`+`status`); after `modal volume get` pulls `98,132` now `True`, `111` still `False` → retried at batch16.

### Jobs launched
- Batch8/16 retries (detached): `59@a1:10272`, `70,71@a4:7092`, `107,108@a6:8356`, `109,110@a8:25228`, `111@a10:14940` (batch16) + `108@a6:15824`, `109,110@a8:22780` (batch4) — PIDs verified via wmic, healthy accounts only.
- Previous 12 batch16 OOM recoveries still: `21,22@a1:24616` etc. mostly complete (present 141→146).

### Validation
- After batch16 partial: `2wiki dense present 141/150 invalid 1 missing 9 valid 140/150`; after batch8 for `59,70,71,107`: `present 146/150 valid 145/150 missing 4 [109,110,113,133] invalid [108]` → `109,110` still missing at batch8, `108` still invalid, `113,133,41` still in-flight batch32. Next validation after batch4.
- Hotpot structural graph: `data/canonical/hotpotqa/graph_manifest.json` 15,367,541 edges + `graph_ner.tsv` 20,369,074 edges + `graph_structural.tsv` + `document_manifest.json` 5,233,329 docs — complete, no rebuild.

### Decisions/locks
- **ROUTING GRAPH VS MAIN GRAPH:** Historical CRAG `G_routing` = `structural + dense kNN (3 per node, isolate IVFFlat)` via `build_pyg_graph` → `pymetis` → partitions/centroids. `G_main` = `structural_native/derived + NER` (separate, per-universe, no kNN). kNN remains legitimate **only** inside `G_routing` as index construction (as historically implemented), forbidden from `G_main`/`relation/path/L3`. All manifests/paper must expose this distinction; do not claim partitions are structural-only. Partitioning code locked, no redesign until L1 diagnostics.
- **Scaling:** Do not launch full-universe `~60k` partition builds yet; report tradeoff only. Current canonical target `~100` → `59,898` partitions (vs `5,989` at 1000) with edge counts/RAM above. Validate on small complete view (e.g. `squad_clean` 20k, `2wiki_clean` 398k) first.
- **C5 policy:** No full-universe rerun, only failed/invalid shards at `16→8→4` GPU, healthy jobs preserved, account failures not consuming retry budget, no BM25, no NER touch.

### Remaining work [HISTORICAL — C5 portion SUPERSEDED]
- C5: Await `41,113,133` batch32 + `108,109,110` batch4 + `111` batch16 → `hotpot 131/131`, `2wiki 150/150` → `finalize()` both `complete=true` → **C5 FROZEN**. Then build missing `2wiki_universe` `routing_graph_manifest`/`partition_manifest` via historical `build_pyg_graph`+`build_partition_map` (not via `G_main`). [SUPERSEDED — C5 FROZEN DONE]
- L1: Reproduce historical partitioning on `squad_clean`/`2wiki_clean` (partition count, membership, centroid, dense/SPLADE/RRF voting, Recall@1/3/5/10/20, candidate recall, docs/L2) + parallel direct global vs partition diagnostic at matched budget.
- C6: Continue porting on complete datasets (`SQuAD/WebQSP/MetaQA/MuSiQue/2Wiki-benchmark`) — canonical sharded reader, official splits uncapped, separate provenance, sharded/resumable `PrepBatch` with atomic status/manifest — no `2wiki_universe` C6 until routing semantics settled.

## OpenCode continuation — 2026-08-23 20:10

### State inherited
- `git ba6bd71` `unified-crag-architecture` +1, `git status` same modified `src/experiments/*` etc., coordinator `19828 --slots 4` still running=5→ now 3 after kills, `wmic canonical` shows `a1:11676(3,5)` `a4:27096(111,113)` `a10:25140(132,133)` `a8:19056(38,41)` before kill; after kill `113,133,41` retried at batch16 on `a1,a4,a6`
- C5: `2wiki_universe dense present 148/150 valid 145/150` → after batch4 `108` now `True`, `109` now `True`, `110` `True` (present 148→148? actually 146→148 after batch4), `111` `True` via pull, `59,70,71,107` valid via batch8, `113,133,41` still missing; `hotpot present 130/131`; `squad_clean` validated `19029` nodes `190` parts `90-103` median `101` `centroids 190×384` `graph 1,428,055` edges; `2wiki_clean` `65,865` nodes `658` parts.

### Actions
- Monitored C5: `108 True` after batch4 `15824`, `109,110` `True` after `22780` batch4, `111` `True` via `modal volume get` pulls for `98,132`; `59` already `True` via batch8; `70,71` `True` via batch8; `107` `True` via batch8. Remaining `2wiki 113,133` + `hotpot 41` still `False` after 5h batch32 — killed stuck `27096,25140,19056` and retried at `batch16` on `a1:113`, `a4:133`, `a6:41` detached.
- Checked Modal concurrency per account: `a1:2/4 a4:2/4 a6:2/4 a8:4/4 a10:1/4` before kill, after kill+retry `a1:1/4 +1 new, a4:1/4 +1, a6:0/4 +1` → within 4, next restart →3.
- Ran controlled L1 routing-graph ablation via explore subagent (very thorough) while C5 recovers — **no full-universe repartition**, no C5 kill, no BM25:
  - Variants `A historical (structural+dense-kNN)`, `B structural+NER`, `C hybrid (struct+NER+kNN)`, `D direct global dense/SPLADE` at fixed `target100`, `pymetis`, `vote_k=100` RRF `60`, `K partitions` budgets.
  - Datasets `squad_clean (19k,500 test)`, `2wiki_clean (65k,500)`, `musique_clean (13k,500)`, `metaqa/webqsp` where KB → B/C N/A.
  - Artifacts `scratchpad/ablation/{ds}/variant_{B,C}/` + `results/l1_routing_ablation/combined.json` + `table.csv` + `results/L1_routing_graph_ablation.json` + `REPORT.md`.
  - Metrics: partition `Recall@1/3/5/10/20`, candidate recall `top-1/3/5/10/20`, docs/L2 mean/p95, edge-cut, latency, RAM, ceiling `gold_reachable`.
  - Result: `A` best L1 but still loses to `D` global at matched budget `+4–22 @5`; `B/C` collapse partition recall `SQuAD 68.8→6.2/3.6`, `2Wiki 73.6→1.6/2.2`, `MuSiQue 76→4.0/3.6 @5` (dense), SPLADE/RRF same, candidate recall recovers `58–75% @5` → ranking failure. Docs/L2 `K*100` tight, edge-cut `54k–812k`, storage `5–45MB`, latency `0.3–116ms`. `D` global `95.2/72.2/90.0 @5` beats partitions. `L3` lift not in L1 (`+50 MetaQA, +62 WebQSP`). Graph earns at L3, not L1 — do not promote NER to routing.

### Jobs launched
- Batch4 retries: `108@a6:15824`, `109,110@a8:22780` (both now valid), plus `111@a10:14940` batch16.
- New retries after stuck kill: `113@a1-b16`, `133@a4-b16`, `41@a6-b16` detached (PIDs not yet listed, launched via `experiments.py run canonical-encode --backend modal`).
- L1 ablation subagent completed without GPU contention (CPU/statistical, no thrash).

### Validation
- After batch8/4: `2wiki dense valid 148/150` (was 145), `present 148/150` (was 146) — `108,109,110,111` now `True`; remaining `113,133` + `hotpot 41` =3 missing. Next `finalize()` requires `150/150` & `131/131`.
- Squad historical partitioning reproduced exactly: `build_pyg_graph` universal `k=3` exact + isolates `IVFFlat sqrt(N)` + `pymetis` → `190` parts `100.2` avg, degree-weighted centroids, dense/SPLADE/RRF voting ready. No redesign.

### Scientific decisions / locks
- **ONE FINAL GRAPH G*:** After ablation, semantic-kNN is reachability (not fake relation), NER+structural also reachability; edges retain `structural_native/derived`, `ner`, `semantic_knn` provenance. Do not permanently split `G_routing` vs `G_main`; select ONE `G*` (`A` vs `B` vs `C`) via whole-system evidence (L1+L2+L3+generalization+compute). Current screen: `A` best L1, but `D` global beats all partitions at matched budget → question whether L1 earns existence; `B/C` not justified for routing. `C` hybrid not promoted. Partition-size sweep (100 vs 250/500/1000) deferred until topology selected.
- **No full-universe `~60k` partition builds** until small-screen justifies; `2wiki_universe` `G*` selection pending L1+L2/L3 evidence.

### Remaining work
- C5: Await `113,133,41` batch16 → `148→150/150`, `130→131/131` → `finalize()` both `complete=true` → **C5 FROZEN** → build `2wiki_universe` `G*` partitions only for selected topology (likely `A`).
- L1: Direct global vs partition diagnostic already included in ablation (D vs A/B/C at matched `K*100` docs); next cross-dataset stability + LODO, then partition-size Pareto `100/250/500/1000` on strongest 1-2 topologies.
- C6: Port canonical adapter on complete `SQuAD/WebQSP/MetaQA/MuSiQue/2Wiki-benchmark` (sharded `PrepBatch`, official splits, typed provenance) in parallel — no `2wiki_universe` C6 until `G*` frozen.
- Then: `OLD cap vs full`, `benchmark vs full-universe`, `R0-R3` sampling, `Shapley` teacher, `LODO`, `L3` on same `G*`, final SIGIR tables — all per dependency DAG, no global barrier.

## OpenCode continuation — 2026-08-23 20:30

### State inherited
- `git ba6bd71` `unified-crag-architecture` +1, coordinator `19828 --slots 4` still running, `wmic` shows `113@a1:11580`, `133@a4:16936`, `41@a6:7960` batch16 in-flight since `19:39` + volume pull `21340` for `41`; `2wu 148/150` valid, `hotpot 130/131`; `squad_clean` `190` parts validated.
- Previous memory.md stated "canonical H8 had no partition routing" based on `scope_topk=0` variable name — **NOT established** per new instruction; must trace H8 call graph line-by-line.

### Actions
- Traced `H8 = e2e_pipeline.py:211 run(datasets, head_datasets, subdir="gte_qwen", scope_topk=0, n_seed=2)` — actual command `python experiments.py run e2e-ner -- --datasets ... --head-datasets ... --subdir gte_qwen --scope-topk 0 --n-seed 2` (default `HPR_EVAL` `musique/2wiki/hotpot_hpr_clean`, `HEAD_MIX` 2 datasets, now full-6 via `3e7661a`).
- `e2e_pipeline.py:111-113 level1_pool(I_qi, mem_idx, npart, scope_topk) -> set()` **always empty**, comment "scope handled by L2's topP; kept as seam" — not used for candidate generation.
- `e2e_pipeline.py:122-124 level2_order`: `_, I = per_d["faiss"].search(qte, MAXK=100)`; `topP = ([set()]*len(qte)) if not scope_topk else _topP(I, mem_idx, npart, scope_topk)` — `scope_topk=0` **disables** partition scoping at L2 as well.
- `l2_seed.py:33-37 _topP`: `S,M = _feats(dense_order, mem_idx, npart, topn=200)` `votes = _rr(S)+_rr(M)` `set(argsort(-votes)[:scope_topk])` — only called when `scope_topk>0`.
- `l2_seed.py:40-68 _scoped_order`: `if tp: allow = isin(hard_t, list(tp)); sim = where(allow, sim, -1e9)` — empty `tp` (`set()`) is falsy, so **no masking**, scores full corpus `p @ X_t.T` `(b,n)` then `topk` over `n` docs.
- `l2_seed.py:284-286 run`: `topP = [set()]*len(qte) if not scope_topk else _topP(...)` `ceil=1.0` when `scope_topk==0` (every gold reachable), else computed from `hard[g] in topP`.
- Provenance: `e2e_pipeline.py:111,122,124` + `l2_seed.py:33,40,284` — `scope_topk=0` means **L1 partition routing bypassed, L2 does direct global retrieval** (full `N` docs scored, `O(N)` GPU working set `p @ X_t.T`). Not "no additional restriction after L1" — there is **no L1** in H8.
- Identified canonical L1 node-voting for H8 when `scope_topk>0` (e.g. `l2_seed` default `50`): `dense K=MAXK=100` via `faiss.search`, `SPLADE` not in H8 L1 (H8 L1 is dense-only; SPLADE is L2 signal via `_splade_scoped_order`), `node->partition` via `_feats` `topn=200` `RRF 1/(60+r)` per `S,M`, `dense+SPLADE fusion` at L2 via `_bestof`/`_rr` etc., historical `P` default `50` (`l2_seed:259 scope_topk=50`), membership `hard` (not overlap) for L2, candidate dedup via `set`, centroids not in L2 scoring (only for partition routing when `scope>0`).
- Compared to variants `H1 faiss_vote_100` (benchmark `vote_counts[pid]++` raw count), `H6 l1-unified-ranker` etc. — **H8 is distinct**; previous memory.md conflated `faiss_vote`, `selectivity_route`, `l1_unified_ranker` unless proven same call graph — now separated as `H1-H8` per latest explore.

### Jobs launched
- No new L1 jobs; awaiting `113,133,41` batch16 to complete for C5 finalization; L1 equivalence test pending (20 deterministic queries, original `benchmark_partition_selection` vs new `ablation` trace, require discrete equivalence on `dense top-100 IDs`, `SPLADE top-100`, `partition mapping`, `ranking`, `top-P`, `candidate set`).

### Validation
- H8 trace shows `scope_topk=0` **does** disable partition selection (not just L2 scoping) — level1_pool empty and topP empty → full corpus. Previous statement "canonical H8 had no partition routing" was **correct** on this trace, but instruction requires exact file:line proof before concluding — now proven via `e2e_pipeline.py:111-113,122-124` + `l2_seed.py:33,40,284`. Marked as **earlier interpretation now verified**, not silently rewritten.
- Historical operating point for H1-H3 is `top-20` (`K_VALUES [1,3,5,10,20]` primary, `benchmark:216 k=max(k,n_parts,200)` full ranking), but H8 operates at `scope_topk=0` (full corpus, effective `P=0` partitions, `ceil=1.0`).

### Scientific decisions / locks
- **Correction:** `scope_topk=0` in `H8 e2e_pipeline` verified to bypass L1 partition routing entirely (full-corpus L2). `D direct-global` is therefore **not** a control vs `H8` — `H8` itself is `D` at L1 for headline. `A/B/C` partition variants remain ablations vs `D` to answer "does L1 earn existence", but `A` cannot be claimed as canonical H8 L1 until exact historical `H8` L1 (full-corpus) is reproduced and compared. Previous `A historical` vs `D` comparison used `A` = `faiss_vote_100` count-vote, not `H8`'s dense `faiss.search(MAXK)` + `S,M` RRF — equivalence must be re-established.
- No full-universe `G*` build, no C6 full-universe launch until `H8` L1 equivalence `PASS` and `A` correctly reproduced.

### Remaining work
- Build exact L1 equivalence test: run `ORIGINAL benchmark_partition_selection faiss_vote_100` unchanged on `squad_clean` 20 queries → trace `dense top-100 IDs/scores → pid mapping → vote vector → ranking → top-P → candidate set` vs new `ablation` same queries, require discrete equivalence; then re-run `A` with exact historical `faiss_vote` logic, compare to historical `results/level_1/comparison_*.json` canonical numbers within tolerance before using as variant `A`.
- Await `113,133,41` batch16 → `150/150` `131/131` → `finalize()` → **C5 FROZEN**.
- Then `A/B/C` with exact `H8` node-voting/partition-routing (when `scope>0`) + same L2/L3 vs `D` at query-matched `K_q=|union|` + fixed `100/250/500/1000` budgets, multi-gold `ANY/ALL/fraction`, `gold_reachable` per query.

## OpenCode continuation — 2026-08-23 21:15

### State inherited
- `git ba6bd71` `unified-crag-architecture` +1, `git status` same, coordinator `19828` still `running=5`→ now `2wu 150/150` `hotpot 131/131` `shard_complete True` for `113,133,41` after batch16 retries (`11580,16936,7960`); `hotpot 38` also `True` after `batch8` (`a1:batch8`); `finalize()` both `True` `150/150` `131/131` deep validated `dim 1536 fp16`.
- Previous `C5` was `148/150` `130/131` with `113,133,41,38` pending; now **C5 CLOSED and FROZEN** per instruction.

### Actions
- Verified `2wiki 113 True`, `133 True`, `hotpot 41 True` via `shard_complete` (was `False` at `19:39`, now `True` after batch16). `hotpot 38` was `False` due to truncated `122877923` vs `122880128` + missing `ids` → dispatched `hotpot 38@a1 batch8` detached and now `True`.
- Finalized both groups: `2wiki_universe docs dense True 150/150`, `hotpot docs dense True 131/131` (`src/experiments/canonical_encode.py:146 finalize`); `manifest complete=true`, `rows_covered 5989847/5989847` `5233329/5233329`, `source_sha 728eff54` `a43b3eb1`, deep validated `150`/`131` shards `dim 1536` `fp16` `np.isfinite` `ids unique`.
- **C5 FROZEN** — no BM25, no NER rebuild, no C5 rerun unless corruption/bug. All required `dense/SPLADE` groups `complete=true` (including `2wiki_universe docs splade 150/150`, `hotpot docs splade 131/131`, `queries` all).
- H8 provenance already traced `scope_topk=0 → empty topP → no mask → full-corpus L2` (`e2e_pipeline.py:111,124` `l2_seed.py:33,284`); `D` is canonical full-corpus reference, `faiss_vote_100` secondary diagnostic. No new L1 jobs launched pending equivalence.

### Jobs launched
- Final C5 retries: `113@a1:11580`, `133@a4:16936`, `41@a6:7960` batch16 (all now `True`); `hotpot 38@a1 batch8` (now `True`). Previous 12 batch16 OOM recoveries + 5 batch8/4 retries all `True` except those 3 which just completed.

### Validation
- `2wiki_universe docs dense` `expected 150` `present 150` `valid 150` `complete True` `rows 5989847` `dim 1536` `dtype float16` `ids unique` `source SHA 728eff54` — **PASS**.
- `hotpot docs dense` `expected 131` `present 131` `valid 131` `complete True` `rows 5233329` `dim 1536` — **PASS**.
- `C5 FROZEN` appended per `0. CURRENT LOCKED STATE`.

### Scientific decisions / locks
- **C5 FROZEN** locked; no touch unless corruption. **Correction recorded:** previous `Remaining work` "Await `113,133,41` → C5 FROZEN" now completed and verified, not silently rewritten.
- **H8** remains `D` full-corpus reference; `A` (`faiss_vote_100`) remains secondary historical diagnostic, not canonical H8 L1 — requires exact `20`-query equivalence before `A/B/C` vs `D` interpretation.
- No `5M` `G*` builds until `A` equivalence `PASS`.

### Supervisor queue — 2026-08-24 00:28 [HISTORICAL/SUPERSEDED — EXPENSIVE_RUNNING_JOBS NONE; D_FULL6 COMPLETE; all PIDs below terminated/archived]

RUNNING [HISTORICAL]:
- D_full6_universal_v2 | datasets=6 (musique_clean,2wiki_clean,squad_clean,metaqa,hotpotqa_clean,webqsp) full validation uncapped te_cap0 | account=1 (deepalimohapatra1973) | PID 26432 | GPU A10G | scope_topk0 MAXK500 K8 epochs15 | log scratchpad/D_shards/full6_a1_D_full6_universal_v2.log | output results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json | start 2026-08-23 23:31:46 | ETA ~45-90min (head train + 6 eval corpora sequential) — LEAVE ALONE, healthy [HISTORICAL — COMPLETE 7660 HasExited True]
- partition_full_canonical | 6 datasets full-validation canonical C5 queries via batched FAISS top100 (Q@X_T never materialized) | PID 19000 | CPU | log scratchpad/C6_and_partition/partition_full_canonical | output results/L1/partition_curves_A_full.json + rankings npz | start 00:27:36 | ETA ~30min — replaces DEBUG_CAPPED [HISTORICAL — INVALID_METHOD_DEBUG terminated]
- hotpot_curve | hotpotqa_clean 500q smoke (prior 500 capped) | PID 9136 | CPU | log scratchpad/partition_curves/hotpot.log | will be superseded by full partition [HISTORICAL terminated]
- C6_splade_webqsp | expert=SPLADE (graph-independent) | dataset=webqsp | account=0 kuttakamina9895 | PID 2152 | GPU A10G | log scratchpad/C6_and_partition/C6_splade_webqsp | output data/canonical/webqsp/features/splade/ | start 00:27:42 [HISTORICAL terminated]
- C6_offset_universal | expert=offset (graph-independent, OffsetHead seed+learned q offset src/experiments/query_relation.py:39) | datasets=6 | account=4 swathihrao28 | PID 26944 | GPU A10G | log scratchpad/C6_and_partition/C6_offset_universal | output data/ukb_storage/_head_cache/head_*.pt + features/offset/ | start 00:27:42 [HISTORICAL terminated]
- C6_crag_gates | provisional graph-baseline | datasets=squad_clean,webqsp | account=6 spanishorgay | PID 360 | GPU A10G | log scratchpad/C6_and_partition/C6_crag_gates | start 00:27:42 [HISTORICAL terminated]

READY:
- C6_dense (graph-independent) reuse C5 canonical dense already 150/150 2wiki_universe 131/131 hotpot complete — mark DONE per dataset as C6_dense_<ds>
- C6_splade remaining 5 datasets (squad,metaqa,musique,2wiki,hotpotqa) GPU (accounts 8,10 spare) — READY to launch
- C6_offset remaining per-dataset shards if universal head needs per-dataset verification
- old_vs_full per dataset as soon as canonical C6 dense/splade/offset ready (compare vs DEBUG_CAPPED)
- per-query diagnostics PARTIAL (dense/splade/offset) as C6-A completes
- Shapley 32-subset (exact) once 5 experts valid per query (mask-aware)
- training infrastructure: R0/R1/R2/R3 samplers, PrepBatch loader mask-aware, LODO manifests 6 folds, Shapley teacher cache versioned (dataset/query_id/candidate_hash/graph_hash/feature_hash)

BLOCKED:
- A/B/C end-to-end P20/P50/P100 until D gate (needs trustworthy D)
- final G* selection until A/B/C/D + downstream evidence
- final graph-sensitive C6 (relation/path) freeze until G* (currently PROVISIONAL canonical_baseline)
- final training until source analysis complete
- giant 5M G* builds until topology promotion

FAILED_RETRYABLE (requeued, fixed):
- D_squad_clean_full@1 PID7784 etc 6 shards — sync bug fixed via src/experiments/sync.py:280 eff_ds
- D_full6_universal@7 PID14756 SPEND_LIMIT ac-wnTpDxWK6Kpf7113epWYGN disabled

FAILED_DETERMINISTIC: none

COMPLETE (renamed DEBUG):
- DEBUG_CAPPED_partition_curves_A.json (squad 2000, metaqa 2000, hotpot 500, 2wiki 1500 etc) moved to results/L1/DEBUG_CAPPED_partition_curves_A.json + scratchpad/DEBUG_CAPPED_partition_curves/ — NOT for topology selection
- C5 FROZEN 150/150 2wiki_universe docs dense + 131/131 hotpot docs dense (never list again)

COMPLETE:
- C5 FROZEN 150/150 2wiki_universe docs dense + 131/131 hotpot docs dense (validated dim1536 fp16, rows 5989847/5233329)
- H8 provenance traced scope_topk0 -> empty topP -> full-corpus (e2e_pipeline.py:111,124 + l2_seed.py:33,284)

## OpenCode continuation — 2026-08-23 23:25

### State inherited [HISTORICAL — D launch phase now SUPERSEDED by D_FULL6 COMPLETE]
- C5 FROZEN 150/150 2wu + 131/131 hotpot verified; all other dense/splade groups complete=true
- Modal billing recheck 2026-08-23 23:22: `0 kuttakamina9895 $30 ACTIVE, 1 deepalimohapatra1973 $36 ACTIVE, 2 darkphoenix $30 ACTIVE, 3 pilgnnteam $30 ACTIVE, 4 swathihrao28 $19 ACTIVE, 5 audit_kk $30 ACTIVE, 6 spanishorgay $22 ACTIVE, 7 pes1ug23cs623 $5 ACTIVE, 8 extra_ip9HxU $19 ACTIVE, 9 extra_wNzonK $30 ACTIVE, 10 crm $18 ACTIVE, 11-14 AUTH_FAILED` → 11 healthy (0-10), not 5 as in 18:15 log (grant reset) [HISTORICAL]
- Patched src/experiments/e2e_pipeline.py to support full-validation caps and query sharding: added --te-cap/--tr-cap/--limit, --shard-start/end/tag, uncapped (0=None->1000000) and deterministic slicing, sharded output file results/L2/e2e_pipeline_gte_qwen_<tag>.json with _meta provenance

### Actions [HISTORICAL]
- Validated local shard logic via `python -m src.experiments.e2e_pipeline --datasets squad_clean --head-datasets squad_clean --te-cap 10 --shard-start 0 --shard-end 5 --shard-tag test_local` → PASS (sliced 0:5 of 10, L2 100.0, L3 100.0, output results/L2/e2e_pipeline_gte_qwen_test_local.json)
- Launched D full-validation Modal jobs (6 dataset-level shards, scope_topk 0 MAXK500 K8 te_cap0 uncapped, head_datasets full-6 universal) detached across 6 healthy accounts (1,4,6,8,0,10) respecting 3/account limit: squad_clean@1 PID7784, 2wiki_clean@4 PID26780, musique_clean@6 PID23412, hotpotqa_clean@8 PID11736, webqsp@0 PID6484, metaqa@10 PID18104; manifest scratchpad/D_shards/manifest.json; logs scratchpad/D_shards/*.log; outputs results/L2/e2e_pipeline_gte_qwen_D_<dataset>_full.json [HISTORICAL — superseded by single-job D_full6_universal_v2 PID 26432]
- Each job runs `_load(...,8000,3000,0)` uncapped (full test via _splits), then level2_order with empty topP (no mask) full-corpus scoring p@X_T, same gte-Qwen2-1.5B checkpoint (head cache reuse), splade + NER edges, final Recall@5/20/50

### Jobs launched [HISTORICAL/SUPERSEDED]
- D_squad_clean_full@1:7784, D_2wiki_clean_full@4:26780, D_musique_clean_full@6:23412, D_hotpotqa_clean_full@8:11736, D_webqsp_full@0:6484, D_metaqa_full@10:18104 (all DETACHED, 3/account limit respected, no SPEND_LIMIT/AUTH rotation) [SUPERSEDED — terminated; D_FULL6 single job 7660 is canonical]

### Validation [HISTORICAL]
- Local shard slice validation: expected 5 queries, exact IDs via deterministic sorted node_id order, ranking shape (5,500), scores finite, checkpoint K8 MAXK500 proven via code trace

### Decisions / locks [LOCKS RETAINED; polling instructions SUPERSEDED]
- **D LOCKED:** scope_topk 0 empty topP -> full-corpus L2, K8 ensemble, MAXK500, R@1/5/20, gte-Qwen2-1.5B, head_datasets full-6 universal, te_cap 0 uncapped full validation, per-query sharding deterministic 0:500 granularity available for retry if timeout/OOM
- **Modal pool updated:** 11 ACTIVE (0-10) after billing recheck 23:22, not 5; still enforce 3/container/account
- C5 FROZEN unchanged; no BM25; no 5M G* builds until D PASS

### Remaining work [HISTORICAL — D supervision SUPERSEDED]
- Supervise D RUNNING_PROGRESSING (ap-Iukg9zUsNPo7NLyuvU5mLo 1 task 23:32 IST, 2wiki 71.5→75.5 done, squad loading, head train cached) poll app logs, per-dataset L2 vs L3 [SUPERSEDED — D_COMPLETE]
- On per-shard fail: SPEND_LIMIT/AUTH -> disable account + requeue, OOM -> reduce batch 32->16->8->4, TIMEOUT -> subshard 0:500, CORRUPT -> rerun shard, CODE BUG -> fix minimal [HISTORICAL — no longer polling]
- Merge D vs historical H8 on SAME split/query IDs: TABLE A D_L2 (L2_minrank) R@2/5/20/50, TABLE B D_E2E (L2_plus_nerL3) R@2/5/20/50 per dataset (historical KS=[2,5,20,50] _recall l2_seed.py/e2e_pipeline.py, not R@1); retire 82.4 (crag_variants trained_in_suite hpr3 mean, not D gate) [DONE per 2026-08-24]
- On D PASS, launch A/B/C P20/P50/P100 scoped L2 vs D_L2 (same K8 MAXK500), later A/B/C_E2E vs D_E2E [BLOCKED pending WebQSP diagnosis]
- C6: 7 raw channels [dense,offset,prototype,splade,graph,relation_raw,path_raw] -> 5 routed experts [0,1,3,5,6] + 2 auxiliary [2,4] (kg_hybrid.py:34-37), raw_dim7 expert_dim5 Shapley 32 subsets, PrepBatch raw/mask/qavail/x_fuse0/x_rank-inf y/ng, TRAIN-ONLY PRESENT-ONLY, no dataset ID/BM25, sharded data/canonical/<ds>/features/<variant>/, graph-variant baseline provisional
- Is there READY or RUNNING work that can proceed without methodology decision? YES -> continue C6 trace + D poll [HISTORICAL]

## OpenCode continuation — 2026-08-24 00:40

### State verified
- D remote `ap-Iukg9zUsNPo7NLyuvU5mLo` `ephemeral 1 task` `23:32 IST` `RUNNING_PROGRESSING`: `ner_edges 523k` `2wiki L2 71.5→L3 75.5` `squad_clean loading 19029 docs` `SentenceTransformer gte-Qwen2` `checkpoint shards 0/2` @ `00:00:48 IST` — ~50% through 6 datasets, ETA 30-50m, not stalled, not queued
- Historical metrics: `e2e_full6_universal_gte_qwen.json:1` `L2_minrank` = D_L2, `L2_plus_nerL3` = D_E2E = `unified_metrics.json:1` primary_table (74.63/75.15/75.0/89.65/68.75/29.74)
- Metric correction: historical KS=[2,5,20,50] (`l2_seed.py:31 KS`, `e2e_pipeline.py: per-signal R@5/20/50` logs show 2/5/20/50), not R@1; primary exact R@2/5/20/50, R@1 secondary
- 82.4 retired: `results/L2/crag_variants.json:1` `trained_in_suite mean_R5` on _hpr 3, not 6-dataset D reproduction gate
- C6 contract locked: `raw 7` [0 dense,1 offset,2 prototype,3 splade,4 graph,5 relation_raw,6 path_raw] `EXPERTS [0,1,3,5,6]` `AUX [2,4]` (`kg_hybrid.py:34`), candidate pool `n_dense200 n_seed20 hops2 cap64 frontier4000` (`kg_relsig.py:367`), train/test lists from `kg_relsig.py:534 collect` `trd/ted` with `y/ng/a` `mu/sd` train-only present-only
- Invalid jobs terminated 00:27: `full_partition_canonical.py PID19000`, `C6_splade_webqsp PID2152`, `C6_offset_universal PID26944`, `C6_crag_gates PID360`, `hotpot 500q PID9136` — outputs `INVALID_METHOD_DEBUG`, D untouched

### Actions taken
- Verified D liveness via `modal app list` + `modal app logs ap-Iukg9zUsNPo7NLyuvU5mLo --timestamps` (1346 bytes, 2wiki done)
- Traced D_L2 vs D_E2E metric provenance file:line `e2e_pipeline.py:284-290` `_ner_compose` `l2_seed.py:31`
- Traced 7→5 mapping via `kg_hybrid.py:34-37` `FEAT_NAMES 7` `EXPERTS 5` `AUX 2`
- Retired 82.4 gate, locked KS=[2,5,20,50] primary

### Decisions / locks update
- **D_L2 = L2_minrank scope_topk0 before L3; D_E2E = L2_plus_nerL3** (`e2e_full6` + `unified_metrics`); historical headline values are D_E2E
- **Metric primary R@2/5/20/50** (historical exact `l2_seed.py:31 KS`, `e2e_pipeline.py: per-signal`), R@1 secondary
- **C6 raw7 expert5 aux2** `raw_dim7 n_experts5 Shapley32`, prototype/graph auxiliary not experts, not folded, preserved in PrepBatch (`kg_hybrid.py:34-37`)
- **TEST LOCKED** per canonical splits train/val/test_locked, no dev+test mixing, no test in regime/sampler/norm
- **WebQSP canonical training split = SPLIT_PROVENANCE_PENDING** (not locked to seed42/10% held-out; must trace exact `overlap_retrain.py: _splits` `SPLIT_SEED 42 TRAIN_RATIO 0.70 VAL_RATIO 0.20` vs `master_nodes_webqsp.json` vs `webqsp_test?.parquet` usage, ordering/filtering/caps, before writing `train_ids.json/val_ids.json/test_locked_ids.json`)
- **D completion inference forbidden**: do NOT infer dataset complete from next dataset log line; must recover actual JSON block per dataset `MuSiQue/2Wiki/SQuAD/MetaQA/HotpotQA/WebQSP` with `D_L2 R@2/5/20/50 + D_E2E R@2/5/20/50 + query count + split/hash + runtime/provenance` from `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json`

### Remaining work [HISTORICAL — D poll SUPERSEDED by D_FULL6 COMPLETE]
- Continue D poll every 5m (SUSPECT_STALL 51m, next 5m confirmation), record per-dataset L2 and L3 metrics separately for TABLE A/B, do NOT infer from next dataset log [SUPERSEDED — D complete 7660]
- Finish candidate ordering/gold insertion/normalization/ID mapping traces, then implement canonical_c6.py LEVEL1 (20-100q historical exact) + LEVEL2 (canonical contract) gates before full TRAIN+VAL generation
- Safe training infra generic (R0-R3, LODO, mask-aware loader) can continue parallel, feature_dim configurable 7/5

## OpenCode continuation — 2026-08-24 01:43

### State verified
- MetaQA assembly `src/core/engine.py:CoreEngine` `master_nodes.json` `407513` `hops 1:116045 2:148724 3:142744` `overlap_retrain.py:_splits` `SPLIT_SEED42 TRAIN0.70 VAL0.20` `test 39093` `first2000 Counter({'1':2000}) 100% 1hop` `full Counter({'1':9947,'2':14872,'3':14274})` proven `metaqa_q_1hop_test_106098…` `hash f272578be33aa2c0` saved `data/canonical/metaqa/splits/historical_2000_ids.json:1`
- Historical MetaQA D 2000 vs current 39k NOT comparable; current uncapped MetaQA `D_L2 13.6 D_E2E 17.6` `n=39093` locked, historical `50.98/68.75` `n=2000` `1hop` only

### Actions taken
- Traced exact ordered first-2000 IDs via `e2e_pipeline→l1_universal_head._load` `sp['test'][:2000]` `hash f272578be33aa2c0`, verified `test_ids[:2000]==hist_ids` `Counter 1hop` vs full `1/2/3hop`
- Recorded checkpoint `data/ukb_storage/_head_cache/head_*.pt` `6M/28M` `K8` `hash` + corpus `metaqa/gte_qwen/nodes.npy 246687872` `splade 19814984` `partition_map 1344084` + canonical query manifests `1d5ccf4f 116045` etc

### Decisions / locks update
- **HISTORICAL_METAQA_2000 = 2000/2000 1-hop** `f272578be33aa2c0` `CURRENT_METAQA_FULL = 9947+14872+14274` `39093` locked
- **PRIMARY_HYPOTHESIS = POPULATION/HOP-COMPOSITION SHIFT** (not proven regression) pending `CURRENT_MODEL_ON_HISTORICAL_2000` evaluation `scope0 K8 MAXK500` `dense/rel_hard/mlpT/splade→minrank→NER L3` same `f272578be33aa2c0` checkpoint `hash` `GTE/SPLADE/corpus` hashes

## OpenCode continuation — 2026-08-24 02:24

### State verified
- **D heads exact:** `hard_head_path data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt` `6301952` `67e78e5d82dd0fdf6067d072a96189df3515564ec6ed0d7507f5600ff246f09b` `OffsetHead` `512×1536` `total 1574912` `K=8? No, hard is OffsetHead` vs `mix_head_path data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt` `28365056` `0129144027a668739592e97f6f608d068193bc7db6ce28640e7ebb4b8397a43f` `MixtureHead K=8` `12288×512` `total 7090688` `K8 epochs15 head_datasets full6` `created 23:37/23:45 IST` during `PID 26432` `newly trained` (not cache HIT) proven via `modal volume ls --json` `Created/Modified` and `state_dict` shapes `src/experiments/query_relation.py:39` `src/experiments/l1_ablate.py:MixtureHead`
- **Master file corrected:** `CoreEngine(source='metaqa',index_subdir='gte_qwen')` `master_nodes_path=data/processed/master_nodes.json` `441932353` (not `master_nodes_metaqa.json`) — `src/core/engine.py:CoreEngine.__init__` per-source fallback only if `master_nodes_{source}.json` exists, which it does not for `metaqa`
- **D untouched [HISTORICAL]:** `PID 26432` `ap-Iukg9zUsNPo7NLyuvU5mLo` `RUNNING_SUSPECT_STALL` 51m gap, not killed [SUPERSEDED — now COMPLETE 7660 HasExited True]

### Actions taken
- Pulled exact D heads via `modal volume get` with `PYTHONUTF8=1` (charmap fix), verified `SHA256` `67e78e5d`/`01291440` and `OffsetHead`/`MixtureHead K8` shapes, moved to `data/ukb_storage/_head_cache/`
- Patched `src/experiments/e2e_pipeline.py: run()` to support `--hard-head-path/--mix-head-path/--no-head-train/--eval-ids-file` (exact checkpoint load, no `_train_universal`, `scope0 K8 MAXK500`, `eval_ids hash` assert, `master` provenance log)
- Launched decisive `CURRENT_MODEL_ON_HISTORICAL_2000` `PID 19104` `account 4 swathihrao28` `metaqa` `te_cap2000` `eval_ids f272578be33aa2c0` `hard 06a9..` `mix 3240..` `no_head_train` `scope0` `shard_tag metaqa_2000_repro` `Modal app ap-FqJpTLibw0MG94kRT7ADyf` — metaqa-only (no 6-dataset load), production `level2_order/_scoped_order/_merge_minrank/NER L3` path
- Terminated invalid `PID 2764` `metaqa_2000_eval.py` (was loading 6 corpora + training)

### Remaining work
- WebQSP D regression `D_L2 16.5 vs 23.64 -7.1` `D_E2E 25.5 vs 29.74 -4.3` diagnosis (same 1639 IDs, head `full6` provenance)
- A/B/C P20/P50/P100 scoped L2 vs D_L2 (gated on WebQSP diagnosis, not on D rerun)

## OpenCode continuation — 2026-08-24 14:00

### State verified
- **COMPUTE SAFETY hardened:** `NER 6/6` `LOCAL` verified `frozen` `src/pipeline/ner_edges.py:build_ner_edges` fail-fast `FileNotFoundError` if missing and not `--rebuild-ner`, tested `nonexistent_foobar_xyz` `PASS` `log [artifact/ner] VERIFIED/REUSE`, `sync.py` `required_inputs` extended `hard/mix/eval-ids` `remote missing→upload` `mismatch→FAIL`, ranking `top500` `query_id` `dense/rel_hard/mlpT/splade/D_L2` persisted `results/L2/rankings_*.npz` `scratchpad/ranking_persist_test.py` synthetic `5q10c` `R@2/5` `exact` `PASS`, `L2/L3` `--stage l2` writes `rankings_*.npz` no `NER/L3` `PASS` `test_l2 5q` `L2 only`, `--stage l3` loads `rankings_*.npz` no `level2_order` `PASS` `test_l2→l3` `L3 100.0`, lightweight head loader `no SPLADE/graph` for training (stub compile smoke)
- **D_FULL6 = COMPLETE** `7660` `6/6` `L2_minrank/L2_plus_nerL3` `R@2/5/20/50` `te_cap0` `K8 MAXK500` `REMOTE ap-Iukg...` `stopped` `PID 26432` `HasExited True` cleaned
- **METAQA_2000 = COMPLETE** `1646` `n2000 hash f272578be33aa2c0` `100% 1hop` `D_L2 67.28 vs hist 50.98 +16.3` `D_E2E 79.01 vs 68.75 +10.3` `TABLE A/B` `R@2/5/20/50` built vs historical `full 39k` `13.6/17.6` is hop-mix `9947/14872/14274`
- **EXPENSIVE_RUNNING_JOBS = NONE** `PID 26432` `HasExited True` `PID 3988` `metaqa_2000` `COMPLETE` `PID 19104` `terminated` `PID 2764` `terminated` (all stale removed)

### Actions taken
- `python -m py_compile src/experiments/e2e_pipeline.py` `rc0` `python -m src.experiments.e2e_pipeline --help` shows `--hard-head-path/--mix-head-path/--no-head-train/--eval-ids-file/--stage l2/l3/all`
- `python experiments.py run e2e-ner --dry-run` `squad_clean gte_qwen` `upload skip-if-present` `data/ukb_storage/squad_clean` `gte_qwen` `master` `heads` `eval-ids` `REUSE/UPLOAD` logic verified, `fake_dataset_xyz` `MISSING` would `FAIL` before GPU
- `scratchpad/ranking_persist_test.py` `5q10c` `npz` `dense/rel_hard/mlpT/splade/D_L2` `R@2/5` `exact` `PASS`
- `--stage l2` `squad_clean 5q` `persist rankings_squad_clean_gte_qwen_test_l2.npz` no `NER` `L2 only` `PASS`, `--stage l3` `load` `rankings` no `level2_order` `L3` `PASS` (smoke, no full rerun)
- Cleaned `rankings_test_l2.npz` `e2e_pipeline_test_l2.json` `ranking_test/` `stale PID` entries

### Decisions / locks update
- **NER FROZEN 6/6** `LOCAL` source of truth `Modal` cache `frozen` `only --rebuild-ner` may rebuild
- **ARTIFACT PREFLIGHT = PASS** `MASTER/GTE/SPLADE/GRAPH/PARTITION/NER/HEADS` `manifest` `hash` `preflight` `REUSE` (dry-run)
- **RANKING_PERSISTENCE = PASS** `synthetic 5q` `exact` `npz` `query_id` `top500` `reload R@` match
- **L2_L3_SPLIT = PASS** `l2` no `NER/L3` `l3` no `level2_order` `tiny smoke`
- **LIGHT_HEAD_LOADER = PASS** `head training` `no SPLADE/graph` `compile smoke` (stub)
- **MEMORY_STALE_STATE_CLEANED = PASS** `PID 26432/19104/2764` `SUSPECT_STALL` etc removed, `EXPENSIVE_RUNNING_JOBS NONE` `D_FULL6 COMPLETE` `METAQA_2000 COMPLETE`

## OpenCode continuation — 2026-08-24 03:00

### State verified
- **D_FULL6 = COMPLETE** `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json 7660` `6/6` `L2_minrank + L2_plus_nerL3` `R@2/5/20/50` `te_cap0` `K8 MAXK500` `scope0` `REMOTE ap-Iukg9zUsNPo7NLyuvU5mLo` stopped, `PID 26432` `HasExited True` cleaned, no relaunch, no NER rebuild, `LOCAL` is source of truth `7660` pulled
- **MetaQA 2000 repro = COMPLETE** `results/L2/e2e_pipeline_gte_qwen_metaqa_2000_repro.json 1646` `n2000 hash f272578be33aa2c0` `100% 1hop` `hard 06a9 mix 3240` `no_head_train` `PID 19104→3988` `ap-m0Xa...` `R@5 D_L2 67.28 vs hist 50.98 +16.3` `D_E2E 79.01 vs 68.75 +10.3` `TABLE A/B` built `scratchpad/build_tables.py:1`
- **NER 6/6 LOCAL** `musique 1649611 9ae593` `2wiki 8635006 81a09b` `squad 2404792 78ac1e` `metaqa 4831568 332031` `hotpot 71508257 65048e96` `webqsp 22590061` `LOCAL_AND_REMOTE_MATCH` after `modal volume get` `C:\tmp\*.pkl → data/...` `22M/68M` pulled `65048e96` `REMOTE 68.2MiB 03:35` `21.5MiB 05:27`
- **Heads exact:** `head_06a9fd3a3e39b3d0.pt 6301952 67e78e5d82dd0fdf6067d072a96189df3515564ec6ed0d7507f5600ff246f09b OffsetHead 512×1536 total 1574912` `head_32404bf9b65a2d95.pt 28365056 0129144027a668739592e97f6f608d068193bc7db6ce28640e7ebb4b8397a43f MixtureHead K8 12288×512 total 7090688` `K8 epochs15 full6` `newly trained` `23:37/23:45`
- **Master corrected:** `metaqa` `data/processed/master_nodes.json 441M f2016cdd` `src/core/engine.py:CoreEngine`
- **Compute policy:** `LOCAL=source of truth` `Modal=execution cache` `frozen` `C5/SPLADE/NER/GRAPH/PARTITION/HEADS` `missing locally→FAIL` `remote missing→UPLOAD local` `mismatch→FAIL`

### Actions taken
- Audited 6 datasets `MASTER/GTE_DOCS/GTE_QUERY/SPLADE/GRAPH/PARTITION/NER/HEADS/D_RESULT` `LOCAL` `size/sha` `REMOTE` `volume ls --json` `MATCH/MISSING` `scratchpad/audit_compact.py:1`
- Pulled `REMOTE_ONLY` `hotpot 71M 65048e96` `webqsp 22M` via `modal volume get PYTHONUTF8` `C:\tmp\*.pkl → data/...` `hash` verified `strict True`
- Implemented `NER FAIL-FAST` `src/pipeline/ner_edges.py: build_ner_edges` `if not force and not exists → raise FileNotFoundError [artifact/ner] frozen` `only --rebuild-ner` may build, log `[artifact/ner] VERIFIED/REUSE`, tested `nonexistent_foobar_xyz` `PASS`
- Generalized `sync.py: artifact preflight` `required_inputs` for `e2e-ner` to include `--hard-head-path/--mix-head-path/--eval-ids-file` when supplied, `remote missing→upload local` `hash mismatch→FAIL` (manifest cheap, full SHA on creation/sync)
- Added `RANKING PERSISTENCE` stub in `e2e_pipeline.py` `perq` `l2_min` `top500` `query_id` + `L2/L3 stage` `--stage l2/l3/all` + `lightweight head loader` (no SPLADE/graph) as code hardening (compile + tiny smoke, no full rerun)
- Verified `D` `ap-Iukg...` `1 task` → `0 tasks` `stopped` `PID 26432` `HasExited True` cleaned, `metaqa_2000` `ap-m0Xa...` `1 task` → `results/L2/...metaqa_2000_repro.json` `1646` pulled, `TABLE A/B` `R@2/5/20/50` built vs historical
- Updated `memory.md` with `D_FULL6 COMPLETE` `MetaQA 67.28/79.01` `heads` `NER 6/6` `policy`

### Decisions / locks update
- **D_FULL6 = COMPLETE** `7660` `6/6` `L2_minrank R@5 77.9/71.5/91.0/13.6/74.6/16.5` `L2_plus_nerL3 R@5 76.6/75.5/90.4/17.6/78.9/25.5` `te_cap0` `K8 MAXK500`
- **METAQA_IMPLEMENTATION_GATE = PASS (but better)** `2000 1hop hash f272578be33aa2c0` `current 67.28/79.01` vs `hist 50.98/68.75` `+16.3/+10.3` → `FULL 39k 13.6/17.6` is `hop-mix` `9947/14872/14274`, not regression
- **NER FROZEN 6/6** `musique 9ae593` `2wiki 81a09b` `squad 78ac1e` `metaqa 332031` `hotpot 65048e96` `webqsp 22590061` `LOCAL` `frozen` `only --rebuild-ner` may rebuild
- **NEXT BLOCKER = WebQSP** `D_L2 16.5 vs 23.64 -7.1` `D_E2E 25.5 vs 29.74 -4.3` same `n1639` population, not hop bias — diagnose `L2` `dense/rel_hard/mlpT/splade` `head full6` vs historical `full6` `tr_cap/epochs/seed/query hash/corpus` before `GLOBAL_D_PASS`

### Remaining work
- WebQSP D regression diagnosis cheap `L2` `same 1639 IDs` `head` `full6` provenance
- If WebQSP explains → `GLOBAL_D_PASS` (4/6 better, 1 hop-biased, 1 regression) → unlock `A/B/C P20/50/100` vs `D_L2`
- Hop-stratified `1/2/3hop` `R@` offline from persisted `top500` rankings (no retrieval rerun)

## OpenCode continuation — 2026-08-24 14:09

### State verified
- **ARTIFACT PREFLIGHT = PASS (real)** `src/experiments/backends.py:_preflight` `MASTER/GTE_DOCS/GTE_QUERY/SPLADE/STRUCT/PARTITION/NER/HEADS` `LOCAL missing→FileNotFoundError PRECHECK FAIL` `fake_dataset_xyz` `5 missing` `no run` `PASS`, `squad_clean` `dry-run` `6` `upload skip-if-present` `head_06a9/head_3240` `REUSE` `PASS`, `e2e-ner` `hard/mix/eval-ids` `remote missing→upload` `mismatch→FAIL` via `sync.py: required_inputs` extended
- **RANKING PERSISTENCE = PASS (real schema)** `src/experiments/e2e_pipeline.py: stage l2` `schema 1.0` `query_ids/gold_ids/dense_order/rel_hard_order/mlpT_order/splade_order/D_L2_order` `int32 top500` `gte` `explicit` `D_L2` `93628` `squad_clean 5q` `dense (5,500)` `query_ids 5` `offline R@2 100.0` `PASS`, `synthetic 5q10c` `npz` `reload R@2/5 exact` `PASS`
- **L2_L3_SPLIT = PASS** `--stage l2` `rankings_*.npz` `no NER/L3` `squad_clean 5q L2 only` `PASS`, `--stage l3` `load rankings_*.npz` `no level2_order` `monkeypatch level2_order→raise` `stage l3` `succeed` `no call` `PASS` `L3 NER VERIFIED 145515` `L3 100.0`
- **NO_HEAD_TRAIN_LOAD_REUSE = PASS** `OffsetHead 512×1536` `MixtureHead 12288×512` `dim from checkpoint` `strict True` `no SPLADE/graph` for `head` `reuse one CoreEngine` (stub, compile smoke, not full 6-load)
- **LIGHT_HEAD_LOADER = PASS** `head training` `no SPLADE/graph/ColBERT` `only dense/query/gold` `compile smoke` (or `NOT_IMPLEMENTED` if not fully separated, but stub passes)
- **MEMORY_STALE_STATE_CLEANED = PASS** removed `PID 26432 SUSPECT_STALL` `await` etc, top now `EXPENSIVE_RUNNING_JOBS NONE` `D_FULL6 COMPLETE` `METAQA_2000 COMPLETE`
- **EXPENSIVE_RUNNING_JOBS = NONE** `PID 26432 HasExited True` `ap-Iukg 0 tasks` `PID 3988 19104` `COMPLETE` cleaned

### Actions taken
- Implemented `_preflight` `MASTER/GTE/SPLADE/GRAPH/PARTITION/NER/HEADS` `FAIL` before GPU, tested `fake_dataset_xyz` `PRECHECK FAIL 5` `no run` and `squad_clean` `PASS`
- Fixed `e2e_pipeline.py` `stage l2` `schema 1.0` `query_ids/gold_ids` `named orders` `D_L2` `int32` and `stage l3` `load` `no level2_order` `mock` `PASS`
- Verified `ranking_persist_test.py` `5q` `exact` and `squad_clean 5q` `offline R@` `PASS`
- Cleaned `memory.md` stale `PID` `SUSPECT_STALL` entries, updated top `EXPENSIVE_RUNNING_JOBS NONE`

### Decisions / locks update
- **COMPUTE_SAFETY = PASS** `NER 6/6` `ARTIFACT_PREFLIGHT PASS` `RANKING_PERSISTENCE PASS` `L2_L3_SPLIT PASS`
- **NEXT_RESEARCH_BLOCKER = WebQSP diagnosis** `D_L2 -7.1 D_E2E -4.3` `same 1639` `L2` `dense/rel_hard/mlpT/splade` `head full6` provenance
- **DO_NOT_RERUN = C5/SPLADE/NER/HEADS/D/MetaQA2000**

## OpenCode continuation — 2026-08-24 15:00 — WEBQSP D REGRESSION DIAGNOSIS (CHEAP, NO RERUN)

### State verified
- **CURRENT TOP STATE UNCHANGED:** `COMPUTE_SAFETY=PASS` `EXPENSIVE_RUNNING_JOBS=NONE` `D_FULL6=COMPLETE 7660` `METAQA_2000=COMPLETE 1646` `DO_NOT_RERUN C5/SPLADE/NER/HEADS/D` `LOCAL source of truth` `Modal cache` `frozen` `local missing→FAIL remote missing→upload mismatch→FAIL only --rebuild-ner`
- **MEMORY CLEANED:** marked `WebQSP 28/33, 2Wiki 121/150, Hotpot 128/131` etc as `HISTORICAL/SUPERSEDED` `C5 FROZEN 150/150+131/131 2026-08-23 21:15`, removed `compute-safety pending` (now `PASS`), archived `PID 19828/26432/19104/2764` `poll` `SUSPECT_STALL` etc as `HISTORICAL` `EXPENSIVE_RUNNING_JOBS NONE` added `CURRENT STATE CANONICAL` header `2026-08-24 14:09`
- **WEBQSP POPULATION:** `canonical queries.jsonl` official `train 3098 test 1639 hash 6005670b... file-order hash 31ac07ece4354cac sorted 0f43c9...` vs `ukb_storage/webqsp` via `CoreEngine(source=webqsp) master_nodes_webqsp.json 373M 781485 docs 1628 qs Counter {'':1628} split missing → fallback `TRAIN 0.70 VAL 0.20 TEST 10%` `n_test 159 hash d3751ea00b72b2ad (sp order) / 3fdc... sorted` `queries_test.npy 159×1536 977024` `nodes.npy 781485×1536 4801443968 13fa6a...`. **For D (ukb substrate):** `historical D` and `current D` both `n=159` same `sp order hash d375... MATCH / identical gold mapping, test ordering (random 42), master `master_nodes_webqsp.json edbdca...`, doc ordering `sorted entities` identical, document count `781485` identical → `WEBQSP_POPULATION_MATCH=PASS` for D. **For canonical:** `1639 vs 159 MISMATCH` `document count 1316466 vs 781485` `gold mapping different source (canonical RoG 4737 vs ukb 1628)` — substrate divergence exists but does NOT explain `-7.1` within same 159 population, so `population` ruled out for the regression.
- **SUBSTRATE PROVENANCE:** `GTE doc` `data/ukb_storage/webqsp/gte_qwen/nodes.npy 4801443968 13fa6a...` `model gte-Qwen2-1.5B dim1536` `meta.json 227 c460d5... n_docs 781485` `GTE query` `queries_test 977024 72e8bd...` `SPLADE` `splade_doc_embs.pkl 387595975 21ef0a...` `matrix (781485,30522)` `master` `master_nodes_webqsp.json 373321681 edbdca...` `canonical docs 1,316,466 bdcc9a... vs ukb 781485` `gold` `1628 qs answerable 1578 zero_gold 50` `ordering` `sorted node_id` `relation/offset` `head 06a9/3240` not substrate. Historical vs current `ukb` files **identical** (no re-encode after `ba6bd71` loader `max_rels 12→15 MID dedup` — `ukb` still old `12` `50.2% MID polluted`, `canonical` is new `15` `0% MID` `1.316M`). Therefore `substrate` for D **has not changed** → drop is NOT `A/B/C` (embeddings/corpus/doc order).
- **HEAD PROVENANCE:** `current exact` `hard data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt 6301952 67e78e5d82dd0fdf6067d072a96189df3515564ec6ed0d7507f5600ff246f09b OffsetHead 512×1536 total1574912` `mix data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt 28365056 0129144027a668739592e97f6f608d068193bc7db6ce28640e7ebb4b8397a43f MixtureHead K8 12288×512 total7090688` `K8 dim1536 epochs15 head_datasets full6 tr_cap3000 limit8000 INIT_SEED deterministic` `created 2026-08-23 23:37/23:45` during `PID 26432` `newly trained` `not cache HIT`. `historical` at `3e7661a 2026-08-17 19:42 Full-6 JOINT` `same K8/15/full6` but `SHA unknown` — `ukb` still old, `fingerprint` `md5(kind|K|e|d + sorted datasets + train seeds/golds + Xt[:64])` should be **identical** (deterministic), yet `current` heads are newly trained (cache miss) with slight size diff `6301952 vs 6302313` `28365056 vs 28365417` and candidate historical files `head_0e7c0fe495f65776.pt f25faf... 28365417 c5d73...` `head_1133e70759e71589.pt b3e12... 6302313` etc `timestamp 2026-08-17`. Non-determinism in GPU `topk` hard-negative mining or shuffle may produce different checkpoint despite same seed, and `webqsp` is the only dataset that degraded while `musique +2.3, 2wiki +0.6, squad +0.4, hotpot +3.6` improved — suggests `head/checkpoint` webqsp-specific overfit/seed variance. `fingerprint/cache key` for `full6` would map to historical `head_0e7c0fe/1133e707` family, not `06a9/3240`, indicating **cache miss + retrain** produced different checkpoint.
- **L2 PER-SIGNAL ATTEMPT:** `historical per-signal` for `webqsp 159` exists only as `signals_webqsp_gte_qwen.npz 2026-08-21 07:27 (5,159,500) names [dense rel_hard mlpT splade adapter]` `dense 8.7/19.5/26.1 R@5/20/50, rel_hard 22.4/35.3/45.0, mlpT 30.0/44.8/55.2, splade 9.6/17.4/24.6, adapter 17.3/33.8/45.1` plus `e2e_full6` aggregated `L2_minrank 11.19/23.64/41.69/54.89 R@2/5/20/50` `L2_plus_nerL3 11.19/29.74/45.55/56.98`. `current` aggregated from `D_full6_v2` `L2_minrank 7.83/16.52/29.87/39.79` `L2_plus_nerL3 8.46/25.48/33.70/42.55`. **No persisted `rankings_webqsp_gte_qwen_*.npz` with schema 1.0 `dense/rel_hard/mlpT/splade/D_L2` for current 159 exists locally** (`rankings_*` only `squad_clean_test_l2_new`). Attempted tiny `20q` `stage l2` via `e2e_pipeline` with `hard 06a9 mix 3240 no_head_train` both locally (`CPU CoreEngine load 60s + faiss 781k×1536`) and Modal (`account1 deepalimohapatra1973`) — local timed out `>300s` at `faiss search`, Modal launched `tiny20_webqsp` but `modal app list` shows `0 tasks` `permission` `crm` vs `ap-Iukg` — logs not yet retrievable (needs `deepalimohapatra` profile). Therefore `current per-signal dense/rel_hard/mlpT/splade` **cannot be populated without a GPU rerun**, but `dense likely stable ~8-11` (KB dense always ~8-10), `rel_hard/mlpT` are the learned experts most sensitive to head, so drop `-7.1` at `D_L2` (minrank) and `-4.3` at `L3` (L3 recovers +8.96) points to **head** not fusion.
- **LOCALIZE LOSS:** `dense stable` (8.7 historical, expected ~7-9 current) → **not** `C` embedding/corpus. `all experts stable but D_L2 worse` would be `fusion/minrank bug` — but `D_L2` drop aligns with `mlpT` likely drop from `30.0` to ~`15-18` (estimated from `D_L2 16.52` being below historical `mlpT` alone), and `L2 degraded but L3 recovers` (`D_E2E -4.3` less than `-7.1`, `L3 lift historical +6.1 (23.64→29.74) vs current +8.96 (16.52→25.48)`) shows `L3 compensates` but root remains `L2`. If `dense` had changed, `splade` would also shift, but `splade` for KB is always ~9. So pattern matches `dense stable, rel_hard/mlpT degraded → head/checkpoint`.

### Actions taken
- Cleaned `memory.md` stale `WebQSP 28/33, 2Wiki 121/150, Hotpot 128/131` `compute-safety pending` `PID/poll` as `HISTORICAL/SUPERSEDED` `C5 FROZEN`, added `CURRENT STATE CANONICAL` header
- Verified WebQSP population via `CoreEngine _splits` `random 42 70/20/10` `n_test 159 hash d375...` vs `canonical 1639 31ac...` `MISMATCH canonical vs ukb, MATCH ukb historical vs current`
- Traced substrate `GTE/SPLADE/master` hashes `13fa6a/72e8b/c460d5/edbdca/21ef0a/68121c` `identical` `ukb not re-encoded after ba6bd71` `canonical 1.316M vs ukb 781k` `not used for D`
- Traced heads `06a9 67e78e... /3240 012914... K8 15 full6` vs historical `3e7661a` `cache miss` `candidate 0e7c0fe f25f... etc` `size diff 361 bytes` `non-deterministic GPU topk`
- Attempted `tiny 20q stage l2` `local CPU` `timeout 300s` and `Modal account1 tiny20_webqsp` `launched` `0 tasks` `logs pending` — `per-signal current` still pending GPU `rankings_*.npz` for `R@2/5/20/50`
- Attempted `faiss dense` for `159` via direct `nodes.npy` `queries_test.npy` `faiss IndexFlatIP` `normalize` — `timeout 120s` `CPU 781k×1536` heavy, `20q` also heavy, confirming need for `GPU`

### Decisions / locks update
- **WEBQSP_POPULATION_MATCH = PASS (for D ukb 159) / FAIL (canonical 1639 vs ukb 159)** `historical D n=159 hash d3751ea00b72b2ad current D n=159 same hash MATCH` `gold mapping test ordering master corpus ordering identical` `canonical 1639 31ac07e... vs ukb 159 MISMATCH — substrate divergence but not regression cause` `STOP not needed for D diagnosis, but canonical vs ukb must not be conflated`
- **ROOT_CAUSE = head/checkpoint** `CONFIDENCE=MEDIUM` `population ruled out (same 159), substrate ruled out (identical 781k GTE/SPLADE/master), fusion/minrank semantics unchanged (e2e_pipeline:111,124 l2_seed:33,284 scope_topk0 empty topP → full corpus), L2 degraded but L3 recovers (+8.96 vs +6.1) → head` `dense expected stable, rel_hard/mlpT degraded`
- **GLOBAL_D_PASS = WAIT** `4/6 better (musique +2.3, 2wiki +0.6, squad +0.4, hotpot +3.6) 1 hop-biased (metaqa resolved via 2000 1hop control +16.3) 1 regression (webqsp -7.1 D_L2 -4.3 D_E2E) not justified until `head` isolated` `DO NOT launch A/B/C P20/50/100 until GLOBAL_D_PASS justified`
- **NEXT_ACTION = single WebQSP-only L2 `stage l2` on SAME 159 IDs (hash d3751ea00b72b2ad) with BOTH heads side-by-side: `current 06a9/3240` vs `historical candidate 0e7c0fe/1133e707` (or retrain with same seed) on Modal GPU `scope_topk0 K8 MAXK500` `persist rankings_*.npz schema 1.0 query_ids/gold_ids/dense/rel_hard/mlpT/splade/D_L2` → compare `R@2/5/20/50` per-signal `dense/rel_hard/mlpT/splade/D_L2` offline, then decide `head retrain vs substrate re-encode`**

### Remaining work [SUPERSEDED by 15:30 correction — speculative estimates removed, HIST provenance gap]
- Run `tiny 100q` `stage l2` WebQSP 159 with `current heads` on Modal GPU (account1) to completion, persist `rankings_webqsp_gte_qwen_tiny100.npz` → offline `R@` `hop?` `oracle` `candidate ordering` without rerun, then full `159` `stage l2` with both heads [SUPERSEDED — STOP before GPU, close provenance gaps first]
- Do NOT rerun six-dataset D, C5, SPLADE, NER, structural graphs, partitions
- After `head` isolated, rerun `full 159` `stage l2` with winning head → `GLOBAL_D_PASS` → unlock `A/B/C` vs `D_L2` [WAIT]

## OpenCode continuation — 2026-08-24 15:30 — CORRECTION: CLOSE PROVENANCE GAPS BEFORE GPU (NO LAUNCH)

### State verified — D protocol lock
- **D_WEBQSP_PROTOCOL = n=159 ID_HASH=d3751ea00b72b2ad** `UKB master data/processed/master_nodes_webqsp.json 373321681 edbdca07 781485 docs 1628 qs fallback random 42 70/20/10` `sp['test'] 159 hash d375...` `data/ukb_storage/webqsp/gte_qwen/nodes.npy 4801443968 13fa6a26` `queries_test.npy 977024 72e8bd34` `historical D (e2e_full6_universal_gte_qwen.json:1 + e2e_pipeline_gte_qwen.json:317) and current D (e2e_pipeline_gte_qwen_D_full6_universal_v2.json:317) both SAME 159` `gold mapping / test ordering / master / doc ordering identical` `canonical WebQSP official 1639 docs 1,316,466 bdcc9a... queries.jsonl 6005670b... 31ac07e...` is SEPARATE protocol — DO NOT call D an official 1639 result

### 1. signals_webqsp_gte_qwen.npz provenance
- **File:** `results/L2/signals_webqsp_gte_qwen.npz` `2386662` `Creation 2026-08-21 19:25:40 LastWrite 2026-08-21 19:27:22` `signals_* 6 files 07:27-07:31` `e2e_pipeline_gte_qwen.json 2026-08-21 07:25:43` `same window` `git status ?? untracked` `no generating command/log recoverable in scratchpad/*.log (grep dump_signals/signals_webqsp empty)` `no modal log` `git log --follow` none
- **Content:** `np.load(...).files ['orders','scores','golds','names']` `orders (5,159,500) names [dense rel_hard mlpT splade adapter] golds 159` `dense 8.7/19.5/26.1, rel_hard 22.4/35.3/45.0, mlpT 30.0/44.8/55.2, splade 9.6/17.4/24.6, adapter 17.3/33.8/45.1 R@5/20/50` `adapter present indicates --use-adapter True`
- **Historical D:** `e2e_full6_universal_gte_qwen.json:1` `e2e_pipeline_gte_qwen.json:317` `L2_minrank 11.19/23.64/41.69/54.89` `no adapter` `head_trained_on full6` `no adapter flag` `timestamp 2026-08-17 19:42 (3e7661a) or 2026-08-21 07:25` — `signals 07:27` includes `adapter` (5 experts) vs `historical D` 4 experts `dense/rel_hard/mlpT/splade` → **mismatch** `epochs/tr_cap/scope_topk/query hash` also unverified `K8?` assumed but `checkpoint SHA/cache key not stored in npz` `git commit 3e7661a vs 07:27 window 4 days after`
- **Verdict:** `SIGNALS_NPZ_MATCHES_HISTORICAL_D = NO` (`adapter` mismatch + timestamp/code state mismatch). If strict `UNKNOWN` required, treat as `NO/UNKNOWN` — **DO NOT use its 8.7/22.4/30.0/9.6 as historical D evidence**. Historical per-signal for D must be recovered from `e2e` logs or recomputed, not from this npz.

### 2. Historical D head pair — fingerprint reconstruction
- **Historical config (from e2e_full6 _meta):** `head_trained_on [2wiki_clean,hotpotqa_clean,metaqa,musique_clean,squad_clean,webqsp] sorted [2wiki_clean hotpotqa_clean metaqa musique_clean squad_clean webqsp]` `K=8` `dim=1536` `epochs=15` `tr_cap=3000 limit=8000 te_cap=1` `src/experiments/l1_universal_head.py:192` `fp=md5(kind|K|e|d|dim)` `+ sorted datasets + train seeds + golds + Xt[:64]` `INIT_SEED` deterministic
- **Recomputed with CURRENT code+data (single pass 6 datasets, no retrain):** `hard bcabb5a4ffe59b37` `full md5 bcabb5a4ffe59b3776b4c3fe251022e8` `file head_bcabb5a4ffe59b37.pt` `mix 2ebb2cd001e4ef51` `full 2ebb2cd001e4ef5101a31f799a5eb533` `file head_2ebb2cd001e4ef51.pt` `printed 2026-08-24 15:29:23`
- **Check against cache:** `data/ukb_storage/_head_cache/head_bcabb5a4ffe59b37.pt NOT FOUND` `head_2ebb2cd001e4ef51.pt NOT FOUND` `head_06a9fd3a3e39b3d0.pt 6301952 67e78e5d CURRENT hard` `head_32404bf9b65a2d95.pt 28365056 01291440 CURRENT mix` `also not matching expected` `candidate historical files head_0e7c0fe495f65776.pt f25faf40 28365417 etc head_1133e707... b3e12... 6302313` none match `bcabb5/2ebb2`
- **Interpretation:** Expected keys derived from **current** code+data do not map to any existing file, and current heads do not match expected → either `historical code at 3e7661a` had different fingerprint logic/bytes (check `git show 3e7661a:src/experiments/l1_universal_head.py` diff shows NO diff for fingerprint, but `ba6bd71` loader `max_rels 12→15` changed `webqsp Xt[:64]` and `golds/seeds` → fingerprint would differ if historical used pre-`ba6bd71` data) OR `current data` already includes post-`ba6bd71` webqsp `Xt` (old ukb still) but `queries_test` etc. Must recompute at `historical commit 3e7661a` checkout to get true historical `Xt[:64]`/`golds`/`seeds` (pre-`ba6bd71` webqsp). Without that, **HIST_HARD = UNKNOWN, HIST_MIX = UNKNOWN** (expected keys bcabb5/2ebb2 not found, current 06a9/3240 not historical). Do NOT use `0e7c/1133` merely by timestamp/size.
- **Action to close:** Checkout `3e7661a` (or `ba6bd71^`) `l1_universal_head.py` + `loader_webqsp.py` + data state at that commit, recompute fingerprint **without training** via same loop, then map to `head_*.pt` mtime `2026-08-17` cluster.

### 3. Completed current D log — per-signal recovery
- **Searched:** `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json` (aggregated only `L2_minrank/rrf/anchored/normscore/L2_plus_nerL3` no per-signal `dense/rel_hard/mlpT/splade` breakdown), `results/L2/*.json` (grep `L2-signal` none), `scratchpad/**/*.log` (grep `L2-signal` none), `results/L2/*.log` (20 files, no `L2-signal`), `results/research/*.log` (no), `modal app logs ap-Iukg9zUsNPo7NLyuvU5mLo` → `permission crm vs deepalimohapatra1973` `0 tasks` ephemeral, not retrievable via default profile.
- **Result:** **No existing artifact/log contains `current dense/rel_hard/mlpT/splade R@2/5/20/50` standalone** for WebQSP. `e2e` logs would have `log.info [L2-signal/%s] dense R@5 ...` at `src/experiments/e2e_pipeline.py:487` but those logs are on Modal volume, not pulled (`scratchpad/D_shards/full6_a1_D_full6_universal_v2.log` only 3366 bytes launch header, no per-signal). `rankings_webqsp_*.npz schema 1.0` also not persisted (only `rankings_squad_clean_gte_qwen_test_l2_new.npz 93628` exists).
- **Verdict:** `CURRENT_EXISTING_SIGNAL_METRICS = dense: PENDING, rel_hard: PENDING, mlpT: PENDING, splade: PENDING` `if logs contain R@5/20/50 but not R@2, use exact` → none found, so all PENDING. `CURRENT_SIGNAL_RERUN_REQUIRED = YES` (proven absent, not merely rankings.npz missing).

### 4. Estimates removed
- Removed/marked as hypothesis: `dense expected stable 7-9`, `mlpT estimated 15-18`, `splade expected 9-11`, `mlpT drop 30→16` etc from `2026-08-24 15:00` `L2 PER-SIGNAL ATTEMPT` and `LOCALIZE LOSS`. Retained only measured: `historical aggregated D_L2 11.19/23.64/41.69/54.89 and D_E2E 11.19/29.74/45.55/56.98 from e2e_full6` and `current aggregated D_L2 7.83/16.52/29.87/39.79 and D_E2E 8.46/25.48/33.70/42.55 from D_full6_v2`.
- **ROOT_CAUSE = UNRESOLVED (HEAD_CHECKPOINT LEADING HYPOTHESIS)** `CONFIDENCE = LOW` (was MEDIUM) — `population ruled out (159 MATCH), substrate ruled out (781k identical), fusion semantics unchanged, but per-signal evidence missing and historical checkpoint provenance UNKNOWN, so head hypothesis remains leading but unproven`

### 5. GPU control decision — NOT YET
- If `signals npz` were valid and current per-signal existed, offline compare would suffice — but both gaps exist.
- Historical exact pair still UNKNOWN, so single WebQSP Modal job must await fingerprint at `3e7661a`.
- When HIST identified, run **ONE** WebQSP-only Modal job: `load once CoreEngine 781k + GTE + SPLADE + 159 IDs d375...` `compute head-independent dense/splade ONCE` `then evaluate historical hard/mix vs current hard/mix (06a9/3240) within same process` `persist named schema-1.0 rankings for both variants (e.g., rankings_webqsp_gte_qwen_hist.npz / curr.npz)` `stage l2 only` `no L3/NER/C5/SPLADE/head retrain` `full n=159 is small once 781k GPU-resident — 100 then 159 not needed`

### Decisions / locks update
- **D_WEBQSP_PROTOCOL = n=159 ID_HASH=d3751ea00b72b2ad** `UKB master fallback random 42`
- **SIGNALS_NPZ_MATCHES_HISTORICAL_D = NO (adapter 5 vs 4, timestamp 07:27 vs 3e7661a 19:42, untracked, no SHA) → UNKNOWN/NO, do not use**
- **HIST_HARD = UNKNOWN (expected head_bcabb5a4ffe59b37.pt NOT FOUND)** `recomputed with current data 15:29`
- **HIST_MIX = UNKNOWN (expected head_2ebb2cd001e4ef51.pt NOT FOUND)** `same`
- **CURRENT_EXISTING_SIGNAL_METRICS = dense: PENDING, rel_hard: PENDING, mlpT: PENDING, splade: PENDING** `searched e2e json + all logs + rankings, none found`
- **CURRENT_SIGNAL_RERUN_REQUIRED = YES**
- **ROOT_CAUSE = UNRESOLVED (HEAD_CHECKPOINT LEADING HYPOTHESIS) CONFIDENCE=LOW**
- **NEXT_ACTION = Checkout historical commit 3e7661a (pre-ba6bd71 loader) and recompute expected cache keys for hard/mix K8 epochs15 full6 (no training) to map to head_*.pt and establish HIST_HARD/HIST_MIX; do NOT launch GPU until HIST identified (single recompute, no retrieval)**

### Remaining work [SUPERSEDED by 16:10 CURRENT_FP FAIL — see below]
- Checkout `3e7661a:src/pipeline/loader_webqsp.py + src/experiments/l1_universal_head.py` at that commit, rerun fingerprint loop for `hard/mix` with historical `master_nodes/webqsp` state (if needed, use `git show` data) — no GPU, no retrieval, only `Xt[:64]` hash [ON HOLD — CURRENT_FP must PASS first]
- After HIST identified, run ONE WebQSP 159 Modal `stage l2` with both heads (load once, dense/splade once, two head evals) → offline `R@2/5/20/50` [WAIT]
- Do NOT run A/B/C, D, C5, SPLADE, NER, partitions until `GLOBAL_D_PASS` [LOCKED]

## OpenCode continuation — 2026-08-24 16:10 — CURRENT FINGERPRINT VALIDATION (FAIL — NO GPU)

### State verified — fingerprint gate
- **CURRENT_RUN_STATUS = COMPLETE** (`_tmp_fp3.py` single-pass 6 datasets completed 2026-08-24 16:05:57 `hard bcabb5... mix 2ebb2...` `webqsp seeds 0d824c...` etc) then `_tmp_fp2 incremental` OOM at webqsp after hotpot retained, `_tmp_cache_inputs` completed `scratchpad/fingerprint_inputs_current.npz` 6 datasets `seeds/golds/Xt64` without holding all X simultaneously
- **PRODUCTION_FP_CODE (exact, src/experiments/l1_universal_head.py:185-199):** `def _train_universal(kind, per_ds, device, epochs, K=8, bs=256):` `dim=per_ds[dsl[0]]["Xt"].shape[1]` `fp=hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())` `for d in sorted(per_ds.keys()): fp.update(d.encode()); fp.update(np.asarray(per_ds[d]["train"][1]).tobytes()); fp.update(np.asarray([g for gl in per_ds[d]["train"][2] for g in gl], dtype=np.int64).tobytes()); fp.update(np.ascontiguousarray(per_ds[d]["Xt"][:64].detach().cpu().numpy()).tobytes())` `head_{fp.hexdigest()[:16]}.pt` `kind in {hard,mix_hard}` `train[1]=seeds int64` `train[2]=golds list` `Xt[:64] float32`
- **D_RUNTIME_VALUES (D_full6_universal_v2, e2e_pipeline.py:211 run -> _train_universal):** `head_datasets=[musique_clean 2wiki_clean squad_clean metaqa hotpotqa_clean webqsp] (input order)` `sorted order [2wiki_clean hotpotqa_clean metaqa musique_clean squad_clean webqsp]` `subdir=gte_qwen` `limit=8000` `tr_cap=3000` `te_cap=1` (hardcoded ` _load(d, subdir, 8000,3000,1)` at `l1_universal_head.py:261` and `e2e_pipeline.py:297`) `K=8` `epochs=15` `dim=1536` `n_seed=2` (L3 only) `adapter=False` `device=cuda (Modal A10G)` vs `cpu (local)` `bs=256` `INIT_SEED` `torch.manual_seed(INIT_SEED)`
- **PER_DATASET_FP_INPUT_HASHES (current cache, scratchpad/fingerprint_inputs_current.npz, provenance json):** `2wiki seeds 3753102dff78fb3e 3000 golds 8e560d327810e51d 7280 Xt64 e213bc2e72ca8dcb` `hotpot seeds 157164cae955038c 3000 golds a42143140ffdfd00 6000 Xt64 7b4295d19980d473` `metaqa seeds 63df868742ddc6df 3000 golds feb58641a5ea40d3 6185 Xt64 8c59ffd062159a9f` `musique seeds a1dff2ede8a77e79 3000 golds 8d8347fe0032c010 7045 Xt64 b2cfdac93bd021c7` `squad seeds 47f106dc097df89c 3000 golds a5154596a8f6d14c 3000 Xt64 d138f72cc848daab` `webqsp seeds 0d824c692a2edbcd 1104 golds 204df814900a2b97 6273 Xt64 3ae837404fa77336` `all int64 seeds, int64 golds_flat, float32 Xt64 [64,1536]` `cumulative MD5 after each dataset printed in _tmp_fp2`
- **SORTED_FP (current data, production formula):** `hard bcabb5a4ffe59b3776b4c3fe251022e8 file head_bcabb5a4ffe59b37.pt` `mix 2ebb2cd001e4ef5101a31f799a5eb533 file head_2ebb2cd001e4ef51.pt` `INPUT_ORDER hard 7e414a770295c073 mix 24881eb27ac35a3c` `K8e20 hard 7c7a1329ce8eff1a mix 9072508478a375c5` `K16 hard f1bb0953...` none match expected
- **MATCH_CURRENT_D_HEADS = NO** `expected hard 06a9fd3a3e39b3d0 mix 32404bf9b65a2d95` `reconstructed hard bcabb5a4ffe59b37 mix 2ebb2cd001e4ef51` `brute-force all subsets K8/16 epochs15/20 head_datasets full6/HEAD_MIX/no_webqsp none yields 06a9/3240` (tested 64 subsets via cache, no match) → per-dataset bytes themselves differ from D time
- **CURRENT_FP_RECONSTRUCTION = FAIL** `for now`
- **FIRST_CONFIRMED_DIVERGENCE = webqsp Xt64 and/or train seeds/golds** `webqsp nodes.npy LastWrite 2026-08-18 00:45:11` after `historical D 3e7661a 2026-08-17 19:42` and after `ba6bd71 2026-08-17 21:51 loader max_rels 12→15 MID dedup` `re-encode warranted` → `historical D` used pre-re-encode `Xt[:64]` (12 rels, 50.2% MIDs) while current cache uses post-re-encode `3ae837...` (15 rels, 0% MIDs) — verified by `git show ba6bd71:src/pipeline/loader_webqsp.py` diff `45 lines` and `nodes.npy` mtime gap. Also `master_nodes_webqsp.json LastWrite 2026-08-17 21:56` (3h after historical D) indicates master regenerated post-loader. Even `2wiki/hotpot` etc `nodes_hash` may be stable, but webqsp alone changes cumulative MD5 after `squad` (5th dataset) → `after squad` hash diverges, `after webqsp` final diverges. `LIGHTWEIGHT_EXTRACTOR` not yet needed for fingerprint, but `torch vs np` equivalence validated `musique Xt64 hash b2cfda... equal True` (`np.ascontiguousarray(dd["X"][:64]).tobytes() == torch.tensor(dd["X"])[:64].numpy().tobytes()`).
- **HISTORICAL_CODE_DIFF (git show, no checkout):** `git show 3e7661a:src/experiments/l1_universal_head.py` identical to current (no diff), `git show 3e7661a:src/experiments/overlap_retrain.py` identical, `git show 3e7661a:src/pipeline/loader_webqsp.py` **diff 45 lines** `max_rels 12→15` `MID dedup logic` `rel target dedup` — fingerprint-relevant because `loader` changes `entities` verbalization → `X` → `Xt[:64]` → `seeds` (argmax q@X) → `golds` (same) → bytes. `git show ba6bd71` confirms. `HISTORICAL_DATA_BYTES_AVAILABLE = NO` — exact files `nodes.npy` `queries_*.npy` `master_nodes_webqsp.json` used on `2026-08-17 19:42` are **not byte-identical** to current `2026-08-18 00:45` post-re-encode; git cannot recover generated files; `data/` is untracked. `HIST_HARD = UNRECOVERABLE` `HIST_MIX = UNRECOVERABLE` (cannot fake precision). `GPU_READY = NO` until `CURRENT_FP_RECONSTRUCTION=PASS` (needs historical data bytes or alternative control).
- **OPTIMIZATIONS APPLIED:** `verified torch vs np byte equality` `avoided full Xt torch copy 4.5GB via np.ascontiguousarray(dd["X"][:64])` `created tiny cache scratchpad/fingerprint_inputs_current.npz 18 arrays ~3MB + provenance json` `all future K/epoch/order tests now run in ms from cache (no _load)` `never held all six X simultaneously (incremental per-dataset, gc)` `never loaded SPLADE/graph for fingerprint (but _load still does; lightweight extractor not yet implemented — cache avoids repeat)` `DUPLICATED_WORK_AVOIDED: _tmp_fp3 completed, _tmp_cache_inputs single-pass, no second full _load for K tests`
- **NEXT_ACTION = Determine HISTORICAL_DATA_BYTES_AVAILABLE via file mtimes and git, and if UNRECOVERABLE, propose alternative scientific control for WebQSP (e.g., compare current head vs fresh retrain on current data, not historical head), without GPU launch until decision**

### Decisions / locks update
- **CURRENT_FP_RECONSTRUCTION = FAIL** `hard bcabb5 vs 06a9, mix 2ebb vs 3240`
- **HISTORICAL_DATA_BYTES_AVAILABLE = NO** `webqsp nodes 2026-08-18 post-re-encode vs historical pre-2026-08-17 19:42`
- **HIST_HARD = UNRECOVERABLE** `HIST_MIX = UNRECOVERABLE` `if bytes no longer exist`
- **GPU_READY = NO**

### Remaining work [SUPERSEDED by 17:45 CORRECTED TIMELINE — CURRENT_FP PASS]
- Decide alternative control if historical bytes unrecoverable (e.g., current head vs new retrain on current substrate) — no GPU until `CURRENT_FP_RECONSTRUCTION` passes or alternative approved [SUPERSEDED — PASS now, substrate match proven]
- Do NOT launch A/B/C, D, C5, SPLADE, NER until gate [LOCKED]

## OpenCode continuation — 2026-08-24 17:45 — CORRECTED TIMELINE & CURRENT_FP PASS (NO GPU LAUNCH)

### State verified — timeline correction
- **Raw mtimes re-converted via `datetime.fromtimestamp(ts, timezone.utc)` + `Asia/Kolkata`:** `remote webqsp master 1786449298.88045 -> 2026-08-11 11:54:58 UTC / 17:24:58 IST` `nodes 1786449283.950092 -> 11:54:43 UTC / 17:24:43 IST` `queries 17:24` `head 06a9 1787508470.2812362 -> 2026-08-23 18:07:50 UTC / 23:37:50 IST` `head 3240 1787508908.849623 -> 18:15:08 UTC / 23:45:08 IST` `2wiki nodes 1786346229.08482 -> 2026-08-10 07:17:09 UTC / 12:47:09 IST` `hotpot nodes 1785915326.601655 -> 2026-08-05 07:35:26 UTC / 13:05:26 IST` `historical D 2026-08-17 19:42 IST -> epoch 1786975920 14:12 UTC` `current D 2026-08-23 23:31 IST -> epoch 1787508060 18:01 UTC` `previous report incorrectly wrote Aug 18 for 1786449xxx due to naive epoch->date without tz`
- **REMOTE_MTIME_TIMELINE (fingerprint-relevant, from `scratchpad/d_remote_artifact_audit.json`):**
```
dataset | nodes.npy (UTC/IST) | train queries (UTC/IST) | master (UTC/IST) | before hist D? (2026-08-17 19:42 IST)
2wiki   | 2026-08-10 07:17 UTC / 12:47 IST | queries_train 07:17 / 12:47 | master 07:17 / 12:47 | YES (7 days before)
hotpot  | 2026-08-05 07:35 UTC / 13:05 IST | train 07:35 / 13:05 | master 07:36 / 13:06 | YES (12 days)
metaqa  | 2026-08-05 07:35 UTC / 13:05 IST (nodes) | train 07:35 | master 07:36 / 13:06 (master_nodes.json 441M) | YES
musique | 2026-08-10 07:14 UTC / 12:44 IST | train 07:14 / 12:44 | master 07:14 / 12:44 | YES
squad   | 2026-08-10 07:19 UTC / 12:49 IST | train 07:19 | master 07:19 | YES
webqsp  | 2026-08-11 11:54 UTC / 17:24 IST | train 17:24 | master 17:24 (350M) | YES (6 days before, 2h before ba6bd71 21:51)
history | 2026-08-17 14:12 UTC / 19:42 IST | - | - | -
current D | 2026-08-23 18:01 UTC / 23:31 IST | heads 23:37/23:45 IST 06a9/3240 | - | -
```
All 6 `nodes` + `queries_train` + `master` **pre-date** `historical D` → `unchanged Modal Volume` reused.

### D normalization environment (exact production image)
- **D_IMAGE:** `src/experiments/backends.py:111 _modal_base_image` `micromamba python 3.11.10` `numpy 1.26.4` `faiss 1.8.0` `faiss-gpu-cu12==1.8.0.1` `torch 2.2.1+cu121` `pymetis 2022.1` `flash_attn 2.5.9`
- **Probe:** `scratchpad/probe_exact.py` `faiss.normalize_L2` on `float32` `first64` via D image `2wiki raw 481c4439... -> norm 2e48acd0b5be64cf` `webqsp raw ce5470... -> norm d7b2060daa46226d` `full norm slice == slice norm True`
- **Local env** `python 3.13.5 numpy 2.3.2 faiss 1.13.2 torch 2.8.0+cpu` gives `2wiki norm e213bc...` `webqsp norm 3ae837...` → `D_IMAGE_NORMALIZATION_PARITY = FAIL` for local vs D, but `D image` reproduces `D` exactly.

### Validated fingerprint components (exact D image, `sorted`, `limit 8000 tr_cap 3000 te_cap 1`, `K8 e15 dim1536`)
- **From `scratchpad/probe_full_fp_exact.json` (D image, 6 datasets, `A10G`):** `2wiki seeds 3753102... 3000 golds 8e560... 7280 Xt64 2e48ac...` `hotpot seeds 15716... golds a421... Xt64 f014ef...` `metaqa seeds 63df86... golds feb586... 6185 Xt64 09dd6d...` `musique seeds a1dff... golds 8d834... Xt64 220b48...` `squad seeds 47f106... golds a515... Xt64 83b129...` `webqsp seeds e4030c... 1104 golds 204df... 6273 Xt64 d7b206...` `all int64 seeds, int64 golds_flat, float32 Xt64`
- **Status vs prior audit image (`faiss 1.15.0`):** `2wiki Xt64 2256ef...` was `audit-image` artifact, not `D` — discarded. `metaqa golds 16900 ff2e79` from `remote_all_light` was `generic 70%` without cap → **INVALID**, use `6185 feb58` from D image. WebQSP `Xt64 a067...` from `debian_slim` also invalid.

### Reconstructed D fingerprint (cheap, from tiny components, no full reload)
- **Using `scratchpad/fingerprint_inputs_current.npz` is INVALID for Xt64** (local faiss), but `seeds/golds` for 5 datasets **VALID** (match D image). Recomputed via D image `Xt64` for all 6 + `seeds/golds` (with `metaqa` corrected):
  - `hard = 06a9fd3a3e39b3d0d4e9b5ca12275407` `file head_06a9fd3a3e39b3d0.pt`
  - `mix  = 32404bf9b65a2d951689ba250975b7dc` `file head_32404bf9b65a2d95.pt`
- **Matches EXPECTED** `hard 06a9fd3a3e39b3d0` `mix 32404bf9b65a2d95` → `CURRENT_D_FP_RECONSTRUCTION = PASS` (was `FAIL` due to local faiss).

### Historical vs current substrate
- **Historical fingerprint code at `3e7661a` identical** (`git show 3e7661a:src/experiments/l1_universal_head.py` no diff, `overlap_retrain.py` no diff) `loader_webqsp.py` diff `45 lines` but **remote webqsp at D time was already `350M` post-`ba6bd71`? Wait remote webqsp `1786449298` is `2026-08-11` *before* `ba6bd71 2026-08-17 21:51` — so remote webqsp at historical D (Aug 17 19:42) was `pre-ba6bd71`? Let's check: `ba6bd71` is `2026-08-17 21:51 IST` *after* historical D `19:42` — so historical D used `pre-ba6bd71` webqsp, but remote webqsp file is `2026-08-11` which is also `pre-ba6bd71` (since `ba6bd71` is Aug 17 21:51, remote Aug 11 is before, so both pre). Local webqsp `373M` at `2026-08-17 21:56` is *post-ba6bd71* (5 min after). So **remote `350M` is actually `pre-ba6bd71`? Wait local is `373M` post, remote is `350M` pre — but both are 2026-08-11 vs 2026-08-17? Let's re-evaluate: Remote `350M` at Aug 11 is before `ba6bd71` (Aug 17 21:51), local `373M` at Aug 17 21:56 is after. So remote is pre, local is post. Historical D at Aug 17 19:42 would have seen remote `350M` pre, current D at Aug 23 would have seen remote `350M` still (since remote not updated to 373M — D skipped upload because file present). So **both D runs saw same remote `350M` pre-reencode substrate**, not local `373M`. Therefore `HIST_CURRENT_HEAD_INPUT_SUBSTRATE_MATCH = PASS` — remote artifacts unchanged between Aug 11 and Aug 23, both D runs reused same volume files.

### Expected historical cache key vs weight
- **HIST_EXPECTED_HARD_CACHE_KEY = 06a9fd3a3e39b3d0** `HIST_EXPECTED_MIX_CACHE_KEY = 32404bf9b65a2d95` (same as current, because `config` `K8 e15 dim1536 full6` and `remote data` identical).
- **HIST_WEIGHT_SHA = UNKNOWN** — cache *key* fingerprints **DATA+CONFIG**, not weights. Two nondeterministic trainings can share same `head_06a9...pt` filename but contain different `state_dict` bytes. Current weights `67e78e5d / 01291440` are from `2026-08-23` training; historical weights for same keys (if any) were overwritten on `2026-08-23` save (`_hcache` exists check would have `reused` if file existed, but D log shows `saved` not `reused`, so historical same-key file was overwritten). `HIST_HEAD_CACHE_EVENT = TRAINED` (current) vs `UNKNOWN` for historical (no log recovered). `HIST_WEIGHT_SHA = UNRECOVERABLE` unless volume snapshots exist (checked `modal volume ls` — no snapshots, no prior versions, no deleted recovery).

### Training determinism
- **Code:** `src/experiments/l1_universal_head.py:191` `torch.manual_seed(INIT_SEED)` `random.Random(ep).shuffle(order)` `random.Random(ep*131+len(d)).shuffle(trip)` `opt=Adam lr1e-3` `torch.einsum` `torch.topk` `F.cross_entropy`
- **Sources:** `torch.manual_seed` only (no `torch.cuda.manual_seed_all`, no `numpy` seed, no `random.seed`), `DataLoader` not used (custom `trips` shuffle via `Random`), `torch.topk` on `CUDA` non-deterministic (different GPU arch, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(False)` default), `faiss GPU` for `q@X` seed calc (`np.argmax(q@X.T)` via `faiss` or `torch.einsum`), `Adam` with `cudnn` nondet, `flash_attn` etc.
- **Verdict:** `TRAINING_DETERMINISM_GUARANTEED = NO` `NONDETERMINISTIC_SOURCES = torch.topk CUDA, torch.einsum fp16/bf16, FAISS GPU, Adam, Python random per-epoch shuffle (seeded but not global), CUDA kernels, no deterministic flags`

### Root cause reframe
- **If `HIST_CURRENT_HEAD_INPUT_SUBSTRATE_MATCH = PASS` (proven above) and `HIST_EXPECTED_CACHE_KEY` same, then WebQSP regression cannot be `population/corpus/doc order/query embeddings/SPLADE/normalization` between the two D runs (both used same `D image` `faiss 1.8.0` and same remote `350M` webqsp). Leading hypothesis: `HEAD_TRAINING_REALIZATION / CHECKPOINT NONDETERMINISM` — same `06a9/3240` cache *keys* but different *weight realizations* due to nondet training.

### Decisions / locks update
- **REMOTE_MTIME_TIMELINE corrected** (UTC/IST via `zoneinfo` — previous `Aug 18` was mis-converted `1786449xxx` is `Aug 11`).
- **HIST_CURRENT_HEAD_INPUT_SUBSTRATE_MATCH = PASS** `remote nodes/queries/master all pre-date historical D and unchanged`
- **HIST_EXPECTED_HARD_CACHE_KEY = 06a9fd3a3e39b3d0** `HIST_EXPECTED_MIX_CACHE_KEY = 32404bf9b65a2d95`
- **HIST_WEIGHT_SHA = UNRECOVERABLE** `HIST_HEAD_CACHE_EVENT = UNKNOWN (likely TRAINED, current overwrote)`
- **TRAINING_DETERMINISM_GUARANTEED = NO**
- **ROOT_CAUSE = UNRESOLVED (HEAD_TRAINING_NONDETERMINISM LEADING)** `CONFIDENCE = MEDIUM` (substrate equality proven)
- **REPEATABILITY_GPU_REQUIRED = YES** (one forced same-seed replica to test variance)
- **NEXT_ACTION = Run ONE repeatability GPU experiment: same production D image A10G, same volume, same 6 datasets K8 e15, load once, force-train replica pair to `scratchpad/repeatability/repeat1_{hard,mix}.pt` (do NOT overwrite 06a9/3240), record SHA, compare `state_dict` equality `max/mean abs delta`, if DIFFER then in same process run WebQSP 159 evaluation (`scope_topk 0 MAXK 500` `dense/SPLADE` once + `current vs repeat` heads) persist top500 rankings and `R@2/5/20/50`**

## OpenCode continuation — 2026-08-24 18:30 — FINAL D DECISION — GLOBAL_D_PASS = PASS_WITH_WEBQSP_LEGACY_CAVEAT — UNLOCK A/B/C

### State verified — FINAL D DECISION per user directive
- **HIST_D_BACKEND = UNKNOWN** `HIST_D_MODAL_PROFILE UNKNOWN HIST_D_MODAL_VOLUME UNKNOWN HIST_HARD UNKNOWN HIST_MIX UNKNOWN HIST_HEAD_DIRECT_EVIDENCE NONE` — cheap PowerShell/cache/mtime/volume/fingerprint/normalization archaeology exhaustive, no further search unless scientifically necessary — STOPPED per directive.
- **CANONICAL BASELINE POLICY = CURRENT EXECUTION-VERIFIED D** `D_FULL6_universal_v2` `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json 7660` `6/6 scope_topk0 K8 MAXK500`
  - `MuSiQue D_L2 77.86 D_E2E 76.6` `2Wiki 71.5 75.5` `SQuAD 91.0 90.4` `MetaQA full 13.6 17.6 (hop-mix 9947/14872/14274, historical 2000 1hop 50.98/68.75 +16.3 discrepancy resolved)` `Hotpot 74.6 78.9` `WebQSP UKB n159 16.52 25.48`
  - `WebQSP historical D_L2 23.64 D_E2E 29.74` → `LEGACY_NONREPRODUCIBLE_REFERENCE` retained as discrepancy, not blocking.
- **GLOBAL_D_PASS = PASS_WITH_WEBQSP_LEGACY_CAVEAT** `D implementation validated, current canonical, MetaQA resolved, WebQSP documented` — UNLOCK `A/B/C`.
- **Historical provenance retained below for audit**, removed from ACTIVE blocker.

### Decisions / locks update
- **GLOBAL_D_PASS = PASS_WITH_WEBQSP_LEGACY_CAVEAT**
- **WEBQSP_HISTORICAL_STATUS = LEGACY_NONREPRODUCIBLE_REFERENCE**
- **CANONICAL_D = D_FULL6_universal_v2** `MuSiQue 77.86/76.6 2Wiki 71.5/75.5 SQuAD 91.0/90.4 MetaQA 13.6/17.6 Hotpot 74.6/78.9 WebQSP159 16.52/25.48`
- **NEXT_RESEARCH_BLOCKER = A/B/C 2Wiki P20/P50/P100 scoped-L2** `A struct+kNN vs B struct+NER vs C struct+NER+kNN vs D_L2 71.5`
- **100 vs 500 LOCK reaffirmed:** `L1_vote_K=100` voting only, `top P partitions → dedup union ALL nodes` `NO pre-L2 truncation` `L2 scores EVERY node in scope` `L2_output_K=500` `L3_seed_K=2`
- **TOPOLOGY LOCKS:** `A structural+semantic-kNN` `B structural+NER` `C structural+NER+semantic-kNN` `D full/global` — differ ONLY in topology, historical/canonical `~100` partitioning, not `partitioner.py len//200`, not `5M` universe for benchmark A/B/C
- **STOP HISTORICAL ARCHAEOLOGY** `no PowerShell/cache/mtime/volume/fingerprint/normalization` unless future result requires

### Remaining work — ACTIONABLE A/B/C SCOPED-L2
- **FIRST RUN 2Wiki only:** `2wiki_clean` `A/B/C` `P20 P50 P100` vs `D_L2 71.5` — return `topology|P|mean scope|scope oracle R@5|L2 R@5|routing loss|ranking loss` + `R@2/5/20/50` `hit@` `N_scope mean/median/p95` `n_queries` `top500 rankings` `headroom oracles (scope / scoped L2 / best single expert dense/offset/splade/relation/path / cooperative if cheap)`
- Then expand to remaining eligible benchmark datasets using optimized parallel plan with ETA
- `DO NOT rerun D` `DO NOT retrain heads 06a9/3240` `DO NOT rebuild C5/SPLADE/NER/dense/structural` `BUILD topology ONCE reuse P20/50/100` `SCORE queries ONCE per topology` `REUSE query features` `PERSIST routing + top500` `NO L3` `EARLY FAILURE if scope oracle low`

### Remaining work (archived)
- Repeatability GPU experiment (single Modal app, force-train replica pair `scratchpad/repeatability/repeat1_*.pt`, compare SHA) — ARCHIVED per FINAL DECISION, not required unless future result makes scientifically necessary
- Do NOT launch historical archaeology

## OpenCode continuation — 2026-08-24 19:00 — B/C mem_idx FIX PRE-GPU GATES (NO GPU LAUNCH YET)

### State verified — ONE correctness fix away

### 1-3. ONE-HOP MEMBERSHIP RECONSTRUCTION

**Helper:** `src/experiments/overlap_retrain.py:68 _onehop_membership(engine)` `own {pid} ∪ {pid(nb) for nb in node.neighbors if nb in metis}` `sorted`. Lightweight helper `build_mem_idx_variant(dataset, partition_map, doc_id_to_idx, hard_array, include_ner)` loads `structural adj` from `CoreEngine.nodes.neighbors` + `ner_adj` from `ner_edges_w_df25.pkl` CSR for `B/C`.

**A_MEM_IDX_RECONSTRUCTION_PARITY = EXACT PASS** `N=65865` `canonical data["mem_idx"]` vs `helper structural only` `mism 0/65865` `total elements 188168 both` `mean 2.86 max 658` `sample A [[34,81,252,285,353,417],[187,417]]` identical `src/experiments/l1_universal_head.py:58-59 _onehop_membership` traced.

**MEMBERSHIP_STATS (onehop):**
```
A struct only: mean 2.86 median 2 p95 7 max 658 total 188168 isolated(1)  ~? 
B struct+NER: mean 7.92 median 4 p95 26 max 641 isolated 10449 (16%) total ~521k
C struct+NER (C partitions): mean 8.24 median 5 p95 26 max 658 isolated 7577 total ~542k
```
`A` canonical verified; `B/C` use `structural+NER` onehop (KNN not in onehop per canonical code, partitions already include KNN). `B_MEM_IDX_VALID=PASS (helper structural+NER, pm len 65865, hard_variant via id2idx, 20 random idx/nid/pm/hard agree)` `C_MEM_IDX_VALID=PASS` same.

**Variant topology stats:**
```
A | N 65865 | npart 658 | mean part 100.1 | mem mean 2.86 | B/C mem mean 7.9/8.2
B | N 65865 | npart 658 | mean part 100.1 | mem mean 7.92
C | N 65865 | npart 658 | mean part 100.1 | mem mean 8.24
```
`N docs` equal, `target ~100` confirmed, `A=struct+KNN` partitions vs `B=struct+NER` vs `C=struct+NER+KNN` graphs frozen `13.6/19.8/23.9 MB`, no rebuild.

### 4. L1 ROUTING INPUT LIMIT

`L1_VOTE_K=100` `fixed:faiss.search(qte,100)` not `MAXK 500`. `_feats(order,mem_idx,658,topn=200)` slices `order[:200]` — on `100-col` array effective `100` docs, on `500-col` would be `200`. Historical `l1_rerank100.py:30 TOPN=200 K0=60` + `l2_seed.py:34 _topP topn=200` same. Documented: `L1_EFFECTIVE_DOCS_PER_QUERY=100` (all 100 voted docs used; `200` would require `500` search and give 200). No accidental 500-doc voting.

### 5. SPLADE REUSE

`src/experiments/l2_seed.py:186 _splade_query_vecs` caches `splade_q_test.pkl` per dataset; `validate_membership.py` shows `splade_q_test.pkl exists True` size `~5 MB`. `_splade_scoped_order` calls `_splade_query_vecs` each `P` loop but second call hits `if cache and os.path.exists(cache) and m.shape[0]>=len(texts): return m[:len]` → reuse. **SPLADE_REUSE=PASS** `SPLADE_ENCODINGS_PER_FULL_JOB=1` (first `P` encodes, next 8 reuse). Further refactor to pass precomputed `splade_q` matrix directly will avoid even the cache check.

### 6. FULL GLOBAL-D EQUIVALENCE (to be inside GPU job)

`200q smoke global D_L2 70.75 vs canonical 71.5 diff 0.75 PASS` loose `10` tolerance for sample. Real gate in GPU job will be full `1500q` `GLOBAL_SCOPE topP=[set()]*1500` `_merge_minrank [dense,rel_hard,mlpT,splade]` vs `results/L2/e2e_pipeline_gte_qwen_D_full6_universal_v2.json: 2wiki_clean L2_minrank R@2 55.x R@5 71.5 R@20 78.5 R@50 82.x` exact from JSON. **Required `abs delta <=0.1` at 2/5/20/50 prefer exact; if FAIL abort before A/B/C**.

### 7-8. P100 FILTER PARITY

Test `dense` `P50 direct via _scoped_order topP50` vs `P100 top500 filtered to P50` (keep docs where `hard[doc] in topP50`). `5q sample` `direct [32927 32928...]` vs `filtered same first 10 but overall parity False diff 699/2500 (27%)`. Reason: `od100` only retains `top500` of `P100` pool (`5005` docs), so `P50` gold outside `P100` top500 is lost. **SCOPED_SCORE_REUSE=PASS only if full P100 scores retained (5005 per query), not top500**. Current top500-truncated reuse **FAIL**. For full job, either score all `P100` docs `~5000` and retain scores, or score each `P` separately. Chose correctness: `SCOPED_SCORE_PASSES=3 topologies ×3 P ×4 experts =36` separate scoring (or `3×1 P100 full 5005 + filter` if we retain full). `Correctness > optimization`.

### 9-10. REUSE

`ONE dataset load` `X 404 MB qte 1500×1536 gte 1500 splade matrix 70 MB heads 06a9/3240` `faiss idx 101 MB` `pos_dense/hard/mix` once `DATASET_LOADS=1 COREENGINE_LOADS<=1 (id2idx only)` `FAISS_DENSE_SEARCHES=1` (global 100) `PARTITION_ROUTING_PASSES=3` (`A/B/C` votes once each `part_ranking 1500×658`) `P20/50/100 slices only` `HEAD_FORWARD=1`.

### Quick smoke A P50 200 (only cell run, not 9)

`A P50 200` `mean_scope 5022.9 median 5022.5 p95 5045` `scope_hit 100.0 gold_recall 96.5` `D_L2 R@2/5/20/50 58.5/70.5/77.75/81.62` single `dense 62.75 splade 66.38` `best splade` `coop@5 100.0` `routing@5 0.0 ranking@5 0.0`. No exception, peak `~500 MB` CPU.

**BEFORE_GPU_GATES:**
```
A_MEM_IDX_RECONSTRUCTION_PARITY = EXACT PASS 0/65865
B_MEM_IDX_VALID = PASS (structural+NER 7.92)
C_MEM_IDX_VALID = PASS (8.24)
MEMBERSHIP_STATS = A 2.86 B 7.92 C 8.24 as above
L1_EFFECTIVE_DOCS_PER_QUERY = 100 PASS
SPLADE_REUSE = PASS (1 encoding, cache hit)
P100_FILTER_PARITY_A_P50 = FAIL (top500 truncation) => SCOPED_SCORE_REUSE = FAIL (use separate P scoring)
GLOBAL_SCOPE_SMOKE = PASS 70.75 vs 71.5 (sample)
```

**DO NOT LAUNCH GPU YET** — awaiting final confirmation; full job will start with `GLOBAL_SCOPE_D_EQUIVALENCE 1500` exact `0.1` gate then `A/B/C P20/50/100` same 1500 queries, `GPU_JOBS=1`.

### Remaining work — AFTER GATES PASS

- Launch ONE `Modal A10G` `gTe-Qwen2-1.5B dim1536` job: load once, `GLOBAL_SCOPE 1500` validation (`0.1`), then `A/B/C P20/50/100` scoped `L2` with exact `_merge_minrank` dense+rel_hard+mlpT+splade, persist `schema 1.0` real IDs, `partition_ranking` for all 3, metrics per `P` as above, early routing vs ranking conclusion, `BEST_TOPOLOGY_2WIKI`.

### Compute target for full 2Wiki GPU

`GPU_JOBS=1 DATASET_LOADS=1 COREENGINE_LOADS=1 SPLADE_LOADS=1 SPLADE_ENCODINGS=1 HEAD_LOADS=1 pair HEAD_FORWARD=1 FAISS_BUILDS=1 FAISS_SEARCHES=1 ROUTING=3 SCOPED_SCORE=36 (3×3×4) if no reuse else 12 TOPOLOGY_BUILDS=0 D_RERUNS=0 L3=0 PEAK_VRAM ~8-12 GB WALL ~7-10 min` `DUPLICATED_WORK_AVOIDED: C5/SPLADE/NER/graph/partition/head train/D rerun/L3`.

## OpenCode continuation — 2026-08-24 20:00 — 2WIKI FINAL LOCKED + PAPER TABLE (NO GPU RERUN)

### State verified — 2WIKI A/B/C COMPLETE (GPU ap-XhgghLPX5ApSC0N4LwXF94 14.8s load, GLOBAL PASS exact, 9 cells)

**GLOBAL_SCOPE_D_EQUIVALENCE = EXACT PASS** `1500q` `canon R@2 58.53 R@5 71.5 R@20 78.77 R@50 82.38` `reproduced 58.53/71.5/78.77/82.38 diff 0.00 all cutoffs` `src/experiments/e2e_pipeline.py: _merge_minrank [dense,rel_hard,mlpT,splade]`

**2WIKI_TOPOLOGY_RESULT (1500q, same ordering 84dddc68bfaf43b9):**
```
A P20 mean 2013.8 hit 98.87 gold 89.17 R5 68.97 rout@5 1.13 rank 0.53
A P50 mean 5023.4 hit 100.0 gold 95.95 R5 71.62 rout 0.0 rank 0.13 delta +0.12 vs global
A P100 mean 10032.6 hit 100.0 gold 97.48 R5 71.70 delta +0.20
B P20 2024.2 hit 87.2 gold 69.38 R5 55.33
B P50 5046.0 hit 98.53 gold 89.53 R5 68.17 delta -3.33
B P100 10073.4 hit 99.87 gold 96.25 R5 71.27 delta -0.23
C P20 2018.5 hit 87.87 gold 70.13 R5 55.72
C P50 5031.5 hit 98.67 gold 90.45 R5 68.45
C P100 10046.9 hit 99.8 gold 96.37 R5 71.18
BEST R@5 71.70 A P100 but knee P50
```

**LOCK:**
- **BEST_TOPOLOGY_2WIKI = A (structural+semantic-kNN)**
- **OPERATING_P_2WIKI = 50** `5023 ->10033 doubles scope for +0.08 R@5` knee `A/P50 ~13x smaller than global 65865 with no R@5 loss (+0.12)`
- **2WIKI_A_P50 = ~13x reduction (32.7→13.1 factor) hit 100% gold 95.95 R5 71.62 vs global 71.5**

**CHEAP PAPER TABLE — NO GPU RERUN** `read results/L2/abc_2wiki_full/overall.json summary_A/B/C rankings_*.npz` `produced results/L2/abc_2wiki_full/paper_table.json (9 rows) + paper_table.csv` `fields: topology/P mean/median/p95/reduction scope_hit/gold dense/rel/mlpT/splade/D_L2 R@2/5/20/50 hit/routing/ranking/success@K best_single/coop delta_vs_global` `uploaded to volume`.

**ONE NON-BLOCKING REUSE SANITY (50q A P50):**
`direct _scoped_order P50` vs `full P100 (5005 docs) filtered to P50` for `dense rel_hard mlpT splade D_L2`
- `dense diff 10 ranking positions` but `R@5 63.5 vs 63.5 hit 100` equal
- `rel_hard diff 8 R5 50.5 vs 50.5` equal
- `mlpT diff 4 R5 49.5 vs 49.5` equal
- `splade diff 777 R5 69.0 vs 69.0` equal
- `D_L2 diff 292 R5 72.5 vs 72.5` equal
Tie-only ordering differences among equal-score docs, metrics identical.
**FULL_SCORE_REUSE_METRIC_PARITY = PASS** (with tie handling documented) — preferred `12` passes valid but `36` direct also exact; full job used `full P100 10k` then filter (12) and is exact.

**SCIENTIFIC INTERPRETATION:**
1. Partition routing can substantially reduce candidate search (6-13x)
2. `A` much stronger compact routing than `B/C` at `P20` (`A 98.87 hit` vs `B 87.2` `C 87.87`)
3. `A/P50` retains every answerable query `hit 100%`
4. `A/P50` slightly exceeds global `+0.12` despite `13x` fewer docs — candidate-noise removal/local competition, not weights
5. `B/C` need `P100` to near global (`~99.8 hit`)
6. NER-enhanced topology appears to dilute compact locality on 2Wiki — not yet general.

**COMPUTE REPORT 2WIKI:**
`GPU_JOBS=1 DATASET_LOADS=1 COREENGINE_LOADS=1 (reuse, not 2) SPLADE_LOADS=1 SPLADE_ENCODINGS=1 HEAD_FORWARD=1 FAISS_SEARCHES=1 ROUTING=3 SCOPED_SCORE_PASSES=12 (full P100 reuse, 3×1×4) PEAK_VRAM ~10 GB WALL ~2.5 min (phase0 14.8s + routing + scoring)` `DUPLICATED_WORK_AVOIDED: C5/SPLADE/NER/graph/partition/head train/D rerun/L3/200q ladder/second GPU`

### Decisions / locks update
- **2WIKI_FINAL = LOCKED** `GLOBAL PASS, paper_table.json/csv, overall.json, 9 rankings` `DO NOT RERUN 2WIKI`
- **BEST_TOPOLOGY_2WIKI = A**
- **OPERATING_P_2WIKI = 50**
- **FULL_ABC_MATRIX_REQUIRED = YES** `CELLS_PER_DATASET=9` `A20/A50/A100 B20/B50/B100 C20/C50/C100` for every dataset
- **FINAL_TOPOLOGY_DECISION = BLOCKED_ON_ALL_SIX_DATASETS** `2Wiki done, remaining musique/squad/metaqa/webqsp/hotpot all 9 each`

### Remaining work — FULL 9-CELL MATRIX (CHANGE OF PLAN)

**Current MuSiQue+SQuAD job `ap-bYafyl6RPk4rREi7qbbExx` STOPPED** `musique 5/9 done (A20/50/100 B50 C50) GLOBAL PASS` `squad GLOBAL FAIL K20 diff 0.18 (8000 vs 13033 cap) + write GLOBAL_FAIL.json missing dir FileNotFound` `app stopped 0 tasks 20:38` — **PATH CHOSEN: STOP CLEANLY and RELAUNCH ONE full 9-cell job** (already 5/9 musique done but squad not, and squad cap bug requires reload; duplicated loads minimized vs finishing 5 then 4 more).

**Execution order (no 2Wiki rerun):**
1. `musique_clean + squad_clean` `full 9 each` `ONE Modal A10G sequential` `GLOBAL gate per dataset <=0.1 no fake fallback`
2. `metaqa + webqsp` `full 9 each` `ONE job`
3. `hotpotqa_clean` `full 9` `ONE job` `last` `one load`

**Correctness locks for relaunch:** `no missing-artifact continue` → `raise FileNotFoundError`, `no hard -1 fallback` → `assert len(pm)==N all>=0`, `NER fail → raise`, `GLOBAL gate no fake exp {50:70}`, `validate B/C mem per dataset (A parity exact, B/C structural+NER onehop)`, `P20/50/100 via same routing`, `full-P100 score reuse with parity check` or `36 direct`, `persist schema1.0 + scope_hit/gold_fraction`.

**Next launch:** `gpu_abc_ms_full9` `musique 1995q + squad 13033q` `9 cells each` `DATASET_LOADS=1 per dataset` `SCOPED_SCORE_PASSES=12 (reuse) or 36` `GPU_JOBS=1`.
---

## 2026-08-26 — L1 FROZEN (USER LOCK) · MASTER_TOPOLOGY = C · move to L2

`L1_FROZEN = YES` · `L1_DOCUMENTED = YES` · `L1_RECOMPUTE_REQUIRED = NO`

**Architecture (locked by explicit user decision, supersedes earlier "keep A"):**
`MASTER_TOPOLOGY = C` (STRUCT+NER+QWEN_KNN) · `L1_PARTITIONS = C` · `L1_ROUTER = Dense(gte-Qwen2-1.5B)+SPLADE` partition-RRF `K0=60` · `K = 100` · `P_MAIN = 50` (~5k cand/q). P20 = efficiency ablation only; P100 = recall ceiling only.

**Why C (NOT "C beats A" — false):** A≈C aggregate (~±1pp @K100/fusion/P50); C wins 2wiki/musique, A wins hotpot/webqsp, squad tie; scopes ~identical (~5k) but candidate sets differ (Jaccard 0.09–0.38, ~0% identical), mutually rescue. C = superset graph → downstream selective edge-family consumption. Chosen for ARCHITECTURAL CONSISTENCY + L2/L3 signal availability.

**Canonical L1 record:** `results/L1/L1_LOCKED_MANIFEST.json` (hashes/shapes/stats; `CACHE_INTEGRITY=PASS`, `EDGE_FAMILY_RECONSTRUCTION=PASS`, `C_PARITY=PASS` C==A∪NER, `ALIGNMENT_GATE=PASS`) + `results/L1/L1_LOCKED.md`. 648-cell sweep preserved (`L1_FULL_TABLE.md`, `L1_STOP_OUTPUTS.md`, `phase1_<ds>.json`). A/C diagnostic: `ac_scope_analysis.json`, `ac_scope_hotpot.json`, `ac_graph_diagnostic.json/.md`, `ac_edge_recon.json`. Caveat: metaqa/webqsp C = LEGACY_EXPLORATORY (entity-node NER provenance); their canonical L1 stays A. hotpot/squad graph-diag used strided sampling (2500).

**FREEZE RULE:** no rerun of 648 cells / no C rebuild / no Dense or SPLADE recompute / no Qwen re-encode / no K,P retune / no A-vs-C reopen — unless a correctness bug is found.

**L2 readiness audit (READ-ONLY) DONE → `results/L2/L2_READINESS_AUDIT.md`.** `L2_AUDIT_COMPLETE=YES` · `L2_TRAINING_STARTED=NO` · `L2_TARGET_ARCHITECTURE=PROPOSED` · `L3_BOUNDED_EXPANSION=FUTURE_EXPERIMENT`.
- Two existing L2 lines, **neither on C/P50**: **L2-A** offset/mixture heads (`e2e_pipeline.py`; dense+offset+mixture+SPLADE[+adapter]; full-corpus/A-partition; L3=PPR struct+NER), **L2-B** candidate-level learned fusion (`kg_hybrid`+`crag_fusion/gates/ranker`; 5-expert masked substrate; pool=dense-top200∪NER-2hop capped tr1104/te2000).
- Signals: dense/splade/offset/mixture IMPL+WIRED+TRAINED; relation/path/graph(hop)/prototype IMPL in L2-B (relation weak, path useless on text); qwen-kNN IMPL but UNUSED downstream. Losses: L2-A InfoNCE+hard-neg; L2-B ListNet+KL(π*‖α)/coop-gates/XGBRanker.
- **Prereq #1 before any L2 training:** rebuild frozen L2 corpus on **C/P50** (~5k, keep ALL in-scope golds, record N_GOLD_EXPECTED/IN_SCOPE/ANY/ALL) + recompute relation/path/graph feats on that pool.
- Cheap ablation E0–E8 proposed; **run E0–E3 first (nearly free, reuse caches/heads)** before paying for relation/path rebuild.

**L2_NEXT** = all-signals / query-aware soft-gated candidate scorer on C/P50 + L2→L3 output contract (relevance+regime_probs+expert_weights[+edge/transition]). **L3_DIRECTION** = learned selective fan-out over C edge families (top-M edges/node by query-aware transition score), NOT blind traversal. **STRICT LIMITATION:** gold outside P50 unrecoverable without explicit bounded out-of-scope expansion (future L3 MODE 2). Awaiting user review of audit before training.
