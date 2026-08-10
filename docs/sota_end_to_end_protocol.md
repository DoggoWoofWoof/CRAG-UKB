# End-to-End SOTA Reproduction Protocol

## Purpose

`sota_e2e_v1` is the publication benchmark for the complete CRAG RAG system.
It does not combine the best numbers from independent Level 1, Level 2, and
Level 3 experiments. Every reported matched-track row must execute:

```text
full corpus -> full graph/index ingestion -> retrieval -> fixed context budget
-> common reader -> common evaluator
```

The source of truth is `configs/sota_end_to_end.yaml`. Generated corpora,
indexes, retrievals, predictions, logs, environment locks, and metric files live
under `data/ukb_storage/_sota/` and are intentionally excluded from Git.

## Two Tracks

### Native reproduction

Run the authors' repository with its paper corpus, split, indexer, retriever,
reader, prompts, and evaluator. This answers whether our pinned checkout can
reproduce the paper. Native numbers must not be used to claim that CRAG is
better because corpora, readers, budgets, and evaluators differ.

### Matched-corpus comparison

Every method ingests the same complete CRAG corpus and receives the same test
queries. It uses its own graph/index and retriever, but all systems are hydrated
to the same document IDs and use:

- top-100 retrieval output for retrieval curves;
- at most 20 context documents;
- at most 4,096 reader tokens;
- `meta-llama/Llama-3.3-70B-Instruct` at the pinned revision;
- temperature 0 and the same evidence-only prompt;
- the same answer and retrieval evaluator.

Only this track supports the end-to-end superiority claim.

## Frozen Inputs

The exporter writes content-addressed, immutable bundles:

```text
data/ukb_storage/_sota/sota_e2e_v1/bundles/<dataset>/<fingerprint>/
  canonical/documents.jsonl
  canonical/edges.jsonl
  canonical/queries/{train,val,test}.jsonl
  adapters/hipporag/
  adapters/gfmrag/
  adapters/text_files/
  manifest.json
```

The manifest records every artifact hash, document/query counts, graph
statistics, split statistics, and the exact bundle fingerprint. Questions,
answers, and gold support labels are excluded from index input. Synthetic edges
are disabled in the main protocol. A separate synthetic-edge run is an
ablation, not a replacement for the main result.

Where an official workflow insists on preprocessing a test-query file during
index setup, the runtime adapter supplies only query IDs/text with empty answer
and support fields. Gold labels remain evaluator-side.

Current full-corpus bundles:

| Dataset | Documents | Original edges | Train / val / test |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 65,865 | 252,140 | 10,500 / 3,000 / 1,500 |
| MuSiQue | 13,672 | 103,086 | 13,956 / 3,987 / 1,995 |
| HotpotQA | 66,573 | 197,932 | 4,313 / 1,232 / 617 |
| SQuAD v2 | 19,029 | 1,389,160 | 91,223 / 26,063 / 13,033 |
| MetaQA | 40,151 | 218,960 | 329,282 / 39,138 / 39,093 |

MetaQA retains official split boundaries. The other matched-track datasets use
CRAG's deterministic local split, not the paper's official test split. Native
reproduction uses the paper split. SQuAD unanswerable questions intentionally
have no answer string.

## Pinned Baselines

| System | Role | Exact checkout | Matched ingestion |
|---|---|---|---|
| CRAG | system under test | Git base plus source fingerprint | partition graph, persistent dense/sparse indexes, Level 1 -> 2 -> 3 |
| HippoRAG 2 | primary | `e37fba2af1a951ac340d837a7c02efb9d8c9544a` | authors' OpenIE graph from all documents |
| GFM-RAG 8M | primary | paper `v1.0.0`, `f38361d9c4b754e1a800763e3f9e7cb4b841aa0f` | paper-v1 graph constructor and 8M checkpoint |
| G-reasoner 34M | primary | `57e3e28045fffff5411e2454a4323fbe4dff9b91` | current typed graph constructor and 34M checkpoint |
| HopRAG | primary | `a6e425b8f8a5d8131dd7805db40185ac76e09903` | full document nodes; author pseudo-query edges scoped by label-free graph neighborhoods |
| SiReRAG | secondary | `70c9434776ca0eaac17590285285c26313817365` | full similarity/relatedness trees |
| KG2RAG | secondary | `7d626c77b7af30b55aa3f960cde755b9549a0616` | full fact KG; HotpotQA only |
| RAPTOR | secondary | `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767` | global recursive abstraction tree with canonical leaf provenance |
| LightRAG paper | secondary | `c91f15b2a87d949bb3488934019fbc13b5f1bff5`, last algorithm commit before arXiv v1 | paper entity-relation graph and dual-level retrieval |
| LightRAG current | diagnostic | `cb7cc70bc6b29055d9eeeccc32714679feb33704` | current full entity-relation graph; post-paper reranker disabled |
| GraphRAG current | diagnostic | `14a00ad88fc33cf2b52f4f113f25807556f8e25e`, v3.1.1 | current entity/relation/community index, not exact 2024 paper code |

The official repositories are cached at
`data/ukb_storage/_sota/external/repos/`. GFM-RAG 8M, G-reasoner 34M, and
NV-Embed-v2 resolve by immutable Hugging Face commit into
`data/ukb_storage/_sota/external/huggingface/`.

HippoRAG 2, GFM-RAG 8M, G-reasoner 34M, HopRAG, KG2RAG, RAPTOR, both LightRAG
revisions, and CRAG have direct matched-track stage commands. HopRAG still
requires an installed isolated environment, Neo4j 5.26.0, and an explicitly
configured OpenAI-compatible traversal service before its first smoke can be
marked publication-ready. KG2RAG requires its isolated environment plus a
reviewed Ollama service containing the exact locked models. RAPTOR requires an
isolated environment and an explicit summarization endpoint. SiReRAG retains a
notebook-driven official workflow.

HopRAG needs a protocol distinction beyond the normal native/matched split.
Its official edge builder groups documents using every benchmark question's
supplied context before pseudo-question matching. Reproducing that unchanged is
valid only in the native table: it is transductive and exposes test-time
candidate-set structure to graph construction. The matched adapter never reads
questions, answers, or support labels during ingestion. It materializes every
canonical document as one provenance-preserving node and deterministically
groups documents by the immutable label-free dataset edges. The pinned author
code then generates answerable/pending pseudo-questions and logical edges
inside those neighborhoods. This is an explicit matched-corpus adaptation, not
a claim that the official repository natively supports an unconstrained global
corpus.

HopRAG preparation, node batches, edge batches, model revision, and Neo4j
namespace are content-addressed. The adapter verifies every author batch,
waits for all four Neo4j indexes to become online, and checks that the final
node count equals the complete corpus. Because the author builder is not
transactional across a Neo4j batch, interruption during an online-node or edge
batch fails closed rather than risking a duplicated partial graph. Retrieval
also fails if any returned node cannot be mapped exactly to one canonical
document ID.

RAPTOR also needs a strict native/matched distinction. The paper evaluates a
tree over the long text associated with a question on NarrativeQA, QASPER, and
QuALITY. The CRAG datasets instead require retrieval across a shared
open-domain corpus. The matched adapter therefore builds one explicitly
labeled global cross-document tree from all canonical documents. It calls the
official sentence-aware splitter separately for each document, which preserves
the paper's 100-token leaf policy without allowing a leaf to cross a canonical
document boundary.

The matched RAPTOR index keeps the paper's
`sentence-transformers/multi-qa-mpnet-base-cos-v1` encoder at immutable revision
`d51b22a1dfa8184e9258074e56e2875e50612dca`, soft-cluster threshold `0.1`,
`gpt-3.5-turbo` summarization target, and 2,000-token collapsed-tree retrieval.
Every paid summary is cached by the exact prompt, context hash, endpoint, model,
and output budget, so an interrupted run reuses completed calls. Final tree,
provenance, and FAISS artifacts are content-addressed and hash-verified.

Internal summary nodes are not canonical evidence documents. For matched
retrieval metrics, the adapter projects selected nodes through their complete
descendant provenance and fuses those RRF votes with a dense centroid ranking
of each document's leaf embeddings. Every raw row records this projection
policy. It is a necessary matched-corpus adapter and must not be described as a
native RAPTOR result. The global tree is also a scale stress test beyond the
paper's reported trees of at most roughly 80,000 tokens, so build feasibility
and cost are publication gates rather than assumed properties.

KG2RAG also needs a strict native/matched distinction. Its paper's fullwiki
experiment uses 66,581 HotpotQA documents pooled from the author evaluation
artifact. That is a valid native, transductive benchmark protocol. The matched
run instead ingests all 66,573 immutable canonical HotpotQA documents and never
reads evaluation questions, answers, distractor groups, or supporting facts
during preparation or indexing. These are materially different corpora rather
than an eight-document count discrepancy: 5,521 author-pool titles are absent
from the canonical corpus and 5,513 canonical titles are absent from the
author pool. The native runner materializes 306,487 sentence occurrences
because repeated contexts are retained, while the deduplicated author pool has
269,602 sentence chunks and the matched canonical corpus has 275,204.

The matched preparation recovers HotpotQA's retained sentence boundaries from
the source double whitespace present in 62,246 canonical documents. A
deterministic punctuation rule is used for the remaining documents. Every
sentence keeps canonical document provenance, and the first-sentence title
prefix policy matches the author's extraction script. Preparation is
content-addressed by corpus hash and sentence-policy version.

The index requires the paper models at exact immutable identities:
`llama3:8b` at Ollama manifest
`365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1`,
`mxbai-embed-large:latest` at manifest
`468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8`,
and `BAAI/bge-reranker-large` at Hugging Face revision
`55611d7bca2a7133960a6d3b71e083071bbfc312`. The adapter checks Ollama's
installed tags and fails on a missing or mismatched digest. It never pulls a
model, starts a service, or substitutes a checkpoint. The isolated matched
environment installs only the author packages imported by this path plus
locked NumPy, FAISS, and Hugging Face Hub versions; obsolete unused
`pandas==1.1.5` and `ujson==1.35` pins from the repository requirements are
not introduced into the Python 3.10 adapter environment. The live Ollama
server version is stored in the index identity and must also match at
retrieval. The released environment is also internally unsatisfiable:
`llama-index==0.12.20` requires `tqdm>=4.66.1` but the repository pins
`tqdm==4.64.0`, and it requires `networkx>=3.0` while the repository pins
`networkx==2.5.1`. The isolated adapter uses `tqdm==4.67.1` and
`networkx==3.3` and records these non-algorithmic compatibility corrections.
It also pins `transformers==4.44.2`, the minimum declared by
`FlagEmbedding==1.3.4`; an unconstrained install currently resolves to
Transformers 4.57 and fails at import because FlagEmbedding references removed
Gemma2 symbols.

Triplets are cached per canonical document and embedding vectors are cached in
content-addressed batches. Interrupted indexing resumes from those caches.
Final KG, chunk map, and FAISS index hashes are verified before retrieval.
Calls to non-local Ollama endpoints fail closed unless the command is
explicitly amended with `--allow-remote-ollama`; remote extraction also
requires a positive `--max-extraction-calls` budget.

Two released-code discrepancies are recorded rather than hidden. The fullwiki
runner omits required `dataset` and `use_tpt` fields when constructing its
postprocessors, and the released filter reranks MST text while Eq. 8 of the
paper specifies triplet representations. The matched adapter supplies the
missing constructor fields and uses the author's `use_tpt` representation with
a transparent projection that passes only the final cumulative triplet string
to BGE. Graph expansion, MST filtering, and context organization remain in the
pinned author module. This is labeled a paper-spec correction; a
`released_text` diagnostic remains available. KG2RAG has a top-10 *chunk*
budget, so document metrics above the returned distinct-document count plateau
instead of fabricating a top-100 ranking.

The LightRAG split is deliberate. The current checkout contains features added
after the paper, so it is a diagnostic rather than the published baseline. The
paper checkout exposes formatted contexts but not ranked document IDs, and its
`hybird` formatter deduplicates rows with an unordered Python set. The matched
adapter therefore uses the author's unchanged high-level and low-level keyword,
entity, relation, and source-selection functions, captures each ranked source
list before the lossy formatter, and applies deterministic high/low round-robin
fusion. Chunk provenance is then mapped to canonical document IDs. This adapter
policy is stored in every retrieval manifest.

LightRAG indexing is persistent and corpus-hash guarded. Successful document
batches are checkpointed; repeated runs reuse the graph and LLM cache. Because
the paper insertion path is not transactional, a process interrupted inside an
active batch fails closed instead of silently accepting a partial graph.

## Execution

Inspect all bundle, repository, and stage state:

```bash
python experiments.py exec sota-e2e -- status
```

Run the fail-closed publication audit before launching or reporting a method.
It reports only whether required variables are present, never their values:

```bash
python experiments.py exec sota-e2e -- audit --methods crag hoprag --datasets 2wiki_clean musique_clean
python experiments.py exec sota-e2e -- audit --methods crag hoprag --datasets 2wiki_clean musique_clean --verify-hashes
```

The fast form checks bundle fingerprints, repository/config locks, isolated
environments, configured stages, cached inputs, and completed retrieval stages.
`--verify-hashes` additionally streams every immutable bundle artifact through
SHA-256 and is the required pre-publication check.

Re-export a full immutable corpus only when source data or the contract changes:

```bash
python experiments.py exec sota-e2e -- export --datasets 2wiki_clean musique_clean hotpotqa_clean squad_clean metaqa
```

Pin repositories. `--install` creates one isolated environment per method and
writes `crag_environment.lock.txt`:

```bash
python experiments.py exec sota-e2e -- lock --methods hipporag2 gfmrag8m greasoner34m --install
python experiments.py exec sota-e2e -- lock --methods lightrag_paper lightrag_current --install
python experiments.py exec sota-e2e -- lock --methods crag
```

Run HippoRAG 2 full ingestion and retrieval:

```bash
python experiments.py exec sota-e2e -- stage --method hipporag2 --dataset 2wiki_clean --track matched --stage index
python experiments.py exec sota-e2e -- stage --method hipporag2 --dataset 2wiki_clean --track matched --stage retrieve
```

GFM-RAG and G-reasoner use the same two stage names. The GFM-RAG adapter
materializes the paper-v1 schema in its run directory; it never mutates the
immutable canonical bundle.

Run paper-faithful or current LightRAG with the same stage sequence:

```bash
python experiments.py exec sota-e2e -- stage --method lightrag_paper --dataset 2wiki_clean --track matched --stage index
python experiments.py exec sota-e2e -- stage --method lightrag_paper --dataset 2wiki_clean --track matched --stage retrieve
```

`OPENAI_API_KEY` is required. `OPENAI_BASE_URL` may point to a compatible
endpoint for engineering smoke tests, but a paper-faithful native result must
actually serve `gpt-4o-mini` and `text-embedding-3-small`.

Run HopRAG full ingestion and retrieval after installing its isolated
environment and starting Neo4j 5.26.0:

```bash
python experiments.py exec sota-e2e -- lock --methods hoprag --install
python experiments.py exec sota-e2e -- stage --method hoprag --dataset 2wiki_clean --track matched --stage prepare
python experiments.py exec sota-e2e -- stage --method hoprag --dataset 2wiki_clean --track matched --stage index
python experiments.py exec sota-e2e -- stage --method hoprag --dataset 2wiki_clean --track matched --stage retrieve
```

Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`HOPRAG_LLM_BASE_URL`, `HOPRAG_LLM_MODEL`, and
`HOPRAG_LLM_API_KEY`. The BGE checkpoint is locked by immutable Hugging Face
revision and reused from the shared model cache. A localhost-compatible LLM
endpoint is accepted for the matched engineering run. Calls to
`api.openai.com` fail closed unless the adapter command is deliberately amended
with `--allow-paid-api`; the suite never starts paid ingestion silently.
HopRAG returns its paper budget of 20 documents, so metrics above `@20` plateau
rather than pretending the method produced a 100-document ranking.

Run the RAPTOR matched adaptation only after reviewing the recursive summary
cost:

```bash
python experiments.py exec sota-e2e -- lock --methods raptor --install
python experiments.py exec sota-e2e -- stage --method raptor --dataset 2wiki_clean --track matched --stage prepare
python experiments.py exec sota-e2e -- stage --method raptor --dataset 2wiki_clean --track matched --stage index
python experiments.py exec sota-e2e -- stage --method raptor --dataset 2wiki_clean --track matched --stage retrieve
```

Set `RAPTOR_LLM_BASE_URL`, `RAPTOR_LLM_MODEL`, and `RAPTOR_LLM_API_KEY`.
The paper model target is `gpt-3.5-turbo`; the adapter never substitutes
another summarizer. Local OpenAI-compatible endpoints are accepted. Every
non-local endpoint fails closed unless the reviewed index command explicitly
adds both `--allow-paid-api` and a positive `--max-summary-calls` budget.
Retrieval itself is local and reuses the locked SBERT snapshot plus the
completed FAISS artifacts.

Run KG2RAG only after installing its isolated environment and configuring a
reviewed Ollama service:

```bash
python experiments.py exec sota-e2e -- lock --methods kg2rag --install
python experiments.py exec sota-e2e -- stage --method kg2rag --dataset hotpotqa_clean --track matched --stage prepare
python experiments.py exec sota-e2e -- stage --method kg2rag --dataset hotpotqa_clean --track matched --stage index
python experiments.py exec sota-e2e -- stage --method kg2rag --dataset hotpotqa_clean --track matched --stage retrieve
```

Set `KG2RAG_OLLAMA_BASE_URL` to the reviewed service. The service-free prepare
stage may be run in advance and is reused unchanged. Indexing performs the
paper's query-independent extraction over every canonical sentence, so it is a
large local-model job; its per-document and per-batch caches are publication
artifacts and must be retained. The default matched run uses
`paper_triplet`; use `released_text` only as a separately labeled
released-code diagnostic.

Run the complete CRAG retriever. Top 100 are retained for retrieval metrics,
while hydration and generation enforce the common 20-document context budget:

```bash
python experiments.py exec sota-e2e -- stage --method crag --dataset 2wiki_clean --track matched --stage retrieve
```

Hydrate any retriever output into the common contract:

```bash
python experiments.py exec sota-e2e -- hydrate --method hipporag2 --dataset 2wiki_clean --split test --source data/ukb_storage/_sota/sota_e2e_v1/runs/matched/hipporag2/2wiki_clean/sota_e2e_v1/retrieval.jsonl --output data/ukb_storage/_sota/sota_e2e_v1/runs/matched/hipporag2/2wiki_clean/sota_e2e_v1/hydrated.jsonl
```

For CRAG, hydrate `retrieval.raw.jsonl`. Then run the common reader and
evaluator:

```bash
python experiments.py exec sota-e2e -- generate --source <hydrated.jsonl> --output <predictions.jsonl>
python experiments.py exec sota-e2e -- evaluate --source <predictions.jsonl> --output-dir <metrics-directory>
```

After evaluating two systems, compute query-paired deltas:

```bash
python experiments.py exec sota-e2e -- compare --baseline <baseline-predictions.jsonl> --treatment <crag-predictions.jsonl> --output-dir <comparison-directory>
```

The comparison writes `comparison.json` and `per_query_deltas.jsonl`. It reports
paired bootstrap 95% intervals, win/tie/loss counts, exact McNemar tests for
binary Full Coverage/EM metrics, and paired sign-flip tests for continuous
metrics. A paired comparison refuses different query IDs, questions, answers,
or supporting-document labels.

Set `SOTA_READER_BASE_URL` and `SOTA_READER_API_KEY` to an OpenAI-compatible
server hosting the pinned reader revision. A model name alone is insufficient:
the final run record must also retain server/runtime provenance.

## No-Rerun Rules

Each stage signature includes the suite config hash, official repository commit,
full bundle fingerprint, track, command, and reader configuration where
relevant. CRAG stages also include a source-tree fingerprint. A completed stage
with the same signature is reused. A changed prompt reruns generation only; a
changed metric reruns evaluation only; a changed reader reruns generation and
evaluation; a changed retriever reruns retrieval onward; a changed corpus or
graph contract invalidates ingestion.

Partial HippoRAG/GFM retrieval and common-reader files resume by query ID.
Offline indexes, OpenIE outputs, downloaded model snapshots, raw retrieval
rankings, predictions, stdout/stderr, wall time, peak RSS, and disk usage are
retained. Never use `--force` merely to regenerate a table.

## Reported Metrics

Retrieval is evaluated at K = 2, 5, 10, 20, 50, and 100:

- document recall and hit rate;
- Full Coverage@K over all required evidence documents;
- support precision, recall, and F1;
- MRR and nDCG;
- weakest-positive rank.

Generation and grounding:

- answer EM and token F1;
- joint retrieval-answer EM/F1;
- answer-in-context grounding proxy;
- optional frozen faithfulness and answer-relevance judge.

Efficiency:

- online retrieval, generation, and total p50/p95 latency;
- ingestion wall time, peak RSS/VRAM, and persistent index bytes;
- throughput;
- prompt/completion tokens and configured monetary cost.

All headline metrics require paired query-level bootstrap 95% confidence
intervals. Statistical comparisons must be paired on identical test IDs and
run through the `compare` command rather than comparing overlapping
single-system confidence intervals.

## Publication Gates

A result is publishable only when:

1. The full corpus was ingested and the bundle fingerprint matches.
2. The repository, model revisions, dependency lock, command, and output hashes exist.
3. Hyperparameters and the Level 2 winner were selected on validation, never test.
4. Retrieval output covers every test query and contains no duplicate document IDs.
5. The same reader, context budget, and evaluator were used for all matched rows.
6. The CRAG Git tree is clean at the final tagged commit.
7. At least three seeds are reported for trained CRAG components.
8. Failures, unsupported datasets, and API/service substitutions are explicit.

CRAG currently supports this launch path for 2Wiki, MuSiQue, SQuAD, and MetaQA.
HotpotQA is not eligible for a CRAG headline row until a frozen Level 1
checkpoint is trained and selected. The current `auto` Level 2 selection is
also a temporary integration setting; replace it with the validation-selected,
explicit reranker before the final test run.
