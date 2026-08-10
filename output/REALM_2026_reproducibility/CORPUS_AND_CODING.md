# REALM Audit Corpus and Coding Protocol

Status: final and reproducible as of 2026-08-03.

## Audit question

Do iterative, graph-based, and agentic multi-hop RAG papers report the process signals implied by
their own failure modes, or mainly endpoint answer metrics?

The audit is a documented purposive sample, not a random sample of the full literature. All reported
percentages therefore describe this 32-paper corpus and are not population-prevalence estimates.

## Corpus construction

We searched arXiv and ACL, EMNLP, NAACL, NeurIPS, ICLR, and SIGIR proceedings from 2020 through
2026. Queries combined `multi-hop` with `RAG`, `retrieval-augmented`, `iterative retrieval`,
`agentic retrieval`, `graph RAG`, and `knowledge-graph QA`.

The corpus deliberately spans four primary design families:

- 10 iterative retrieval or decomposition systems
- 5 explicitly tool-using or agentic systems
- 15 graph-based retrieval or reasoning systems
- 2 iterative dense multi-hop retrievers

Influential adjacent systems are included when their evaluation contains retrieval-like tool use or
multi-step control. The sample characterizes reporting practice across these families; it is not used
to estimate architecture prevalence. `coding_sheet.csv` is the canonical corpus list and records a
source URL and evidence note for every paper.

## Coding dimensions

Each cell is `1` when the paper reports a quantitative measure, `0` when it does not, and `?` only
during unresolved review. The released reconciled sheet contains no `?` values.

| Code | Signal | Positive-code rule |
|---|---|---|
| S1 | Answer accuracy | EM, F1, accuracy, or equivalent endpoint score |
| S2 | Retrieval quality | Recall, hit rate, answer coverage, or another retrieval-specific score |
| A1 | Evidence completeness | Joint/all-support measure; aggregate partial-credit recall does not count |
| A2 | Retrieval-control quality | Quantitative quality of routing, retrieve/continue, or stop decisions |
| A3 | Failed-retrieval recovery | Quantitative recovery after an empty or wrong intermediate retrieval |
| A4 | Trajectory correctness | Quantified path/trajectory metric or a stated-sample manual error audit |
| A5 | Tool-error rate | Quantitative malformed, failed, or unusable tool/action call rate |
| C1 | Retrieval use/calls | Actual calls, rounds, searches, or retrieval-trigger frequency |
| C2 | Tokens/cost | Token, API, monetary, compute, or explicit indexing-cost measure |
| C3 | Latency | Online wall-clock, throughput, time per query, or relative inference speed |

Additional conventions:

- Offline indexing time alone does not count as C3, but it can count as C2.
- Sweeping a maximum step count does not count as reporting actual retrieval use.
- A qualitative example does not count as A4; a quantified manual audit over a stated sample does.
- The corpus-wide figure keeps denominator 32 for descriptive comparability.
- A2/A3 are additionally reported within the 15 iterative/agentic papers.
- A5 is additionally reported within the five explicitly tool-using agent papers.

## Repeat-coding consistency audit

`consistency_audit.py` uses seed `20260803` to reproduce a stratified seven-paper sample: two
iterative, one agentic, three graph, and one dense system. The retained sample is FLARE, IRCoT,
Toolformer, SimGRAG, HippoRAG 2, GNN-RAG, and MDR.

`consistency_primary_snapshot.csv` preserves the pre-reconciliation labels. A source-verification
pass is retained in `consistency_second_pass.csv`, including pinned paper URLs and evidence notes.
Across 70 paired cells, raw agreement is 68/70 (97.1%) and pooled Cohen's kappa is 0.934. The two
disagreements were source-resolved as follows:

- MDR A4: `0 -> 1`, because Figure 2 reports a quantified 50-error passage-sequence audit.
- HippoRAG 2 C3: `0 -> 1`, because Appendix F reports online time per query.

Per-dimension kappa is released but is unstable at seven papers and undefined when both passes are
invariant. The manuscript therefore emphasizes pooled agreement and exposes every paired cell.

## Final corpus statistics

The generated `coding_summary.json` is authoritative:

- Answer accuracy: 31/32 (97%)
- Retrieval quality: 14/32 (44%)
- Joint evidence completeness: 4/32 (13%)
- Retrieval-control quality: 1/32 (3%); 1/15 in iterative/agentic families
- Failed-retrieval recovery: 0/32; 0/15 in iterative/agentic families
- Trajectory correctness: 5/32 (16%)
- Tool-error rate: 0/32; 0/5 among tool-using agents
- Retrieval use/calls: 12/32 (38%)
- Tokens/cost: 13/32 (41%)
- Latency: 13/32 (41%)

## Reproduction

```powershell
python review_paper/consistency_audit.py
python review_paper/make_figure.py
python -m pytest tests/test_review_coding.py -q
```

The first command regenerates `consistency_results.json` with input hashes. The second regenerates
`coding_summary.json` and both versions of `fig_reporting_gap`.
