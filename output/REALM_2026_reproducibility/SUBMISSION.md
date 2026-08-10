# REALM 2026 Submission — final status and package

**Title:** *Do Multi-Hop RAG Evaluations Measure Agentic Behavior? A Failure-Mode Audit*
**Track:** short archival (4 pg) — REALM "Agent Quality Evaluation". **Deadline:** Aug 5 AoE.

## Headline result (from the released coding sheet, 32 systems)
Answer accuracy **97%**; retrieval recall 44%; **evidence completeness 13%, retrieval-control quality 3%,
and trajectory 16%**; cost signals are 38–41%. Within the relevant families, recovery is **0/15**
iterative/agentic papers and tool-error reporting is **0/5** tool-using agent papers.
A fixed-seed stratified 22% repeat-coding audit gives **97.1% raw agreement and pooled Cohen's κ = 0.93**.
Both paired matrices, evidence notes, the two reconciled disagreements, and the recomputation script are released.

The empirical illustration now evaluates **all 22,398 official development questions** from 2Wiki,
MuSiQue, and HotpotQA over 144,341 deduplicated candidate passages. At @20, aggregate recall exceeds
joint evidence completeness by **27.9, 30.9, and 8.3 points**, respectively. Every estimate is backed by
5,000 paired bootstrap resamples and a validated per-query artifact.

## Files (all in `review_paper/`)
| file | what |
|---|---|
| `realm_paper.tex` | the paper (ACL style), numbers + κ filled |
| `references_realm.bib` | references cited in the manuscript, including datasets and encoder |
| `fig_reporting_gap.pdf/.png` | Figure 1 (the reporting-gap bar chart) |
| `coding_sheet.csv` | the released artifact: 32 papers × 10 dims + metrics + URLs |
| `codings_raw.md` | per-batch coding with evidence notes (audit trail) |
| `make_figure.py` | regenerates stats + figure from the sheet |
| `coding_summary.json` | generated corpus-wide and family-relevant rates |
| `consistency_*.csv`, `consistency_audit.py`, `consistency_results.json` | retained paired labels and reproducible agreement audit |
| `experiment.py` | standalone pinned official-data retrieval experiment |
| `experiment_results.json` | aggregate estimates, CIs, support strata, paired tests, runtime pins |
| `experiment_per_query.jsonl` | all 22,398 per-query records |
| `validate_experiment.py`, `validation_manifest.json` | independent arithmetic checks + artifact hashes |
| `requirements_experiment.txt` | pinned local reproduction environment |
| `CORPUS_AND_CODING.md`, `REALM_PLAN.md` | protocol + plan |

## Submission artifacts
- `../output/pdf/REALM_2026_failure_mode_audit.pdf` is the anonymous review PDF.
- `../output/REALM_2026_overleaf_source.zip` is the minimal compilable ACL/Overleaf source package.
- `../output/REALM_2026_reproducibility.zip` contains the coding and experiment evidence.

The final paper has exactly four content pages, followed by references and a one-page appendix (six pages total).
All six pages were rendered and visually inspected. Submit the PDF anonymously through REALM OpenReview;
replace the author block only for camera-ready.

## Reproduction and validation
```powershell
python review_paper/consistency_audit.py
python review_paper/make_figure.py
python review_paper/experiment.py --datasets 2wiki musique hotpotqa --no-reuse-existing
python review_paper/validate_experiment.py --results review_paper/experiment_results.json --per-query review_paper/experiment_per_query.jsonl
```

The experiment asserts upstream checksums, model revision, support mappings, corpus fingerprints, and
query counts before evaluation. Content-addressed embeddings and rankings are reused only when all
matching fingerprints exist.

## What this deliberately avoids
The empirical section is a small metric-divergence illustration, not a new RAG architecture or SOTA
claim. CRAG and its end-to-end results remain reserved for the main paper.
