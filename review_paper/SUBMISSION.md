# REALM 2026 Submission — final status and package

**Title:** *Do Multi-Hop RAG Evaluations Measure Agentic Behavior? A Failure-Mode Audit*
**Track:** short archival (4 content pages) — REALM "Agent Quality Evaluation".
**Deadline:** August 5, 2026, 11:59 p.m. AoE = August 6, 2026, 5:29 p.m. IST.

## Headline result (32-system coding audit)
Answer accuracy **97%**; retrieval recall 44%; **evidence completeness 13%, retrieval-control quality 3%,
and trajectory 16%**; cost signals are 38–41%. Within the relevant families, recovery is **0/15**
iterative/agentic papers and tool-error reporting is **0/5** tool-using agent papers.
A fixed-seed stratified 22% repeat-coding audit gives **97.1% raw agreement and pooled Cohen's κ = 0.93**.
The complete 32-system matrix, source links, and paired repeat-coding matrix are included directly in Appendix A.

The empirical illustration now evaluates **all 22,398 official development questions** from 2Wiki,
MuSiQue, and HotpotQA over 144,341 deduplicated candidate passages. At @20, aggregate recall exceeds
joint evidence completeness by **27.9, 30.9, and 8.3 points**, respectively. Every estimate is backed by
5,000 paired bootstrap resamples and an independently validated per-query record.

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

## Submission artifact
- `../output/pdf/REALM_2026_failure_mode_audit.pdf` is the anonymous review PDF.

The REALM OpenReview form exposes a single PDF upload and no supplementary-file field. Upload only the PDF;
do not upload the source or reproducibility ZIP. The final paper has exactly four content pages, followed by
references and appendices (seven pages total). All pages were rendered and visually inspected. Replace the
author block only for camera-ready.

## OpenReview fields
- Enter the PDF title and abstract verbatim as plain text.
- Add all author OpenReview profiles; they remain hidden from reviewers.
- Add keywords and an optional TL;DR.
- Select `Archival` only if the work is unpublished and not under review elsewhere during REALM review.
- Leave `cross_submission_to` empty for an archival submission.
- Nominate at least one author profile in `serve_as_reviewer`.
- The submission uses the OpenReview form's CC BY 4.0 license.

## Internal backups — do not upload with the review PDF
- `../output/REALM_2026_overleaf_source.zip` is the minimal compilable ACL/Overleaf source package.
- `../output/REALM_2026_reproducibility.zip` contains the full coding and experiment evidence for later release.

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
