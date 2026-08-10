# REALM 2026 Resubmission Plan and Completion Status

Status: completed. The final manuscript is an ACL-style short archival paper with four pages of main
content plus references and appendix.

## Reframed contribution

The rejected ICANN version was a broad narrative review without a reproducible review method or a
comparable empirical result. The REALM paper is now a failure-mode audit of evaluation practice:

- A purposive, documented corpus of 32 iterative, graph-based, agentic, and dense multi-hop systems
- Ten prespecified outcome, process, and cost reporting dimensions
- A retained fixed-seed, seven-paper repeat-coding audit with raw paired labels
- An operational failure-mode table and practitioner metric checklist
- A full-split retrieval illustration over 22,398 official benchmark questions

The paper does not claim to estimate prevalence across the entire literature. Corpus-wide rates are
descriptive, and control/recovery/tool-error conclusions also use relevant family denominators.

## Completed publication gates

- Search and inclusion protocol documented
- `coding_sheet.csv` contains 32 fully resolved rows and no unclear labels
- Agreement artifacts reproduce 97.1% raw agreement and pooled kappa 0.93
- Reporting statistics and figure regenerate from the coding sheet
- Official 2Wiki, MuSiQue, and HotpotQA experiment completed and independently validated
- Dataset revisions, checksums, model revision, per-query output, bootstrap CIs, and paired tests retained
- Citations checked against primary sources
- AI-assistance disclosure retained at the end
- PDF compiled and visually inspected

## Deliberate scope boundaries

- This is an evaluation audit, not an architecture leaderboard.
- The retrieval experiment diagnoses aggregate-versus-joint metric behavior; it is not a CRAG result.
- CRAG architecture and SOTA claims remain reserved for the main paper.
- The corpus was not expanded ad hoc before submission because doing so without a new systematic search
  and repeat-coding cycle would weaken rather than strengthen the methodology.

## Reproduction commands

```powershell
python review_paper/consistency_audit.py
python review_paper/make_figure.py
python review_paper/validate_experiment.py --results review_paper/experiment_results.json --per-query review_paper/experiment_per_query.jsonl
python -m pytest tests/test_review_coding.py tests/test_review_experiment.py -q
```
