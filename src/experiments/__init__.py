"""
Unified experiment runner for C-RAG.
====================================
A single source of truth for launching experiments across compute backends:

    experiments.py           # thin root CLI
    src/experiments/
        tasks.py             # backend-agnostic task registry (the "what")
        backends.py          # Modal / Lightning / Local backends (the "where")
        credentials.py       # account pool + rotation from configs/compute.local.yaml
        coverage.py          # coverage-loss sweep body (was run_coverage_eval.py)
        ablations.py         # loss / temperature / HNM ablation bodies
    src/evaluation/level2.py # Level-2 reranking body (was run_level2_eval.py)

Routing: GPU tasks default to Modal, CPU tasks default to Lightning AI. Any task
can be forced onto any backend. When a backend account runs out of credits the
runner rotates to the next account in the pool.
"""
