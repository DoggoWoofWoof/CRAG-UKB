"""
v4 Graph Selection — Standalone Component Package

This package is a self-contained port of the `v4-query-graph-selection` GitHub branch.
It implements the complete Teleport → Stitch → ColBERT Traverse agentic pipeline and
can be tested independently before integration with the PipelineFactory.

Usage:
    from src.retrievers.v4_graph_selection import build_v4_pipeline, V4GraphSelectionAdapter
    
    pipeline = build_v4_pipeline(config)
    result = pipeline.retrieve("Who is the CEO of the company that acquiring GitHub in 2018?")
"""

from .adapter import V4GraphSelectionAdapter, build_v4_pipeline

__all__ = ["V4GraphSelectionAdapter", "build_v4_pipeline"]
