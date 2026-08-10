"""Compatibility wrapper for the Paper 2 generation benchmark CLI."""

from src.evaluation.benchmark_generation import *  # noqa: F401,F403
from src.evaluation.benchmark_generation import main


if __name__ == "__main__":
    main()
