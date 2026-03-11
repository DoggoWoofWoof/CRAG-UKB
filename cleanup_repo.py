"""
Repo cleanup script for unified-crag-architecture branch.
Removes all legacy files and directories NOT part of the target directory structure.

Run from the repo root:
    python cleanup_repo.py [--dry-run]
    
  --dry-run: Print what would be deleted without actually deleting anything.
             Always run --dry-run first to verify the list.
"""

import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Files to DELETE at repo root level ────────────────────────────────────────
ROOT_FILES_TO_DELETE = [
    "benchmark_results_c-rag-colbert-query.json",
    "benchmark_results_graph.json",
    "benchmark_results_crag_base.json",
    "benchmark_results_vector.json",
    "benchmark_results_v4-query-graph-selection.json",
    "evaluate_results.py",
    "run_benchmark.py",
    "run_benchmark_main.py",
    "fix_csv.py",
    "fix_repo.py",
    "debug_agent.py",
    "evaluation_summary.txt",
    "results.tex",
    "pyproject.toml",          # Replaced by setup.py in the new structure
    "ADVANCED_REFACTORING.md",
    "CODE_REVIEW.md",
    "PROJECT_STATUS.md",
    "TEST_REPORT.md",
]

# ─── Directories to DELETE at repo root level ──────────────────────────────────
ROOT_DIRS_TO_DELETE = [
    "deploy",
    "docs",
    "examples",
    "paper",
    "scripts",
    "experiments",
    "tests",
    "monitoring",
]

# ─── Old src/crag/* sub-packages to DELETE ─────────────────────────────────────
# All legacy code lives in src/crag/. The new structure uses src/common/, src/ingestion/ etc.
# We will delete src/crag/ entirely AFTER migrating any code that's still needed
# (already ported into v4_graph_selection/ and base retrievers).
LEGACY_SRC_DIRS = [
    os.path.join("src", "crag"),   # Entire old package — replaced by src/common/ etc.
]

# ─── Files INSIDE src/ to delete (old benchmark scripts that ended up there) ───
SRC_FILES_TO_DELETE: list[str] = []   # Add any stray .py files at src/ level here

# ─── Files / Dirs to KEEP (whitelist check) ────────────────────────────────────
KEEP = {
    ".git", ".gitignore", "LICENSE", "README.md",
    "requirements.txt", "setup.py",
    "benchmark_400.csv",   # The primary benchmark dataset
    "configs",
    "data",
    "results",
    "src",
    "checkpoints",         # GNN checkpoints if present
}


def path_in_keep(name: str) -> bool:
    return name in KEEP


def delete(path: str, dry_run: bool):
    if not os.path.exists(path):
        return
    tag = "[DRY-RUN] Would delete" if dry_run else "Deleting"
    print(f"  {tag}: {os.path.relpath(path, ROOT)}")
    if not dry_run:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(description="Unified C-RAG repo cleanup script")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting anything")
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        print("=== DRY RUN — nothing will be deleted ===\n")
    else:
        confirm = input("This will PERMANENTLY delete legacy files. Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    print("\n[1] Root-level file cleanup:")
    for f in ROOT_FILES_TO_DELETE:
        delete(os.path.join(ROOT, f), dry)

    print("\n[2] Root-level directory cleanup:")
    for d in ROOT_DIRS_TO_DELETE:
        delete(os.path.join(ROOT, d), dry)

    print("\n[3] Legacy src/crag package removal:")
    for d in LEGACY_SRC_DIRS:
        delete(os.path.join(ROOT, d), dry)

    print("\n[4] Stray src/ file cleanup:")
    for f in SRC_FILES_TO_DELETE:
        delete(os.path.join(ROOT, "src", f), dry)

    print("\n[5] Unknown root-level entries (not in keeplist):")
    for entry in sorted(os.listdir(ROOT)):
        if not path_in_keep(entry) and entry not in ROOT_FILES_TO_DELETE and entry not in ROOT_DIRS_TO_DELETE:
            full = os.path.join(ROOT, entry)
            if entry.startswith(".") and entry != ".gitignore":
                continue  # Skip hidden files (IDE configs etc.)
            print(f"  [WARNING] Unknown entry (not whitelisted, not scheduled): {entry}")

    print("\nDone." if not dry else "\nDry run complete. Run without --dry-run to actually delete.")


if __name__ == "__main__":
    main()
