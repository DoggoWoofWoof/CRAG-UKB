#!/usr/bin/env python
"""
C-RAG unified experiment runner — the single entrypoint for all experiments.
============================================================================
One script, two interchangeable cloud backends (Modal for GPU, Lightning AI for
CPU), with automatic account rotation when one runs out of credits.

    python experiments.py list
    python experiments.py run <task> [--backend auto|modal|lightning|local]
                                     [--gpu | --cpu] [--dry-run] [-- <task args>]
    python experiments.py exec <task> [-- <task args>]     # run the body in-process

Routing: GPU tasks default to Modal, CPU tasks to Lightning (your "train on
Modal, benchmark on Lightning" workflow). Override per run with --backend.

Examples:
    python experiments.py run train-coverage -- --datasets 2wiki musique --lambdas 0.1 0.25 0.5 1.0
    python experiments.py run bench-level1  --backend lightning -- --datasets 2wiki
    python experiments.py run bench-level2  -- --datasets squad --methods bm25 splade
    python experiments.py run train-coverage --backend local --dry-run -- --datasets 2wiki --limit 500

Accounts for both backends live in the git-ignored configs/compute.local.yaml
(copy configs/compute.local.yaml.example). Nothing secret is committed or passed
on a command line.
"""
import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experiments")


def cmd_list(_):
    from src.experiments import tasks
    from src.experiments.backends import resolve_backend
    print(f"{'TASK':<16} {'RESOURCE':<9} {'DEFAULT BACKEND':<16} DESCRIPTION")
    print("-" * 90)
    for t in tasks.all_tasks():
        print(f"{t.name:<16} {t.resource:<9} {resolve_backend(t):<16} {t.help}")


def cmd_exec(args):
    """Run a task body in-process (used by remote backends and for local debugging)."""
    from src.experiments import tasks
    tasks.get(args.task).body(args.task_args)


def cmd_smoke(args):
    """Run a task LOCALLY on a tiny slice to confirm the end-to-end path before a
    big cloud sweep. Uses the task's registered smoke preset; extra args after --
    are appended."""
    from src.experiments import tasks
    from src.experiments.backends import get_backend
    task = tasks.get(args.task)
    if not task.smoke_args:
        log.warning("No smoke preset for '%s'; running locally as-is (may be heavy).", task.name)
    argv = list(task.smoke_args) + list(args.task_args)
    log.info("SMOKE (local): %s %s", task.name, argv)
    get_backend("local").launch(task, argv, gpu=False, dry_run=False)


def cmd_run(args):
    from src.experiments import tasks
    from src.experiments.backends import get_backend, resolve_backend
    task = tasks.get(args.task)
    backend_name = resolve_backend(task, args.backend)
    gpu = task.resource == "gpu"
    if args.gpu:
        gpu = True
    if args.cpu:
        gpu = False
    log.info("task=%s  backend=%s  gpu=%s  args=%s", task.name, backend_name, gpu, args.task_args)
    get_backend(backend_name).launch(task, args.task_args, gpu, dry_run=args.dry_run, account=args.account)


def build_parser():
    p = argparse.ArgumentParser(description="C-RAG unified experiment runner")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all available tasks and their default backend")

    r = sub.add_parser("run", help="Provision a backend and run a task")
    r.add_argument("task", help="Task name (see `list`)")
    r.add_argument("--backend", default="auto", choices=["auto", "modal", "lightning", "local"])
    r.add_argument("--gpu", action="store_true", help="Force GPU machine")
    r.add_argument("--cpu", action="store_true", help="Force CPU machine")
    r.add_argument("--dry-run", action="store_true", help="Print the plan without launching")
    r.add_argument("--account", default=None,
                   help="Pin to one account by name or index (disables rotation) — for 1-run-per-account parallelism")

    e = sub.add_parser("exec", help="Run a task body in-process (remote/local executor)")
    e.add_argument("task")

    s = sub.add_parser("smoke", help="Run a task locally on a tiny slice (pre-flight check)")
    s.add_argument("task")

    return p


def main():
    # Split on the first `--`: everything before is adapter args (subcommand,
    # task, flags in any order); everything after is passed to the task verbatim.
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        adapter_argv, task_argv = argv[:i], argv[i + 1:]
    else:
        adapter_argv, task_argv = argv, []
    args = build_parser().parse_args(adapter_argv)
    args.task_args = task_argv
    {"list": cmd_list, "run": cmd_run, "exec": cmd_exec, "smoke": cmd_smoke}[args.command](args)


if __name__ == "__main__":
    main()
