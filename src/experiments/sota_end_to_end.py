"""Reproducible full-ingestion and end-to-end SOTA RAG benchmark runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from src.evaluation.sota_contract import (
    _atomic_json,
    _atomic_jsonl,
    _stable_json,
    compare_end_to_end,
    directory_size,
    evaluate_end_to_end,
    export_dataset_bundle,
    latest_bundle,
    load_jsonl,
    sha256_file,
    validate_run_rows,
)


log = logging.getLogger("experiments.sota_end_to_end")
DEFAULT_CONFIG = Path("configs/sota_end_to_end.yaml")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported or invalid SOTA config: {path}")
    suite = config.get("suite") or {}
    if not suite.get("id") or not suite.get("output_root"):
        raise ValueError("SOTA config requires suite.id and suite.output_root.")
    for method, specification in (config.get("methods") or {}).items():
        commit = str(specification.get("commit") or "")
        if not specification.get("repository") or len(commit) != 40:
            raise ValueError(
                f"Method {method!r} must pin an official repository and 40-character commit."
            )
    config["_path"] = path.as_posix()
    config["_sha256"] = sha256_file(path)
    return config


def _root(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config["suite"][key]))


def _selected(
    requested: Optional[Sequence[str]],
    available: Iterable[str],
) -> List[str]:
    available = list(available)
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown values: {', '.join(unknown)}")
    return list(requested)


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    log.info("exec: %s", " ".join(map(str, command)))
    return subprocess.run(
        list(map(str, command)),
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
    )


def _repo_path(config: Mapping[str, Any], method: str) -> Path:
    specification = config["methods"][method]
    if specification.get("internal"):
        return Path(str(specification["repository"])).resolve()
    return _root(config, "external_root") / "repos" / method / str(
        specification["commit"]
    )[:16]


def _internal_source_fingerprint(repository: Path) -> str:
    digest = hashlib.sha256()
    roots = [
        repository / "src",
        repository / "configs",
        repository / "experiments.py",
        repository / "requirements.txt",
    ]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(item for item in root.rglob("*") if item.is_file())
    for path in sorted(files):
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prepared_path(config: Mapping[str, Any], method: str) -> Path:
    specification = config["methods"][method]
    if specification.get("internal"):
        return _root(config, "external_root") / "internal" / f"{method}_prepared.json"
    return _repo_path(config, method) / "crag_prepared.json"


def _environment_python(repo: Path) -> Optional[Path]:
    candidates = (
        repo / ".venv" / "Scripts" / "python.exe",
        repo / ".venv" / "bin" / "python",
    )
    return next((path for path in candidates if path.exists()), None)


def _freeze_environment(repo: Path, destination: Path) -> Optional[str]:
    python = _environment_python(repo)
    if python is None:
        return None
    python = python.resolve()
    try:
        result = _run_checked(
            [python, "-m", "pip", "freeze", "--all"],
            cwd=repo,
            capture=True,
        )
    except subprocess.CalledProcessError:
        if shutil.which("uv") is None:
            raise
        result = _run_checked(
            ["uv", "pip", "freeze", "--python", python],
            cwd=repo,
            capture=True,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.stdout, encoding="utf-8")
    return sha256_file(destination)


def _environment_install_signature(specification: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _stable_json(
            {
                "python": str(specification.get("python") or ""),
                "install": list(specification.get("install") or []),
            }
        ).encode("utf-8")
    ).hexdigest()


def lock_repository(
    config: Mapping[str, Any],
    method: str,
    *,
    install: bool = False,
) -> Dict[str, Any]:
    specification = config["methods"][method]
    repository = str(specification["repository"])
    commit = str(specification["commit"])
    target = _repo_path(config, method)
    if specification.get("internal"):
        actual = _run_checked(
            ["git", "rev-parse", "HEAD"], cwd=target, capture=True
        ).stdout.strip()
        if actual != commit:
            raise RuntimeError(
                f"Internal base commit mismatch for {method}: {actual} != {commit}"
            )
        dirty = bool(
            _run_checked(
                ["git", "status", "--porcelain"], cwd=target, capture=True
            ).stdout.strip()
        )
        prepared = {
            "method": method,
            "repository": target.as_posix(),
            "commit": actual,
            "source_fingerprint": _internal_source_fingerprint(target),
            "dirty": dirty,
            "platform": platform.platform(),
            "prepared_at": _utc_now(),
            "installed": True,
            "config_sha256": config["_sha256"],
        }
        _atomic_json(_prepared_path(config, method), prepared)
        return prepared
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        _run_checked(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repository, target]
        )
    remote = _run_checked(
        ["git", "remote", "get-url", "origin"], cwd=target, capture=True
    ).stdout.strip()
    if remote.lower().removesuffix(".git") != repository.lower().removesuffix(".git"):
        raise RuntimeError(
            f"Repository mismatch for {method}: expected {repository}, found {remote}"
        )
    _run_checked(["git", "fetch", "origin", commit, "--depth", "1"], cwd=target)
    _run_checked(["git", "checkout", "--detach", commit], cwd=target)
    actual = _run_checked(["git", "rev-parse", "HEAD"], cwd=target, capture=True).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"Commit mismatch for {method}: {actual} != {commit}")

    install_signature = _environment_install_signature(specification)
    prepared_path = target / "crag_prepared.json"
    existing = (
        json.loads(prepared_path.read_text(encoding="utf-8"))
        if prepared_path.exists()
        else {}
    )
    freeze_path = target / "crag_environment.lock.txt"
    reusable_lock_sha = None
    if (
        not install
        and existing.get("installed")
        and existing.get("install_signature") == install_signature
        and existing.get("environment_lock_sha256")
        and _environment_python(target) is not None
    ):
        current_lock_sha = _freeze_environment(target, freeze_path)
        if current_lock_sha == existing.get("environment_lock_sha256"):
            reusable_lock_sha = current_lock_sha

    prepared = {
        "method": method,
        "repository": repository,
        "commit": actual,
        "ref": specification.get("ref"),
        "python_requested": str(specification.get("python") or ""),
        "platform": platform.platform(),
        "prepared_at": _utc_now(),
        "installed": reusable_lock_sha is not None,
        "install_signature": install_signature,
        "config_sha256": config["_sha256"],
    }
    if reusable_lock_sha is not None:
        prepared["environment_lock_sha256"] = reusable_lock_sha
    if install:
        if shutil.which("uv") is None:
            raise RuntimeError(
                "The pinned external environments use uv; install uv before --install."
            )
        if not (target / ".venv").exists() and specification.get("python"):
            _run_checked(
                ["uv", "venv", "--python", str(specification["python"]), ".venv"],
                cwd=target,
            )
        for command in specification.get("install") or []:
            _run_checked_shell(str(command), cwd=target)
        prepared["environment_lock_sha256"] = _freeze_environment(
            target, freeze_path
        )
        prepared["installed"] = True
    _atomic_json(target / "crag_prepared.json", prepared)
    return prepared


def _run_checked_shell(command: str, *, cwd: Path) -> None:
    log.info("shell: %s", command)
    subprocess.run(command, cwd=str(cwd), shell=True, check=True)


def _process_rss(process: subprocess.Popen) -> int:
    try:
        import psutil

        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except Exception:
        return 0


def _process_gpu_memory(process: subprocess.Popen) -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        import psutil

        root = psutil.Process(process.pid)
        process_ids = {
            root.pid,
            *(child.pid for child in root.children(recursive=True)),
        }
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_mib = 0
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 1)]
            if len(fields) == 2 and int(fields[0]) in process_ids:
                total_mib += int(fields[1])
        return total_mib * 1024 * 1024
    except Exception:
        return 0


def _stage_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _resolve_native_dataset(method_spec: Mapping[str, Any], dataset: str) -> str:
    candidates = list(method_spec.get("native_datasets") or [])
    stem = dataset.replace("_clean", "")
    aliases = {
        "2wiki": ("2wiki", "2wikimultihopqa", "2wikimultihopqa_test"),
        "musique": ("musique", "musique_test"),
        "hotpotqa": ("hotpot", "hotpotqa", "hotpotqa_test"),
        "squad": ("squad",),
        "metaqa": ("metaqa",),
    }
    for candidate in aliases.get(stem, (stem,)):
        if candidate in candidates:
            return candidate
    return stem


def _template_values(
    config: Mapping[str, Any],
    method: str,
    dataset: str,
    run_dir: Path,
    variables: Mapping[str, str],
) -> Dict[str, str]:
    method_spec = config["methods"][method]
    repository = _repo_path(config, method).resolve()
    bundle = latest_bundle(_root(config, "output_root"), dataset).resolve()
    run_dir = run_dir.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    alias = str(manifest["adapter_alias"])
    python = Path(sys.executable) if method_spec.get("internal") else _environment_python(repository)
    if python is None:
        python = repository / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
    hipporag_root = bundle / "adapters" / "hipporag" / "reproduce" / "dataset"
    values = {
        "method": method,
        "dataset": dataset,
        "crag_dataset": dataset.replace("_clean", ""),
        "native_dataset": _resolve_native_dataset(method_spec, dataset),
        "repo": repository.as_posix(),
        "bundle": bundle.as_posix(),
        "run_dir": run_dir.as_posix(),
        "python": python.resolve().as_posix(),
        "external_adapter": (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "external_sota_adapter.py"
        ).as_posix(),
        "adapter_alias": alias,
        "canonical_documents": (
            bundle / "canonical" / "documents.jsonl"
        ).as_posix(),
        "canonical_edges": (
            bundle / "canonical" / "edges.jsonl"
        ).as_posix(),
        "canonical_queries": (
            bundle / "canonical" / "queries" / "test.jsonl"
        ).as_posix(),
        "hipporag_corpus": (hipporag_root / f"{alias}_corpus.json").as_posix(),
        "hipporag_queries": (hipporag_root / f"{alias}.json").as_posix(),
        "gfm_data_root": (bundle / "adapters" / "gfmrag" / "data").as_posix(),
        "gfm_data_name": alias,
        "model_cache": (
            _root(config, "external_root").resolve() / "huggingface"
        ).as_posix(),
        "native_data_root": (repository / "data").as_posix(),
        "index_name": f"{method}_{dataset}",
    }
    values.update(variables)
    return values


def _configured_command(
    method_spec: Mapping[str, Any],
    track: str,
    stage: str,
) -> Optional[str]:
    track_spec = method_spec.get(track) or {}
    value = track_spec.get(f"{stage}_command")
    if value is None and stage in {"run", "qa", "native"}:
        value = track_spec.get("command")
    return str(value) if value else None


CONFIGURABLE_STAGES = ("prepare", "index", "retrieve", "qa", "generate")


def _configured_stages(
    method_spec: Mapping[str, Any],
    track: str,
) -> List[str]:
    return [
        stage
        for stage in CONFIGURABLE_STAGES
        if _configured_command(method_spec, track, stage)
    ]


def _requirement_status(method_spec: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        str(name): bool(os.environ.get(str(name)))
        for name in method_spec.get("requirements") or []
    }


def execute_stage(
    config: Mapping[str, Any],
    method: str,
    dataset: str,
    *,
    track: str,
    stage: str,
    command_override: Optional[str],
    variables: Mapping[str, str],
    force: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    method_spec = config["methods"][method]
    supported = (method_spec.get(track) or {}).get("supported_datasets")
    if supported and dataset not in supported:
        raise ValueError(
            f"{method} does not support {dataset} on the {track} track; "
            f"supported datasets: {', '.join(supported)}"
        )
    repository = _repo_path(config, method)
    if not (repository / ".git").exists():
        raise FileNotFoundError(
            f"{method} is not prepared at {repository}; run the lock command first."
        )
    prepared_path = _prepared_path(config, method)
    prepared = (
        json.loads(prepared_path.read_text(encoding="utf-8"))
        if prepared_path.exists()
        else None
    )
    if not dry_run:
        if prepared is None or prepared.get("commit") != method_spec["commit"]:
            raise RuntimeError(
                f"{method} has no matching repository lock; run the lock command."
            )
        if prepared.get("config_sha256") != config["_sha256"]:
            raise RuntimeError(
                f"{method} was locked against an older SOTA config; lock it again."
            )
        if not method_spec.get("internal") and not prepared.get("installed"):
            raise RuntimeError(
                f"{method}'s isolated environment is not installed; "
                "run lock --install before executing a stage."
            )
    run_dir = (
        _root(config, "output_root")
        / "runs"
        / track
        / method
        / dataset
        / str(config["suite"]["id"])
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    template = _template_values(config, method, dataset, run_dir, variables)
    command_template = command_override or _configured_command(
        method_spec, track, stage
    )
    if not command_template:
        raise ValueError(
            f"No {track}.{stage} command is configured for {method}; "
            "provide --command after reviewing the official adapter."
        )
    try:
        command = command_template.format(**template)
    except KeyError as exc:
        raise ValueError(f"Missing template variable {exc.args[0]!r}.") from exc

    bundle = latest_bundle(_root(config, "output_root"), dataset)
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    signature_payload = {
        "suite": config["suite"]["id"],
        "config_sha256": config["_sha256"],
        "method": method,
        "repository_commit": method_spec["commit"],
        "source_fingerprint": (
            _internal_source_fingerprint(repository)
            if method_spec.get("internal")
            else None
        ),
        "dataset": dataset,
        "bundle_fingerprint": bundle_manifest["fingerprint"],
        "track": track,
        "stage": stage,
        "command": command,
        "reader": (
            config.get("reader")
            if track == "matched" and stage in {"qa", "generate"}
            else None
        ),
    }
    signature = _stage_signature(signature_payload)
    stage_dir = run_dir / "stages" / stage
    record_path = stage_dir / "stage.json"
    if record_path.exists() and not force:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and existing.get("status") == "completed":
            log.info("Reusing completed %s/%s/%s stage %s", method, dataset, track, stage)
            return existing
    plan = {
        **signature_payload,
        "signature": signature,
        "repository": repository.as_posix(),
        "run_dir": run_dir.as_posix(),
    }
    if dry_run:
        return {"status": "dry_run", **plan}

    stage_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    before_bytes = directory_size(run_dir)
    started = time.perf_counter()
    started_at = _utc_now()
    peak_rss = 0
    peak_gpu_memory = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(repository),
            shell=True,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=os.environ.copy(),
        )
        while process.poll() is None:
            peak_rss = max(peak_rss, _process_rss(process))
            peak_gpu_memory = max(
                peak_gpu_memory,
                _process_gpu_memory(process),
            )
            time.sleep(0.5)
        return_code = int(process.returncode)
    duration = time.perf_counter() - started
    record = {
        **plan,
        "status": "completed" if return_code == 0 else "failed",
        "return_code": return_code,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_time_seconds": round(duration, 3),
        "peak_rss_bytes": peak_rss or None,
        "peak_gpu_memory_bytes": peak_gpu_memory or None,
        "run_directory_bytes_before": before_bytes,
        "run_directory_bytes_after": directory_size(run_dir),
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    _atomic_json(record_path, record)
    if return_code != 0:
        raise RuntimeError(
            f"{method}/{dataset}/{stage} failed with exit code {return_code}; "
            f"see {stderr_path}"
        )
    return record


def _extract_retrieved_ids(row: Mapping[str, Any]) -> List[str]:
    for field in (
        "retrieved_document_ids",
        "retrieved_doc_ids",
        "document_ids",
        "doc_ids",
        "retrieved_docs",
    ):
        values = row.get(field)
        if values is None:
            continue
        if isinstance(values, Mapping):
            values = (
                values.get("document")
                or values.get("documents")
                or values.get("passage")
                or values.get("passages")
                or []
            )
        if isinstance(values, (str, bytes)):
            values = [values]
        output = []
        for value in values:
            if isinstance(value, Mapping):
                document_id = (
                    value.get("id")
                    or value.get("document_id")
                    or value.get("title")
                    or value.get("name")
                )
            else:
                document_id = value
            if document_id is not None:
                output.append(str(document_id))
        return list(dict.fromkeys(output))
    return []


def hydrate_retrieval(
    config: Mapping[str, Any],
    method: str,
    dataset: str,
    source: Path,
    output: Path,
    *,
    split: str = "test",
    allow_partial: bool = False,
) -> Dict[str, Any]:
    """Join an external retriever's IDs to immutable questions and full contexts."""
    bundle = latest_bundle(_root(config, "output_root"), dataset)
    documents = {
        str(row["id"]): row for row in load_jsonl(bundle / "canonical" / "documents.jsonl")
    }
    questions = load_jsonl(bundle / "canonical" / "queries" / f"{split}.jsonl")
    retrieval_rows = load_jsonl(source)
    retrieval_by_id = {
        str(row.get("id") or row.get("query_id")): row for row in retrieval_rows
    }
    missing = [row["id"] for row in questions if row["id"] not in retrieval_by_id]
    if missing and not allow_partial:
        raise ValueError(
            f"{source} is missing {len(missing)} of {len(questions)} {split} queries."
        )
    max_retrieval = max(map(int, config["suite"]["retrieval_k"]))
    context_budget = int(config["suite"]["context_document_budget"])
    hydrated = []
    for question in questions:
        external = retrieval_by_id.get(str(question["id"]))
        if external is None:
            continue
        retrieved_ids = [
            document_id
            for document_id in _extract_retrieved_ids(external)
            if document_id in documents
        ][:max_retrieval]
        contexts = []
        for rank, document_id in enumerate(retrieved_ids[:context_budget], start=1):
            document = documents[document_id]
            contexts.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "title": document["title"],
                    "text": document["text"],
                }
            )
        hydrated.append(
            {
                "contract_version": 1,
                "id": str(question["id"]),
                "dataset": dataset,
                "split": split,
                "method": method,
                "question": question["question"],
                "answers": question["answers"],
                "supporting_document_ids": question["supporting_document_ids"],
                "retrieved_document_ids": retrieved_ids,
                "contexts": contexts,
                "latency_ms": external.get("latency_ms") or {},
                "native_output": {
                    key: external[key]
                    for key in ("prediction", "answer", "scores", "path")
                    if key in external
                },
            }
        )
    validate_run_rows(hydrated)
    _atomic_jsonl(output, hydrated)
    manifest = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "source": source.as_posix(),
        "source_sha256": sha256_file(source),
        "output": output.as_posix(),
        "output_sha256": sha256_file(output),
        "rows": len(hydrated),
        "missing_queries": len(missing),
        "bundle_fingerprint": json.loads(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "created_at": _utc_now(),
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def materialize_text_input(
    config: Mapping[str, Any],
    method: str,
    dataset: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Materialize one text file per document only for systems that require it."""
    bundle = latest_bundle(_root(config, "output_root"), dataset)
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = str(bundle_manifest["fingerprint"])
    target = (
        _root(config, "output_root")
        / "materialized"
        / method
        / dataset
        / fingerprint[:16]
    )
    marker = target / "materialization.json"
    signature = _stage_signature(
        {
            "method": method,
            "dataset": dataset,
            "bundle_fingerprint": fingerprint,
            "layout": "numbered_text_files_v1",
        }
    )
    if marker.exists() and not force:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and existing.get("status") == "completed":
            return existing
    target.mkdir(parents=True, exist_ok=True)
    documents = load_jsonl(bundle / "canonical" / "documents.jsonl")
    mapping_path = target / "document_map.jsonl"
    mapping_rows = []
    for document in documents:
        filename = f"{int(document['index']):08d}.txt"
        path = target / filename
        if force or not path.exists():
            path.write_text(
                f"{document['title']}\n{document['text']}",
                encoding="utf-8",
            )
        mapping_rows.append(
            {
                "filename": filename,
                "document_id": document["id"],
                "title": document["title"],
            }
        )
    _atomic_jsonl(mapping_path, mapping_rows)
    record = {
        "status": "completed",
        "signature": signature,
        "method": method,
        "dataset": dataset,
        "bundle_fingerprint": fingerprint,
        "document_count": len(documents),
        "input_dir": target.as_posix(),
        "input_size_bytes": directory_size(target),
        "mapping_sha256": sha256_file(mapping_path),
        "completed_at": _utc_now(),
    }
    _atomic_json(marker, record)
    _atomic_json(
        _root(config, "output_root") / "materialized" / method / dataset / "latest.json",
        {
            "signature": signature,
            "input_dir": target.as_posix(),
            "marker": marker.as_posix(),
        },
    )
    return record


def _load_tokenizer(reader: Mapping[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(reader["model"]),
        revision=reader.get("revision"),
    )


def _fit_context(
    contexts: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    token_budget: int,
) -> str:
    pieces = []
    used = 0
    for context in contexts:
        piece = f"[{context.get('rank', len(pieces) + 1)}] {context.get('title', '')}\n{context.get('text', '')}"
        token_ids = tokenizer.encode(piece, add_special_tokens=False)
        remaining = token_budget - used
        if remaining <= 0:
            break
        if len(token_ids) > remaining:
            piece = tokenizer.decode(token_ids[:remaining], skip_special_tokens=True)
            pieces.append(piece)
            break
        pieces.append(piece)
        used += len(token_ids)
    return "\n\n".join(pieces)


def generate_common_reader(
    config: Mapping[str, Any],
    source: Path,
    output: Path,
    *,
    force: bool = False,
    limit: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    reader = config["reader"]
    rows = load_jsonl(source)
    if limit:
        rows = rows[:limit]
    validate_run_rows(rows)
    signature_payload = {
        "source_sha256": sha256_file(source),
        "reader": reader,
        "context_document_budget": config["suite"]["context_document_budget"],
        "context_token_budget": config["suite"]["context_token_budget"],
        "limit": limit,
    }
    signature = _stage_signature(signature_payload)
    manifest_path = output.with_suffix(".generation.json")
    if manifest_path.exists() and output.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature and manifest.get("status") == "completed":
            return manifest
    if dry_run:
        return {
            "status": "dry_run",
            "signature": signature,
            "rows": len(rows),
            **signature_payload,
        }

    base_url = os.environ.get(str(reader["base_url_env"]))
    api_key = os.environ.get(str(reader["api_key_env"]))
    if not base_url or not api_key:
        raise RuntimeError(
            f"Set {reader['base_url_env']} and {reader['api_key_env']} for the common reader."
        )
    from openai import OpenAI

    tokenizer = _load_tokenizer(reader)
    client = OpenAI(base_url=base_url, api_key=api_key)
    partial = output.with_suffix(output.suffix + ".partial")
    completed: Dict[str, Dict[str, Any]] = {}
    if partial.exists() and not force:
        completed = {str(row["id"]): row for row in load_jsonl(partial)}
    elif force and partial.exists():
        partial.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    with partial.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            row_id = str(row["id"])
            if row_id in completed and completed[row_id].get("generation_signature") == signature:
                continue
            context = _fit_context(
                row.get("contexts") or [],
                tokenizer,
                int(config["suite"]["context_token_budget"]),
            )
            user_prompt = str(reader["user_prompt"]).format(
                context=context,
                question=row["question"],
            )
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=str(reader["model"]),
                messages=[
                    {"role": "system", "content": str(reader["system_prompt"])},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=float(reader["temperature"]),
                max_tokens=int(reader["max_output_tokens"]),
                seed=int(reader["seed"]),
            )
            generation_ms = (time.perf_counter() - started) * 1000.0
            usage = response.usage
            enriched = dict(row)
            enriched["prediction"] = response.choices[0].message.content or ""
            enriched["generation_signature"] = signature
            enriched["latency_ms"] = {
                **(row.get("latency_ms") or {}),
                "generation": round(generation_ms, 3),
                "total": round(
                    generation_ms
                    + float((row.get("latency_ms") or {}).get("retrieval") or 0.0),
                    3,
                ),
            }
            enriched["usage"] = {
                "prompt_tokens": int(usage.prompt_tokens) if usage else None,
                "completion_tokens": int(usage.completion_tokens) if usage else None,
                "total_tokens": int(usage.total_tokens) if usage else None,
            }
            enriched["reader_provenance"] = {
                "model": reader["model"],
                "revision": reader.get("revision"),
                "response_id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "prompt_sha256": hashlib.sha256(
                    user_prompt.encode("utf-8")
                ).hexdigest(),
            }
            handle.write(_stable_json(enriched) + "\n")
            handle.flush()
            completed[row_id] = enriched

    ordered = [completed[str(row["id"])] for row in rows]
    _atomic_jsonl(output, ordered)
    partial.unlink(missing_ok=True)
    manifest = {
        "status": "completed",
        "signature": signature,
        "source": source.as_posix(),
        "output": output.as_posix(),
        "output_sha256": sha256_file(output),
        "rows": len(ordered),
        "reader": reader,
        "completed_at": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def evaluate_run(
    config: Mapping[str, Any],
    source: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    rows = load_jsonl(source)
    summary, per_query = evaluate_end_to_end(
        rows,
        ks=tuple(map(int, config["suite"]["retrieval_k"])),
        bootstrap_samples=int(config["suite"]["bootstrap_samples"]),
        bootstrap_seed=int(config["suite"]["bootstrap_seed"]),
        pricing=config["reader"].get("pricing_per_million_tokens"),
    )
    summary["source"] = source.as_posix()
    summary["source_sha256"] = sha256_file(source)
    summary["suite_id"] = config["suite"]["id"]
    summary["config_sha256"] = config["_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_jsonl(output_dir / "per_query.jsonl", per_query)
    return summary


def compare_runs(
    config: Mapping[str, Any],
    baseline: Path,
    treatment: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    baseline_rows = load_jsonl(baseline)
    treatment_rows = load_jsonl(treatment)
    summary, per_query = compare_end_to_end(
        baseline_rows,
        treatment_rows,
        ks=tuple(map(int, config["suite"]["retrieval_k"])),
        bootstrap_samples=int(config["suite"]["bootstrap_samples"]),
        bootstrap_seed=int(config["suite"]["bootstrap_seed"]),
    )
    summary.update(
        {
            "suite_id": config["suite"]["id"],
            "config_sha256": config["_sha256"],
            "baseline": baseline.as_posix(),
            "baseline_sha256": sha256_file(baseline),
            "treatment": treatment.as_posix(),
            "treatment_sha256": sha256_file(treatment),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "comparison.json", summary)
    _atomic_jsonl(output_dir / "per_query_deltas.jsonl", per_query)
    return summary


def _parse_variables(values: Sequence[str]) -> Dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        output[key] = item
    return output


def _command_export(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    datasets = _selected(args.datasets, config["suite"]["datasets"])
    manifests = [
        export_dataset_bundle(
            dataset,
            _root(config, "output_root"),
            include_synthetic_edges=bool(config["suite"]["include_synthetic_edges"]),
        )
        for dataset in datasets
    ]
    print(json.dumps(manifests, indent=2))


def _command_lock(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    methods = _selected(args.methods, config["methods"])
    results = [
        lock_repository(config, method, install=args.install) for method in methods
    ]
    print(json.dumps(results, indent=2))


def _command_stage(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = execute_stage(
        config,
        args.method,
        args.dataset,
        track=args.track,
        stage=args.stage,
        command_override=args.command,
        variables=_parse_variables(args.var),
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


def _command_hydrate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = hydrate_retrieval(
        config,
        args.method,
        args.dataset,
        Path(args.source),
        Path(args.output),
        split=args.split,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, indent=2))


def _command_materialize(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = materialize_text_input(
        config,
        args.method,
        args.dataset,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


def _command_generate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = generate_common_reader(
        config,
        Path(args.source),
        Path(args.output),
        force=args.force,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


def _command_evaluate(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = evaluate_run(config, Path(args.source), Path(args.output_dir))
    print(json.dumps(result, indent=2))


def _command_compare(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    result = compare_runs(
        config,
        Path(args.baseline),
        Path(args.treatment),
        Path(args.output_dir),
    )
    print(json.dumps(result, indent=2))


def _audit_bundle(
    output_root: Path,
    dataset: str,
    *,
    verify_hashes: bool,
) -> Dict[str, Any]:
    pointer_path = output_root / "bundles" / dataset / "latest.json"
    output: Dict[str, Any] = {
        "dataset": dataset,
        "pointer_exists": pointer_path.exists(),
        "manifest_exists": False,
        "fingerprint_matches": False,
        "artifact_count": 0,
        "hashes_verified": verify_hashes,
        "missing_artifacts": [],
        "hash_mismatches": [],
        "ready": False,
    }
    if not pointer_path.exists():
        return output
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(pointer.get("manifest") or ""))
    if not manifest_path.exists():
        return output
    output["manifest_exists"] = True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output["fingerprint"] = manifest.get("fingerprint")
    output["fingerprint_matches"] = (
        bool(manifest.get("fingerprint"))
        and manifest.get("fingerprint") == pointer.get("fingerprint")
    )
    output["labels_excluded_from_index"] = bool(
        manifest.get("labels_excluded_from_index")
    )
    output["questions_excluded_from_index"] = bool(
        manifest.get("questions_excluded_from_index")
    )
    artifacts = manifest.get("artifacts") or {}
    output["artifact_count"] = len(artifacts)
    if verify_hashes:
        bundle_dir = Path(str(pointer.get("bundle_dir") or manifest_path.parent))
        for relative, expected in sorted(artifacts.items()):
            path = bundle_dir / str(relative)
            if not path.exists():
                output["missing_artifacts"].append(str(relative))
            elif sha256_file(path) != str(expected):
                output["hash_mismatches"].append(str(relative))
    output["ready"] = bool(
        output["manifest_exists"]
        and output["fingerprint_matches"]
        and output["labels_excluded_from_index"]
        and output["questions_excluded_from_index"]
        and not output["missing_artifacts"]
        and not output["hash_mismatches"]
    )
    return output


def _audit_method(
    config: Mapping[str, Any],
    method: str,
    datasets: Sequence[str],
) -> Dict[str, Any]:
    method_spec = config["methods"][method]
    matched = method_spec.get("matched") or {}
    prepared_path = _prepared_path(config, method)
    prepared = (
        json.loads(prepared_path.read_text(encoding="utf-8"))
        if prepared_path.exists()
        else None
    )
    repository_pinned = bool(
        prepared and prepared.get("commit") == method_spec.get("commit")
    )
    config_current = bool(
        prepared and prepared.get("config_sha256") == config["_sha256"]
    )
    current_source_fingerprint = (
        _internal_source_fingerprint(_repo_path(config, method))
        if method_spec.get("internal")
        else None
    )
    source_current = bool(
        not method_spec.get("internal")
        or (
            prepared
            and prepared.get("source_fingerprint") == current_source_fingerprint
        )
    )
    environment_installed = bool(prepared and prepared.get("installed"))
    requirements = _requirement_status(method_spec)
    missing_requirements = sorted(
        name for name, available in requirements.items() if not available
    )
    configured_stages = _configured_stages(method_spec, "matched")
    supported = matched.get("supported_datasets")
    supported_set = set(supported or config["suite"]["datasets"])
    dataset_state = {}
    output_root = _root(config, "output_root")
    for dataset in datasets:
        if dataset not in supported_set:
            dataset_state[dataset] = {"supported": False}
            continue
        run_dir = (
            output_root
            / "runs"
            / "matched"
            / method
            / dataset
            / str(config["suite"]["id"])
        )
        stages = {}
        for stage in configured_stages:
            record_path = run_dir / "stages" / stage / "stage.json"
            record = (
                json.loads(record_path.read_text(encoding="utf-8"))
                if record_path.exists()
                else None
            )
            stages[stage] = {
                "status": record.get("status") if record else "not_run",
                "signature": record.get("signature") if record else None,
            }
        input_marker = run_dir / "index" / "input" / "preparation.json"
        dataset_state[dataset] = {
            "supported": True,
            "configured_stages": stages,
            "content_addressed_input_prepared": (
                input_marker.exists() if "prepare" in configured_stages else None
            ),
        }

    blockers = []
    if not repository_pinned:
        blockers.append("repository_lock_missing_or_mismatched")
    if not config_current:
        blockers.append("repository_lock_uses_stale_config")
    if not source_current:
        blockers.append("internal_source_fingerprint_stale")
    if not environment_installed:
        blockers.append("isolated_environment_not_installed")
    if missing_requirements:
        blockers.append("required_runtime_variables_missing")
    if "retrieve" not in configured_stages:
        blockers.append("matched_retrieve_stage_not_configured")
    completed_retrievals = [
        dataset
        for dataset, state in dataset_state.items()
        if state.get("supported")
        and (state.get("configured_stages") or {})
        .get("retrieve", {})
        .get("status")
        == "completed"
    ]
    return {
        "tier": method_spec.get("tier"),
        "repository_pinned": repository_pinned,
        "config_current": config_current,
        "source_current": source_current,
        "current_source_fingerprint": current_source_fingerprint,
        "environment_installed": environment_installed,
        "requirements_present": requirements,
        "missing_requirements": missing_requirements,
        "matched_configured_stages": configured_stages,
        "datasets": dataset_state,
        "launch_ready": not blockers,
        "matched_retrievals_completed": completed_retrievals,
        "publication_result_ready": bool(not blockers and completed_retrievals),
        "blockers": blockers,
    }


def _command_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    datasets = _selected(args.datasets, config["suite"]["datasets"])
    methods = _selected(args.methods, config["methods"])
    bundles = {
        dataset: _audit_bundle(
            _root(config, "output_root"),
            dataset,
            verify_hashes=args.verify_hashes,
        )
        for dataset in datasets
    }
    method_audits = {
        method: _audit_method(config, method, datasets) for method in methods
    }
    result = {
        "suite_id": config["suite"]["id"],
        "config_sha256": config["_sha256"],
        "hashes_verified": args.verify_hashes,
        "bundles": bundles,
        "methods": method_audits,
        "all_bundles_ready": all(item["ready"] for item in bundles.values()),
        "all_selected_methods_launch_ready": all(
            item["launch_ready"] for item in method_audits.values()
        ),
        "all_selected_methods_have_publication_result": all(
            item["publication_result_ready"] for item in method_audits.values()
        ),
        "audited_at": _utc_now(),
    }
    print(json.dumps(result, indent=2))


def _command_status(_: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output_root = _root(config, "output_root")
    status = {
        "suite_id": config["suite"]["id"],
        "config_sha256": config["_sha256"],
        "bundles": {},
        "repositories": {},
        "method_readiness": {},
        "runs": [],
    }
    for dataset in config["suite"]["datasets"]:
        pointer = output_root / "bundles" / dataset / "latest.json"
        status["bundles"][dataset] = (
            json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None
        )
    for method in config["methods"]:
        repo = _repo_path(config, method)
        prepared = _prepared_path(config, method)
        prepared_record = (
            json.loads(prepared.read_text(encoding="utf-8"))
            if prepared.exists()
            else None
        )
        status["repositories"][method] = prepared_record
        method_spec = config["methods"][method]
        matched = method_spec.get("matched") or {}
        configured_stages = _configured_stages(method_spec, "matched")
        requirements = _requirement_status(method_spec)
        current_source_fingerprint = (
            _internal_source_fingerprint(repo) if method_spec.get("internal") else None
        )
        status["method_readiness"][method] = {
            "tier": method_spec.get("tier"),
            "repository_pinned": bool(
                prepared_record
                and prepared_record.get("commit") == method_spec.get("commit")
            ),
            "config_current": bool(
                prepared_record
                and prepared_record.get("config_sha256") == config["_sha256"]
            ),
            "source_current": bool(
                not method_spec.get("internal")
                or (
                    prepared_record
                    and prepared_record.get("source_fingerprint")
                    == current_source_fingerprint
                )
            ),
            "environment_installed": bool(
                prepared_record and prepared_record.get("installed")
            ),
            "matched_input_adapter": matched.get("input_adapter"),
            "matched_configured_stages": configured_stages,
            "matched_launch_configured": "retrieve" in configured_stages,
            "supported_datasets": matched.get("supported_datasets"),
            "requirements_present": requirements,
            "missing_requirements": sorted(
                name for name, available in requirements.items() if not available
            ),
        }
    runs_root = output_root / "runs"
    if runs_root.exists():
        for stage_file in sorted(runs_root.rglob("stage.json")):
            record = json.loads(stage_file.read_text(encoding="utf-8"))
            status["runs"].append(
                {
                    "method": record.get("method"),
                    "dataset": record.get("dataset"),
                    "track": record.get("track"),
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                    "signature": record.get("signature"),
                }
            )
    for method, readiness in status["method_readiness"].items():
        completed = [
            {
                "dataset": run["dataset"],
                "track": run["track"],
                "stage": run["stage"],
            }
            for run in status["runs"]
            if run["method"] == method and run["status"] == "completed"
        ]
        readiness["completed_stages"] = completed
        readiness["matched_smoke_validated"] = any(
            run["track"] == "matched" and run["stage"] == "retrieve"
            for run in completed
        )
    print(json.dumps(status, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-ingestion, end-to-end SOTA RAG benchmark"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    export = subparsers.add_parser("export", help="Export immutable full-corpus bundles")
    export.add_argument("--datasets", nargs="+")
    export.set_defaults(func=_command_export)

    lock = subparsers.add_parser("lock", help="Clone and pin official repositories")
    lock.add_argument("--methods", nargs="+")
    lock.add_argument("--install", action="store_true")
    lock.set_defaults(func=_command_lock)

    stage = subparsers.add_parser("stage", help="Run one resumable official stage")
    stage.add_argument("--method", required=True)
    stage.add_argument("--dataset", required=True)
    stage.add_argument("--track", choices=("native", "matched"), required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--command")
    stage.add_argument("--var", action="append", default=[])
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--dry-run", action="store_true")
    stage.set_defaults(func=_command_stage)

    hydrate = subparsers.add_parser(
        "hydrate", help="Join external retrieval IDs to canonical questions and contexts"
    )
    hydrate.add_argument("--method", required=True)
    hydrate.add_argument("--dataset", required=True)
    hydrate.add_argument("--source", required=True)
    hydrate.add_argument("--output", required=True)
    hydrate.add_argument("--split", default="test", choices=("train", "val", "test"))
    hydrate.add_argument("--allow-partial", action="store_true")
    hydrate.set_defaults(func=_command_hydrate)

    materialize = subparsers.add_parser(
        "materialize-text",
        help="Create persistent one-file-per-document input for a baseline",
    )
    materialize.add_argument("--method", required=True)
    materialize.add_argument("--dataset", required=True)
    materialize.add_argument("--force", action="store_true")
    materialize.set_defaults(func=_command_materialize)

    generate = subparsers.add_parser(
        "generate", help="Run the frozen common reader with resumable outputs"
    )
    generate.add_argument("--source", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.set_defaults(func=_command_generate)

    evaluate = subparsers.add_parser(
        "evaluate", help="Compute retrieval, answer, grounding, and efficiency metrics"
    )
    evaluate.add_argument("--source", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=_command_evaluate)

    compare = subparsers.add_parser(
        "compare",
        help="Compute paired deltas, confidence intervals, and significance tests",
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--treatment", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=_command_compare)

    audit = subparsers.add_parser(
        "audit",
        help="Audit immutable bundles, locks, runtime requirements, and stages",
    )
    audit.add_argument("--methods", nargs="+")
    audit.add_argument("--datasets", nargs="+")
    audit.add_argument("--verify-hashes", action="store_true")
    audit.set_defaults(func=_command_audit)

    status = subparsers.add_parser("status", help="Show reusable bundles and stages")
    status.set_defaults(func=_command_status)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(Path(args.config))
    args.func(args, config)


if __name__ == "__main__":
    main(sys.argv[1:])
