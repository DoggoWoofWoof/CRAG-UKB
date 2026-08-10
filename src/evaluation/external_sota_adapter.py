"""Adapters executed inside pinned third-party SOTA environments.

Keep this module stdlib-only at import time. Each subcommand imports the
official implementation only after adding its pinned checkout to sys.path.
"""
from __future__ import annotations

import argparse
import asyncio
import builtins
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    handle.flush()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            _append_jsonl(handle, row)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_openai_credentials() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "LightRAG indexing and keyword extraction require OPENAI_API_KEY. "
            "OPENAI_BASE_URL may be set for a compatible endpoint, but a "
            "paper-faithful run must serve the named OpenAI models."
        )


def _canonical_documents(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl(path)
    seen = set()
    for index, row in enumerate(rows):
        document_id = str(row.get("id", ""))
        if not document_id:
            raise ValueError(f"Canonical document {index} has no id.")
        if document_id in seen:
            raise ValueError(f"Duplicate canonical document id {document_id!r}.")
        seen.add(document_id)
    return rows


def _document_text(document: Mapping[str, Any]) -> str:
    return f"{document.get('title', '')}\n{document.get('text', '')}".strip()


def _lightrag_identity(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "adapter_version": 1,
        "implementation": args.implementation,
        "corpus_sha256": _sha256_file(args.corpus),
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 100,
        "entity_extract_max_gleaning": 1,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    }


def _lightrag_progress(
    args: argparse.Namespace,
    documents: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    path = args.working_dir / "crag_adapter_progress.json"
    identity = _lightrag_identity(args)
    if path.exists():
        progress = _load_json(path)
        if progress.get("identity") != identity:
            raise RuntimeError(
                "The persistent LightRAG index belongs to a different corpus "
                "or adapter configuration. Use a new working directory."
            )
        if progress.get("active_document_ids"):
            raise RuntimeError(
                "A previous LightRAG insertion stopped inside a non-transactional "
                "author batch. The index is fail-closed; use a new working directory "
                "rather than silently accepting a potentially partial graph."
            )
        next_index = int(progress.get("next_document_index", 0))
        if not 0 <= next_index <= len(documents):
            raise RuntimeError("Invalid LightRAG progress checkpoint.")
        return progress
    return {
        "status": "pending",
        "identity": identity,
        "document_count": len(documents),
        "next_document_index": 0,
        "active_document_ids": [],
    }


def _write_lightrag_progress(
    working_dir: Path,
    progress: Mapping[str, Any],
) -> None:
    _write_json(working_dir / "crag_adapter_progress.json", progress)


def _lightrag_document_paths(
    documents: Sequence[Mapping[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
    paths = [f"crag_docs/{index:08d}.txt" for index in range(len(documents))]
    mapping = {
        path: str(document["id"]) for path, document in zip(paths, documents)
    }
    return paths, mapping


def _paper_lightrag_index(
    args: argparse.Namespace,
    documents: Sequence[Mapping[str, Any]],
    progress: Dict[str, Any],
) -> None:
    sys.path.insert(0, str(args.repo.resolve()))
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(args.working_dir.resolve()),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
    )
    batch_size = max(1, int(args.batch_size))
    start = int(progress["next_document_index"])
    for offset in range(start, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        progress["status"] = "running"
        progress["active_document_ids"] = [str(row["id"]) for row in batch]
        _write_lightrag_progress(args.working_dir, progress)
        rag.insert([_document_text(row) for row in batch])
        progress["next_document_index"] = offset + len(batch)
        progress["active_document_ids"] = []
        _write_lightrag_progress(args.working_dir, progress)


async def _current_lightrag_instance(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo.resolve()))
    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    rag = LightRAG(
        working_dir=str(args.working_dir.resolve()),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
        llm_model_name="gpt-4o-mini",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    return rag


async def _current_lightrag_index(
    args: argparse.Namespace,
    documents: Sequence[Mapping[str, Any]],
    progress: Dict[str, Any],
) -> None:
    rag = await _current_lightrag_instance(args)
    paths, mapping = _lightrag_document_paths(documents)
    _write_json(args.working_dir / "crag_document_map.json", mapping)
    batch_size = max(1, int(args.batch_size))
    start = int(progress["next_document_index"])
    try:
        for offset in range(start, len(documents), batch_size):
            batch = documents[offset : offset + batch_size]
            batch_paths = paths[offset : offset + len(batch)]
            progress["status"] = "running"
            progress["active_document_ids"] = [str(row["id"]) for row in batch]
            _write_lightrag_progress(args.working_dir, progress)
            await rag.ainsert(
                [_document_text(row) for row in batch],
                ids=[str(row["id"]) for row in batch],
                file_paths=batch_paths,
            )
            progress["next_document_index"] = offset + len(batch)
            progress["active_document_ids"] = []
            _write_lightrag_progress(args.working_dir, progress)
    finally:
        await rag.finalize_storages()


def lightrag_index(args: argparse.Namespace) -> None:
    _require_openai_credentials()
    documents = _canonical_documents(args.corpus)
    args.working_dir.mkdir(parents=True, exist_ok=True)
    progress = _lightrag_progress(args, documents)
    started = time.perf_counter()
    if args.implementation == "paper":
        _paper_lightrag_index(args, documents, progress)
    else:
        asyncio.run(_current_lightrag_index(args, documents, progress))
    progress["status"] = "completed"
    progress["active_document_ids"] = []
    _write_lightrag_progress(args.working_dir, progress)
    _write_json(
        args.output,
        {
            "status": "completed",
            "method": f"lightrag_{args.implementation}",
            "documents": len(documents),
            "identity": progress["identity"],
            "index_directory": str(args.working_dir.resolve()),
            "wall_time_seconds": round(time.perf_counter() - started, 3),
        },
    )


def _parse_keyword_json(value: str) -> Mapping[str, Any]:
    value = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("LightRAG keyword extraction did not return an object.")
    return parsed


def _paper_source_contents(context: str | None) -> List[str]:
    """Parse the author's CSV source section without changing selected chunks."""
    if not context:
        return []
    match = re.search(
        r"-----Sources-----\s*```csv\s*(.*?)\s*```",
        context,
        flags=re.DOTALL,
    )
    if not match:
        return []
    rows = list(csv.reader(io.StringIO(match.group(1).strip())))
    if not rows:
        return []
    header = [field.strip().lower() for field in rows[0]]
    try:
        content_index = header.index("content")
    except ValueError as exc:
        raise ValueError("LightRAG paper context has no source content column.") from exc
    return [
        row[content_index]
        for row in rows[1:]
        if len(row) > content_index and row[content_index]
    ]


def _round_robin_unique(*rankings: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    width = max((len(ranking) for ranking in rankings), default=0)
    for index in range(width):
        for ranking in rankings:
            if index >= len(ranking):
                continue
            value = ranking[index]
            if value not in seen:
                seen.add(value)
                output.append(value)
    return output


def _paper_chunk_document_map(
    working_dir: Path,
    documents: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    full_doc_ids = {
        "doc-" + hashlib.md5(_document_text(row).encode()).hexdigest(): str(row["id"])
        for row in documents
    }
    chunks_path = working_dir / "kv_store_text_chunks.json"
    chunks = _load_json(chunks_path)
    output = {}
    for chunk in chunks.values():
        document_id = full_doc_ids.get(str(chunk.get("full_doc_id")))
        content = str(chunk.get("content") or "")
        if document_id and content:
            output[content] = document_id
    if not output:
        raise RuntimeError("The LightRAG paper index has no mappable text chunks.")
    return output


async def _paper_lightrag_sources(
    rag: Any,
    question: str,
    top_k: int,
) -> List[str]:
    from lightrag.base import QueryParam
    from lightrag.operate import (
        _build_global_query_context,
        _build_local_query_context,
    )
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["keywords_extraction"].format(query=question)
    keywords = _parse_keyword_json(await rag.llm_model_func(prompt))
    query_param = QueryParam(
        mode="hybird",
        only_need_context=True,
        top_k=top_k,
    )
    high_context = await _build_global_query_context(
        keywords.get("high_level_keywords", []),
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.text_chunks,
        query_param,
    )
    low_context = await _build_local_query_context(
        keywords.get("low_level_keywords", []),
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        rag.text_chunks,
        query_param,
    )
    await rag._query_done()
    return _round_robin_unique(
        _paper_source_contents(high_context),
        _paper_source_contents(low_context),
    )


def _rank_documents_from_chunks(
    chunks: Sequence[str],
    chunk_to_document: Mapping[str, str],
    top_k: int,
) -> List[str]:
    output = []
    seen = set()
    for chunk in chunks:
        document_id = chunk_to_document.get(chunk)
        if document_id is None:
            raise KeyError("LightRAG returned a chunk absent from its persistent index.")
        if document_id not in seen:
            seen.add(document_id)
            output.append(document_id)
        if len(output) >= top_k:
            break
    return output


async def _paper_lightrag_retrieve(
    args: argparse.Namespace,
    queries: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    completed: Dict[str, Dict[str, Any]],
) -> None:
    sys.path.insert(0, str(args.repo.resolve()))
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(args.working_dir.resolve()),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
    )
    chunk_map = _paper_chunk_document_map(args.working_dir, documents)
    with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            query_id = str(query["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            chunks = await _paper_lightrag_sources(
                rag,
                str(query["question"]),
                args.top_k,
            )
            document_ids = _rank_documents_from_chunks(
                chunks,
                chunk_map,
                args.top_k,
            )
            row = {
                "id": query_id,
                "retrieved_document_ids": document_ids,
                "scores": [1.0 / rank for rank in range(1, len(document_ids) + 1)],
                "latency_ms": {
                    "retrieval": round((time.perf_counter() - started) * 1000.0, 3)
                },
            }
            _append_jsonl(handle, row)
            completed[query_id] = row


def _normalize_lightrag_path(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def _current_document_id(
    file_path: str,
    mapping: Mapping[str, str],
) -> str:
    normalized = _normalize_lightrag_path(file_path)
    normalized_mapping = {
        _normalize_lightrag_path(key): value for key, value in mapping.items()
    }
    if normalized in normalized_mapping:
        return normalized_mapping[normalized]
    basename = Path(normalized).name
    basename_mapping = {Path(key).name: value for key, value in mapping.items()}
    if basename in basename_mapping:
        return basename_mapping[basename]
    raise KeyError(f"Unknown LightRAG source path {file_path!r}.")


async def _current_lightrag_retrieve(
    args: argparse.Namespace,
    queries: Sequence[Mapping[str, Any]],
    completed: Dict[str, Dict[str, Any]],
) -> None:
    from lightrag import QueryParam

    rag = await _current_lightrag_instance(args)
    mapping = _load_json(args.working_dir / "crag_document_map.json")
    try:
        with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
            for query in queries:
                query_id = str(query["id"])
                if query_id in completed:
                    continue
                started = time.perf_counter()
                result = await rag.aquery_data(
                    str(query["question"]),
                    QueryParam(
                        mode="mix",
                        top_k=args.top_k,
                        chunk_top_k=args.top_k,
                        enable_rerank=False,
                    ),
                )
                chunks = (result.get("data") or {}).get("chunks") or []
                document_ids = []
                seen = set()
                for chunk in chunks:
                    document_id = _current_document_id(
                        str(chunk.get("file_path") or ""),
                        mapping,
                    )
                    if document_id not in seen:
                        seen.add(document_id)
                        document_ids.append(document_id)
                    if len(document_ids) >= args.top_k:
                        break
                row = {
                    "id": query_id,
                    "retrieved_document_ids": document_ids,
                    "scores": [
                        1.0 / rank for rank in range(1, len(document_ids) + 1)
                    ],
                    "latency_ms": {
                        "retrieval": round(
                            (time.perf_counter() - started) * 1000.0,
                            3,
                        )
                    },
                }
                _append_jsonl(handle, row)
                completed[query_id] = row
    finally:
        await rag.finalize_storages()


def lightrag_retrieve(args: argparse.Namespace) -> None:
    _require_openai_credentials()
    documents = _canonical_documents(args.corpus)
    queries = _load_jsonl(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    progress = _lightrag_progress(args, documents)
    if progress.get("status") != "completed":
        raise RuntimeError("Complete the LightRAG index stage before retrieval.")
    completed = {
        str(row["id"]): row for row in _load_jsonl(args.partial_output)
    }
    expected_ids = [str(row["id"]) for row in queries]
    if set(completed) - set(expected_ids):
        raise ValueError("Partial LightRAG output belongs to a different query set.")
    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    if args.implementation == "paper":
        asyncio.run(
            _paper_lightrag_retrieve(args, queries, documents, completed)
        )
        adapter_policy = (
            "author high/low keyword and graph retrieval; source chunks extracted "
            "before the paper code's unordered set formatting; deterministic "
            "high/low round-robin document fusion"
        )
    else:
        asyncio.run(_current_lightrag_retrieve(args, queries, completed))
        adapter_policy = (
            "official query_data mix retrieval; post-paper reranker disabled; "
            "chunk file provenance mapped to canonical document ids"
        )
    if set(completed) != set(expected_ids):
        raise RuntimeError("LightRAG retrieval did not produce every expected query.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in expected_ids:
            _append_jsonl(handle, completed[query_id])
    os.replace(temporary, args.output)
    args.partial_output.unlink(missing_ok=True)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": f"lightrag_{args.implementation}",
            "queries": len(queries),
            "top_k": args.top_k,
            "adapter_policy": adapter_policy,
            "identity": progress["identity"],
            "output": str(args.output.resolve()),
        },
    )


def _hipporag_components(repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    return HippoRAG, BaseConfig


def _hipporag_config(args: argparse.Namespace):
    _, base_config = _hipporag_components(args.repo)
    embedding_model = _locked_model_path(
        args.embedding_name,
        args.embedding_revision,
        args.model_cache,
    )
    return base_config(
        save_dir=str(args.save_dir.resolve()),
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=args.dataset_alias,
        embedding_model_name=embedding_model,
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        rerank_dspy_file_path=str(
            (
                args.repo
                / "src/hipporag/prompts/dspy_prompts/"
                "filter_llama3.3-70B-Instruct.json"
            ).resolve()
        ),
        retrieval_top_k=args.top_k,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode=args.openie_mode,
    )


def _hipporag_documents(corpus_path: Path) -> tuple[List[str], Dict[str, str]]:
    corpus = _load_json(corpus_path)
    documents = [f"{row['title']}\n{row['text']}" for row in corpus]
    document_ids = {document: str(row["title"]) for document, row in zip(documents, corpus)}
    if len(document_ids) != len(documents):
        raise ValueError("HippoRAG corpus contains duplicate title/text documents.")
    return documents, document_ids


def hipporag_index(args: argparse.Namespace) -> None:
    hipporag, _ = _hipporag_components(args.repo)
    documents, _ = _hipporag_documents(args.corpus)
    started = time.perf_counter()
    rag = hipporag(global_config=_hipporag_config(args))
    rag.index(documents)
    elapsed = time.perf_counter() - started
    _write_json(
        args.output,
        {
            "status": "completed",
            "method": "hipporag2",
            "dataset_alias": args.dataset_alias,
            "documents": len(documents),
            "index_directory": str(args.save_dir.resolve()),
            "wall_time_seconds": round(elapsed, 3),
        },
    )


def _scores(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def hipporag_retrieve(args: argparse.Namespace) -> None:
    hipporag, _ = _hipporag_components(args.repo)
    _, document_ids = _hipporag_documents(args.corpus)
    samples = _load_json(args.queries)
    if args.limit:
        samples = samples[: args.limit]

    completed = {
        str(row["id"]): row for row in _load_jsonl(args.partial_output)
    }
    expected_ids = [str(row["id"]) for row in samples]
    if set(completed) - set(expected_ids):
        raise ValueError("Partial HippoRAG output belongs to a different query set.")

    startup = time.perf_counter()
    rag = hipporag(global_config=_hipporag_config(args))
    rag.prepare_retrieval_objects()
    startup_seconds = time.perf_counter() - startup

    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            query_id = str(sample["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            result = rag.retrieve(
                [str(sample["question"])],
                num_to_retrieve=args.top_k,
            )[0]
            latency_ms = (time.perf_counter() - started) * 1000.0
            missing = [document for document in result.docs if document not in document_ids]
            if missing:
                raise KeyError(
                    "HippoRAG returned a document not present in the immutable corpus."
                )
            row = {
                "id": query_id,
                "retrieved_document_ids": [
                    document_ids[document] for document in result.docs
                ],
                "scores": _scores(result.doc_scores),
                "latency_ms": {"retrieval": round(latency_ms, 3)},
            }
            _append_jsonl(handle, row)
            completed[query_id] = row

    if set(completed) != set(expected_ids):
        raise RuntimeError("HippoRAG retrieval did not produce every expected query.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in expected_ids:
            _append_jsonl(handle, completed[query_id])
    os.replace(temporary, args.output)
    args.partial_output.unlink(missing_ok=True)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": "hipporag2",
            "dataset_alias": args.dataset_alias,
            "queries": len(samples),
            "top_k": args.top_k,
            "warm_start_seconds": round(startup_seconds, 3),
            "output": str(args.output.resolve()),
        },
    )


def _prepare_gfm_data(
    source_root: Path,
    working_root: Path,
    data_name: str,
    *,
    paper_v1: bool,
) -> Path:
    source = source_root / data_name / "raw"
    target = working_root / data_name / "raw"
    target.mkdir(parents=True, exist_ok=True)
    documents = _load_json(source / "documents.json")
    if paper_v1:
        _write_json(target / f"{data_name}_corpus.json", documents)
    else:
        _write_json(target / "documents.json", documents)

    # The matched retriever receives no answers or support labels. Gold data
    # stays in CRAG's evaluator bundle even if the official index workflow also
    # preprocesses its test query file.
    for split in ("test",):
        source_path = source / f"{split}.json"
        if not source_path.exists():
            continue
        rows = [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": "",
                "answer_aliases": [],
                **(
                    {"supporting_facts": []}
                    if paper_v1
                    else {"supporting_documents": []}
                ),
            }
            for row in _load_json(source_path)
        ]
        temporary = target / f"{split}.json.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, target / f"{split}.json")
    return working_root


def _run_module(repo: Path, arguments: Sequence[str]) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=str(repo.resolve()),
        check=True,
    )


def gfmrag_index(args: argparse.Namespace) -> None:
    data_root = _prepare_gfm_data(
        args.source_data_root,
        args.working_data_root,
        args.data_name,
        paper_v1=args.paper_v1,
    )
    started = time.perf_counter()
    if args.paper_v1:
        module = "gfmrag.workflow.stage1_index_dataset"
    else:
        module = "gfmrag.workflow.index_dataset"
    _run_module(
        args.repo,
        [
            "-m",
            module,
            f"dataset.root={data_root.resolve().as_posix()}",
            f"dataset.data_name={args.data_name}",
            f"hydra.run.dir={args.hydra_output.resolve().as_posix()}",
        ],
    )
    _write_json(
        args.output,
        {
            "status": "completed",
            "method": "gfmrag8m" if args.paper_v1 else "greasoner34m",
            "data_name": args.data_name,
            "data_root": str(data_root.resolve()),
            "index_directory": str(
                (data_root / args.data_name / "processed" / "stage1").resolve()
            ),
            "wall_time_seconds": round(time.perf_counter() - started, 3),
        },
    )


def _load_gfm_retriever(args: argparse.Namespace):
    sys.path.insert(0, str(args.repo.resolve()))
    from gfmrag import GFMRetriever
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    model_path = _locked_model_path(args.model, args.model_revision, args.model_cache)
    if args.paper_v1:
        config = OmegaConf.load(
            args.repo
            / "gfmrag"
            / "workflow"
            / "config"
            / "stage3_qa_ircot_inference.yaml"
        )
        config.dataset.root = str(args.working_data_root.resolve())
        config.dataset.data_name = args.data_name
        config.graph_retriever.model_path = model_path
        return GFMRetriever.from_config(config)

    config = OmegaConf.load(
        args.repo
        / "gfmrag"
        / "workflow"
        / "config"
        / "gfm_rag"
        / "qa_ircot_inference.yaml"
    )
    return GFMRetriever.from_index(
        data_dir=str(args.working_data_root.resolve()),
        data_name=args.data_name,
        model_path=model_path,
        ner_model=instantiate(config.ner_model),
        el_model=instantiate(config.el_model),
        graph_constructor=instantiate(config.graph_constructor),
    )


def _gfm_documents(result: Any) -> List[Mapping[str, Any]]:
    if isinstance(result, Mapping):
        result = (
            result.get("document")
            or result.get("documents")
            or result.get("passage")
            or result.get("passages")
            or []
        )
    return list(result)


def _locked_model_path(repo_id: str, revision: str, cache_dir: Path) -> str:
    if not revision:
        raise ValueError(f"Model {repo_id!r} must have an immutable revision.")
    from huggingface_hub import snapshot_download

    return str(
        Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(cache_dir.resolve()),
            )
        ).resolve()
    )


def gfmrag_retrieve(args: argparse.Namespace) -> None:
    _prepare_gfm_data(
        args.source_data_root,
        args.working_data_root,
        args.data_name,
        paper_v1=args.paper_v1,
    )
    queries = _load_json(
        args.source_data_root / args.data_name / "raw" / f"{args.split}.json"
    )
    if args.limit:
        queries = queries[: args.limit]
    completed = {
        str(row["id"]): row for row in _load_jsonl(args.partial_output)
    }
    expected_ids = [str(row["id"]) for row in queries]

    startup = time.perf_counter()
    retriever = _load_gfm_retriever(args)
    startup_seconds = time.perf_counter() - startup
    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            query_id = str(query["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            documents = _gfm_documents(
                retriever.retrieve(str(query["question"]), top_k=args.top_k)
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            document_ids = []
            scores = []
            for rank, document in enumerate(documents):
                document_id = (
                    document.get("id")
                    or document.get("title")
                    or document.get("name")
                )
                if document_id is None:
                    raise KeyError("GFM-RAG result has no document identifier.")
                document_ids.append(str(document_id))
                scores.append(
                    float(
                        document.get(
                            "score",
                            document.get("norm_score", len(documents) - rank),
                        )
                    )
                )
            row = {
                "id": query_id,
                "retrieved_document_ids": document_ids,
                "scores": scores,
                "latency_ms": {"retrieval": round(latency_ms, 3)},
            }
            _append_jsonl(handle, row)
            completed[query_id] = row

    if set(completed) != set(expected_ids):
        raise RuntimeError("GFM-RAG retrieval did not produce every expected query.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in expected_ids:
            _append_jsonl(handle, completed[query_id])
    os.replace(temporary, args.output)
    args.partial_output.unlink(missing_ok=True)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": "gfmrag8m" if args.paper_v1 else "greasoner34m",
            "data_name": args.data_name,
            "queries": len(queries),
            "top_k": args.top_k,
            "warm_start_seconds": round(startup_seconds, 3),
            "model": args.model,
            "model_revision": args.model_revision,
            "output": str(args.output.resolve()),
        },
    )


def _hoprag_document_text(document: Mapping[str, Any]) -> str:
    text = _document_text(document).replace("\r\n", "\n").replace("\r", "\n")
    # HopRAG treats a blank line as a node boundary. Matched runs use one node
    # per canonical document so every returned node has unambiguous provenance.
    return re.sub(r"\n[ \t]*\n+", "\n", text).strip()


def _hoprag_namespace(dataset_alias: str, corpus_sha256: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]", "_", dataset_alias).strip("_").lower()
    if not alias or not alias[0].isalpha():
        alias = f"d_{alias}"
    return f"hprg_{alias[:30]}_{corpus_sha256[:12]}"


def _prepare_hoprag_inputs(
    corpus_path: Path,
    edges_path: Path,
    working_dir: Path,
    *,
    dataset_alias: str,
    group_size: int,
) -> Dict[str, Any]:
    if group_size < 2:
        raise ValueError("HopRAG graph-neighborhood groups require group_size >= 2.")
    documents = _canonical_documents(corpus_path)
    document_by_id = {str(row["id"]): row for row in documents}
    edges = _load_jsonl(edges_path)
    corpus_sha256 = _sha256_file(corpus_path)
    edges_sha256 = _sha256_file(edges_path)
    identity = {
        "adapter_version": 1,
        "protocol": "label_free_graph_neighborhoods",
        "corpus_sha256": corpus_sha256,
        "edges_sha256": edges_sha256,
        "dataset_alias": dataset_alias,
        "group_size": group_size,
    }
    input_root = working_dir / "input"
    marker = input_root / "preparation.json"
    if marker.exists():
        prepared = _load_json(marker)
        if prepared.get("identity") != identity:
            raise RuntimeError(
                "The HopRAG input directory belongs to a different corpus or "
                "grouping policy. Use a new working directory."
            )
        required = (
            Path(prepared["documents_dir"]),
            Path(prepared["document_map"]),
            Path(prepared["problems"]),
        )
        if prepared.get("status") == "completed" and all(path.exists() for path in required):
            return prepared
        raise RuntimeError("The existing HopRAG input preparation is incomplete.")
    if input_root.exists() and any(input_root.iterdir()):
        raise RuntimeError(
            "HopRAG input files exist without a matching completion marker. "
            "Use a new working directory rather than accepting partial input."
        )

    documents_dir = input_root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    filename_by_id: Dict[str, str] = {}
    map_rows = []
    text_seen: Dict[str, str] = {}
    for index, document in enumerate(documents):
        document_id = str(document["id"])
        filename = f"{index:08d}.txt"
        text = _hoprag_document_text(document)
        if not text:
            raise ValueError(f"Canonical document {document_id!r} has no text.")
        duplicate = text_seen.get(text)
        if duplicate is not None:
            raise ValueError(
                "HopRAG exact provenance requires unique normalized document text; "
                f"{duplicate!r} and {document_id!r} collide."
            )
        text_seen[text] = document_id
        filename_by_id[document_id] = filename
        (documents_dir / filename).write_text(text, encoding="utf-8")
        map_rows.append(
            {
                "document_id": document_id,
                "filename": filename,
                "title": str(document.get("title", "")),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    adjacency: Dict[str, set[str]] = {document_id: set() for document_id in document_by_id}
    ignored_edges = 0
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in document_by_id or target not in document_by_id or source == target:
            ignored_edges += 1
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    groups: set[tuple[str, ...]] = set()
    neighbor_budget = group_size - 1
    for source in sorted(adjacency):
        neighbors = sorted(adjacency[source])
        for offset in range(0, len(neighbors), neighbor_budget):
            group = tuple(sorted([source, *neighbors[offset : offset + neighbor_budget]]))
            if len(group) > 1:
                groups.add(group)

    problems = []
    for index, group in enumerate(sorted(groups)):
        problems.append(
            {
                "_id": f"graph_{index:08d}",
                "question": "",
                "answer": "",
                "supporting_facts": [],
                "context": [
                    [Path(filename_by_id[document_id]).stem, []]
                    for document_id in group
                ],
            }
        )

    document_map = input_root / "document_map.jsonl"
    problems_path = input_root / "graph_neighborhoods.jsonl"
    _write_jsonl(document_map, map_rows)
    _write_jsonl(problems_path, problems)
    prepared = {
        "status": "completed",
        "identity": identity,
        "namespace": _hoprag_namespace(dataset_alias, corpus_sha256),
        "document_count": len(documents),
        "edge_count": len(edges),
        "ignored_edge_count": ignored_edges,
        "neighborhood_count": len(problems),
        "documents_dir": str(documents_dir.resolve()),
        "document_map": str(document_map.resolve()),
        "problems": str(problems_path.resolve()),
        "created_at": time.time(),
    }
    _write_json(marker, prepared)
    return prepared


def hoprag_prepare(args: argparse.Namespace) -> None:
    prepared = _prepare_hoprag_inputs(
        args.corpus,
        args.edges,
        args.working_dir,
        dataset_alias=args.dataset_alias,
        group_size=args.group_size,
    )
    _write_json(args.output, prepared)


def _require_hoprag_runtime(args: argparse.Namespace) -> Dict[str, str]:
    values = {
        "neo4j_uri": getattr(args, "neo4j_uri", None)
        or os.environ.get("NEO4J_URI", ""),
        "neo4j_user": getattr(args, "neo4j_user", None)
        or os.environ.get("NEO4J_USER", ""),
        "neo4j_password": getattr(args, "neo4j_password", None)
        or os.environ.get("NEO4J_PASSWORD", ""),
        "neo4j_database": getattr(args, "neo4j_database", None)
        or os.environ.get("NEO4J_DATABASE", "neo4j"),
        "llm_base_url": getattr(args, "llm_base_url", None)
        or os.environ.get("HOPRAG_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", ""),
        "llm_api_key": getattr(args, "llm_api_key", None)
        or os.environ.get("HOPRAG_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY", ""),
        "llm_model": getattr(args, "llm_model", None)
        or os.environ.get("HOPRAG_LLM_MODEL", ""),
    }
    missing = [
        name
        for name in (
            "neo4j_uri",
            "neo4j_user",
            "neo4j_password",
            "llm_base_url",
            "llm_model",
        )
        if not values[name]
    ]
    if missing:
        raise RuntimeError(
            "HopRAG runtime is not configured; missing " + ", ".join(missing) + "."
        )
    if not values["llm_api_key"]:
        if values["llm_base_url"].startswith(("http://localhost", "http://127.0.0.1")):
            values["llm_api_key"] = "EMPTY"
        else:
            raise RuntimeError("HopRAG's non-local LLM endpoint requires an API key.")
    paid_openai = "api.openai.com" in values["llm_base_url"].lower()
    if paid_openai and not getattr(args, "allow_paid_api", False):
        raise RuntimeError(
            "HopRAG would call the paid OpenAI API. Re-run with --allow-paid-api "
            "only after explicitly approving that cost, or configure a local "
            "OpenAI-compatible endpoint."
        )
    return values


def _load_hoprag_config(
    repo: Path,
    args: argparse.Namespace,
    runtime: Mapping[str, str],
    embedding_path: str,
    namespace: str,
) -> Any:
    config_path = repo.resolve() / "config.py"
    specification = importlib.util.spec_from_file_location("config", config_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load HopRAG config from {config_path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules["config"] = module
    specification.loader.exec_module(module)

    old_node_name = str(getattr(module, "node_name", ""))
    old_edge_name = str(getattr(module, "edge_name", ""))
    edge_name = f"edge_{namespace}"
    module.dataset_name = args.dataset_alias
    module.node_name = namespace
    module.edge_name = edge_name
    module.generator_label = f"{namespace}_"
    module.node_dense_index_name = f"{namespace}_node_dense"
    module.edge_dense_index_name = f"{namespace}_edge_dense"
    module.node_sparse_index_name = f"{namespace}_node_sparse"
    module.edge_sparse_index_name = f"{namespace}_edge_sparse"
    module.embed_model = "crag_locked_embedding"
    module.embed_model_dict = {module.embed_model: embedding_path}
    module.embed_dim = args.embedding_dim
    module.llm_device = args.device
    module.local_model_name = runtime["llm_model"]
    module.query_generator_model = runtime["llm_model"]
    module.traversal_model = runtime["llm_model"]
    module.default_gpt_model = runtime["llm_model"]
    module.deployment_sign = {
        runtime["llm_model"]: {
            "base": runtime["llm_base_url"],
            "key": runtime["llm_api_key"],
        }
    }
    module.neo4j_url = runtime["neo4j_uri"]
    module.neo4j_user = runtime["neo4j_user"]
    module.neo4j_password = runtime["neo4j_password"]
    module.neo4j_dbname = runtime["neo4j_database"]
    module.signal = "\n\n"
    module.max_thread_num = args.max_threads

    # The checkout builds several Cypher strings eagerly while importing config.py.
    # Rewrite only the old generated labels so the pinned source remains untouched.
    for name, value in list(vars(module).items()):
        if not isinstance(value, str):
            continue
        rewritten = value
        if old_edge_name:
            rewritten = rewritten.replace(old_edge_name, edge_name)
        if old_node_name:
            rewritten = rewritten.replace(old_node_name, namespace)
        if rewritten != value:
            setattr(module, name, rewritten)
    return module


def _import_hoprag_module(repo: Path, name: str) -> Any:
    repo_string = str(repo.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    for stale in ("tool", "HopBuilder", "HopRetriever", "HopGenerator"):
        sys.modules.pop(stale, None)
    return __import__(name)


def _patch_hoprag_neo4j(module: Any, default_database: str) -> None:
    original = module.GraphDatabase

    class DriverProxy:
        def __init__(self, driver: Any, database: str):
            self._driver = driver
            self._database = database

        def session(self, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("database", self._database)
            return self._driver.session(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._driver, name)

    class GraphDatabaseProxy:
        @staticmethod
        def driver(*args: Any, **kwargs: Any) -> DriverProxy:
            database = str(kwargs.pop("database", default_database))
            return DriverProxy(original.driver(*args, **kwargs), database)

    module.GraphDatabase = GraphDatabaseProxy


def _hoprag_graph_counts(runtime: Mapping[str, str], namespace: str) -> Dict[str, int]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        runtime["neo4j_uri"],
        auth=(runtime["neo4j_user"], runtime["neo4j_password"]),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=runtime["neo4j_database"]) as session:
            node_count = int(
                session.run(
                    f"MATCH (n:`{namespace}`) RETURN count(n) AS count"
                ).single()["count"]
            )
            edge_count = int(
                session.run(
                    f"MATCH ()-[r:`edge_{namespace}`]->() RETURN count(r) AS count"
                ).single()["count"]
            )
        return {"nodes": node_count, "edges": edge_count}
    finally:
        driver.close()


def _hoprag_await_indexes(
    runtime: Mapping[str, str],
    index_names: Sequence[str],
) -> List[Dict[str, str]]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        runtime["neo4j_uri"],
        auth=(runtime["neo4j_user"], runtime["neo4j_password"]),
    )
    try:
        with driver.session(database=runtime["neo4j_database"]) as session:
            session.run("CALL db.awaitIndexes(3600)").consume()
            records = session.run(
                "SHOW INDEXES YIELD name, state "
                "WHERE name IN $names RETURN name, state",
                names=list(index_names),
            )
            statuses = [
                {"name": str(record["name"]), "state": str(record["state"])}
                for record in records
            ]
    finally:
        driver.close()
    status_by_name = {row["name"]: row["state"] for row in statuses}
    missing = sorted(set(index_names) - set(status_by_name))
    offline = sorted(
        name for name, state in status_by_name.items() if state.upper() != "ONLINE"
    )
    if missing or offline:
        raise RuntimeError(
            "HopRAG Neo4j indexes are not ready; "
            f"missing={missing}, non_online={offline}."
        )
    return statuses


def _hoprag_quiet_main_nodes(builder: Any, **kwargs: Any) -> None:
    previous = getattr(builder, "print", None)

    def quiet_print(*values: Any, **print_kwargs: Any) -> None:
        if values and isinstance(values[0], dict):
            builtins.print(
                f"HopRAG cache contains {len(values[0])} documents.",
                **print_kwargs,
            )
        else:
            builtins.print(*values, **print_kwargs)

    builder.print = quiet_print
    try:
        builder.main_nodes(**kwargs)
    finally:
        if previous is None:
            delattr(builder, "print")
        else:
            builder.print = previous


def _hoprag_index_identity(
    args: argparse.Namespace,
    prepared: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "adapter_version": 1,
        "preparation": prepared["identity"],
        "namespace": prepared["namespace"],
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "embedding_dim": args.embedding_dim,
        "llm_model": runtime["llm_model"],
        "llm_base_url": runtime["llm_base_url"],
        "node_batch_size": args.batch_size,
        "edge_batch_size": args.edge_batch_size,
    }


def _hoprag_load_index_state(
    state_path: Path,
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "status": "pending",
            "identity": identity,
            "active_phase": None,
            "next_node_batch": 0,
            "next_edge_batch": 0,
        }
    state = _load_json(state_path)
    if state.get("identity") != identity:
        raise RuntimeError(
            "The HopRAG Neo4j namespace belongs to a different corpus or model "
            "configuration. Use a new working directory."
        )
    active_phase = str(state.get("active_phase") or "")
    if active_phase.startswith("offline_nodes:"):
        # The offline author call writes its cache only after a complete batch and
        # cannot have mutated Neo4j, so replaying this batch is safe.
        state["active_phase"] = None
    elif active_phase:
        raise RuntimeError(
            "A previous HopRAG run stopped while mutating Neo4j "
            f"({active_phase}). The namespace is fail-closed because "
            "the author builder is non-transactional across a batch. Clean that "
            "dedicated namespace explicitly, then use a new working directory."
        )
    return state


def _hoprag_batch_directory(
    documents: Sequence[Path],
    root: Path,
    batch_index: int,
) -> Path:
    target = root / f"{batch_index:06d}"
    target.mkdir(parents=True, exist_ok=True)
    for source in documents:
        destination = target / source.name
        if destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return target


def hoprag_index(args: argparse.Namespace) -> None:
    prepared = _prepare_hoprag_inputs(
        args.corpus,
        args.edges,
        args.working_dir,
        dataset_alias=args.dataset_alias,
        group_size=args.group_size,
    )
    runtime = _require_hoprag_runtime(args)
    identity = _hoprag_index_identity(args, prepared, runtime)
    state_path = args.working_dir / "index_progress.json"
    state = _hoprag_load_index_state(state_path, identity)
    if state.get("status") == "completed":
        _write_json(args.output, state)
        return

    namespace = str(prepared["namespace"])
    graph_counts = _hoprag_graph_counts(runtime, namespace)
    if state.get("status") == "pending" and (graph_counts["nodes"] or graph_counts["edges"]):
        raise RuntimeError(
            f"Neo4j namespace {namespace!r} is non-empty without a completed "
            "adapter state. Refusing to mix or duplicate a partial graph."
        )
    embedding_path = _locked_model_path(
        args.embedding_model,
        args.embedding_revision,
        args.model_cache,
    )
    _load_hoprag_config(args.repo, args, runtime, embedding_path, namespace)
    builder = _import_hoprag_module(args.repo, "HopBuilder")
    _patch_hoprag_neo4j(builder, runtime["neo4j_database"])

    documents_dir = Path(str(prepared["documents_dir"]))
    document_paths = sorted(documents_dir.glob("*.txt"))
    node_batches = [
        document_paths[offset : offset + args.batch_size]
        for offset in range(0, len(document_paths), args.batch_size)
    ]
    batch_root = args.working_dir / "input" / "node_batches"
    online_cache = args.working_dir / "cache" / "online"
    for batch_index in range(int(state["next_node_batch"]), len(node_batches)):
        batch_paths = node_batches[batch_index]
        batch_dir = _hoprag_batch_directory(batch_paths, batch_root, batch_index)
        offline_cache = args.working_dir / "cache" / "offline" / f"{batch_index:06d}"
        state["active_phase"] = f"offline_nodes:{batch_index}"
        _write_json(state_path, state)
        _hoprag_quiet_main_nodes(
            builder,
            cache_dir=str(offline_cache),
            docs_dir=str(batch_dir),
            label=namespace,
            start_index=0,
            span=len(batch_paths),
            original_cache_dir=None,
            offline=True,
        )
        offline_documents = _load_json(offline_cache / "docid2nodes.json")
        expected = {path.name for path in batch_paths}
        if not expected.issubset(offline_documents):
            missing = sorted(expected - set(offline_documents))
            raise RuntimeError(
                f"HopRAG failed to prepare {len(missing)} documents in node "
                f"batch {batch_index}; first missing file: {missing[0]}"
            )
        state["active_phase"] = f"online_nodes:{batch_index}"
        _write_json(state_path, state)
        _hoprag_quiet_main_nodes(
            builder,
            cache_dir=str(online_cache),
            docs_dir=str(batch_dir),
            label=namespace,
            start_index=0,
            span=len(batch_paths),
            original_cache_dir=str(offline_cache),
            offline=False,
        )
        online_documents = _load_json(online_cache / "docid2nodes.json")
        if not expected.issubset(online_documents):
            missing = sorted(expected - set(online_documents))
            raise RuntimeError(
                f"HopRAG failed to upload {len(missing)} documents in node "
                f"batch {batch_index}; first missing file: {missing[0]}"
            )
        state["active_phase"] = None
        state["status"] = "indexing"
        state["next_node_batch"] = batch_index + 1
        _write_json(state_path, state)

    problems = _load_jsonl(Path(str(prepared["problems"])))
    edge_batches = [
        problems[offset : offset + args.edge_batch_size]
        for offset in range(0, len(problems), args.edge_batch_size)
    ]
    edge_batch_root = args.working_dir / "input" / "edge_batches"
    for batch_index in range(int(state["next_edge_batch"]), len(edge_batches)):
        problem_path = edge_batch_root / f"{batch_index:06d}.jsonl"
        if not problem_path.exists():
            _write_jsonl(problem_path, edge_batches[batch_index])
        state["active_phase"] = f"edges:{batch_index}"
        _write_json(state_path, state)
        builder.main_edges_index(
            cache_dir=str(online_cache),
            problems_path=str(problem_path),
            label=namespace,
        )
        import pickle

        with (online_cache / "edges_done.pkl").open("rb") as handle:
            completed_edges = pickle.load(handle)
        expected_ids = {str(row["_id"]) for row in edge_batches[batch_index]}
        if not expected_ids.issubset(completed_edges):
            missing = sorted(expected_ids - set(completed_edges))
            raise RuntimeError(
                f"HopRAG silently skipped {len(missing)} graph neighborhoods in "
                f"edge batch {batch_index}; first missing id: {missing[0]}"
            )
        state["active_phase"] = None
        state["next_edge_batch"] = batch_index + 1
        _write_json(state_path, state)

    index_names = (
        f"{namespace}_node_dense",
        f"{namespace}_edge_dense",
        f"{namespace}_node_sparse",
        f"{namespace}_edge_sparse",
    )
    index_status = _hoprag_await_indexes(runtime, index_names)
    graph_counts = _hoprag_graph_counts(runtime, namespace)
    if graph_counts["nodes"] != len(document_paths):
        raise RuntimeError(
            "HopRAG node count does not match the full canonical corpus: "
            f"{graph_counts['nodes']} != {len(document_paths)}."
        )
    state.update(
        {
            "status": "completed",
            "active_phase": None,
            "namespace": namespace,
            "document_count": len(document_paths),
            "neighborhood_count": len(problems),
            "neo4j_counts": graph_counts,
            "neo4j_indexes": index_status,
            "embedding_path": embedding_path,
            "completed_at": time.time(),
        }
    )
    _write_json(state_path, state)
    _write_json(args.output, state)


def hoprag_retrieve(args: argparse.Namespace) -> None:
    prepared = _prepare_hoprag_inputs(
        args.corpus,
        args.edges,
        args.working_dir,
        dataset_alias=args.dataset_alias,
        group_size=args.group_size,
    )
    runtime = _require_hoprag_runtime(args)
    identity = _hoprag_index_identity(args, prepared, runtime)
    state = _hoprag_load_index_state(
        args.working_dir / "index_progress.json",
        identity,
    )
    if state.get("status") != "completed":
        raise RuntimeError("HopRAG retrieval requires a completed matched index.")

    embedding_path = _locked_model_path(
        args.embedding_model,
        args.embedding_revision,
        args.model_cache,
    )
    namespace = str(prepared["namespace"])
    _load_hoprag_config(args.repo, args, runtime, embedding_path, namespace)
    module = _import_hoprag_module(args.repo, "HopRetriever")
    _patch_hoprag_neo4j(module, runtime["neo4j_database"])
    retriever = module.HopRetriever(
        llm=runtime["llm_model"],
        max_hop=args.max_hop,
        entry_type="node",
        if_hybrid=args.hybrid,
        topk=args.top_k,
        traversal=args.traversal,
        embedding_model="crag_locked_embedding",
    )

    documents = _canonical_documents(args.corpus)
    id_by_text = {_hoprag_document_text(row): str(row["id"]) for row in documents}
    queries = _load_jsonl(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    expected_ids = [str(row["id"]) for row in queries]
    completed = {str(row["id"]): row for row in _load_jsonl(args.partial_output)}
    progress_path = args.partial_output.with_suffix(".progress.json")
    retrieval_identity = {
        "adapter_version": 1,
        "index_identity": identity,
        "queries_sha256": _sha256_file(args.queries),
        "top_k": args.top_k,
        "max_hop": args.max_hop,
        "traversal": args.traversal,
        "hybrid": args.hybrid,
    }
    if progress_path.exists():
        if _load_json(progress_path).get("identity") != retrieval_identity:
            raise RuntimeError(
                "The HopRAG partial retrieval belongs to a different query or "
                "retrieval configuration."
            )
    elif completed:
        raise RuntimeError("HopRAG partial output exists without a provenance marker.")
    else:
        _write_json(progress_path, {"status": "running", "identity": retrieval_identity})

    startup_seconds = 0.0
    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
            for query in queries:
                query_id = str(query["id"])
                if query_id in completed:
                    continue
                started = time.perf_counter()
                contexts, scores = retriever.search_docs(str(query["question"]))
                latency_ms = (time.perf_counter() - started) * 1000.0
                document_ids = []
                output_scores = []
                seen = set()
                for context, score in zip(contexts or [], scores or []):
                    normalized = str(context).replace("\r\n", "\n").replace("\r", "\n")
                    document_id = id_by_text.get(normalized.strip())
                    if document_id is None:
                        raise RuntimeError(
                            "HopRAG returned a node that cannot be mapped exactly "
                            "to a canonical document; refusing lossy provenance."
                        )
                    if document_id in seen:
                        continue
                    seen.add(document_id)
                    document_ids.append(document_id)
                    try:
                        output_scores.append(float(score))
                    except (TypeError, ValueError):
                        output_scores.append(float(len(contexts) - len(output_scores)))
                row = {
                    "id": query_id,
                    "retrieved_document_ids": document_ids,
                    "scores": output_scores,
                    "latency_ms": {"retrieval": round(latency_ms, 3)},
                }
                _append_jsonl(handle, row)
                completed[query_id] = row
    finally:
        if getattr(retriever, "driver", None) is not None:
            retriever.driver.close()
            retriever.driver = None

    if set(completed) != set(expected_ids):
        raise RuntimeError("HopRAG retrieval did not produce every expected query.")
    _write_jsonl(args.output, [completed[query_id] for query_id in expected_ids])
    args.partial_output.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": "hoprag",
            "queries": len(queries),
            "top_k": args.top_k,
            "max_hop": args.max_hop,
            "traversal": args.traversal,
            "namespace": namespace,
            "warm_start_seconds": startup_seconds,
            "output": str(args.output.resolve()),
        },
    )


KG2RAG_ADAPTER_VERSION = 2
KG2RAG_PROMPT_VERSION = 1
KG2RAG_SENTENCE_SPLIT_VERSION = 1
KG2RAG_PAPER_LLM = "llama3:8b"
KG2RAG_PAPER_EMBEDDING = "mxbai-embed-large:latest"
KG2RAG_PAPER_LLM_DIGEST = (
    "365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1"
)
KG2RAG_PAPER_EMBEDDING_DIGEST = (
    "468836162de7f81e041c43663fedbbba921dcea9b9fefea135685a39b2d83dd8"
)
KG2RAG_RELATIONAL_FACTS_MARKER = " Relational facts: "


def _kg2rag_sentences(text: str) -> tuple[List[str], str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return [], "empty"
    if re.search(r"\s{2,}", text):
        sentences = re.split(r"\s{2,}", text)
        method = "preserved_hotpot_whitespace"
    else:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        method = "deterministic_punctuation_fallback"
    normalized = [re.sub(r"\s+", " ", value).strip() for value in sentences]
    return [value for value in normalized if value], method


def _prepare_kg2rag_inputs(
    corpus_path: Path,
    working_dir: Path,
) -> Dict[str, Any]:
    documents = _canonical_documents(corpus_path)
    input_dir = working_dir / "input"
    chunks_path = input_dir / "chunks.jsonl"
    marker_path = input_dir / "preparation.json"
    identity = {
        "adapter_version": KG2RAG_ADAPTER_VERSION,
        "sentence_split_version": KG2RAG_SENTENCE_SPLIT_VERSION,
        "corpus_sha256": _sha256_file(corpus_path),
        "document_count": len(documents),
        "chunking": (
            "source_double_whitespace_then_deterministic_punctuation_fallback"
        ),
        "author_first_sentence_extraction_prefix_policy": True,
    }
    if marker_path.exists() and chunks_path.exists():
        existing = _load_json(marker_path)
        if (
            existing.get("identity") == identity
            and existing.get("chunks_sha256") == _sha256_file(chunks_path)
        ):
            return existing

    chunks = []
    split_counts: Dict[str, int] = {}
    seen_titles = set()
    for document in documents:
        document_id = str(document["id"])
        title = str(document.get("title") or "").strip()
        if not title:
            raise ValueError(f"Canonical document {document_id!r} has no title.")
        if "##" in title:
            raise ValueError(
                f"KG2RAG author node identifiers cannot encode title {title!r}."
            )
        if title in seen_titles:
            raise ValueError(
                f"KG2RAG requires unique HotpotQA titles; duplicate {title!r}."
            )
        seen_titles.add(title)
        sentences, split_method = _kg2rag_sentences(str(document.get("text") or ""))
        if not sentences:
            raise ValueError(f"Canonical document {document_id!r} has no text.")
        split_counts[split_method] = split_counts.get(split_method, 0) + 1
        for sequence, sentence in enumerate(sentences):
            chunks.append(
                {
                    "id": f"kg2rag_chunk_{len(chunks):08d}",
                    "author_id": f"{title}##{sequence}",
                    "document_id": document_id,
                    "title": title,
                    "sequence": sequence,
                    "sentence": sentence,
                    "retrieval_text": f"{title}: {sentence}",
                    "extraction_text": (
                        sentence if sequence == 0 else f"{title}: {sentence}"
                    ),
                    "split_method": split_method,
                }
            )

    _write_jsonl(chunks_path, chunks)
    prepared = {
        "status": "completed",
        "identity": identity,
        "chunks": str(chunks_path.resolve()),
        "chunks_sha256": _sha256_file(chunks_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "split_document_counts": split_counts,
        "unique_titles": len(seen_titles),
        "questions_used": False,
        "labels_used": False,
        "document_boundaries_preserved": True,
    }
    _write_json(marker_path, prepared)
    return prepared


def kg2rag_prepare(args: argparse.Namespace) -> None:
    prepared = _prepare_kg2rag_inputs(args.corpus, args.working_dir)
    _write_json(args.output, prepared)
    print(json.dumps(prepared, indent=2, sort_keys=True))


def _kg2rag_ollama_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if value and "://" not in value:
        value = f"http://{value}"
    return value


def _kg2rag_ollama_json(
    base_url: str,
    endpoint: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    data = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        f"{base_url}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(
            f"KG2RAG could not reach the configured Ollama endpoint "
            f"{base_url}{endpoint}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Ollama {endpoint} returned a non-object response.")
    return value


def _kg2rag_model_record(
    tags: Mapping[str, Any],
    model: str,
    expected_digest: str,
) -> Dict[str, str]:
    expected = expected_digest.removeprefix("sha256:").lower()
    candidates = {
        model,
        model.removesuffix(":latest"),
        f"{model}:latest" if ":" not in model else model,
    }
    for record in tags.get("models") or []:
        names = {
            str(record.get("name") or ""),
            str(record.get("model") or ""),
        }
        if not names.intersection(candidates):
            continue
        digest = str(record.get("digest") or "").removeprefix("sha256:").lower()
        if not digest:
            raise RuntimeError(f"Ollama model {model!r} has no manifest digest.")
        if digest != expected and not digest.startswith(expected):
            raise RuntimeError(
                f"Ollama model {model!r} digest {digest} does not match the "
                f"locked paper digest {expected}."
            )
        return {
            "name": next(value for value in names if value),
            "digest": digest,
        }
    raise RuntimeError(
        f"Locked KG2RAG model {model!r} is not installed in Ollama. "
        "The benchmark never pulls or substitutes models automatically."
    )


def _require_kg2rag_runtime(
    args: argparse.Namespace,
    *,
    require_llm: bool,
) -> Dict[str, Any]:
    from urllib.parse import urlparse

    base_url = _kg2rag_ollama_base_url(
        args.ollama_base_url
        or os.environ.get("KG2RAG_OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or ""
    )
    if not base_url:
        raise RuntimeError(
            "KG2RAG requires KG2RAG_OLLAMA_BASE_URL (for example, "
            "http://localhost:11434); no model service is started automatically."
        )
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    local_hosts = {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "host.docker.internal",
    }
    is_local = host in local_hosts or host.endswith(".local")
    if not is_local and not args.allow_remote_ollama:
        raise RuntimeError(
            "KG2RAG is fail-closed for non-local Ollama endpoints. Pass "
            "--allow-remote-ollama only after reviewing the service and cost."
        )
    if not is_local:
        budget = int(getattr(args, "max_extraction_calls", 0) or 0)
        if require_llm and budget < 1:
            raise RuntimeError(
                "Remote KG2RAG extraction requires a positive, reviewed "
                "--max-extraction-calls budget."
            )

    if (
        args.llm_model != KG2RAG_PAPER_LLM
        or args.embedding_model != KG2RAG_PAPER_EMBEDDING
    ) and not args.allow_model_substitution:
        raise RuntimeError(
            "KG2RAG's matched paper protocol requires llama3:8b and "
            "mxbai-embed-large:latest. Use --allow-model-substitution only for "
            "a separately labeled diagnostic."
        )

    tags = _kg2rag_ollama_json(base_url, "/api/tags", timeout=30.0)
    version_response = _kg2rag_ollama_json(
        base_url,
        "/api/version",
        timeout=30.0,
    )
    ollama_version = str(version_response.get("version") or "").strip()
    if not ollama_version:
        raise RuntimeError("The configured Ollama service returned no version.")
    embedding = _kg2rag_model_record(
        tags,
        args.embedding_model,
        args.embedding_digest,
    )
    llm = (
        _kg2rag_model_record(tags, args.llm_model, args.llm_digest)
        if require_llm
        else None
    )
    return {
        "base_url": base_url,
        "service_scope": "local" if is_local else "explicit_remote",
        "ollama_version": ollama_version,
        "embedding": embedding,
        "llm": llm,
    }


def _kg2rag_prompt(context: str) -> str:
    return (
        "Extract triplets informative from the text following the examples. "
        "Make sure the triplet texts are only directly from the given text! "
        "Complete directly and strictly following the instructions without any "
        "additional words, line break nor space!\n"
        "--------------------\n"
        "Text: Scott Derrickson (born July 16, 1966) is an American director, "
        "screenwriter and producer.\n"
        "Triplets:<Scott Derrickson##born in##1966>$$"
        "<Scott Derrickson##nationality##America>$$"
        "<Scott Derrickson##occupation##director>$$"
        "<Scott Derrickson##occupation##screenwriter>$$"
        "<Scott Derrickson##occupation##producer>$$\n"
        "--------------------\n"
        "Text: A Kiss for Corliss is a 1949 American comedy film directed by "
        "Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple "
        "in her final starring role as well as her final film appearance. Shirley "
        "Temple was named United States ambassador to Ghana and to Czechoslovakia "
        "and also served as Chief of Protocol of the United States.\n"
        "Triplets:<A Kiss for Corliss##cast member##Shirley Temple>$$"
        "<Shirley Temple##served as##Chief of Protocol>$$\n"
        "--------------------\n"
        f"Text: {context}\nTriplets:"
    )


def _parse_kg2rag_triplets(response: str, context: str) -> List[List[str]]:
    triplets = set()
    for raw in response.split("$$"):
        raw = raw.strip()
        if len(raw) <= 6:
            continue
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        tokens = raw.split("##")
        if len(tokens) != 3:
            continue
        head, relation, tail = (value.strip() for value in tokens)
        blocked = ("no ", "unknown", "null")
        lower_head = head.lower()
        lower_tail = tail.lower()
        if (
            not head
            or not relation
            or not tail
            or head == tail
            or any(value in lower_head or value in lower_tail for value in blocked)
            or "NO" in head
            or "NO" in relation
            or "NO" in tail
        ):
            continue
        if relation not in context and tail not in context:
            continue
        triplets.add((head, relation, tail))
    return [list(value) for value in sorted(triplets)]


def _kg2rag_extract_triplets(
    runtime: Mapping[str, Any],
    context: str,
) -> tuple[List[List[str]], Dict[str, Any]]:
    response = _kg2rag_ollama_json(
        str(runtime["base_url"]),
        "/api/generate",
        {
            "model": str(runtime["llm"]["name"]),
            "prompt": _kg2rag_prompt(context),
            "stream": False,
        },
        timeout=300.0,
    )
    text = response.get("response")
    if not isinstance(text, str):
        raise RuntimeError("Ollama KG2RAG extraction returned no response text.")
    return _parse_kg2rag_triplets(text, context), {
        key: response.get(key)
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "eval_count",
        )
        if response.get(key) is not None
    }


def _kg2rag_embeddings(
    runtime: Mapping[str, Any],
    texts: Sequence[str],
) -> List[List[float]]:
    response = _kg2rag_ollama_json(
        str(runtime["base_url"]),
        "/api/embed",
        {
            "model": str(runtime["embedding"]["name"]),
            "input": list(texts),
        },
        timeout=600.0,
    )
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama KG2RAG embedding response does not match the input batch."
        )
    return embeddings


def _kg2rag_source_hashes(repo: Path) -> Dict[str, str]:
    files = {
        "full_runner": repo / "code" / "kg_rag_full.py",
        "postprocessors": repo / "code" / "util" / "kg_post_processor.py",
        "extraction": repo / "code" / "preprocess" / "hotpot_extraction.py",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Pinned KG2RAG checkout is missing: {', '.join(missing)}."
        )
    return {name: _sha256_file(path) for name, path in files.items()}


def _kg2rag_index_artifacts_valid(state: Mapping[str, Any]) -> bool:
    artifacts = state.get("artifacts") or {}
    if not artifacts:
        return False
    return all(
        Path(str(record.get("path") or "")).exists()
        and _sha256_file(Path(str(record["path"]))) == record.get("sha256")
        for record in artifacts.values()
    )


def kg2rag_index(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np

    prepared = _prepare_kg2rag_inputs(args.corpus, args.working_dir)
    runtime = _require_kg2rag_runtime(args, require_llm=True)
    source_hashes = _kg2rag_source_hashes(args.repo)
    identity = {
        "adapter_version": KG2RAG_ADAPTER_VERSION,
        "preparation_identity": prepared["identity"],
        "prompt_version": KG2RAG_PROMPT_VERSION,
        "source_hashes": source_hashes,
        "llm_model": args.llm_model,
        "llm_digest": runtime["llm"]["digest"],
        "embedding_model": args.embedding_model,
        "embedding_digest": runtime["embedding"]["digest"],
        "ollama_version": runtime["ollama_version"],
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_dir = args.working_dir / "artifacts" / identity_sha256
    state_path = args.working_dir / "index_state.json"
    if state_path.exists():
        state = _load_json(state_path)
        if (
            state.get("status") == "completed"
            and state.get("identity") == identity
            and _kg2rag_index_artifacts_valid(state)
        ):
            _write_json(args.output, state)
            print(json.dumps(state, indent=2, sort_keys=True))
            return

    chunks = _load_jsonl(Path(prepared["chunks"]))
    documents: Dict[str, List[Mapping[str, Any]]] = {}
    for chunk in chunks:
        documents.setdefault(str(chunk["document_id"]), []).append(chunk)
    extraction_cache = args.working_dir / "cache" / "triplets"
    embedding_cache = args.working_dir / "cache" / "embeddings"
    state: Dict[str, Any] = {
        "status": "running",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "document_count": prepared["document_count"],
        "chunk_count": prepared["chunk_count"],
        "new_extraction_calls": 0,
        "cached_extraction_chunks": 0,
        "new_embedding_batches": 0,
        "cached_embedding_batches": 0,
    }
    _write_json(state_path, state)
    started = time.perf_counter()
    try:
        kg_rows = []
        total_triplets = 0
        entities = set()
        relations = set()
        for document_id, document_chunks in documents.items():
            document_key = {
                "adapter_version": KG2RAG_ADAPTER_VERSION,
                "prompt_version": KG2RAG_PROMPT_VERSION,
                "document_id": document_id,
                "llm_model": args.llm_model,
                "llm_digest": runtime["llm"]["digest"],
                "chunk_keys": [
                    {
                        "sequence": int(chunk["sequence"]),
                        "text_sha256": hashlib.sha256(
                            str(chunk["extraction_text"]).encode("utf-8")
                        ).hexdigest(),
                    }
                    for chunk in document_chunks
                ],
            }
            document_digest = hashlib.sha256(
                json.dumps(
                    document_key,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cache_path = (
                extraction_cache
                / document_digest[:2]
                / f"{document_digest}.json"
            )
            cached = (
                _load_json(cache_path)
                if cache_path.exists()
                else {"key": document_key, "chunks": {}}
            )
            if cached.get("key") != document_key:
                raise RuntimeError(f"Invalid KG2RAG extraction cache {cache_path}.")
            cached_chunks = cached.setdefault("chunks", {})
            for chunk in document_chunks:
                sequence = str(chunk["sequence"])
                chunk_key = document_key["chunk_keys"][int(sequence)]
                entry = cached_chunks.get(sequence)
                if entry and entry.get("key") == chunk_key:
                    state["cached_extraction_chunks"] += 1
                    triplets = entry.get("triplets") or []
                else:
                    limit = int(args.max_extraction_calls or 0)
                    if limit and state["new_extraction_calls"] >= limit:
                        raise RuntimeError(
                            "KG2RAG stopped before exceeding the reviewed "
                            f"--max-extraction-calls budget of {limit}. Cached "
                            "triplets are retained; increase the budget to resume."
                        )
                    triplets, usage = _kg2rag_extract_triplets(
                        runtime,
                        str(chunk["extraction_text"]),
                    )
                    cached_chunks[sequence] = {
                        "key": chunk_key,
                        "triplets": triplets,
                        "usage": usage,
                    }
                    _write_json(cache_path, cached)
                    state["new_extraction_calls"] += 1
                kg_rows.append(
                    {
                        "chunk_id": str(chunk["id"]),
                        "author_id": str(chunk["author_id"]),
                        "document_id": document_id,
                        "title": str(chunk["title"]),
                        "sequence": int(chunk["sequence"]),
                        "triplets": triplets,
                    }
                )
                total_triplets += len(triplets)
                for head, relation, tail in triplets:
                    entities.update((head, tail))
                    relations.add(relation)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        kg_path = artifact_dir / "chunk_kg.jsonl"
        _write_jsonl(kg_path, kg_rows)

        batch_records = []
        batch_size = max(1, int(args.embedding_batch_size))
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            batch_key = {
                "adapter_version": KG2RAG_ADAPTER_VERSION,
                "embedding_model": args.embedding_model,
                "embedding_digest": runtime["embedding"]["digest"],
                "chunks": [
                    {
                        "id": str(chunk["id"]),
                        "text_sha256": hashlib.sha256(
                            str(chunk["retrieval_text"]).encode("utf-8")
                        ).hexdigest(),
                    }
                    for chunk in batch
                ],
            }
            batch_digest = hashlib.sha256(
                json.dumps(
                    batch_key,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            batch_dir = embedding_cache / batch_digest[:2]
            batch_path = batch_dir / f"{batch_digest}.npz"
            marker_path = batch_dir / f"{batch_digest}.json"
            valid = False
            if batch_path.exists() and marker_path.exists():
                marker = _load_json(marker_path)
                valid = (
                    marker.get("key") == batch_key
                    and marker.get("vectors_sha256") == _sha256_file(batch_path)
                )
            if valid:
                state["cached_embedding_batches"] += 1
            else:
                limit = int(args.max_embedding_batches or 0)
                if limit and state["new_embedding_batches"] >= limit:
                    raise RuntimeError(
                        "KG2RAG stopped before exceeding the reviewed "
                        f"--max-embedding-batches budget of {limit}. Cached "
                        "vectors are retained; increase the budget to resume."
                    )
                vectors = np.asarray(
                    _kg2rag_embeddings(
                        runtime,
                        [str(chunk["retrieval_text"]) for chunk in batch],
                    ),
                    dtype=np.float32,
                )
                if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                    raise RuntimeError("Invalid KG2RAG embedding batch shape.")
                _atomic_npz(batch_path, vectors=vectors)
                _write_json(
                    marker_path,
                    {
                        "key": batch_key,
                        "vectors_sha256": _sha256_file(batch_path),
                        "shape": list(vectors.shape),
                    },
                )
                state["new_embedding_batches"] += 1
            batch_records.append(
                {
                    "path": str(batch_path.resolve()),
                    "marker": str(marker_path.resolve()),
                    "count": len(batch),
                }
            )

        index = None
        embedding_dimension = None
        for record in batch_records:
            with np.load(record["path"]) as payload:
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
            faiss.normalize_L2(vectors)
            if index is None:
                embedding_dimension = int(vectors.shape[1])
                index = faiss.IndexFlatIP(embedding_dimension)
            if vectors.shape[1] != embedding_dimension:
                raise RuntimeError("KG2RAG embedding dimensions changed by batch.")
            index.add(vectors)
        if index is None or index.ntotal != len(chunks):
            raise RuntimeError("KG2RAG did not build a complete sentence index.")

        faiss_path = artifact_dir / "chunks.faiss"
        chunks_path = artifact_dir / "chunks.jsonl"
        _atomic_faiss(faiss_path, index)
        _write_jsonl(chunks_path, chunks)
        artifact_paths = {
            "chunks": chunks_path,
            "chunk_kg": kg_path,
            "faiss": faiss_path,
        }
        state.update(
            {
                "status": "completed",
                "artifacts": {
                    name: {
                        "path": str(path.resolve()),
                        "sha256": _sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                },
                "embedding_dimension": embedding_dimension,
                "triplet_count": total_triplets,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "wall_time_seconds": round(time.perf_counter() - started, 3),
                "runtime": runtime,
            }
        )
        _write_json(state_path, state)
        _write_json(args.output, state)
        print(json.dumps(state, indent=2, sort_keys=True))
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "wall_time_seconds": round(time.perf_counter() - started, 3),
            }
        )
        _write_json(state_path, state)
        raise


def _patch_kg2rag_paper_triplet_reranker(reranker: Any) -> Any:
    original_compute_score = reranker.compute_score

    def compute_score(
        pairs: Sequence[Sequence[str]],
        **kwargs: Any,
    ) -> Any:
        transformed = []
        for query, representation in pairs:
            representation = str(representation)
            if KG2RAG_RELATIONAL_FACTS_MARKER in representation:
                representation = representation.rsplit(
                    KG2RAG_RELATIONAL_FACTS_MARKER,
                    1,
                )[1].rstrip(".")
            transformed.append((query, representation))
        return original_compute_score(transformed, **kwargs)

    reranker.compute_score = compute_score
    return reranker


def _kg2rag_official_modules(repo: Path) -> Dict[str, Any]:
    code_dir = (repo / "code").resolve()
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    postprocessors = importlib.import_module("util.kg_post_processor")
    source_path = Path(postprocessors.__file__).resolve()
    if code_dir not in source_path.parents:
        raise RuntimeError(
            f"Imported KG2RAG from {source_path}, not pinned checkout {repo}."
        )
    schema = importlib.import_module("llama_index.core.schema")
    return {"postprocessors": postprocessors, "schema": schema}


def kg2rag_retrieve(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np
    from FlagEmbedding import FlagReranker

    prepared = _prepare_kg2rag_inputs(args.corpus, args.working_dir)
    runtime = _require_kg2rag_runtime(args, require_llm=False)
    state_path = args.working_dir / "index_state.json"
    if not state_path.exists():
        raise RuntimeError("KG2RAG index state is missing; run kg2rag-index first.")
    state = _load_json(state_path)
    if state.get("status") != "completed":
        raise RuntimeError(
            f"KG2RAG index is not complete (status={state.get('status')!r})."
        )
    if state.get("identity", {}).get("preparation_identity") != prepared["identity"]:
        raise RuntimeError("KG2RAG index belongs to a different canonical corpus.")
    if (
        state.get("identity", {}).get("embedding_digest")
        != runtime["embedding"]["digest"]
    ):
        raise RuntimeError("The active Ollama embedding model changed after indexing.")
    if (
        state.get("identity", {}).get("ollama_version")
        != runtime["ollama_version"]
    ):
        raise RuntimeError("The Ollama runtime version changed after indexing.")
    if not _kg2rag_index_artifacts_valid(state):
        raise RuntimeError("KG2RAG index artifacts are missing or have changed.")

    artifacts = state["artifacts"]
    chunks = _load_jsonl(Path(artifacts["chunks"]["path"]))
    kg_rows = _load_jsonl(Path(artifacts["chunk_kg"]["path"]))
    index = faiss.read_index(str(artifacts["faiss"]["path"]))
    if index.ntotal != len(chunks):
        raise RuntimeError("KG2RAG FAISS row count does not match canonical chunks.")

    chunks_index: Dict[str, Dict[str, str]] = {}
    author_to_document = {}
    chunk_by_author = {}
    for chunk in chunks:
        title = str(chunk["title"])
        sequence = str(chunk["sequence"])
        chunks_index.setdefault(title, {})[sequence] = str(chunk["retrieval_text"])
        author_to_document[str(chunk["author_id"])] = str(chunk["document_id"])
        chunk_by_author[str(chunk["author_id"])] = chunk
    doc2kg: Dict[str, Dict[str, List[List[str]]]] = {}
    for row in kg_rows:
        triplets = row.get("triplets") or []
        if triplets:
            doc2kg.setdefault(str(row["title"]), {})[
                str(row["sequence"])
            ] = triplets
    entities = set(chunks_index)

    modules = _kg2rag_official_modules(args.repo)
    postprocessors = modules["postprocessors"]
    schema = modules["schema"]
    reranker_path = _locked_model_path(
        args.reranker_model,
        args.reranker_revision,
        args.model_cache,
    )
    base_reranker = FlagReranker(model_name_or_path=reranker_path)
    reranker = (
        _patch_kg2rag_paper_triplet_reranker(base_reranker)
        if args.mst_rerank_representation == "paper_triplet"
        else base_reranker
    )
    use_tpt = args.mst_rerank_representation == "paper_triplet"
    expansion = postprocessors.KGRetrievePostProcessor(
        dataset="hotpotqa",
        ents=entities,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
    )
    graph_filter = postprocessors.GraphFilterPostProcessor(
        dataset="hotpotqa",
        use_tpt=use_tpt,
        topk=args.top_k,
        ents=entities,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
        reranker=reranker,
    )
    organizer = postprocessors.NaivePostprocessor(dataset="hotpotqa")

    queries = _load_jsonl(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    completed_rows = _load_jsonl(args.partial_output)
    completed = {}
    for row in completed_rows:
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in completed:
            raise ValueError("KG2RAG partial output has missing or duplicate query ids.")
        completed[query_id] = row
    expected_ids = {str(row["id"]) for row in queries}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise ValueError(
            f"KG2RAG partial output contains {len(unexpected)} unexpected queries."
        )

    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            query_id = str(query["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            vector = np.asarray(
                _kg2rag_embeddings(runtime, [str(query["question"])]),
                dtype=np.float32,
            )
            faiss.normalize_L2(vector)
            initial_k = min(args.initial_chunk_top_k, len(chunks))
            similarities, positions = index.search(vector, initial_k)
            initial_nodes = []
            for similarity, position in zip(similarities[0], positions[0]):
                if int(position) < 0:
                    continue
                chunk = chunks[int(position)]
                node = schema.TextNode(
                    text=str(chunk["retrieval_text"]),
                    id_=str(chunk["author_id"]),
                )
                initial_nodes.append(
                    schema.NodeWithScore(node=node, score=float(similarity))
                )
            query_bundle = schema.QueryBundle(query_str=str(query["question"]))
            expanded = expansion._postprocess_nodes(initial_nodes, query_bundle)
            selected = graph_filter._postprocess_nodes(expanded, query_bundle)
            organized = organizer._postprocess_nodes(selected, query_bundle)

            document_ids = []
            seen_documents = set()
            retrieved_chunks = []
            for node in organized:
                author_id = str(node.node.id_)
                document_id = author_to_document.get(author_id)
                if document_id is None:
                    raise RuntimeError(
                        f"KG2RAG returned unmapped author chunk {author_id!r}."
                    )
                retrieved_chunks.append(
                    {
                        "author_id": author_id,
                        "document_id": document_id,
                        "score": float(node.score or 0.0),
                    }
                )
                if document_id not in seen_documents:
                    seen_documents.add(document_id)
                    document_ids.append(document_id)
            row = {
                "query_id": query_id,
                "doc_ids": document_ids,
                "latency_ms": {
                    "retrieval": (time.perf_counter() - started) * 1000.0
                },
                "retrieved_chunks": retrieved_chunks,
                "adapter": {
                    "initial_chunk_top_k": args.initial_chunk_top_k,
                    "final_chunk_budget": args.top_k,
                    "graph_expansion": "pinned_author_postprocessor",
                    "graph_filter": "pinned_author_postprocessor",
                    "mst_rerank_representation": args.mst_rerank_representation,
                    "constructor_fields_restored": ["dataset", "use_tpt"],
                    "document_projection": "stable_unique_from_organized_chunks",
                },
            }
            _append_jsonl(handle, row)
            completed[query_id] = row

    ordered = [completed[str(query["id"])] for query in queries]
    _write_jsonl(args.output, ordered)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": "kg2rag",
            "queries": len(ordered),
            "initial_chunk_top_k": args.initial_chunk_top_k,
            "final_chunk_budget": args.top_k,
            "mst_rerank_representation": args.mst_rerank_representation,
            "output": str(args.output.resolve()),
        },
    )


RAPTOR_ADAPTER_VERSION = 1
RAPTOR_EMBEDDING_KEY = "SBERT"
RAPTOR_SUMMARY_PROMPT_VERSION = 1


def _prepare_raptor_inputs(
    corpus_path: Path,
    working_dir: Path,
) -> Dict[str, Any]:
    documents = _canonical_documents(corpus_path)
    normalized = []
    for document in documents:
        text = _document_text(document)
        if not text:
            raise ValueError(
                f"Canonical document {document['id']!r} has no title or text."
            )
        normalized.append({"id": str(document["id"]), "text": text})

    input_dir = working_dir / "input"
    documents_path = input_dir / "documents.jsonl"
    marker_path = input_dir / "preparation.json"
    identity = {
        "adapter_version": RAPTOR_ADAPTER_VERSION,
        "corpus_sha256": _sha256_file(corpus_path),
        "document_count": len(normalized),
        "tree_scope": "global_cross_document",
        "leaf_chunking": "official_split_text_per_canonical_document",
        "max_leaf_tokens": 100,
    }
    if marker_path.exists() and documents_path.exists():
        existing = _load_json(marker_path)
        if (
            existing.get("identity") == identity
            and existing.get("documents_sha256") == _sha256_file(documents_path)
        ):
            return existing

    _write_jsonl(documents_path, normalized)
    prepared = {
        "status": "completed",
        "identity": identity,
        "documents": str(documents_path.resolve()),
        "documents_sha256": _sha256_file(documents_path),
        "document_count": len(normalized),
        "questions_used": False,
        "labels_used": False,
        "document_boundaries_preserved": True,
    }
    _write_json(marker_path, prepared)
    return prepared


def raptor_prepare(args: argparse.Namespace) -> None:
    prepared = _prepare_raptor_inputs(args.corpus, args.working_dir)
    _write_json(args.output, prepared)
    print(json.dumps(prepared, indent=2, sort_keys=True))


def _require_raptor_summary_runtime(args: argparse.Namespace) -> Dict[str, str]:
    from urllib.parse import urlparse

    base_url = (
        args.llm_base_url
        or os.environ.get("RAPTOR_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
    api_key = (
        args.llm_api_key
        or os.environ.get("RAPTOR_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = (
        args.llm_model
        or os.environ.get("RAPTOR_LLM_MODEL")
        or ""
    )
    missing = [
        name
        for name, value in (
            ("RAPTOR_LLM_BASE_URL", base_url),
            ("RAPTOR_LLM_API_KEY", api_key),
            ("RAPTOR_LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "RAPTOR indexing requires an explicit OpenAI-compatible summarization "
            f"service; missing {', '.join(missing)}."
        )

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    local_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
    is_local = host in local_hosts or host.endswith(".local")
    if not is_local and not args.allow_paid_api:
        raise RuntimeError(
            "RAPTOR indexing is fail-closed for remote model endpoints because it "
            "generates recursive summaries. Pass --allow-paid-api only after "
            "reviewing cost and confirming the configured model."
        )
    max_summary_calls = int(getattr(args, "max_summary_calls", 0) or 0)
    if not is_local and max_summary_calls < 1:
        raise RuntimeError(
            "Remote RAPTOR indexing also requires --max-summary-calls with a "
            "positive reviewed budget."
        )
    if model != "gpt-3.5-turbo" and not args.allow_model_substitution:
        raise RuntimeError(
            "The matched RAPTOR protocol requires the paper's gpt-3.5-turbo "
            "summary target. Pass --allow-model-substitution only for a separately "
            "labeled engineering diagnostic."
        )
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "service_scope": "local" if is_local else "explicit_remote",
        "max_summary_calls": str(max_summary_calls),
    }


def _raptor_modules(repo: Path) -> Dict[str, Any]:
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    package = importlib.import_module("raptor")
    package_path = Path(package.__file__).resolve()
    if repo not in package_path.parents:
        raise RuntimeError(
            f"Imported RAPTOR from {package_path}, not pinned checkout {repo}."
        )
    return {
        "embedding_models": importlib.import_module("raptor.EmbeddingModels"),
        "summarization_models": importlib.import_module(
            "raptor.SummarizationModels"
        ),
        "cluster_tree_builder": importlib.import_module(
            "raptor.cluster_tree_builder"
        ),
        "tree_structures": importlib.import_module("raptor.tree_structures"),
        "utils": importlib.import_module("raptor.utils"),
    }


def _raptor_embedding_model(
    modules: Mapping[str, Any],
    model_path: str,
    device: str,
) -> Any:
    from sentence_transformers import SentenceTransformer

    base = modules["embedding_models"].BaseEmbeddingModel

    class LockedSBertEmbeddingModel(base):
        def __init__(self) -> None:
            kwargs = {} if device == "auto" else {"device": device}
            self.model = SentenceTransformer(model_path, **kwargs)

        def create_embedding(self, text: str) -> List[float]:
            return self.model.encode(text, show_progress_bar=False).tolist()

        def encode_batch(
            self,
            texts: Sequence[str],
            batch_size: int,
        ) -> Any:
            return self.model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

    return LockedSBertEmbeddingModel()


def _raptor_summary_model(
    modules: Mapping[str, Any],
    runtime: Mapping[str, str],
    cache_dir: Path,
) -> Any:
    from openai import OpenAI

    base = modules["summarization_models"].BaseSummarizationModel

    class CachedOpenAISummaryModel(base):
        def __init__(self) -> None:
            self.client = OpenAI(
                api_key=runtime["api_key"],
                base_url=runtime["base_url"],
            )
            self.cache_dir = cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_hits = 0
            self.cache_misses = 0

        def summarize(self, context: str, max_tokens: int = 150) -> str:
            key = {
                "prompt_version": RAPTOR_SUMMARY_PROMPT_VERSION,
                "model": runtime["model"],
                "base_url": runtime["base_url"],
                "max_tokens": int(max_tokens),
                "context_sha256": hashlib.sha256(
                    context.encode("utf-8")
                ).hexdigest(),
            }
            digest = hashlib.sha256(
                json.dumps(key, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            path = self.cache_dir / f"{digest}.json"
            if path.exists():
                cached = _load_json(path)
                if cached.get("key") != key or not cached.get("summary"):
                    raise RuntimeError(f"Invalid RAPTOR summary cache entry {path}.")
                self.cache_hits += 1
                return str(cached["summary"])

            call_limit = int(runtime["max_summary_calls"])
            if call_limit and self.cache_misses >= call_limit:
                raise RuntimeError(
                    "RAPTOR stopped before exceeding the reviewed "
                    f"--max-summary-calls budget of {call_limit}. Cached summaries "
                    "are retained; increase the explicit budget to resume."
                )
            response = self.client.chat.completions.create(
                model=runtime["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": (
                            "Write a summary of the following, including as many "
                            f"key details as possible: {context}:"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )
            summary = response.choices[0].message.content
            if not isinstance(summary, str) or not summary.strip():
                raise RuntimeError("RAPTOR summarizer returned an empty response.")
            _write_json(path, {"key": key, "summary": summary})
            self.cache_misses += 1
            return summary

    return CachedOpenAISummaryModel()


def _atomic_pickle(path: Path, value: Any) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_faiss(path: Path, index: Any) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _raptor_node_documents(
    all_nodes: Mapping[int, Any],
    leaf_to_document: Mapping[int, str],
) -> Dict[int, tuple[str, ...]]:
    memo: Dict[int, tuple[str, ...]] = {}
    visiting = set()

    def resolve(node_index: int) -> tuple[str, ...]:
        if node_index in memo:
            return memo[node_index]
        if node_index in visiting:
            raise ValueError(f"Cycle detected in RAPTOR tree at node {node_index}.")
        node = all_nodes.get(node_index)
        if node is None:
            raise ValueError(f"RAPTOR node {node_index} is missing.")
        visiting.add(node_index)
        if node_index in leaf_to_document:
            documents = (leaf_to_document[node_index],)
        else:
            children = sorted(int(value) for value in node.children)
            if not children:
                raise ValueError(
                    f"Non-leaf RAPTOR node {node_index} has no children."
                )
            documents = tuple(
                sorted(
                    {
                        document_id
                        for child in children
                        for document_id in resolve(child)
                    }
                )
            )
        visiting.remove(node_index)
        memo[node_index] = documents
        return documents

    for node_index in sorted(int(value) for value in all_nodes):
        resolve(node_index)
    return memo


def _raptor_index_identity(
    args: argparse.Namespace,
    prepared: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "adapter_version": RAPTOR_ADAPTER_VERSION,
        "preparation_identity": prepared["identity"],
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "summary_model": runtime["model"],
        "summary_base_url": runtime["base_url"],
        "summary_prompt_version": RAPTOR_SUMMARY_PROMPT_VERSION,
        "max_leaf_tokens": args.max_leaf_tokens,
        "max_layers": args.max_layers,
        "summarization_length": args.summarization_length,
        "cluster_threshold": args.cluster_threshold,
        "max_cluster_tokens": args.max_cluster_tokens,
        "tree_scope": "global_cross_document",
    }


def _raptor_index_artifacts_valid(state: Mapping[str, Any]) -> bool:
    artifacts = state.get("artifacts") or {}
    if not artifacts:
        return False
    for record in artifacts.values():
        path = Path(str(record.get("path") or ""))
        if not path.exists() or _sha256_file(path) != record.get("sha256"):
            return False
    return True


def raptor_index(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np

    prepared = _prepare_raptor_inputs(args.corpus, args.working_dir)
    runtime = _require_raptor_summary_runtime(args)
    identity = _raptor_index_identity(args, prepared, runtime)
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_dir = args.working_dir / "artifacts" / identity_sha256
    state_path = args.working_dir / "index_state.json"
    if state_path.exists():
        existing = _load_json(state_path)
        if (
            existing.get("status") == "completed"
            and existing.get("identity") == identity
            and _raptor_index_artifacts_valid(existing)
        ):
            _write_json(args.output, existing)
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    state: Dict[str, Any] = {
        "status": "in_progress",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "artifact_dir": str(artifact_dir.resolve()),
    }
    _write_json(state_path, state)
    summary = None
    try:
        modules = _raptor_modules(args.repo)
        model_path = _locked_model_path(
            args.embedding_model,
            args.embedding_revision,
            args.model_cache,
        )
        embedding = _raptor_embedding_model(modules, model_path, args.device)
        summary = _raptor_summary_model(
            modules,
            runtime,
            args.working_dir / "cache" / "summaries",
        )
        tokenizer = modules["utils"].tiktoken.get_encoding("cl100k_base")
        normalized_documents = _load_jsonl(Path(str(prepared["documents"])))

        chunks: List[str] = []
        leaf_rows: List[Dict[str, Any]] = []
        for document in normalized_documents:
            document_chunks = modules["utils"].split_text(
                str(document["text"]),
                tokenizer,
                args.max_leaf_tokens,
            )
            if not document_chunks:
                raise ValueError(
                    f"RAPTOR produced no leaf chunks for {document['id']!r}."
                )
            for ordinal, text in enumerate(document_chunks):
                leaf_index = len(chunks)
                chunks.append(text)
                leaf_rows.append(
                    {
                        "leaf_index": leaf_index,
                        "document_id": str(document["id"]),
                        "chunk_ordinal": ordinal,
                        "text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                    }
                )

        vectors = embedding.encode_batch(chunks, args.embedding_batch_size)
        node_class = modules["tree_structures"].Node
        leaf_nodes = {
            index: node_class(
                text,
                index,
                set(),
                {RAPTOR_EMBEDDING_KEY: vectors[index]},
            )
            for index, text in enumerate(chunks)
        }
        config = modules["cluster_tree_builder"].ClusterTreeConfig(
            tokenizer=tokenizer,
            max_tokens=args.max_leaf_tokens,
            num_layers=args.max_layers,
            summarization_length=args.summarization_length,
            summarization_model=summary,
            embedding_models={RAPTOR_EMBEDDING_KEY: embedding},
            cluster_embedding_model=RAPTOR_EMBEDDING_KEY,
            clustering_params={
                "threshold": args.cluster_threshold,
                "max_length_in_cluster": args.max_cluster_tokens,
                "tokenizer": tokenizer,
            },
        )
        builder = modules["cluster_tree_builder"].ClusterTreeBuilder(config)
        all_nodes = dict(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}
        root_nodes = builder.construct_tree(
            leaf_nodes,
            all_nodes,
            layer_to_nodes,
            use_multithreading=False,
        )
        tree = modules["tree_structures"].Tree(
            all_nodes,
            root_nodes,
            leaf_nodes,
            builder.num_layers,
            layer_to_nodes,
        )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        tree_path = artifact_dir / "tree.pkl"
        leaf_path = artifact_dir / "leaf_provenance.jsonl"
        node_ids_path = artifact_dir / "node_ids.json"
        document_ids_path = artifact_dir / "document_ids.json"
        provenance_path = artifact_dir / "node_document_provenance.npz"
        node_index_path = artifact_dir / "nodes.faiss"
        document_index_path = artifact_dir / "documents.faiss"
        _atomic_pickle(tree_path, tree)
        _write_jsonl(leaf_path, leaf_rows)

        node_ids = sorted(int(value) for value in all_nodes)
        _write_json(node_ids_path, {"node_ids": node_ids})
        document_ids = [str(row["id"]) for row in normalized_documents]
        _write_json(document_ids_path, {"document_ids": document_ids})
        document_position = {
            document_id: index for index, document_id in enumerate(document_ids)
        }
        leaf_to_document = {
            int(row["leaf_index"]): str(row["document_id"]) for row in leaf_rows
        }
        node_documents = _raptor_node_documents(all_nodes, leaf_to_document)
        offsets = [0]
        provenance_indices = []
        for node_index in node_ids:
            provenance_indices.extend(
                document_position[value] for value in node_documents[node_index]
            )
            offsets.append(len(provenance_indices))
        _atomic_npz(
            provenance_path,
            offsets=np.asarray(offsets, dtype=np.int64),
            document_indices=np.asarray(provenance_indices, dtype=np.int32),
        )

        node_vectors = np.asarray(
            [
                all_nodes[node_index].embeddings[RAPTOR_EMBEDDING_KEY]
                for node_index in node_ids
            ],
            dtype=np.float32,
        )
        faiss.normalize_L2(node_vectors)
        node_index = faiss.IndexFlatIP(node_vectors.shape[1])
        node_index.add(node_vectors)
        _atomic_faiss(node_index_path, node_index)

        document_vectors = np.zeros(
            (len(document_ids), node_vectors.shape[1]),
            dtype=np.float32,
        )
        document_counts = np.zeros(len(document_ids), dtype=np.int32)
        for leaf_index, document_id in leaf_to_document.items():
            position = document_position[document_id]
            document_vectors[position] += np.asarray(
                all_nodes[leaf_index].embeddings[RAPTOR_EMBEDDING_KEY],
                dtype=np.float32,
            )
            document_counts[position] += 1
        if np.any(document_counts == 0):
            raise RuntimeError("At least one canonical document has no RAPTOR leaves.")
        document_vectors /= document_counts[:, None]
        faiss.normalize_L2(document_vectors)
        document_index = faiss.IndexFlatIP(document_vectors.shape[1])
        document_index.add(document_vectors)
        _atomic_faiss(document_index_path, document_index)

        artifact_paths = {
            "tree": tree_path,
            "leaf_provenance": leaf_path,
            "node_ids": node_ids_path,
            "document_ids": document_ids_path,
            "node_document_provenance": provenance_path,
            "node_faiss": node_index_path,
            "document_faiss": document_index_path,
        }
        state = {
            **state,
            "status": "completed",
            "documents": len(document_ids),
            "leaf_nodes": len(leaf_nodes),
            "all_nodes": len(all_nodes),
            "tree_layers": tree.num_layers,
            "summary_cache_hits": summary.cache_hits,
            "summary_cache_misses": summary.cache_misses,
            "embedding_model_path": model_path,
            "summary_service_scope": runtime["service_scope"],
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                }
                for name, path in artifact_paths.items()
            },
        }
        _write_json(state_path, state)
        _write_json(args.output, state)
        print(json.dumps(state, indent=2, sort_keys=True))
    except Exception as exc:
        failed = {
            **state,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if summary is not None:
            failed["summary_cache_hits"] = summary.cache_hits
            failed["summary_cache_misses"] = summary.cache_misses
        _write_json(state_path, failed)
        raise


def _raptor_layer_map(tree: Any) -> Dict[int, int]:
    return {
        int(node.index): int(layer)
        for layer, nodes in tree.layer_to_nodes.items()
        for node in nodes
    }


def _raptor_select_nodes(
    query_vector: Any,
    index: Any,
    node_ids: Sequence[int],
    tree: Any,
    tokenizer: Any,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    import numpy as np

    if not node_ids:
        return []
    requested = min(128, len(node_ids))
    while True:
        similarities, positions = index.search(query_vector, requested)
        selected = []
        total_tokens = 0
        budget_reached = False
        for similarity, position in zip(similarities[0], positions[0]):
            if int(position) < 0:
                continue
            node_index = int(node_ids[int(position)])
            node = tree.all_nodes[node_index]
            tokens = len(tokenizer.encode(node.text))
            if total_tokens + tokens > max_tokens:
                budget_reached = True
                break
            selected.append(
                {
                    "node_index": node_index,
                    "similarity": float(similarity),
                    "tokens": tokens,
                    "text": node.text,
                }
            )
            total_tokens += tokens
        if budget_reached or requested == len(node_ids):
            return selected
        requested = min(len(node_ids), requested * 2)


def _rank_raptor_documents(
    selected_node_ids: Sequence[int],
    node_position: Mapping[int, int],
    offsets: Any,
    provenance_indices: Any,
    dense_document_positions: Sequence[int],
    document_ids: Sequence[str],
    top_k: int,
) -> List[str]:
    import numpy as np

    scores = np.zeros(len(document_ids), dtype=np.float64)
    for rank, node_index in enumerate(selected_node_ids, start=1):
        position = node_position[int(node_index)]
        start = int(offsets[position])
        end = int(offsets[position + 1])
        scores[provenance_indices[start:end]] += 1.0 / (60.0 + rank)
    for rank, position in enumerate(dense_document_positions, start=1):
        if int(position) >= 0:
            scores[int(position)] += 1.0 / (60.0 + rank)

    count = min(top_k, len(document_ids))
    if count == 0:
        return []
    if count == len(document_ids):
        candidates = np.arange(len(document_ids))
    else:
        candidates = np.argpartition(scores, -count)[-count:]
    ranked = sorted(
        (int(value) for value in candidates),
        key=lambda value: (-float(scores[value]), document_ids[value]),
    )
    return [document_ids[value] for value in ranked[:count]]


def raptor_retrieve(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np
    import pickle

    prepared = _prepare_raptor_inputs(args.corpus, args.working_dir)
    state_path = args.working_dir / "index_state.json"
    if not state_path.exists():
        raise RuntimeError("RAPTOR index state is missing; run raptor-index first.")
    state = _load_json(state_path)
    if state.get("status") != "completed":
        raise RuntimeError(
            f"RAPTOR index is not complete (status={state.get('status')!r})."
        )
    if state.get("identity", {}).get("preparation_identity") != prepared["identity"]:
        raise RuntimeError("RAPTOR index was built from a different canonical corpus.")
    if not _raptor_index_artifacts_valid(state):
        raise RuntimeError("RAPTOR index artifacts are missing or have changed.")

    modules = _raptor_modules(args.repo)
    model_path = _locked_model_path(
        args.embedding_model,
        args.embedding_revision,
        args.model_cache,
    )
    embedding = _raptor_embedding_model(modules, model_path, args.device)
    artifacts = state["artifacts"]
    with Path(artifacts["tree"]["path"]).open("rb") as handle:
        tree = pickle.load(handle)
    node_ids = [
        int(value)
        for value in _load_json(Path(artifacts["node_ids"]["path"]))["node_ids"]
    ]
    document_ids = [
        str(value)
        for value in _load_json(Path(artifacts["document_ids"]["path"]))[
            "document_ids"
        ]
    ]
    provenance = np.load(Path(artifacts["node_document_provenance"]["path"]))
    offsets = provenance["offsets"]
    provenance_indices = provenance["document_indices"]
    node_index = faiss.read_index(artifacts["node_faiss"]["path"])
    document_index = faiss.read_index(artifacts["document_faiss"]["path"])
    node_position = {node_id: index for index, node_id in enumerate(node_ids)}
    layer_map = _raptor_layer_map(tree)
    tokenizer = modules["utils"].tiktoken.get_encoding("cl100k_base")

    queries = _load_jsonl(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    completed_rows = _load_jsonl(args.partial_output)
    completed = {}
    for row in completed_rows:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in completed:
            raise ValueError("RAPTOR partial output has missing or duplicate query ids.")
        completed[query_id] = row
    expected_ids = {str(row["id"]) for row in queries}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise ValueError(
            f"RAPTOR partial output contains {len(unexpected)} unexpected queries."
        )

    args.partial_output.parent.mkdir(parents=True, exist_ok=True)
    with args.partial_output.open("a", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            query_id = str(query["id"])
            if query_id in completed:
                continue
            started = time.perf_counter()
            query_vector = np.asarray(
                [embedding.create_embedding(str(query["question"]))],
                dtype=np.float32,
            )
            faiss.normalize_L2(query_vector)
            selected = _raptor_select_nodes(
                query_vector,
                node_index,
                node_ids,
                tree,
                tokenizer,
                args.max_context_tokens,
            )
            dense_k = min(
                len(document_ids),
                max(args.top_k * args.dense_projection_multiplier, args.top_k),
            )
            _, dense_positions = document_index.search(query_vector, dense_k)
            ranked_documents = _rank_raptor_documents(
                [row["node_index"] for row in selected],
                node_position,
                offsets,
                provenance_indices,
                [int(value) for value in dense_positions[0]],
                document_ids,
                args.top_k,
            )
            row = {
                "query_id": query_id,
                "doc_ids": ranked_documents,
                "latency_ms": {
                    "retrieval": (time.perf_counter() - started) * 1000.0
                },
                "retrieved_nodes": [
                    {
                        **value,
                        "layer": layer_map[value["node_index"]],
                    }
                    for value in selected
                ],
                "adapter": {
                    "tree_scope": "global_cross_document",
                    "retrieval": "paper_collapsed_tree_token_budget",
                    "max_context_tokens": args.max_context_tokens,
                    "document_projection": (
                        "rrf_selected_node_descendants_plus_dense_leaf_centroids"
                    ),
                },
            }
            _append_jsonl(handle, row)
            completed[query_id] = row

    ordered = [completed[str(query["id"])] for query in queries]
    _write_jsonl(args.output, ordered)
    _write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "status": "completed",
            "method": "raptor",
            "queries": len(ordered),
            "top_k": args.top_k,
            "max_context_tokens": args.max_context_tokens,
            "tree_scope": "global_cross_document",
            "output": str(args.output.resolve()),
        },
    )


def _add_gfm_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--working-data-root", type=Path, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--paper-v1", action="store_true")
    parser.add_argument("--model-cache", type=Path, required=True)


def _add_hipporag_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dataset-alias", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--llm-name", default="gpt-4o-mini")
    parser.add_argument("--embedding-name", default="nvidia/NV-Embed-v2")
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--openie-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--top-k", type=int, default=200)


def _add_lightrag_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--implementation", choices=("paper", "current"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)


def _add_hoprag_input_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--dataset-alias", required=True)
    parser.add_argument("--group-size", type=int, default=10)


def _add_hoprag_runtime_shared(parser: argparse.ArgumentParser) -> None:
    _add_hoprag_input_shared(parser)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--edge-batch-size", type=int, default=2000)
    parser.add_argument("--neo4j-uri")
    parser.add_argument("--neo4j-user")
    parser.add_argument("--neo4j-database")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="Explicitly allow calls to api.openai.com; local endpoints need no flag.",
    )


def _add_raptor_input_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)


def _add_kg2rag_input_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, required=True)


def _add_kg2rag_runtime_shared(parser: argparse.ArgumentParser) -> None:
    _add_kg2rag_input_shared(parser)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ollama-base-url")
    parser.add_argument("--llm-model", default=KG2RAG_PAPER_LLM)
    parser.add_argument("--llm-digest", default=KG2RAG_PAPER_LLM_DIGEST)
    parser.add_argument(
        "--embedding-model",
        default=KG2RAG_PAPER_EMBEDDING,
    )
    parser.add_argument(
        "--embedding-digest",
        default=KG2RAG_PAPER_EMBEDDING_DIGEST,
    )
    parser.add_argument(
        "--allow-remote-ollama",
        action="store_true",
        help="Allow an explicitly reviewed non-local Ollama endpoint.",
    )
    parser.add_argument(
        "--allow-model-substitution",
        action="store_true",
        help="Allow non-paper model names for a separately labeled diagnostic.",
    )


def _add_kg2rag_model_shared(parser: argparse.ArgumentParser) -> None:
    _add_kg2rag_runtime_shared(parser)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--reranker-revision", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--initial-chunk-top-k", type=int, default=10)


def _add_raptor_model_shared(parser: argparse.ArgumentParser) -> None:
    _add_raptor_input_shared(parser)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/multi-qa-mpnet-base-cos-v1",
    )
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--device", default="auto")


def _add_raptor_index_shared(parser: argparse.ArgumentParser) -> None:
    _add_raptor_model_shared(parser)
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="Explicitly allow recursive summaries on a non-local endpoint.",
    )
    parser.add_argument(
        "--allow-model-substitution",
        action="store_true",
        help="Allow a non-paper summarizer for a separately labeled diagnostic.",
    )
    parser.add_argument(
        "--max-summary-calls",
        type=int,
        default=0,
        help="Maximum new summary calls; required for any non-local endpoint.",
    )
    parser.add_argument("--max-leaf-tokens", type=int, default=100)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--summarization-length", type=int, default=100)
    parser.add_argument("--cluster-threshold", type=float, default=0.1)
    parser.add_argument("--max-cluster-tokens", type=int, default=3500)
    parser.add_argument("--embedding-batch-size", type=int, default=64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned external SOTA adapters")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("hipporag-index")
    _add_hipporag_shared(index)
    index.add_argument("--output", type=Path, required=True)
    index.set_defaults(func=hipporag_index)

    retrieve = commands.add_parser("hipporag-retrieve")
    _add_hipporag_shared(retrieve)
    retrieve.add_argument("--queries", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--partial-output", type=Path, required=True)
    retrieve.add_argument("--limit", type=int, default=0)
    retrieve.set_defaults(func=hipporag_retrieve)

    gfm_index = commands.add_parser("gfmrag-index")
    _add_gfm_shared(gfm_index)
    gfm_index.add_argument("--hydra-output", type=Path, required=True)
    gfm_index.add_argument("--output", type=Path, required=True)
    gfm_index.set_defaults(func=gfmrag_index)

    gfm_retrieve = commands.add_parser("gfmrag-retrieve")
    _add_gfm_shared(gfm_retrieve)
    gfm_retrieve.add_argument("--model", required=True)
    gfm_retrieve.add_argument("--model-revision", required=True)
    gfm_retrieve.add_argument("--split", choices=("train", "val", "test"), default="test")
    gfm_retrieve.add_argument("--top-k", type=int, default=100)
    gfm_retrieve.add_argument("--output", type=Path, required=True)
    gfm_retrieve.add_argument("--partial-output", type=Path, required=True)
    gfm_retrieve.add_argument("--limit", type=int, default=0)
    gfm_retrieve.set_defaults(func=gfmrag_retrieve)

    lightrag_index_parser = commands.add_parser("lightrag-index")
    _add_lightrag_shared(lightrag_index_parser)
    lightrag_index_parser.add_argument("--batch-size", type=int, default=1)
    lightrag_index_parser.add_argument("--output", type=Path, required=True)
    lightrag_index_parser.set_defaults(func=lightrag_index)

    lightrag_retrieve_parser = commands.add_parser("lightrag-retrieve")
    _add_lightrag_shared(lightrag_retrieve_parser)
    lightrag_retrieve_parser.add_argument("--queries", type=Path, required=True)
    lightrag_retrieve_parser.add_argument("--top-k", type=int, default=100)
    lightrag_retrieve_parser.add_argument("--output", type=Path, required=True)
    lightrag_retrieve_parser.add_argument(
        "--partial-output",
        type=Path,
        required=True,
    )
    lightrag_retrieve_parser.add_argument("--limit", type=int, default=0)
    lightrag_retrieve_parser.set_defaults(func=lightrag_retrieve)

    hoprag_prepare_parser = commands.add_parser("hoprag-prepare")
    _add_hoprag_input_shared(hoprag_prepare_parser)
    hoprag_prepare_parser.add_argument("--output", type=Path, required=True)
    hoprag_prepare_parser.set_defaults(func=hoprag_prepare)

    hoprag_index_parser = commands.add_parser("hoprag-index")
    _add_hoprag_runtime_shared(hoprag_index_parser)
    hoprag_index_parser.add_argument("--output", type=Path, required=True)
    hoprag_index_parser.set_defaults(func=hoprag_index)

    hoprag_retrieve_parser = commands.add_parser("hoprag-retrieve")
    _add_hoprag_runtime_shared(hoprag_retrieve_parser)
    hoprag_retrieve_parser.add_argument("--queries", type=Path, required=True)
    hoprag_retrieve_parser.add_argument("--top-k", type=int, default=100)
    hoprag_retrieve_parser.add_argument("--max-hop", type=int, default=4)
    hoprag_retrieve_parser.add_argument(
        "--traversal",
        choices=("dfs", "bfs", "bfs_sim_node", "bfs_node", "bfs_hop2"),
        default="bfs_node",
    )
    hoprag_retrieve_parser.add_argument("--hybrid", action="store_true")
    hoprag_retrieve_parser.add_argument("--output", type=Path, required=True)
    hoprag_retrieve_parser.add_argument(
        "--partial-output",
        type=Path,
        required=True,
    )
    hoprag_retrieve_parser.add_argument("--limit", type=int, default=0)
    hoprag_retrieve_parser.set_defaults(func=hoprag_retrieve)

    kg2rag_prepare_parser = commands.add_parser("kg2rag-prepare")
    _add_kg2rag_input_shared(kg2rag_prepare_parser)
    kg2rag_prepare_parser.add_argument("--output", type=Path, required=True)
    kg2rag_prepare_parser.set_defaults(func=kg2rag_prepare)

    kg2rag_index_parser = commands.add_parser("kg2rag-index")
    _add_kg2rag_runtime_shared(kg2rag_index_parser)
    kg2rag_index_parser.add_argument("--embedding-batch-size", type=int, default=128)
    kg2rag_index_parser.add_argument("--max-extraction-calls", type=int, default=0)
    kg2rag_index_parser.add_argument("--max-embedding-batches", type=int, default=0)
    kg2rag_index_parser.add_argument("--output", type=Path, required=True)
    kg2rag_index_parser.set_defaults(func=kg2rag_index)

    kg2rag_retrieve_parser = commands.add_parser("kg2rag-retrieve")
    _add_kg2rag_model_shared(kg2rag_retrieve_parser)
    kg2rag_retrieve_parser.add_argument("--queries", type=Path, required=True)
    kg2rag_retrieve_parser.add_argument("--top-k", type=int, default=10)
    kg2rag_retrieve_parser.add_argument(
        "--mst-rerank-representation",
        choices=("paper_triplet", "released_text"),
        default="paper_triplet",
    )
    kg2rag_retrieve_parser.add_argument("--output", type=Path, required=True)
    kg2rag_retrieve_parser.add_argument(
        "--partial-output",
        type=Path,
        required=True,
    )
    kg2rag_retrieve_parser.add_argument("--limit", type=int, default=0)
    kg2rag_retrieve_parser.set_defaults(func=kg2rag_retrieve)

    raptor_prepare_parser = commands.add_parser("raptor-prepare")
    _add_raptor_input_shared(raptor_prepare_parser)
    raptor_prepare_parser.add_argument("--output", type=Path, required=True)
    raptor_prepare_parser.set_defaults(func=raptor_prepare)

    raptor_index_parser = commands.add_parser("raptor-index")
    _add_raptor_index_shared(raptor_index_parser)
    raptor_index_parser.add_argument("--output", type=Path, required=True)
    raptor_index_parser.set_defaults(func=raptor_index)

    raptor_retrieve_parser = commands.add_parser("raptor-retrieve")
    _add_raptor_model_shared(raptor_retrieve_parser)
    raptor_retrieve_parser.add_argument("--queries", type=Path, required=True)
    raptor_retrieve_parser.add_argument("--top-k", type=int, default=100)
    raptor_retrieve_parser.add_argument("--max-context-tokens", type=int, default=2000)
    raptor_retrieve_parser.add_argument(
        "--dense-projection-multiplier",
        type=int,
        default=10,
    )
    raptor_retrieve_parser.add_argument("--output", type=Path, required=True)
    raptor_retrieve_parser.add_argument(
        "--partial-output",
        type=Path,
        required=True,
    )
    raptor_retrieve_parser.add_argument("--limit", type=int, default=0)
    raptor_retrieve_parser.set_defaults(func=raptor_retrieve)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
