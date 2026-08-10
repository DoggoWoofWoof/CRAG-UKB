"""Standalone REALM metric-divergence experiment on official public data.

The experiment compares aggregate supporting-passage recall with joint evidence
completeness for a single-shot dense retriever and a fixed pseudo-relevance-
feedback (PRF) baseline. It intentionally does not import CRAG code.

Stages are content-addressed under data/ukb_storage/_review/realm_metric_v2.
Matching source files, parsed corpora, embeddings, rankings, and statistics are
reused. Run `python review_paper/experiment.py --help` for the entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LOG = logging.getLogger("realm.metric_divergence")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "review_public"
CACHE = ROOT / "data" / "ukb_storage" / "_review" / "realm_metric_v2"

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PARSER_VERSION = 2
RETRIEVAL_VERSION = 3
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 5000

TWOWIKI_URL = (
    "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/"
    "612bc5039a457880d9e7d84c3b0a4cf154b70e4f/dev.parquet?download=true"
)
MUSIQUE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SOURCE_PROVENANCE = {
    "2wiki": {
        "upstream": "xanhho/2WikiMultihopQA",
        "revision": "612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
        "artifact": "dev.parquet",
    },
    "musique": {
        "upstream": "StonyBrookNLP/musique",
        "revision": "official v1.0 archive",
        "artifact": "musique_ans_v1.0_dev.jsonl",
    },
    "hotpotqa": {
        "upstream": "hotpotqa/hotpot_qa",
        "revision": HOTPOT_REVISION,
        "artifact": "distractor/validation",
    },
}
EXPECTED_SOURCE_SHA256 = {
    "2wiki": "c0d8b60b9026b728fb07ad74c5252a0f188f6942e8ba5c02df4dfa369502ea8d",
    "musique": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
    "hotpotqa": "91d903fc2d904fa104bf1c4464692cdb3ef255f8b7766597ee09e2d05bba8875",
}


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    golds: tuple[int, ...]


@dataclass
class DatasetView:
    name: str
    display_name: str
    split: str
    source_path: str
    source_sha256: str
    documents: list[str]
    document_keys: list[str]
    queries: list[Query]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def _document_key(title: str, text: str) -> str:
    return _sha256_json([_normalize_text(title).casefold(), _normalize_text(text)])


def _vector_text_hash(text: str) -> str:
    # Existing CRAG BGE vectors encode passage content without the title.
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "CRAG-REALM-reproduction/2"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            output.write(chunk)


def _prepare_sources() -> dict[str, Path]:
    RAW.mkdir(parents=True, exist_ok=True)

    twowiki = RAW / "2wiki_dev.parquet"
    _download(TWOWIKI_URL, twowiki)

    musique_zip = RAW / "musique_v1.0.zip"
    musique = RAW / "musique_v1.0" / "data" / "musique_ans_v1.0_dev.jsonl"
    if not musique.exists():
        if not musique_zip.exists():
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("Install gdown to fetch the official MuSiQue archive") from exc
            gdown.download(id=MUSIQUE_DRIVE_ID, output=str(musique_zip), quiet=False)
        with zipfile.ZipFile(musique_zip) as archive:
            archive.extractall(RAW / "musique_v1.0")

    hotpot = RAW / "hotpot_dev_distractor.jsonl"
    if not hotpot.exists():
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets to fetch the official HotpotQA split") from exc
        split = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split="validation",
            revision=HOTPOT_REVISION,
        )
        with hotpot.open("w", encoding="utf-8", newline="\n") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    sources = {"2wiki": twowiki, "musique": musique, "hotpotqa": hotpot}
    for name, path in sources.items():
        observed = _sha256_file(path)
        if observed != EXPECTED_SOURCE_SHA256[name]:
            raise ValueError(
                f"{name} source checksum mismatch: expected {EXPECTED_SOURCE_SHA256[name]}, got {observed}"
            )
    return sources


class _CorpusBuilder:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.document_keys: list[str] = []
        self._key_to_index: dict[str, int] = {}

    def add(self, title: str, text: str) -> int:
        text = str(text)
        key = _document_key(title, text)
        index = self._key_to_index.get(key)
        if index is None:
            index = len(self.documents)
            self._key_to_index[key] = index
            self.documents.append(text)
            self.document_keys.append(key)
        return index


def _json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def _load_2wiki(path: Path) -> DatasetView:
    import pandas as pd

    frame = pd.read_parquet(path)
    corpus = _CorpusBuilder()
    queries: list[Query] = []
    invalid: list[str] = []
    for row in frame.to_dict(orient="records"):
        context = _json_field(row["context"])
        supports = _json_field(row["supporting_facts"])
        by_title: dict[str, int] = {}
        for title, sentences in context:
            by_title[str(title)] = corpus.add(str(title), " ".join(map(str, sentences)))
        support_titles = list(dict.fromkeys(str(title) for title, _ in supports))
        golds = tuple(sorted({by_title[title] for title in support_titles if title in by_title}))
        if len(golds) < 2 or len(golds) != len(support_titles):
            invalid.append(str(row["_id"]))
            continue
        queries.append(Query(str(row["_id"]), str(row["question"]), golds))
    if invalid:
        raise ValueError(f"2Wiki source has {len(invalid)} invalid support mappings; first={invalid[:3]}")
    return DatasetView(
        "2wiki",
        "2Wiki",
        "official dev",
        path.relative_to(ROOT).as_posix(),
        _sha256_file(path),
        corpus.documents,
        corpus.document_keys,
        queries,
    )


def _load_musique(path: Path) -> DatasetView:
    corpus = _CorpusBuilder()
    queries: list[Query] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            golds: list[int] = []
            for paragraph in row["paragraphs"]:
                index = corpus.add(paragraph.get("title", ""), paragraph["paragraph_text"])
                if paragraph.get("is_supporting"):
                    golds.append(index)
            unique_golds = tuple(sorted(set(golds)))
            if len(unique_golds) < 2:
                raise ValueError(f"MuSiQue query {row['id']} has fewer than two supporting passages")
            queries.append(Query(str(row["id"]), str(row["question"]), unique_golds))
    return DatasetView(
        "musique",
        "MuSiQue",
        "official answerable dev",
        path.relative_to(ROOT).as_posix(),
        _sha256_file(path),
        corpus.documents,
        corpus.document_keys,
        queries,
    )


def _load_hotpotqa(path: Path) -> DatasetView:
    corpus = _CorpusBuilder()
    queries: list[Query] = []
    invalid: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_title: dict[str, int] = {}
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                by_title[str(title)] = corpus.add(str(title), " ".join(map(str, sentences)))
            support_titles = list(dict.fromkeys(map(str, row["supporting_facts"]["title"])))
            golds = tuple(sorted({by_title[title] for title in support_titles if title in by_title}))
            if len(golds) != 2:
                invalid.append(str(row["id"]))
                continue
            queries.append(Query(str(row["id"]), str(row["question"]), golds))
    if invalid:
        raise ValueError(f"HotpotQA source has {len(invalid)} invalid support mappings; first={invalid[:3]}")
    return DatasetView(
        "hotpotqa",
        "HotpotQA",
        "official distractor validation",
        path.relative_to(ROOT).as_posix(),
        _sha256_file(path),
        corpus.documents,
        corpus.document_keys,
        queries,
    )


LOADERS = {"2wiki": _load_2wiki, "musique": _load_musique, "hotpotqa": _load_hotpotqa}
REUSE_DATASETS = {
    "2wiki": "2wiki_clean",
    "musique": "musique_clean",
    "hotpotqa": "hotpotqa_clean",
}


def _view_fingerprint(view: DatasetView) -> str:
    return _sha256_json(
        {
            "parser_version": PARSER_VERSION,
            "dataset": view.name,
            "source_sha256": view.source_sha256,
            "documents": view.document_keys,
            "queries": [[q.query_id, q.text, q.golds] for q in view.queries],
        }
    )


def _write_view_manifest(view: DatasetView, stage: Path) -> None:
    support_counts = Counter(len(query.golds) for query in view.queries)
    manifest = {
        "parser_version": PARSER_VERSION,
        "dataset": view.name,
        "display_name": view.display_name,
        "split": view.split,
        "source_path": view.source_path,
        "source_sha256": view.source_sha256,
        "source_provenance": SOURCE_PROVENANCE[view.name],
        "fingerprint": _view_fingerprint(view),
        "document_count": len(view.documents),
        "query_count": len(view.queries),
        "support_count_distribution": {str(k): v for k, v in sorted(support_counts.items())},
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "view_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_encoder(device: str | None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=device)


def _encode(encoder, texts: Sequence[str], batch_size: int) -> np.ndarray:
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _reuse_existing_document_vectors(view: DatasetView, output: np.ndarray) -> np.ndarray:
    missing_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(view.documents):
        missing_by_hash[_vector_text_hash(text)].append(index)
    reused = np.zeros(len(view.documents), dtype=bool)
    source = REUSE_DATASETS[view.name]
    master = ROOT / "data" / "processed" / f"master_nodes_{source}.json"
    vectors_path = ROOT / "data" / "ukb_storage" / source / "bge_large" / "nodes.npy"
    if not master.exists() or not vectors_path.exists():
        return reused

    nodes = json.loads(master.read_text(encoding="utf-8"))
    documents = [node for node in nodes if node.get("metadata", {}).get("type") != "question"]
    vectors = np.load(vectors_path, mmap_mode="r")
    if len(documents) != len(vectors):
        LOG.warning("Cannot reuse %s: %d documents but %d vectors", source, len(documents), len(vectors))
        return reused
    for row, document in enumerate(documents):
        targets = missing_by_hash.get(_vector_text_hash(document.get("content", "")))
        if not targets:
            continue
        for target in targets:
            output[target] = vectors[row]
            reused[target] = True
    return reused


def _embedding_cache_path(stage: Path, kind: str, fingerprint: str) -> tuple[Path, Path]:
    stem = stage / f"{kind}_{fingerprint[:16]}"
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def _document_embeddings(
    view: DatasetView,
    stage: Path,
    encoder,
    batch_size: int,
    reuse_existing: bool,
) -> tuple[np.ndarray, dict]:
    fingerprint = _sha256_json(
        {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "kind": "documents",
            "texts": [_vector_text_hash(text) for text in view.documents],
        }
    )
    path, meta_path = _embedding_cache_path(stage, "documents", fingerprint)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(path, mmap_mode="r")
        if tuple(vectors.shape) == tuple(meta["shape"]):
            return np.asarray(vectors), meta

    output = np.empty((len(view.documents), 1024), dtype=np.float32)
    reused = _reuse_existing_document_vectors(view, output) if reuse_existing else np.zeros(len(view.documents), bool)
    missing = np.flatnonzero(~reused)
    if len(missing):
        encoded = _encode(encoder, [view.documents[i] for i in missing], batch_size)
        output[missing] = encoded
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    output /= np.maximum(norms, 1e-12)
    np.save(path, output)
    meta = {
        "fingerprint": fingerprint,
        "shape": list(output.shape),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "reused_existing": int(reused.sum()),
        "newly_encoded": int(len(missing)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return output, meta


def _query_embeddings(view: DatasetView, stage: Path, encoder, batch_size: int) -> tuple[np.ndarray, dict]:
    texts = [QUERY_PREFIX + query.text for query in view.queries]
    fingerprint = _sha256_json(
        {"model": MODEL_ID, "revision": MODEL_REVISION, "kind": "queries", "texts": texts}
    )
    path, meta_path = _embedding_cache_path(stage, "queries", fingerprint)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(path, mmap_mode="r")
        if tuple(vectors.shape) == tuple(meta["shape"]):
            return np.asarray(vectors), meta
    vectors = _encode(encoder, texts, batch_size)
    np.save(path, vectors)
    meta = {
        "fingerprint": fingerprint,
        "shape": list(vectors.shape),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "query_prefix": QUERY_PREFIX,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return vectors, meta


def _rrf_merge(first: np.ndarray, second: np.ndarray, k: int, rrf_k: int = 60) -> list[set[int]]:
    outputs: list[set[int]] = []
    for left, right in zip(first, second):
        scores: dict[int, float] = defaultdict(float)
        for rank, document in enumerate(left, start=1):
            scores[int(document)] += 1.0 / (rrf_k + rank)
        for rank, document in enumerate(right, start=1):
            scores[int(document)] += 1.0 / (rrf_k + rank)
        ranked = sorted(scores, key=lambda document: (-scores[document], document))[:k]
        outputs.append(set(ranked))
    return outputs


def _per_query(topk: Sequence[set[int]], golds: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    recall = np.empty(len(golds), dtype=np.float64)
    joint = np.empty(len(golds), dtype=np.float64)
    for index, (retrieved, gold) in enumerate(zip(topk, golds)):
        hits = sum(document in retrieved for document in gold)
        recall[index] = hits / len(gold)
        joint[index] = float(hits == len(gold))
    return recall, joint


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        width = min(100, samples - start)
        indices = rng.integers(0, len(values), size=(width, len(values)))
        means[start : start + width] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [round(100 * float(low), 2), round(100 * float(high), 2)]


def _estimate(values: np.ndarray, rng: np.random.Generator, samples: int) -> dict:
    return {
        "estimate": round(100 * float(values.mean()), 2),
        "ci95": _bootstrap_mean(values, rng, samples),
    }


def _runtime_versions() -> dict[str, str]:
    packages = ["datasets", "faiss-gpu-cu12", "numpy", "pandas", "pyarrow", "sentence-transformers", "torch", "transformers"]
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _mcnemar_exact(first: np.ndarray, second: np.ndarray) -> dict:
    from scipy.stats import binom

    first_bool = first.astype(bool)
    second_bool = second.astype(bool)
    lost = int(np.sum(first_bool & ~second_bool))
    gained = int(np.sum(~first_bool & second_bool))
    discordant = lost + gained
    if discordant == 0:
        p_value = 1.0
    else:
        cutoff = min(lost, gained)
        p_value = min(1.0, 2.0 * float(binom.cdf(cutoff, discordant, 0.5)))
    return {"single_only": lost, "prf_rrf_only": gained, "p_value_two_sided": p_value}


def _weakest_link_ranks(rankings: np.ndarray, golds: Sequence[tuple[int, ...]]) -> np.ndarray:
    missing_rank = rankings.shape[1] + 1
    output = np.empty(len(golds), dtype=np.int32)
    for index, (ranking, gold) in enumerate(zip(rankings, golds)):
        positions = {int(document): rank for rank, document in enumerate(ranking, start=1)}
        output[index] = max(positions.get(document, missing_rank) for document in gold)
    return output


def _evaluate(
    view: DatasetView,
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ks: Sequence[int],
    rank_depth: int,
    bootstrap_samples: int,
    stage: Path,
) -> tuple[dict, list[dict]]:
    import faiss

    max_k = max(ks)
    rrf_depth = max(100, 5 * max_k)
    search_depth = min(len(view.documents), max(rank_depth, rrf_depth))
    ranking_fingerprint = _sha256_json(
        {
            "retrieval_version": RETRIEVAL_VERSION,
            "view": _view_fingerprint(view),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "search_depth": search_depth,
            "rrf_depth": rrf_depth,
            "feedback": "normalize(query + top1_document)",
        }
    )
    single_path = stage / f"single_rankings_{ranking_fingerprint[:16]}.npy"
    feedback_path = stage / f"feedback_rankings_{ranking_fingerprint[:16]}.npy"
    if single_path.exists() and feedback_path.exists():
        single_rankings = np.load(single_path)
        feedback_rankings = np.load(feedback_path)
        expected = (len(view.queries), search_depth)
        if single_rankings.shape != expected or feedback_rankings.shape != (len(view.queries), rrf_depth):
            single_rankings = feedback_rankings = None
    else:
        single_rankings = feedback_rankings = None

    if single_rankings is None:
        cpu_index = faiss.IndexFlatIP(document_vectors.shape[1])
        cpu_index.add(np.ascontiguousarray(document_vectors, dtype=np.float32))
        index = cpu_index
        try:
            if faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(cpu_index)
                LOG.info("Using %d FAISS GPU(s) for %s retrieval", faiss.get_num_gpus(), view.name)
        except (AttributeError, RuntimeError) as exc:
            LOG.warning("FAISS GPU transfer unavailable; using CPU index: %s", exc)
        _, single_rankings = index.search(
            np.ascontiguousarray(query_vectors, dtype=np.float32), search_depth
        )
        top_passages = document_vectors[single_rankings[:, 0]]
        feedback_queries = np.asarray(query_vectors + top_passages, dtype=np.float32)
        faiss.normalize_L2(feedback_queries)
        _, feedback_rankings = index.search(feedback_queries, rrf_depth)
        np.save(single_path, single_rankings)
        np.save(feedback_path, feedback_rankings)
    golds = [query.golds for query in view.queries]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    result: dict = {}
    per_query_rows = [
        {"dataset": view.name, "query_id": query.query_id, "support_count": len(query.golds)}
        for query in view.queries
    ]
    for k in ks:
        single_sets = [set(map(int, row[:k])) for row in single_rankings]
        prf_sets = _rrf_merge(single_rankings[:, :rrf_depth], feedback_rankings, k)
        single_recall, single_joint = _per_query(single_sets, golds)
        prf_recall, prf_joint = _per_query(prf_sets, golds)
        result[f"k{k}"] = {
            "single": {
                "aggregate_recall": _estimate(single_recall, rng, bootstrap_samples),
                "joint_recall": _estimate(single_joint, rng, bootstrap_samples),
                "aggregate_minus_joint": _estimate(single_recall - single_joint, rng, bootstrap_samples),
            },
            "prf_rrf": {
                "aggregate_recall": _estimate(prf_recall, rng, bootstrap_samples),
                "joint_recall": _estimate(prf_joint, rng, bootstrap_samples),
                "aggregate_minus_joint": _estimate(prf_recall - prf_joint, rng, bootstrap_samples),
            },
            "paired_delta": {
                "aggregate_recall": _estimate(prf_recall - single_recall, rng, bootstrap_samples),
                "joint_recall": _estimate(prf_joint - single_joint, rng, bootstrap_samples),
            },
            "joint_mcnemar": _mcnemar_exact(single_joint, prf_joint),
        }
        for row, sr, sj, pr, pj in zip(
            per_query_rows, single_recall, single_joint, prf_recall, prf_joint
        ):
            row[f"k{k}"] = {
                "single_recall": float(sr),
                "single_joint": int(sj),
                "prf_rrf_recall": float(pr),
                "prf_rrf_joint": int(pj),
            }

        support_sizes = np.asarray([len(gold) for gold in golds])
        for support_count in sorted(set(support_sizes)):
            mask = support_sizes == support_count
            subgroup = result.setdefault("by_support_count", {}).setdefault(
                str(int(support_count)), {"queries": int(mask.sum())}
            )
            subgroup[f"k{k}"] = {
                "single": {
                    "aggregate_recall": _estimate(single_recall[mask], rng, bootstrap_samples),
                    "joint_recall": _estimate(single_joint[mask], rng, bootstrap_samples),
                    "aggregate_minus_joint": _estimate(
                        single_recall[mask] - single_joint[mask], rng, bootstrap_samples
                    ),
                },
                "prf_rrf": {
                    "aggregate_recall": _estimate(prf_recall[mask], rng, bootstrap_samples),
                    "joint_recall": _estimate(prf_joint[mask], rng, bootstrap_samples),
                    "aggregate_minus_joint": _estimate(
                        prf_recall[mask] - prf_joint[mask], rng, bootstrap_samples
                    ),
                },
            }

    weakest = _weakest_link_ranks(single_rankings[:, : min(rank_depth, search_depth)], golds)
    result["weakest_link"] = {
        "rank_depth": min(rank_depth, search_depth),
        "missing_sentinel": min(rank_depth, search_depth) + 1,
        "censored_pct": round(100 * float(np.mean(weakest > min(rank_depth, search_depth))), 2),
        "median_one_based": float(np.median(weakest)),
        "p75_one_based": float(np.percentile(weakest, 75)),
        "within_top20_pct": round(100 * float(np.mean(weakest <= 20)), 2),
    }
    for row, rank in zip(per_query_rows, weakest):
        row["weakest_link_rank"] = int(rank)

    with (stage / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return result, per_query_rows


def run(
    datasets: Sequence[str],
    ks: Sequence[int],
    rank_depth: int,
    bootstrap_samples: int,
    batch_size: int,
    device: str | None,
    reuse_existing: bool,
    audit_only: bool,
    output_dir: Path,
) -> dict:
    sources = _prepare_sources()
    views = {name: LOADERS[name](sources[name]) for name in datasets}
    encoder = None if audit_only else _load_encoder(device)
    output = {
        "schema_version": 3,
        "experiment": "aggregate_vs_joint_evidence_recall",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "query_prefix": QUERY_PREFIX},
        "retrieval": {
            "version": RETRIEVAL_VERSION,
            "single": "exact inner-product ranking over normalized embeddings",
            "feedback": "normalized query-plus-top1 vector",
            "fusion": "RRF over two depth-100 rankings (k0=60)",
        },
        "runtime": _runtime_versions(),
        "bootstrap": {"samples": bootstrap_samples, "seed": BOOTSTRAP_SEED},
        "datasets": {},
    }
    combined_per_query: list[dict] = []

    for name, view in views.items():
        fingerprint = _view_fingerprint(view)
        stage = CACHE / name / fingerprint[:16]
        _write_view_manifest(view, stage)
        support_counts = Counter(len(query.golds) for query in view.queries)
        row = {
            "display_name": view.display_name,
            "split": view.split,
            "source_path": view.source_path,
            "source_sha256": view.source_sha256,
            "source_provenance": SOURCE_PROVENANCE[name],
            "fingerprint": fingerprint,
            "corpus_documents": len(view.documents),
            "queries": len(view.queries),
            "support_count_distribution": {str(k): v for k, v in sorted(support_counts.items())},
        }
        if audit_only:
            output["datasets"][name] = row
            continue
        document_vectors, document_meta = _document_embeddings(
            view, stage, encoder, batch_size, reuse_existing
        )
        query_vectors, query_meta = _query_embeddings(view, stage, encoder, batch_size)
        metrics, per_query = _evaluate(
            view,
            document_vectors,
            query_vectors,
            ks,
            rank_depth,
            bootstrap_samples,
            stage,
        )
        row["embedding_cache"] = {"documents": document_meta, "queries": query_meta}
        row.update(metrics)
        output["datasets"][name] = row
        combined_per_query.extend(per_query)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "experiment_results.json").write_text(
            json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
        )

    if not audit_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "experiment_per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in combined_per_query:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (output_dir / "experiment_results.json").write_text(
            json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(LOADERS), default=sorted(LOADERS))
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--rank-depth", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="sentence-transformers device, e.g. cuda or cpu")
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="validate/describe sources without embedding")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="result directory (relative paths are resolved from the repository root)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run(
        datasets=args.datasets,
        ks=sorted(set(args.ks)),
        rank_depth=args.rank_depth,
        bootstrap_samples=args.bootstrap_samples,
        batch_size=args.batch_size,
        device=args.device,
        reuse_existing=not args.no_reuse_existing,
        audit_only=args.audit_only,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
