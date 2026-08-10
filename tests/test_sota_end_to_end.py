import json
import subprocess
from types import SimpleNamespace

import pytest
import src.experiments.sota_end_to_end as sota_end_to_end

from src.evaluation.sota_contract import (
    answer_exact_match,
    answer_token_f1,
    compare_end_to_end,
    evaluate_end_to_end,
    validate_run_rows,
)
from src.evaluation.external_sota_adapter import (
    KG2RAG_PAPER_LLM_DIGEST,
    _kg2rag_model_record,
    _parse_kg2rag_triplets,
    _patch_hoprag_neo4j,
    _patch_kg2rag_paper_triplet_reranker,
    _prepare_hoprag_inputs,
    _prepare_kg2rag_inputs,
    _prepare_raptor_inputs,
    _rank_raptor_documents,
    _raptor_node_documents,
    _require_hoprag_runtime,
    _require_kg2rag_runtime,
    _require_raptor_summary_runtime,
    _paper_source_contents,
    _round_robin_unique,
)
from src.experiments.sota_end_to_end import (
    _audit_bundle,
    _audit_method,
    _configured_stages,
    _environment_install_signature,
    _extract_retrieved_ids,
    _freeze_environment,
    _requirement_status,
    hydrate_retrieval,
    lock_repository,
)


def test_answer_metrics_match_standard_normalization():
    assert answer_exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert answer_token_f1("Paris, France", ["Paris"]) == pytest.approx(2 / 3)


def test_end_to_end_metrics_cover_retrieval_answer_and_efficiency():
    rows = [
        {
            "id": "q1",
            "question": "Where?",
            "answers": ["Paris"],
            "supporting_document_ids": ["a", "b"],
            "retrieved_document_ids": ["a", "x", "b"],
            "contexts": [{"text": "The answer is Paris."}],
            "prediction": "Paris",
            "latency_ms": {"retrieval": 10, "generation": 20, "total": 30},
            "usage": {"prompt_tokens": 100, "completion_tokens": 2},
        },
        {
            "id": "q2",
            "question": "Who?",
            "answers": ["Ada"],
            "supporting_document_ids": ["c"],
            "retrieved_document_ids": ["z"],
            "contexts": [{"text": "No relevant evidence."}],
            "prediction": "Grace",
            "latency_ms": {"retrieval": 30, "generation": 40, "total": 70},
            "usage": {"prompt_tokens": 120, "completion_tokens": 2},
        },
    ]

    summary, per_query = evaluate_end_to_end(
        rows,
        ks=(2, 100),
        bootstrap_samples=20,
        pricing={"prompt": 1.0, "completion": 2.0},
    )

    assert summary["retrieval"]["recall"]["100"] == 50.0
    assert summary["retrieval"]["full_coverage"]["100"] == 50.0
    assert summary["retrieval"]["hit_rate"]["100"] == 50.0
    assert summary["answers"]["answer_em"] == 50.0
    assert summary["answers"]["answer_f1"] == 50.0
    assert summary["answers"]["joint_f1"] == 40.0
    assert summary["grounding"]["answer_in_context"] == 50.0
    assert summary["efficiency"]["total_latency_ms"]["p50"] == 50.0
    assert summary["efficiency"]["prompt_tokens"]["total"] == 220
    assert summary["efficiency"]["generation_cost_usd"] == pytest.approx(0.000228)
    assert len(per_query) == 2


def test_contract_rejects_duplicate_ids_and_retrieved_documents():
    with pytest.raises(ValueError, match="duplicate retrieved"):
        validate_run_rows(
            [
                {
                    "id": "q1",
                    "question": "Q",
                    "supporting_document_ids": ["a"],
                    "retrieved_document_ids": ["a", "a"],
                }
            ]
        )

    with pytest.raises(ValueError, match="Duplicate query id"):
        validate_run_rows(
            [
                {
                    "id": "q1",
                    "question": "Q",
                    "supporting_document_ids": ["a"],
                    "retrieved_document_ids": ["a"],
                },
                {
                    "id": "q1",
                    "question": "Q2",
                    "supporting_document_ids": ["b"],
                    "retrieved_document_ids": ["b"],
                },
            ]
        )


def test_hydrate_joins_external_ids_to_canonical_bundle(tmp_path):
    output_root = tmp_path / "suite"
    bundle = output_root / "bundles" / "toy" / "abc"
    canonical = bundle / "canonical"
    (canonical / "queries").mkdir(parents=True)
    (canonical / "documents.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "d1", "title": "One", "text": "Alpha"}),
                json.dumps({"id": "d2", "title": "Two", "text": "Beta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (canonical / "queries" / "test.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Question",
                "answers": ["Alpha"],
                "supporting_document_ids": ["d1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"fingerprint": "full-fingerprint"}),
        encoding="utf-8",
    )
    pointer = output_root / "bundles" / "toy" / "latest.json"
    pointer.write_text(
        json.dumps({"bundle_dir": bundle.as_posix()}),
        encoding="utf-8",
    )
    source = tmp_path / "external.jsonl"
    source.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "doc_ids": ["d2", "missing", "d1"],
                "latency_ms": {"retrieval": 12.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "hydrated.jsonl"
    config = {
        "suite": {
            "output_root": output_root.as_posix(),
            "retrieval_k": [2, 100],
            "context_document_budget": 1,
        }
    }

    manifest = hydrate_retrieval(config, "method", "toy", source, output)
    row = json.loads(output.read_text(encoding="utf-8").strip())

    assert manifest["rows"] == 1
    assert row["retrieved_document_ids"] == ["d2", "d1"]
    assert [context["document_id"] for context in row["contexts"]] == ["d2"]
    assert row["supporting_document_ids"] == ["d1"]


def test_sota_audit_verifies_bundle_hashes_and_detects_corruption(tmp_path):
    output_root = tmp_path / "suite"
    bundle = output_root / "bundles" / "toy" / "abc"
    bundle.mkdir(parents=True)
    artifact = bundle / "documents.jsonl"
    artifact.write_text('{"id":"d1"}\n', encoding="utf-8")
    import hashlib

    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = bundle / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fingerprint": "fingerprint",
                "labels_excluded_from_index": True,
                "questions_excluded_from_index": True,
                "artifacts": {"documents.jsonl": expected},
            }
        ),
        encoding="utf-8",
    )
    pointer = output_root / "bundles" / "toy" / "latest.json"
    pointer.write_text(
        json.dumps(
            {
                "bundle_dir": bundle.as_posix(),
                "manifest": manifest.as_posix(),
                "fingerprint": "fingerprint",
            }
        ),
        encoding="utf-8",
    )

    assert _audit_bundle(output_root, "toy", verify_hashes=True)["ready"]
    artifact.write_text('{"id":"changed"}\n', encoding="utf-8")
    corrupted = _audit_bundle(output_root, "toy", verify_hashes=True)
    assert not corrupted["ready"]
    assert corrupted["hash_mismatches"] == ["documents.jsonl"]


def test_sota_readiness_reports_prepare_and_only_secret_presence(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "do-not-return-this-value")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    method = {
        "requirements": ["PRESENT_KEY", "MISSING_KEY"],
        "matched": {
            "prepare_command": "prepare",
            "retrieve_command": "retrieve",
        },
    }

    assert _configured_stages(method, "matched") == ["prepare", "retrieve"]
    assert _requirement_status(method) == {
        "PRESENT_KEY": True,
        "MISSING_KEY": False,
    }


def test_environment_lock_falls_back_to_uv_when_venv_has_no_pip(
    monkeypatch,
    tmp_path,
):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **_):
        calls.append(list(map(str, command)))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="package==1.0\n")

    monkeypatch.setattr(sota_end_to_end, "_environment_python", lambda _: python)
    monkeypatch.setattr(sota_end_to_end, "_run_checked", fake_run)
    monkeypatch.setattr(sota_end_to_end.shutil, "which", lambda _: "uv")
    destination = tmp_path / "environment.lock.txt"

    digest = _freeze_environment(tmp_path, destination)

    assert destination.read_text(encoding="utf-8") == "package==1.0\n"
    assert digest
    assert calls[1] == [
        "uv",
        "pip",
        "freeze",
        "--python",
        str(python),
    ]


def test_repository_relock_preserves_verified_installed_environment(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "external"
    (repository / ".git").mkdir(parents=True)
    python = repository / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    commit = "a" * 40
    specification = {
        "repository": "https://example.com/repository.git",
        "commit": commit,
        "python": "3.10",
        "install": ["install command"],
    }
    (repository / "crag_prepared.json").write_text(
        json.dumps(
            {
                "installed": True,
                "install_signature": _environment_install_signature(specification),
                "environment_lock_sha256": "verified-lock",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "_sha256": "current-config",
        "suite": {"external_root": str(tmp_path / "root")},
        "methods": {"method": specification},
    }

    def fake_run(command, **_):
        if command[:4] == ["git", "remote", "get-url", "origin"]:
            return SimpleNamespace(stdout=specification["repository"] + "\n")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=commit + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(sota_end_to_end, "_repo_path", lambda *_: repository)
    monkeypatch.setattr(sota_end_to_end, "_run_checked", fake_run)
    monkeypatch.setattr(
        sota_end_to_end,
        "_freeze_environment",
        lambda *_: "verified-lock",
    )

    prepared = lock_repository(config, "method")

    assert prepared["installed"]
    assert prepared["environment_lock_sha256"] == "verified-lock"
    assert prepared["config_sha256"] == "current-config"


def test_sota_audit_detects_stale_internal_source_fingerprint(tmp_path):
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    external_root = tmp_path / "external"
    prepared = external_root / "internal" / "crag_prepared.json"
    prepared.parent.mkdir(parents=True)
    prepared.write_text(
        json.dumps(
            {
                "commit": "a" * 40,
                "config_sha256": "config-sha",
                "source_fingerprint": "stale",
                "installed": True,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "_sha256": "config-sha",
        "suite": {
            "id": "suite",
            "datasets": ["toy"],
            "output_root": (tmp_path / "output").as_posix(),
            "external_root": external_root.as_posix(),
        },
        "methods": {
            "crag": {
                "internal": True,
                "repository": repository.as_posix(),
                "commit": "a" * 40,
                "matched": {"retrieve_command": "retrieve"},
            }
        },
    }

    audit = _audit_method(config, "crag", ["toy"])
    assert not audit["source_current"]
    assert "internal_source_fingerprint_stale" in audit["blockers"]
    assert not audit["launch_ready"]


def test_gfm_nested_retrieval_documents_are_normalized():
    row = {
        "retrieved_docs": {
            "document": [
                {"id": "d1"},
                {"document_id": "d2"},
                {"title": "d3"},
            ]
        }
    }

    assert _extract_retrieved_ids(row) == ["d1", "d2", "d3"]


def test_lightrag_paper_context_parser_and_stable_fusion():
    context = '''
-----Sources-----
```csv
"id","content"
0,"first line
second line"
1,"another source"
```
'''

    assert _paper_source_contents(context) == [
        "first line\nsecond line",
        "another source",
    ]
    assert _round_robin_unique(
        ["high-1", "shared", "high-2"],
        ["low-1", "shared", "low-2"],
    ) == ["high-1", "low-1", "shared", "high-2", "low-2"]


def test_hoprag_matched_input_uses_only_label_free_graph_neighborhoods(tmp_path):
    corpus = tmp_path / "documents.jsonl"
    edges = tmp_path / "edges.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"id": "d1", "title": "One", "text": "Alpha"}),
                json.dumps({"id": "d2", "title": "Two", "text": "Beta"}),
                json.dumps({"id": "d3", "title": "Three", "text": "Gamma"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    edges.write_text(
        "\n".join(
            [
                json.dumps({"source": "d1", "target": "d2"}),
                json.dumps({"source": "d1", "target": "d3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = _prepare_hoprag_inputs(
        corpus,
        edges,
        tmp_path / "hoprag",
        dataset_alias="toy",
        group_size=2,
    )
    problems = [
        json.loads(line)
        for line in open(prepared["problems"], encoding="utf-8")
        if line.strip()
    ]

    assert prepared["document_count"] == 3
    assert prepared["neighborhood_count"] == 2
    assert all(row["question"] == "" for row in problems)
    assert all(row["answer"] == "" for row in problems)
    assert all(row["supporting_facts"] == [] for row in problems)
    assert _prepare_hoprag_inputs(
        corpus,
        edges,
        tmp_path / "hoprag",
        dataset_alias="toy",
        group_size=2,
    ) == prepared


def test_hoprag_paid_openai_endpoint_is_fail_closed(monkeypatch):
    for name in (
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "HOPRAG_LLM_BASE_URL",
        "HOPRAG_LLM_API_KEY",
        "HOPRAG_LLM_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="secret",
        neo4j_database="neo4j",
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="secret",
        llm_model="gpt-4o-mini",
        allow_paid_api=False,
    )

    with pytest.raises(RuntimeError, match="paid OpenAI API"):
        _require_hoprag_runtime(args)


def test_hoprag_neo4j_compatibility_shim_moves_database_to_session():
    calls = {}

    class FakeDriver:
        def session(self, *args, **kwargs):
            calls["session"] = (args, kwargs)
            return "session"

    class FakeGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            calls["driver"] = (args, kwargs)
            return FakeDriver()

    module = SimpleNamespace(GraphDatabase=FakeGraphDatabase)
    _patch_hoprag_neo4j(module, "neo4j")
    driver = module.GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "secret"),
        database="research",
    )

    assert driver.session() == "session"
    assert "database" not in calls["driver"][1]
    assert calls["session"][1]["database"] == "research"


def test_kg2rag_preparation_preserves_hotpot_sentence_boundaries(tmp_path):
    corpus = tmp_path / "documents.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "d1",
                        "title": "One",
                        "text": "First sentence.  Second sentence.",
                        "answer": "must not leak",
                    }
                ),
                json.dumps(
                    {
                        "id": "d2",
                        "title": "Two",
                        "text": "Fallback one. Fallback two.",
                        "supporting_facts": [["Two", 0]],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = _prepare_kg2rag_inputs(corpus, tmp_path / "kg2rag")
    rows = [
        json.loads(line)
        for line in open(prepared["chunks"], encoding="utf-8")
        if line.strip()
    ]

    assert prepared["document_count"] == 2
    assert prepared["chunk_count"] == 4
    assert prepared["split_document_counts"] == {
        "deterministic_punctuation_fallback": 1,
        "preserved_hotpot_whitespace": 1,
    }
    assert rows[0]["retrieval_text"] == "One: First sentence."
    assert rows[0]["extraction_text"] == "First sentence."
    assert rows[1]["extraction_text"] == "One: Second sentence."
    assert all("answer" not in row and "supporting_facts" not in row for row in rows)
    assert _prepare_kg2rag_inputs(corpus, tmp_path / "kg2rag") == prepared


def test_kg2rag_triplet_parser_matches_author_filters():
    response = (
        "<Ada Lovelace##born in##London>$$"
        "<Ada Lovelace##occupation##mathematician>$$"
        "<Unknown person##relation##null>$$"
        "<same##relation##same>$$"
    )

    assert _parse_kg2rag_triplets(
        response,
        "Ada Lovelace was a mathematician born in London.",
    ) == [
        ["Ada Lovelace", "born in", "London"],
        ["Ada Lovelace", "occupation", "mathematician"],
    ]


def test_kg2rag_model_digest_and_paper_triplet_reranker_are_locked():
    record = _kg2rag_model_record(
        {
            "models": [
                {
                    "name": "llama3:8b",
                    "model": "llama3:8b",
                    "digest": KG2RAG_PAPER_LLM_DIGEST,
                }
            ]
        },
        "llama3:8b",
        KG2RAG_PAPER_LLM_DIGEST,
    )
    assert record["digest"] == KG2RAG_PAPER_LLM_DIGEST

    class FakeReranker:
        def __init__(self):
            self.pairs = None

        def compute_score(self, pairs, **_):
            self.pairs = pairs
            return [1.0]

    reranker = _patch_kg2rag_paper_triplet_reranker(FakeReranker())
    reranker.compute_score(
        [
            (
                "question",
                "document text Relational facts: Ada has/is occupation mathematician.",
            )
        ]
    )
    assert reranker.pairs == [
        ("question", "Ada has/is occupation mathematician")
    ]


def test_kg2rag_nonlocal_ollama_endpoint_is_fail_closed():
    args = SimpleNamespace(
        ollama_base_url="https://models.example.com",
        allow_remote_ollama=False,
        allow_model_substitution=False,
        llm_model="llama3:8b",
        llm_digest=KG2RAG_PAPER_LLM_DIGEST,
        embedding_model="mxbai-embed-large:latest",
        embedding_digest="468836162de7",
        max_extraction_calls=0,
    )

    with pytest.raises(RuntimeError, match="fail-closed"):
        _require_kg2rag_runtime(args, require_llm=True)


def test_raptor_preparation_preserves_boundaries_without_labels(tmp_path):
    corpus = tmp_path / "documents.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "d1",
                        "title": "One",
                        "text": "Alpha.",
                        "answer": "must not leak",
                    }
                ),
                json.dumps(
                    {
                        "id": "d2",
                        "title": "Two",
                        "text": "Beta.",
                        "supporting_facts": [["Two", 0]],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = _prepare_raptor_inputs(corpus, tmp_path / "raptor")
    rows = [
        json.loads(line)
        for line in open(prepared["documents"], encoding="utf-8")
        if line.strip()
    ]

    assert rows == [
        {"id": "d1", "text": "One\nAlpha."},
        {"id": "d2", "text": "Two\nBeta."},
    ]
    assert prepared["document_boundaries_preserved"]
    assert not prepared["questions_used"]
    assert not prepared["labels_used"]
    assert _prepare_raptor_inputs(corpus, tmp_path / "raptor") == prepared


def test_raptor_provenance_handles_soft_cluster_dag():
    nodes = {
        0: SimpleNamespace(children=set()),
        1: SimpleNamespace(children=set()),
        2: SimpleNamespace(children={0, 1}),
        3: SimpleNamespace(children={1}),
        4: SimpleNamespace(children={2, 3}),
    }

    provenance = _raptor_node_documents(nodes, {0: "d1", 1: "d2"})

    assert provenance[2] == ("d1", "d2")
    assert provenance[3] == ("d2",)
    assert provenance[4] == ("d1", "d2")


def test_raptor_document_projection_combines_tree_and_dense_rrf():
    import numpy as np

    ranked = _rank_raptor_documents(
        selected_node_ids=[20, 21],
        node_position={20: 0, 21: 1},
        offsets=np.asarray([0, 2, 3]),
        provenance_indices=np.asarray([0, 1, 1]),
        dense_document_positions=[2, 0, 1],
        document_ids=["d1", "d2", "d3"],
        top_k=3,
    )

    assert ranked == ["d2", "d1", "d3"]


def test_raptor_remote_summary_endpoint_is_fail_closed(monkeypatch):
    for name in (
        "RAPTOR_LLM_BASE_URL",
        "RAPTOR_LLM_API_KEY",
        "RAPTOR_LLM_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="secret",
        llm_model="gpt-3.5-turbo",
        allow_paid_api=False,
        allow_model_substitution=False,
        max_summary_calls=0,
    )

    with pytest.raises(RuntimeError, match="fail-closed"):
        _require_raptor_summary_runtime(args)


def test_raptor_summary_model_substitution_requires_explicit_flag(monkeypatch):
    for name in (
        "RAPTOR_LLM_BASE_URL",
        "RAPTOR_LLM_API_KEY",
        "RAPTOR_LLM_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        llm_base_url="http://localhost:8000/v1",
        llm_api_key="local",
        llm_model="different-model",
        allow_paid_api=False,
        allow_model_substitution=False,
        max_summary_calls=0,
    )

    with pytest.raises(RuntimeError, match="gpt-3.5-turbo"):
        _require_raptor_summary_runtime(args)


def test_raptor_remote_summary_endpoint_requires_call_budget(monkeypatch):
    for name in (
        "RAPTOR_LLM_BASE_URL",
        "RAPTOR_LLM_API_KEY",
        "RAPTOR_LLM_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    args = SimpleNamespace(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="secret",
        llm_model="gpt-3.5-turbo",
        allow_paid_api=True,
        allow_model_substitution=False,
        max_summary_calls=0,
    )

    with pytest.raises(RuntimeError, match="max-summary-calls"):
        _require_raptor_summary_runtime(args)


def test_paired_end_to_end_comparison_uses_identical_queries():
    common = {
        "question": "Where?",
        "answers": ["Paris"],
        "supporting_document_ids": ["a", "b"],
        "contexts": [{"text": "Paris"}],
        "latency_ms": {"retrieval": 10, "generation": 20, "total": 30},
    }
    baseline = [
        {
            **common,
            "id": "q1",
            "retrieved_document_ids": ["a", "x"],
            "prediction": "London",
        }
    ]
    treatment = [
        {
            **common,
            "id": "q1",
            "retrieved_document_ids": ["a", "b"],
            "prediction": "Paris",
        }
    ]

    summary, per_query = compare_end_to_end(
        baseline,
        treatment,
        ks=(2,),
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    assert summary["n_paired_queries"] == 1
    assert summary["metrics"]["full_coverage@2"]["delta"]["mean"] == 100.0
    assert summary["metrics"]["answer_em"]["test"] == "mcnemar_exact"
    assert per_query[0]["deltas"]["answer_f1"] == 100.0


def test_paired_comparison_rejects_nonidentical_ids():
    row = {
        "id": "q1",
        "question": "Q",
        "answers": ["A"],
        "supporting_document_ids": ["d1"],
        "retrieved_document_ids": ["d1"],
    }
    with pytest.raises(ValueError, match="identical query IDs"):
        compare_end_to_end(
            [row],
            [{**row, "id": "q2"}],
            ks=(1,),
            bootstrap_samples=10,
        )
