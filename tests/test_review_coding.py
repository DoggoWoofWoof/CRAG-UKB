import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_consistency_audit_is_reproducible_and_reconciled():
    module = _load_module(
        "review_consistency_audit", ROOT / "review_paper" / "consistency_audit.py"
    )
    result = module.audit()
    assert result["sample_papers"] == [
        "FLARE",
        "GNN-RAG",
        "HippoRAG2",
        "IRCoT",
        "MDR",
        "SimGRAG",
        "Toolformer",
    ]
    assert result["cells"] == 70
    assert result["disagreement_count"] == 2
    assert abs(result["raw_agreement"] - 68 / 70) < 1e-12
    assert abs(result["pooled_cohen_kappa"] - 0.9337748344370861) < 1e-12


def test_family_relevant_denominators():
    module = _load_module("review_make_figure", ROOT / "review_paper" / "make_figure.py")
    family = module.relevant_family_stats(module.load())
    assert family["iterative_agentic"] == {
        "N": 15,
        "A2_stop_decision": 1,
        "A3_recovery": 0,
    }
    assert family["tool_agents"] == {"N": 5, "A5_tool_error": 0}
