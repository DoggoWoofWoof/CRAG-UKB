"""FULL centralized comparison matrix: every RAG system x dataset x metric.
- Published cells carry {v, src}. Nulls carry {v:None, plan:<disposition>}.
- Disposition codes:
    PUB?   = likely reported in that system's own paper; fetch to fill (FREE)
    RERUN-R = re-runnable by us, RETRIEVAL only, NO LLM  (cheap; our metric)
    RERUN-L = re-runnable but needs an LLM (indexing or reader) -> API $  (gated)
    NA      = system/task doesn't apply to this dataset without redesign
The centralized PRIMARY metric is reader-free retrieval Recall@k. QA (F1/EM/Hit) is secondary and
only centralized under ONE fixed reader (ours) applied to each system's retrieved passages.
"""
import json

DATASETS = ["musique", "2wiki", "hotpot", "squad", "metaqa", "webqsp"]
KIND = {"musique": "text", "2wiki": "text", "hotpot": "text", "squad": "text", "metaqa": "kb", "webqsp": "kb"}

# ---------- METRIC 1: retrieval Recall@5 (text passages) / answer-entity coverage (KB) — the CENTRAL metric ----------
# text Recall@5 from HippoRAG2 paper (same corpora); KB from SubgraphRAG paper (anchored, @100 triples).
R5 = {
    # system: {dataset: value or None}
    "BM25":         {"musique": 43.5, "2wiki": 65.3, "hotpot": 74.0, "squad": None, "metaqa": None, "webqsp": None},
    "Contriever":   {"musique": 46.6, "2wiki": 57.5, "hotpot": 75.3, "squad": None, "metaqa": None, "webqsp": None},
    "GTR":          {"musique": 49.1, "2wiki": 67.9, "hotpot": 73.9, "squad": None, "metaqa": None, "webqsp": None},
    "RAPTOR":       {"musique": 57.8, "2wiki": 66.2, "hotpot": 86.9, "squad": None, "metaqa": None, "webqsp": None},
    "HippoRAG_v1":  {"musique": 53.2, "2wiki": 90.4, "hotpot": 77.3, "squad": None, "metaqa": None, "webqsp": None},
    "NV-Embed-v2":  {"musique": 69.7, "2wiki": 76.5, "hotpot": 94.5, "squad": None, "metaqa": None, "webqsp": None},
    "GFM-RAG":      {"musique": 58.2, "2wiki": 95.6, "hotpot": 87.1, "squad": None, "metaqa": None, "webqsp": None},
    "HippoRAG2":    {"musique": 74.7, "2wiki": 90.4, "hotpot": 96.3, "squad": None, "metaqa": None, "webqsp": None},
    "SiReRAG":      {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": None},
    "HopRAG":       {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": None},
    "KG2RAG":       {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": None},
    # KB retrievers (WebQSP anchored entity coverage @100 triples) — text cells NA (KGQA-only systems)
    "SubgraphRAG":  {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": 86.5},
    "GNN-RAG":      {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": 40.5},
    "RoG":          {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": 38.8},
    "G-Retriever":  {"musique": None, "2wiki": None, "hotpot": None, "squad": None, "metaqa": None, "webqsp": 32.5},
    # CRAG — _hpr for the 3 multi-hop (paper-consistent), _clean for the rest; webqsp anchored hit@50
    "CRAG (ours)":  {"musique": 72.2, "2wiki": 82.5, "hotpot": 92.5, "squad": 90.6, "metaqa": 56.7, "webqsp": 80.5},
}
R5_SRC = {"text_baselines": "HippoRAG 2 paper (arXiv 2502.14802) Table 3, same corpora",
          "kb_baselines": "SubgraphRAG paper (2410.20724), anchored entity coverage @100 triples",
          "CRAG": "ours: _hpr Recall@5 (musique/2wiki/hotpot), _clean (squad/metaqa), webqsp anchored hit@50"}

# ---------- METRIC 2: QA — Hits@1 (KB) / F1 (text). Reader varies -> NOT centralized until one reader. ----------
QA = {
    "HippoRAG2":   {"musique": ("F1", 48.6, "Llama-70B"), "2wiki": ("F1", 71.0, "Llama-70B"), "hotpot": ("F1", 75.5, "Llama-70B")},
    "GFM-RAG":     {"musique": ("F1", 40.4, "GPT-4o-mini"), "2wiki": ("F1", 77.7, "GPT-4o-mini"), "hotpot": ("F1", 66.9, "GPT-4o-mini")},
    "NV-Embed-v2": {"musique": ("F1", 45.7, "Llama-70B"), "2wiki": ("F1", 61.5, "Llama-70B"), "hotpot": ("F1", 75.3, "Llama-70B")},
    "SiReRAG":     {"musique": ("F1", 53.1, "own reader"), "2wiki": ("F1", 67.9, "own reader"), "hotpot": ("F1", 76.5, "own reader")},
    "SubgraphRAG": {"webqsp": ("Hit", 89.8, "GPT-4o")},
    "GNN-RAG":     {"webqsp": ("Hit", 85.7, "GNN+LLM")},
    "RoG":         {"webqsp": ("Hit", 82.2, "LLM")},
    "EmbedKGQA":   {"metaqa": ("Hit", 97.0, "specialized"), "webqsp": ("Hit", 66.6, "specialized")},
    "CRAG (ours)": {"musique": ("F1", 28.4, "Qwen-1.5B"), "2wiki": ("F1", 43.9, "Qwen-1.5B"),
                    "hotpot": ("F1", 53.2, "Qwen-1.5B"), "squad": ("F1", 34.1, "Qwen-1.5B"),
                    "metaqa": ("Hit/EM", 40.8, "Qwen-1.5B"), "webqsp": ("Hit/EM", 38.4, "Qwen-1.5B")},
}

# ---------- disposition for each Recall null ----------
def disposition(system, ds):
    kind = KIND[ds]
    kb_systems = {"SubgraphRAG", "GNN-RAG", "RoG", "G-Retriever"}
    text_systems = {"BM25", "Contriever", "GTR", "RAPTOR", "HippoRAG_v1", "NV-Embed-v2", "HippoRAG2"}
    cloned = {"HippoRAG2", "GFM-RAG", "SiReRAG", "HopRAG", "KG2RAG"}          # we have the code
    # cross-family = NA
    if system in kb_systems and kind == "text":
        return "NA (KGQA-only system; no text-corpus retriever)"
    if system in text_systems and kind == "kb" and system != "HippoRAG2":
        return "NA (text retriever; not run on Freebase in-paper)"
    if ds == "squad":
        return "PUB? single-hop, rarely in these papers -> likely NA / our-only"
    if system in cloned:
        if kind == "text":
            return "RERUN-L (cloned; needs LLM indexing for OpenIE/GPT-4o) — text corpora"
        return "RERUN-L (cloned; needs KG build + LLM) — KB"
    if system in {"SiReRAG", "HopRAG", "KG2RAG"}:
        return "PUB? check own paper; else RERUN-L"
    return "PUB? / RERUN-R"

matrix = {"primary_metric_recall5": {}, "qa_secondary": {}, "sources": R5_SRC,
          "note": "Central table = reader-free Recall@k. QA is reader-confounded (see 'reader' tags); "
                  "centralize QA only by running each system's retrieval through ONE fixed reader."}
for s, row in R5.items():
    matrix["primary_metric_recall5"][s] = {}
    for ds in DATASETS:
        v = row[ds]
        matrix["primary_metric_recall5"][s][ds] = ({"v": v} if v is not None
                                                   else {"v": None, "plan": disposition(s, ds)})
for s, row in QA.items():
    matrix["qa_secondary"][s] = {ds: {"metric": t[0], "v": t[1], "reader": t[2]} for ds, t in row.items()}

json.dump(matrix, open("baselines/full_matrix.json", "w"), indent=2)

# ---- print coverage summary ----
tot = filled = 0
plans = {}
for s, row in matrix["primary_metric_recall5"].items():
    for ds, cell in row.items():
        tot += 1
        if cell["v"] is not None:
            filled += 1
        else:
            key = cell["plan"].split(" ")[0]
            plans[key] = plans.get(key, 0) + 1
print(f"RECALL@5 matrix: {len(R5)} systems x {len(DATASETS)} datasets = {tot} cells")
print(f"  filled (published/ours): {filled}  ({100*filled//tot}%)")
print(f"  null dispositions:")
for k, n in sorted(plans.items(), key=lambda kv: -kv[1]):
    print(f"    {k:10s} {n}")
print("-> baselines/full_matrix.json")
