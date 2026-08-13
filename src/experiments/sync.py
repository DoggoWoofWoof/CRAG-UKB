"""
Per-task data sync spec — "upload only what the job needs".
============================================================
Both backends use this to stage the minimum inputs onto the remote (Modal
Volume / Lightning Studio FS) and to pull the right outputs back. Immutable
dataset/index inputs are uploaded skip-if-present. Mutable Level 1 cache and
checkpoint inputs are enumerated per file so a new fingerprint or run can be
staged even when its parent directory already exists remotely.
"""
from pathlib import Path
from typing import List

import numpy as np

ALL_DATASETS = ["squad_clean", "metaqa", "musique_clean", "2wiki_clean", "hotpotqa_clean"]  # canonical audited substrate
UNIFIED_DATASETS = [
    "2wiki_clean",
    "musique_clean",
    "hotpotqa_clean",
    "squad_clean",
    "metaqa",
]

_RESULT_DIR = {
    "train-coverage": "results/coverage_ablation",
    "train-hnm": "results/hnm_ablation",
    "train-loss": "results/loss_ablation",
    "train-temp": "results/temp_ablation",
    "bench-level1": "results/level_1",
    "bench-level2": "results/level_2",
    "bench-level3": "results/level_3",
}


def parse_datasets(argv: List[str], default=None) -> List[str]:
    """Extract the --datasets values from a task argv (best-effort)."""
    default = default or ALL_DATASETS
    if "--dataset" in argv:
        dataset = parse_option(argv, "--dataset")
        return [dataset] if dataset else default
    if "--datasets" in argv:
        i = argv.index("--datasets")
        vals = []
        for a in argv[i + 1:]:
            if a.startswith("-"):
                break
            vals.append(a)
        return vals or default
    return default


def parse_option(argv: List[str], name: str, default=None):
    if name not in argv:
        return default
    index = argv.index(name) + 1
    if index >= len(argv) or argv[index].startswith("-"):
        return default
    return argv[index]


def local_files(root: str) -> List[str]:
    path = Path(root)
    if not path.exists():
        return []
    return [
        item.as_posix()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def _immutable_ukb_files(dataset: str) -> List[str]:
    root = Path("data") / "ukb_storage" / dataset
    if not root.exists():
        return []
    required = {
        "bm25.pkl",
        "centroid_pids.json",
        "centroids.index",
        "graph.pt",
        "nodes.index",
        "partition_map.json",
    }
    return [
        item.as_posix()
        for item in sorted(root.iterdir())
        if item.is_file() and item.name in required
    ]


def _matching_l1_query_caches(dataset: str, limit: int) -> List[str]:
    """Stage only the largest locally reusable query-cache fingerprint."""
    root = Path("data") / "ukb_storage" / dataset / "cache" / "L1"
    candidates = []
    for cache_dir in sorted(root.glob("*")) if root.exists() else []:
        paths = [cache_dir / f"queries_{split}.npz" for split in ("train", "val", "test")]
        if not all(path.exists() for path in paths):
            continue
        try:
            counts = []
            for path in paths:
                with np.load(path, allow_pickle=False) as payload:
                    counts.append(int(payload["query_vectors"].shape[0]))
        except (KeyError, OSError, ValueError):
            continue
        if limit > 0 and any(count > limit for count in counts):
            continue
        candidates.append((sum(counts), paths))
    if not candidates:
        return []
    best_size = max(size for size, _ in candidates)
    return [
        path.as_posix()
        for size, paths in candidates
        if size == best_size
        for path in paths
    ]


def required_inputs(task_name: str, argv: List[str]) -> List[str]:
    """Repo-relative input paths the task must read on the remote."""
    if task_name == "review-metric":
        source_paths = {
            "2wiki": "data/raw/review_public/2wiki_dev.parquet",
            "musique": "data/raw/review_public/musique_v1.0/data/musique_ans_v1.0_dev.jsonl",
            "hotpotqa": "data/raw/review_public/hotpot_dev_distractor.jsonl",
        }
        datasets = parse_datasets(argv, sorted(source_paths))
        return [source_paths[dataset] for dataset in datasets]
    if task_name == "rebuild-clean":                          # upload raw official data; build happens remotely
        ds = parse_option(argv, "--dataset")
        if ds == "squad":
            return ["data/raw/squad_v2.json"]
        if ds == "hotpotqa":
            return ["data/raw/review_public/hotpot_dev_distractor.jsonl",
                    "data/raw/review_public/hotpot_train_distractor.jsonl"]
        return []
    default_datasets = UNIFIED_DATASETS if task_name == "l1-unified" else ALL_DATASETS
    ds = parse_datasets(argv, default_datasets)
    if task_name == "ukb-build":                          # upload the specified master (per-dataset supported)
        return [parse_option(argv, "--nodes", "data/processed/master_nodes.json")]
    if task_name == "l1-unified":
        paths = []
        limit = int(parse_option(argv, "--limit", 15000))
        for dataset in ds:
            paths.extend(_immutable_ukb_files(dataset))
            paths.extend(_matching_l1_query_caches(dataset, limit))
            paths.append(f"data/processed/master_nodes_{dataset}.json")
        if "metaqa" in ds:
            paths.append("data/processed/master_nodes.json")
        run_id = parse_option(argv, "--run-id")
        if run_id:
            paths.extend(
                local_files(
                    f"data/ukb_storage/_shared/checkpoints/L1/{run_id}"
                )
            )
        return paths
    if task_name in (
        "l1-optimize",
        "sota-baselines",
        "sota-e2e",
        "l3-traverse",
        "l3-methods",
        "baselines-rag",
        "l1-attn-fusion",
        "l1-mlpt-improve",
        "l1-struct-scorer",
        "l1-error-analysis",
        "l1-partition-probe",
        "l1-partition-router",
        "l1-partition-mlpt",
        "l1-partition-select",
        "l1-unified-ranker",
        "l1-rerank100",
        "l1-universal-head",
        "l2-seed",
        "l2-learned-fusion",
        "e2e-ner",
        "l2-mem-bench",
        "l1-candgen",
        "l3-graphlift",
        "l1-overlap-test",
        "encoder-graph",
        "reencode-ukb",
        "partsize-ablation",
        "train-coverage",
        "splade-encode",
    ):
        paths = []                                        # graph/index + per-source master
        for d in ds:
            paths.append(f"data/ukb_storage/{d}")
            if task_name in ("l1-universal-head", "l2-seed", "l2-learned-fusion", "e2e-ner", "l2-mem-bench", "l1-candgen", "l3-graphlift", "l1-overlap-test", "encoder-graph"):   # dir-level skip misses this subdir on stale volumes
                paths.append(f"data/ukb_storage/{d}/{parse_option(argv, '--subdir', 'bge_large')}")
            if task_name in ("l1-optimize", "sota-baselines"):
                paths.extend(local_files(f"data/ukb_storage/{d}/cache/L1"))
                paths.extend(local_files(f"data/ukb_storage/{d}/cache/SOTA"))
                run_id = parse_option(argv, "--run-id")
                if task_name == "l1-optimize" and run_id:
                    paths.extend(
                        local_files(
                            f"data/ukb_storage/{d}/checkpoints/L1/{run_id}"
                        )
                    )
            paths.append(f"data/processed/master_nodes_{d}.json")
            if task_name == "sota-e2e":
                paths.extend(
                    local_files(
                        f"data/ukb_storage/_sota/sota_e2e_v1/bundles/{d}"
                    )
                )
        # MetaQA still lives in the canonical all-source master file.
        if task_name in (
            "l1-optimize",
            "l1-unified",
            "sota-baselines",
            "sota-e2e",
            "l1-attn-fusion",
            "l1-mlpt-improve",
            "l1-struct-scorer",
            "l1-error-analysis",
            "l1-partition-probe",
            "l1-partition-router",
            "l1-partition-mlpt",
            "l1-partition-select",
            "l1-unified-ranker",
            "l1-rerank100",
            "l1-universal-head",
            "l2-seed",
            "l2-learned-fusion",
            "e2e-ner",
            "l2-mem-bench",
            "l1-candgen",
            "l3-graphlift",
            "l1-overlap-test",
            "encoder-graph",
            "reencode-ukb",
            "partsize-ablation",
            "train-coverage",
            "splade-encode",
        ) and "metaqa" in ds:
            paths.append("data/processed/master_nodes.json")
        return paths
    paths = ["data/processed/master_nodes.json"]
    for d in ds:
        paths.append(f"data/ukb_storage/{d}")            # indexes (+ colbert/splade embs)
        paths.append(f"checkpoints/{d}/hnm_ablation")    # frozen KL baseline / prior models
    return paths


def result_outputs(task_name: str, argv: List[str]) -> List[str]:
    """Repo-relative output paths to pull back after the task finishes."""
    if task_name == "review-metric":
        return [parse_option(argv, "--output-dir", "results/review_metric_v2")]
    if task_name == "train-gnn":                              # pull the GNN coverage results (not the default checkpoints)
        return ["results/gnn_ablation"]
    if task_name == "rebuild-clean":                          # pull rebuilt substrate + clean master
        ds = parse_option(argv, "--dataset")
        clean = f"{ds}_clean"
        return [f"data/ukb_storage/{clean}", f"data/processed/master_nodes_{clean}.json"]
    if task_name == "encoder-graph":                          # pull the per-encoder graph/partition/centroids subdir
        eg_ds = parse_datasets(argv)
        subdir = parse_option(argv, "--subdir", "gte_qwen")
        return [f"data/ukb_storage/{d}/{subdir}" for d in eg_ds]
    if task_name == "partsize-ablation":                       # nodes-per-partition sweep results
        return ["results/overlap_partsize"]
    if task_name == "splade-encode":                           # pull the SPLADE doc matrix
        return [f"data/ukb_storage/{d}/splade_doc_embs.pkl" for d in parse_datasets(argv)]
    default_datasets = UNIFIED_DATASETS if task_name == "l1-unified" else ALL_DATASETS
    ds = parse_datasets(argv, default_datasets)
    if task_name == "ukb-build":                          # source inferred from the --nodes master (master_nodes_webqsp.json -> webqsp)
        nodes = parse_option(argv, "--nodes", "data/processed/master_nodes.json")
        src = Path(nodes).stem.replace("master_nodes_", "")
        return [f"data/ukb_storage/{src}"] if src and src != "master_nodes" else [f"data/ukb_storage/{d}" for d in ds]
    if task_name == "l1-universal-head":                  # subdir-specific JSON at repo-root results/
        return [f"results/L1_universal_head_{parse_option(argv, '--subdir', 'bge_large')}.json"]
    if task_name in ("l2-seed", "l2-learned-fusion", "e2e-ner", "l2-mem-bench", "l1-candgen", "l3-graphlift", "l1-overlap-test"):   # pull the whole L2 results dir
        return ["results/L2"]
    if task_name == "reencode-ukb":                       # pull the re-encoded subdir (bge/e5 embeddings)
        subdir = parse_option(argv, "--subdir", "bge_base")
        return [f"data/ukb_storage/{d}/{subdir}" for d in ds]
    if task_name == "l1-finetune-encoder":                # pull the fine-tuned encoder subdir (nodes + cached queries)
        subdir = parse_option(argv, "--subdir", "ft_bge")
        return [f"data/ukb_storage/{d}/{subdir}" for d in ds]
    if task_name in ("l3-traverse", "l3-methods"):
        return [f"data/ukb_storage/{d}/results/L3" for d in ds]
    if task_name == "baselines-rag":
        return [f"data/ukb_storage/{d}/results/baselines" for d in ds]
    if task_name == "l1-partition-mlpt":
        # dedicated explore dir -> pulling it can never clobber canonical results/L1 files
        return [f"data/ukb_storage/{d}/results/L1_explore" for d in ds]
    if task_name in ("l1-partition-select", "l1-unified-ranker", "l1-rerank100"):
        # separate dir again -> no race with a concurrent partition-mlpt pull on same dataset
        return [f"data/ukb_storage/{d}/results/L1_select" for d in ds]
    if task_name in ("l1-attn-fusion", "l1-mlpt-improve", "l1-struct-scorer", "l1-error-analysis",
                     "l1-partition-probe", "l1-partition-router"):
        return [f"data/ukb_storage/{d}/results/L1" for d in ds] + ["data/ukb_storage/_index"]
    if task_name == "sota-baselines":
        return [
            f"data/ukb_storage/{d}/results/baselines" for d in ds
        ] + [
            f"data/ukb_storage/{d}/cache/L1" for d in ds
        ] + [
            f"data/ukb_storage/{d}/cache/SOTA" for d in ds
        ] + [
            "data/ukb_storage/_index"
        ]
    if task_name == "sota-e2e":
        return ["data/ukb_storage/_sota/sota_e2e_v1"]
    if task_name == "l1-optimize":
        return [
            f"data/ukb_storage/{d}/results/L1" for d in ds
        ] + [
            f"data/ukb_storage/{d}/checkpoints/L1" for d in ds
        ] + [
            f"data/ukb_storage/{d}/cache/L1" for d in ds
        ] + [
            "data/ukb_storage/_index"
        ]
    if task_name == "l1-unified":
        return [
            f"data/ukb_storage/{d}/results/L1" for d in ds
        ] + [
            f"data/ukb_storage/{d}/cache/L1" for d in ds
        ] + [
            "data/ukb_storage/_shared/results/L1",
            "data/ukb_storage/_shared/checkpoints/L1",
            "data/ukb_storage/_index",
        ]
    outs = []
    if task_name in _RESULT_DIR:
        outs.append(_RESULT_DIR[task_name])
    if task_name.startswith("train"):
        # trained/updated checkpoints (coverage models, ablation grid) persist here
        for d in ds:
            outs.append(f"checkpoints/{d}/hnm_ablation")
    return outs
