"""
Task registry — the single catalogue of runnable experiments.
==============================================================
Each task is a backend-agnostic body (operates on the local filesystem) plus a
resource hint used for auto-routing: `gpu` -> Modal, `cpu` -> Lightning AI.
Bodies reuse the tested primitives in src/ (no compute logic lives here).
"""
import sys
import logging
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List

log = logging.getLogger("experiments.tasks")


@dataclass
class Task:
    name: str
    resource: str          # "gpu" | "cpu"
    help: str
    body: Callable[[List[str]], None]
    # Tiny end-to-end preset used by `experiments.py smoke <task>` (local, fast).
    smoke_args: List[str] = field(default_factory=list)


def _subprocess(module: str, argv: List[str]) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    log.info("exec: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"{module} exited with code {rc}")


def _script(path: str, argv: List[str]) -> None:
    cmd = [sys.executable, path, *argv]
    log.info("exec: %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"{path} exited with code {rc}")


# ── Task bodies ──────────────────────────────────────────────────────────────
def _ukb_build(argv):
    _subprocess("src.core.indexers", argv)          # builds per-source UKB views


def _train_coverage(argv):
    from src.experiments import coverage
    coverage.main(argv)


def _train_hnm(argv):
    from src.experiments import ablations
    ablations.main(["hnm", *argv])


def _train_loss(argv):
    from src.experiments import ablations
    ablations.main(["loss", *argv])


def _train_temp(argv):
    from src.experiments import ablations
    ablations.main(["temp", *argv])


def _bench_level1(argv):
    from src.evaluation.benchmark_partition_selection import run_benchmark
    p = argparse.ArgumentParser(prog="bench-level1")
    p.add_argument("--datasets", nargs="+", default=["squad_clean", "metaqa", "musique_clean", "2wiki_clean", "hotpotqa_clean"])
    p.add_argument("--limit", type=int, default=0, help="Cap queries per split (0 = all).")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of routing methods (default all). Use to skip the slow "
                        "SPLADE arm when running on CPU.")
    a = p.parse_args(argv)
    for ds in a.datasets:
        run_benchmark(dataset=ds, limit=a.limit, methods_override=a.methods)


def _bench_level2(argv):
    from src.evaluation.level2 import run_level2, ALL_METHODS
    p = argparse.ArgumentParser(prog="bench-level2")
    p.add_argument("--datasets", nargs="+", default=["squad_clean", "metaqa", "musique_clean", "2wiki_clean", "hotpotqa_clean"])
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    a = p.parse_args(argv)
    run_level2(a.datasets, methods=a.methods)


def _bench_level3(argv):
    _subprocess("src.evaluation.benchmark_level3", argv)


def _probe_overlap(argv):
    from src.experiments import overlap_probe
    overlap_probe.main(argv)


def _overlap_retrain(argv):
    from src.experiments import overlap_retrain
    overlap_retrain.main(argv)


def _finetune_encoder(argv):
    from src.experiments import encoder_finetune
    encoder_finetune.main(argv)


def _train_gnn(argv):
    from src.experiments import train_gnn
    train_gnn.main(argv)


def _adaptive_k(argv):
    from src.experiments import adaptive_k
    adaptive_k.main(argv)


def _stack_best(argv):
    from src.experiments import stack_best
    stack_best.main(argv)


def _multiproto(argv):
    from src.experiments import multiproto
    multiproto.main(argv)


def _reencode_ukb(argv):
    from src.experiments import reencode_ukb
    reencode_ukb.main(argv)


def _rebuild_clean(argv):
    """Full clean-substrate rebuild for an audited-broken source (2026-08 gold fix):
    loaders -> build_clean master -> build_all substrate -> reencode (bge_large + gte_qwen).
    Resumable: each stage is skipped when its output already exists, and the Volume is
    committed after each stage so a mid-run failure never loses a completed stage."""
    import os
    from src.pipeline import rebuild_dataset
    from src.core.indexers import build_all
    from src.experiments import reencode_ukb
    p = argparse.ArgumentParser(prog="rebuild-clean")
    p.add_argument("--dataset", required=True, choices=["squad", "hotpotqa"])
    p.add_argument("--batch", type=int, default=32, help="reencode batch (gte-Qwen2 is memory-heavy)")
    p.add_argument("--max-seq", type=int, default=None, help="cap encoder seq length (bounds gte memory on rare long docs)")
    p.add_argument("--target-per-partition", type=int, default=100,
                   help="METIS granularity; 100 = canonical clean-substrate design (matches repartition.py)")
    p.add_argument("--force", action="store_true", help="rebuild every stage even if outputs already exist")
    a = p.parse_args(argv)
    src, clean = a.dataset, f"{a.dataset}_clean"
    try:
        from src.experiments.backends import commit_persistent_storage as _commit
    except Exception:                                         # local run: no Volume to commit
        _commit = lambda: None

    clean_master = f"data/processed/master_nodes_{clean}.json"
    if a.force or not os.path.exists(clean_master):
        log.info("[rebuild-clean] stage 1/4: build clean master for %s", src)
        (rebuild_dataset.rebuild_squad if src == "squad" else rebuild_dataset.rebuild_hotpotqa)()
        _commit()
    else:
        log.info("[rebuild-clean] clean master present, skipping stage 1")

    if a.force or not os.path.exists(f"data/ukb_storage/{clean}/partition_map.json"):
        log.info("[rebuild-clean] stage 2/4: build substrate for %s (~%d docs/partition)", clean, a.target_per_partition)
        build_all(master_nodes_path=clean_master, target_datasets=[clean], skip_colbert=True,
                  target_per_partition=a.target_per_partition)
        _commit()
    else:
        log.info("[rebuild-clean] substrate present, skipping stage 2")

    import numpy as _np, json as _json
    _pm_root = f"data/ukb_storage/{clean}/partition_map.json"
    _ndocs = len(_json.load(open(_pm_root))) if os.path.exists(_pm_root) else -1

    def _rows(path):                                          # row/key count, or -1 if unreadable
        try:
            if path.endswith(".npy"):
                return int(_np.load(path, mmap_mode="r").shape[0])
            return len(_json.load(open(path)))
        except Exception:
            return -1

    for i, (model, subdir) in enumerate(
        [("BAAI/bge-large-en-v1.5", "bge_large"),
         ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "gte_qwen")], start=3):
        npy = f"data/ukb_storage/{clean}/{subdir}/nodes.npy"
        # verify the CURRENT doc count, not just existence — a stale encode (e.g. old 66k gte) must re-run
        if not a.force and _rows(npy) == _ndocs:
            log.info("[rebuild-clean] %s encodings current (%d docs), skipping encode", subdir, _ndocs)
        else:
            log.info("[rebuild-clean] stage %d/4: reencode %s -> %s (batch %d)", i, clean, subdir, a.batch)
            reencode_ukb.run(clean, model_name=model, subdir=subdir, batch=a.batch, max_seq=a.max_seq)
            _commit()
        # per-encoder graph: kNN + METIS in THIS encoder's space (rebuild if its partition is stale)
        if a.force or _rows(f"data/ukb_storage/{clean}/{subdir}/partition_map.json") != _ndocs:
            reencode_ukb.build_encoder_graph(clean, subdir, a.target_per_partition)
            _commit()
    log.info("[rebuild-clean] complete for %s", clean)


def _encoder_graph(argv):
    from src.experiments.reencode_ukb import build_encoder_graph
    p = argparse.ArgumentParser(prog="encoder-graph")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--subdir", required=True, help="encoder subdir whose vectors build the kNN graph + partitions")
    p.add_argument("--target-per-partition", type=int, default=100)
    a = p.parse_args(argv)
    for d in a.datasets:
        build_encoder_graph(d, a.subdir, a.target_per_partition)


def _splade_encode(argv):
    from src.experiments import splade_encode                 # SPLADE doc pre-encode -> splade_doc_embs.pkl
    splade_encode.main(argv)


def _partsize_ablation(argv):
    from src.experiments import overlap_by_partsize          # nodes-per-partition sweep (needs pymetis)
    overlap_by_partsize.main(argv)


def _encoder_upgrade(argv):
    from src.experiments import encoder_upgrade
    encoder_upgrade.main(argv)


def _l3_recovery(argv):
    from src.experiments import l3_recovery
    l3_recovery.main(argv)


def _l3_traverse(argv):
    from src.experiments import l3_traverse
    l3_traverse.main(argv)


def _l3_methods(argv):
    from src.experiments import l3_methods
    l3_methods.main(argv)


def _baselines_rag(argv):
    from src.experiments import baselines_rag
    baselines_rag.main(argv)


def _l1_attn_fusion(argv):
    from src.experiments import l1_attn_fusion
    l1_attn_fusion.main(argv)


def _l1_mlpt_improve(argv):
    from src.experiments import l1_mlpt_improve
    l1_mlpt_improve.main(argv)


def _l1_struct_scorer(argv):
    from src.experiments import l1_struct_scorer
    l1_struct_scorer.main(argv)


def _l1_error_analysis(argv):
    from src.experiments import l1_error_analysis
    l1_error_analysis.main(argv)


def _l1_partition_probe(argv):
    from src.experiments import l1_partition_probe
    l1_partition_probe.main(argv)


def _l1_partition_router(argv):
    from src.experiments import l1_partition_router
    l1_partition_router.main(argv)


def _l1_partition_mlpt(argv):
    from src.experiments import l1_partition_mlpt
    l1_partition_mlpt.main(argv)


def _l1_partition_select(argv):
    from src.experiments import l1_partition_select
    l1_partition_select.main(argv)


def _l1_unified_ranker(argv):
    from src.experiments import l1_unified_ranker
    l1_unified_ranker.main(argv)


def _l1_rerank100(argv):
    from src.experiments import l1_rerank100
    l1_rerank100.main(argv)


def _l1_finetune_encoder(argv):
    from src.experiments import l1_finetune_encoder
    l1_finetune_encoder.main(argv)


def _l1_universal_head(argv):
    from src.experiments import l1_universal_head
    l1_universal_head.main(argv)


def _l2_seed(argv):
    from src.experiments import l2_seed
    l2_seed.main(argv)


def _e2e_ner(argv):
    from src.experiments import e2e_pipeline
    e2e_pipeline.main(argv)


def _l2_learned_fusion(argv):
    from src.experiments import l2_learned_fusion
    l2_learned_fusion.main(argv)


def _l2_mem_bench(argv):
    from src.experiments import l2_mem_bench
    l2_mem_bench.main(argv)


def _l1_candgen(argv):
    from src.experiments import l1_candgen
    l1_candgen.main(argv)


def _l3_graphlift(argv):
    from src.experiments import l3_graphlift
    l3_graphlift.main(argv)


def _l1_overlap_test(argv):
    from src.experiments import l1_overlap_test
    l1_overlap_test.main(argv)


def _sota_baselines(argv):
    from src.experiments import sota_baselines
    sota_baselines.main(argv)


def _sota_end_to_end(argv):
    from src.experiments import sota_end_to_end
    sota_end_to_end.main(argv)


def _query_decomp(argv):
    from src.experiments import query_decomp
    query_decomp.main(argv)


def _pool_narrow(argv):
    from src.experiments import pool_narrow
    pool_narrow.main(argv)


def _train_mlp(argv):
    _subprocess("src.alignment.train_mlp", argv)     # single MLP train (any loss)


def _l1_optimize(argv):
    from src.experiments import l1_optimize
    l1_optimize.main(argv)


def _l1_unified(argv):
    from src.experiments import l1_unified
    l1_unified.main(argv)


def _review_metric(argv):
    _script("review_paper/experiment.py", argv)


_SMOKE_TRAIN = ["--datasets", "2wiki", "--limit", "64", "--epochs", "1"]

TASKS: Dict[str, Task] = {
    "review-metric": Task(
        "review-metric",
        "gpu",
        "Standalone official-data aggregate-vs-joint multi-hop retrieval audit",
        _review_metric,
        smoke_args=["--datasets", "2wiki", "--audit-only"],
    ),
    "ukb-build":      Task("ukb-build", "gpu", "Build per-source UKB indices from master_nodes.json", _ukb_build),
    "train-mlp":      Task("train-mlp", "gpu", "Train one MLP router (any loss/tau/hn_k/lambda)", _train_mlp,
                           smoke_args=["--dataset", "2wiki", "--loss_type", "coverage_kl", "--limit", "64", "--epochs", "1"]),
    "train-coverage": Task("train-coverage", "gpu", "Coverage-loss lambda sweep vs KL baseline (+McNemar)", _train_coverage,
                           smoke_args=["--datasets", "2wiki", "--lambdas", "0.5", "--limit", "64", "--epochs", "1"]),
    "train-hnm":      Task("train-hnm", "gpu", "Hard-negative-mining ablation", _train_hnm, smoke_args=_SMOKE_TRAIN),
    "train-loss":     Task("train-loss", "gpu", "Loss-function ablation", _train_loss, smoke_args=_SMOKE_TRAIN),
    "train-temp":     Task("train-temp", "gpu", "Temperature ablation", _train_temp, smoke_args=_SMOKE_TRAIN),
    "l1-optimize":    Task(
        "l1-optimize",
        "gpu",
        "Validation-locked dense/sparse/relational/partition Level-1 candidate optimization",
        _l1_optimize,
        smoke_args=[
            "--datasets", "2wiki_clean", "--run-id", "smoke", "--limit", "64",
            "--epochs", "1", "--heads", "1", "--coverage-lambdas", "0",
            "--seeds", "42", "--hard-negative-k", "4", "--eval-every", "1",
            "--patience", "1", "--no-bm25",
        ],
    ),
    "l1-unified":     Task(
        "l1-unified",
        "gpu",
        "One dataset-agnostic dense/relational Level-1 router with macro validation",
        _l1_unified,
        smoke_args=[
            "--datasets", "2wiki_clean", "musique_clean",
            "--run-id", "smoke_unified", "--limit", "32", "--epochs", "1",
            "--relational-heads", "1", "--coverage-lambdas", "0",
            "--seeds", "42", "--hard-negative-k", "4", "--eval-every", "1",
            "--patience", "1", "--batch-size", "16",
        ],
    ),
    "bench-level1":   Task("bench-level1", "cpu", "Level-1 partition-routing benchmark (incl. selectivity)", _bench_level1,
                           smoke_args=["--datasets", "squad", "--limit", "64"]),
    "bench-level2":   Task("bench-level2", "gpu", "Level-2 reranking benchmark (bm25/dense/colbert/splade)", _bench_level2,
                           smoke_args=["--datasets", "squad", "--methods", "bm25"]),
    "bench-level3":   Task("bench-level3", "cpu", "Level-3 traversal / context benchmark", _bench_level3,
                           smoke_args=["--dataset", "squad", "--limit", "20"]),
    "probe-overlap":  Task("probe-overlap", "cpu",
                           "Overlapped-partition coverage vs explosion probe (training-free, soft-knn)", _probe_overlap,
                           smoke_args=["--datasets", "2wiki", "--limit", "300", "--overlaps", "0", "1", "2"]),
    "overlap-retrain": Task("overlap-retrain", "gpu",
                            "1-hop-overlap partitions RETRAINED (hard vs overlap1, coverage + explosion)",
                            _overlap_retrain,
                            smoke_args=["--datasets", "2wiki", "--limit", "64", "--epochs", "2"]),
    "finetune-encoder": Task("finetune-encoder", "gpu",
                             "Fine-tune the query encoder end-to-end to partition routing (representation lever)",
                             _finetune_encoder,
                             smoke_args=["--datasets", "2wiki", "--limit", "64", "--epochs", "1"]),
    "train-gnn": Task("train-gnn", "gpu",
                      "Corrected GNN training (live-positive backprop) — valid GNN-vs-MLP test",
                      _train_gnn,
                      smoke_args=["--datasets", "2wiki", "--models", "gin", "--limit", "64", "--epochs", "2"]),
    "adaptive-k": Task("adaptive-k", "gpu",
                       "Adaptive-K oracle probe: K_q proportional to gold-count vs fixed-K (pool-vs-coverage frontier)",
                       _adaptive_k,
                       smoke_args=["--datasets", "2wiki", "--configs", "hard", "--limit", "128", "--epochs", "2"]),
    "stack-best": Task("stack-best", "gpu",
                       "Full-stack combo: frozen+MLP vs fine-tuned-encoder on best overlap membership + adaptive-K",
                       _stack_best,
                       smoke_args=["--datasets", "2wiki", "--configs", "overlap1+knn1", "--epochs", "2", "--ft_epochs", "1", "--limit", "128"]),
    "multiproto": Task("multiproto", "cpu",
                       "Multi-prototype partition routing (max-sim over k sub-centroids) — improve coverage at fixed pool",
                       _multiproto,
                       smoke_args=["--datasets", "2wiki", "--configs", "overlap1", "--protos", "1", "4", "--limit", "300"]),
    "reencode-ukb": Task("reencode-ukb", "gpu",
                         "Re-encode a UKB source with a stronger frozen encoder into a parallel subfolder (non-destructive)",
                         _reencode_ukb,
                         smoke_args=["--datasets", "2wiki", "--model", "BAAI/bge-base-en-v1.5"]),
    "rebuild-clean": Task("rebuild-clean", "gpu",
                          "Rebuild an audited-broken source clean end-to-end: loaders->build_clean->build_all->reencode(bge_large+gte_qwen)+per-encoder graphs, resumable",
                          _rebuild_clean,
                          smoke_args=["--dataset", "squad"]),
    "encoder-graph": Task("encoder-graph", "gpu",
                          "Build per-encoder graph.pt+partition_map+centroids in a subdir (kNN+METIS in the encoder's space; reuses nodes.npy, no re-encode)",
                          _encoder_graph,
                          smoke_args=["--datasets", "2wiki_clean", "--subdir", "bge_large"]),
    "splade-encode": Task("splade-encode", "gpu",
                          "Pre-encode a dataset's docs with SPLADE -> splade_doc_embs.pkl (lexical axis for L2 hybrid; frozen substrate untouched)",
                          _splade_encode,
                          smoke_args=["--datasets", "musique_clean"]),
    "partsize-ablation": Task("partsize-ablation", "gpu",
                          "Nodes-per-partition ablation: overlap FullCov-vs-pool frontier at 100/250/500/1000 docs/part (training-free, MiniLM base)",
                          _partsize_ablation,
                          smoke_args=["--datasets", "2wiki_clean", "--targets", "100"]),
    "l1-finetune-encoder": Task("l1-finetune-encoder", "gpu",
                                "Fine-tune bge-large in-domain on (query,gold-doc) pairs -> rerank-compatible subdir (representation lever for L1 partition coverage)",
                                _l1_finetune_encoder,
                                smoke_args=["--datasets", "2wiki_clean", "--limit", "256", "--epochs", "1"]),
    "l1-universal-head": Task("l1-universal-head", "gpu",
                              "STEP1: best SINGLE universal relational head (base/hard/mix=mlpT/mix_hard) pooled across all datasets -> per-dataset FullCov@20 + dense-fused",
                              _l1_universal_head,
                              smoke_args=["--datasets", "musique_clean", "squad_clean", "--tr-cap", "300",
                                          "--te-cap", "200", "--epochs", "3"]),
    "l2-seed": Task("l2-seed", "gpu",
                    "L2 seed-finding WITHIN L1's top-K partitions: gold-DOC recall dense vs offset/mlpT (mlpT's real home; +55 on metaqa)",
                    _l2_seed,
                    smoke_args=["--datasets", "metaqa", "--tr-cap", "300", "--te-cap", "200",
                                "--epochs", "3", "--scope-topk", "50"]),
    "l2-learned-fusion": Task("l2-learned-fusion", "gpu",
                    "Learned per-query fusion gate vs parameter-free best-of (tests the 'why not one learned model' critique)",
                    _l2_learned_fusion,
                    smoke_args=["--datasets", "metaqa", "--gate-epochs", "5"]),
    "e2e-ner": Task("e2e-ner", "gpu",
                    "End-to-end L1->L2->L3 with struct+weighted-NER edges (kNN dropped) + NER-blend L3; universal head/adapter; final Recall@5",
                    _e2e_ner,
                    smoke_args=["--datasets", "metaqa", "--head-datasets", "metaqa",
                                "--epochs", "3", "--use-adapter", "--adapter-epochs", "5"]),
    "l2-mem-bench": Task("l2-mem-bench", "gpu",
                    "L2 memory microbench: measured GPU peak for full-corpus-resident vs routed-pool-gather scoring (+topk parity)",
                    _l2_mem_bench,
                    smoke_args=["--datasets", "metaqa", "--te-cap", "200", "--sample", "16"]),
    "l1-candgen": Task("l1-candgen", "gpu",
                    "L1 candidate-generator ablation: partition-routed pool vs dense top-P recall at matched budget (defends the graph)",
                    _l1_candgen,
                    smoke_args=["--datasets", "metaqa", "--te-cap", "200"]),
    "l3-graphlift": Task("l3-graphlift", "gpu",
                    "L3 graph-lift: does traversal from dense seeds recover golds dense-top-N misses (the graph's native multi-hop job)?",
                    _l3_graphlift,
                    smoke_args=["--datasets", "metaqa", "--te-cap", "200", "--budget", "100"]),
    "l1-overlap-test": Task("l1-overlap-test", "gpu",
                    "Overlap vs no-overlap voting: does the graph signal help L1 partition routing, or is it wasted at partition granularity?",
                    _l1_overlap_test,
                    smoke_args=["--datasets", "metaqa", "--te-cap", "200"]),
    "encoder-upgrade": Task("encoder-upgrade", "gpu",
                            "Stronger-encoder A/B on the overlap stack (MiniLM vs bge) — fixed-overlap coverage lift",
                            _encoder_upgrade,
                            smoke_args=["--datasets", "2wiki", "--configs", "hard", "--encoders", "minilm"]),
    "l3-recovery": Task("l3-recovery", "gpu",
                        "Measure L3 graph-traversal recovery of L1 FullCov failures (missed golds one hop from found)",
                        _l3_recovery,
                        smoke_args=["--datasets", "2wiki", "--config", "overlap1+knn1", "--epochs", "3"]),
    "l3-traverse": Task("l3-traverse", "gpu",
                        "L3 bounded PPR-guided best-first traversal from seeds (recall + latency + stop-reasons)",
                        _l3_traverse,
                        smoke_args=["--datasets", "2wiki_clean", "--N_seed", "10", "--budget", "200", "--limit", "50"]),
    "l3-methods": Task("l3-methods", "gpu",
                       "L3 all-methods comparison (dense/1hop/2hop/ppr/appnp/best-first) at matched budgets",
                       _l3_methods,
                       smoke_args=["--datasets", "2wiki_clean", "--N_seed", "20", "--limit", "50"]),
    "baselines-rag": Task("baselines-rag", "gpu",
                          "External flat RAG baselines (bm25/dense/hybrid/+cross-encoder/splade) + FullCov (field comparison)",
                          _baselines_rag,
                          smoke_args=["--datasets", "2wiki_clean", "--limit", "50"]),
    "l1-attn-fusion": Task("l1-attn-fusion", "cpu",
                           "L1 doc-conditioned attention-fusion vs ensemble (improvement search; relative, cloud-parallel)",
                           _l1_attn_fusion,
                           smoke_args=["--datasets", "2wiki_clean", "--limit", "300", "--epochs", "2", "--K", "4"]),
    "l1-mlpt-improve": Task("l1-mlpt-improve", "cpu",
                            "L1 MLP-transformer head/epoch/diversity sweep (improvement search; relative, cloud-parallel)",
                            _l1_mlpt_improve,
                            smoke_args=["--datasets", "2wiki_clean", "--Ks", "4", "--epochs_list", "2", "--lam_divs", "0.0", "--limit", "300"]),
    "l1-struct-scorer": Task("l1-struct-scorer", "cpu",
                             "L1 offset-signal set-transformer scorer + combination (improvement search; relative, cloud-parallel)",
                             _l1_struct_scorer,
                             smoke_args=["--datasets", "2wiki_clean", "--limit", "300", "--epochs", "2"]),
    "l1-error-analysis": Task("l1-error-analysis", "cpu",
                              "L1 champion miss error-analysis — buckets each missed gold by the recovering lever (diagnostic)",
                              _l1_error_analysis,
                              smoke_args=["--datasets", "2wiki_clean", "--limit_test", "20", "--limit_train", "300"]),
    "l1-partition-probe": Task("l1-partition-probe", "cpu",
                               "L1 partition-reachability probe — are multi-hop golds routable via their partitions? (diagnostic, no training)",
                               _l1_partition_probe,
                               smoke_args=["--datasets", "2wiki_clean", "--limit", "100"]),
    "l1-partition-router": Task("l1-partition-router", "cpu",
                                "L1 learned multi-label partition router — predict gold-partition SET, route+retrieve (the multi-hop L1 lever)",
                                _l1_partition_router,
                                smoke_args=["--datasets", "2wiki_clean", "--epochs", "3", "--limit", "400"]),
    "l1-partition-mlpt": Task("l1-partition-mlpt", "cpu",
                              "L1 K-head PARTITION predictor (mlpT soft-OR over partitions) + offset signal; partition FullCov, hard vs overlap",
                              _l1_partition_mlpt,
                              smoke_args=["--datasets", "2wiki_clean", "--limit", "400", "--epochs", "3", "--off-epochs", "2", "--Ks", "1", "8"]),
    "l1-partition-select": Task("l1-partition-select", "cpu",
                                "L1 partition selection exploiting node features — node-evidence voting + fusion vs centroid routing; partition FullCov, hard vs overlap",
                                _l1_partition_select,
                                smoke_args=["--datasets", "2wiki_clean", "--limit", "400", "--epochs", "3", "--off-epochs", "2", "--topn", "100"]),
    "l1-unified-ranker": Task("l1-unified-ranker", "cpu",
                              "L1 UNIFIED partition ranker — learned fusion of centroid+route+dense/relhard/rel2hop/mlpT node-votes; per-ds + universal; partition FullCov",
                              _l1_unified_ranker,
                              smoke_args=["--mode", "solo", "--datasets", "2wiki_clean", "--limit", "300", "--epochs", "3", "--off-epochs", "2", "--topn", "100"]),
    "l1-rerank100": Task("l1-rerank100", "cpu",
                         "L1 re-ranking toward 100@20 — per-query attention over 6 sum/max vote signals + pairwise loss; oracle/eqrrf/attn/mlp partition FullCov",
                         _l1_rerank100,
                         smoke_args=["--datasets", "2wiki_clean", "--limit", "400", "--off-epochs", "2", "--tr-cap", "300"]),
    "sota-baselines": Task(
        "sota-baselines",
        "gpu",
        "Validation-locked matched-corpus SOTA retrieval baselines",
        _sota_baselines,
        smoke_args=[
            "--datasets", "2wiki_clean", "--run-id", "smoke",
            "--limit", "64", "--methods", "dense", "bm25",
            "hybrid_rrf", "hybrid_tuned",
        ],
    ),
    "sota-e2e": Task(
        "sota-e2e",
        "cpu",
        "Pinned full-ingestion and end-to-end SOTA RAG reproduction suite",
        _sota_end_to_end,
        smoke_args=["status"],
    ),
    "query-decomp": Task("query-decomp", "gpu",
                         "Query-decomposition routing (route entity sub-queries, union) vs single-query baseline",
                         _query_decomp,
                         smoke_args=["--datasets", "2wiki", "--configs", "hard", "--epochs", "3", "--limit", "300"]),
    "pool-narrow": Task("pool-narrow", "gpu",
                        "Narrow the overlap pool to a small candidate set (dense top-N + 1-hop traversal); recall vs size",
                        _pool_narrow,
                        smoke_args=["--datasets", "2wiki", "--config", "overlap1+knn1", "--epochs", "3", "--limit", "300"]),
}


def get(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {', '.join(sorted(TASKS))}")
    return TASKS[name]


def all_tasks() -> List[Task]:
    return list(TASKS.values())
