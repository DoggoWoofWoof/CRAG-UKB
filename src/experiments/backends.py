"""
Compute backends — the single place that talks to cloud providers.
===================================================================
Three backends behind one interface:
  * LocalBackend    — runs the task body in-process (no cloud; fully testable).
  * ModalBackend    — GPU-preferred; mirrors the proven run_*_modal image/volume
                      pattern. Runs the task on an A10G with a persistent Volume.
  * LightningBackend— CPU-preferred; Lightning AI cloud via lightning_sdk Jobs,
                      run inside a persistent Studio that holds the repo + data.

Auto-routing: task.resource == "gpu" -> Modal, "cpu" -> Lightning. Any task can
be forced onto any backend. When an account errors with a quota/credit/auth
signal, the runner rotates to the next account from configs/compute.local.yaml.

Heavy SDK imports (modal, lightning_sdk) are deferred so this module always
imports even when a provider isn't installed.
"""
import os
import sys
import time
import shutil
import logging
import subprocess
from typing import List

from src.experiments.tasks import Task
from src.experiments import credentials, sync

log = logging.getLogger("experiments.backends")

MODAL_VOLUME = "crag-data-volume"
REMOTE_ROOT = "/root/CRAG"
PERSIST_DIRS = ["data", "checkpoints", "results"]


def commit_persistent_storage() -> bool:
    """Commit a mounted Modal volume at a recoverable artifact boundary."""
    if not os.path.isdir(os.path.join(REMOTE_ROOT, "storage")):
        return False
    volume = globals().get("_MODAL_VOL")
    if volume is None:
        return False
    volume.commit()
    return True


def _configure_model_cache(root):
    """Keep downloaded model weights on the same persistent storage as the UKB."""
    cache_root = os.path.join(root, "ukb_storage", "_models")
    locations = {
        "HF_HOME": os.path.join(cache_root, "huggingface"),
        "SENTENCE_TRANSFORMERS_HOME": os.path.join(
            cache_root, "sentence_transformers"
        ),
        "TORCH_HOME": os.path.join(cache_root, "torch"),
    }
    for name, path in locations.items():
        os.makedirs(path, exist_ok=True)
        os.environ.setdefault(name, path)


# ── credential-rotation helper ───────────────────────────────────────────────
def _run_with_rotation(backend_name, attempt, account=None):
    """Try `attempt(cred)` across the account pool, rotating on quota/auth errors.
    If `account` (name or index) is given, pin to just that account — no rotation
    (for launching one run per account in parallel across accounts)."""
    pool = credentials.load_pool(backend_name)
    if account is not None:
        sel = ([pool[int(account)]] if account.isdigit() and int(account) < len(pool)
               else [c for c in pool if c.name == account])
        pool = sel or pool
        log.info("[%s] pinned to account '%s' (no rotation)", backend_name, pool[0].name)
    last_exc = None
    for i, cred in enumerate(pool):
        cred.activate()
        try:
            return attempt(cred)
        except Exception as exc:  # noqa: BLE001 — we classify below
            last_exc = exc
            is_last = i == len(pool) - 1
            if credentials.should_rotate(exc) and not is_last:
                log.warning("[%s] account '%s' unusable (%s); rotating to next account.",
                            backend_name, cred.name, type(exc).__name__)
                continue
            raise
    if last_exc:
        raise last_exc


class Backend:
    name = "base"

    def launch(self, task: Task, argv: List[str], gpu: bool, dry_run: bool = False, account=None):
        raise NotImplementedError


# ── Local ────────────────────────────────────────────────────────────────────
class LocalBackend(Backend):
    name = "local"

    def launch(self, task, argv, gpu, dry_run=False, account=None):
        if dry_run:
            log.info("[dry-run] local: %s.body(%s)", task.name, argv)
            return
        log.info("Running task '%s' locally (resource=%s).", task.name, task.resource)
        task.body(argv)


# ── Modal (GPU-preferred) ─────────────────────────────────────────────────────
def _modal_base_image(modal):
    return (
        modal.Image.micromamba(python_version="3.11")
        .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
        .apt_install("git", "build-essential", "ninja-build")
        .pip_install("torch==2.2.1", "numpy<2.0")
        .pip_install("torch-geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18",
                     find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
        .pip_install("networkx==3.2.1", "rank_bm25", "spacy", "pyyaml", "pandas", "tqdm",
                     "scipy", "sentence-transformers<3.0", "transformers==4.44.2")
        .pip_install("colbert-ai>=0.2.19", extra_options="--no-deps")
        .pip_install("ragatouille==0.0.9", "langchain<0.2")
        .run_commands("pip uninstall -y faiss-cpu faiss-gpu")
        .pip_install("faiss-gpu-cu12==1.8.0.1")
        .micromamba_install("pymetis=2022.1", "pytorch-cuda=12.1", "cuda-nvcc", "cuda-cudart-dev",
                            channels=["conda-forge", "pytorch", "nvidia"])
        # prebuilt flash-attn wheel (no compile) so gte-Qwen2's custom code imports cleanly; we only
        # need forward-pass encodings, but its modeling file has an unconditional `import flash_attn`.
        .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/"
                     "flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
        .run_commands("python -m spacy download en_core_web_sm")   # NER model for shares-entity edges (no LLM)
    )


def _modal_image(modal):
    return (
        _modal_base_image(modal)
        .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs")
        .add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py")
    )


def _modal_review_image(modal):
    """Isolated image for the paper audit; do not perturb the main CRAG stack."""
    return (
        _modal_base_image(modal)
        .pip_install(
            "datasets==4.8.3",
            "gdown==5.2.1",
            "pyarrow==23.0.1",
            "sentence-transformers==5.2.0",
        )
        .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs")
        .add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py")
        .add_local_dir("review_paper", remote_path=f"{REMOTE_ROOT}/review_paper")
    )


def _symlink_storage(volume_commit):
    """Inside the container: point data/checkpoints/results at the persistent Volume."""
    storage_root = f"{REMOTE_ROOT}/storage"
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.append(REMOTE_ROOT)
    for sd in PERSIST_DIRS:
        remote_sd = os.path.join(storage_root, sd)
        local_sd = os.path.join(REMOTE_ROOT, sd)
        os.makedirs(remote_sd, exist_ok=True)
        if os.path.islink(local_sd):
            continue
        if os.path.exists(local_sd):
            # sync any image-baked content up once, then replace with a symlink
            for item in os.listdir(local_sd):
                s, d = os.path.join(local_sd, item), os.path.join(remote_sd, item)
                if not os.path.exists(d):
                    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
            shutil.rmtree(local_sd) if os.path.isdir(local_sd) else os.remove(local_sd)
        os.symlink(remote_sd, local_sd)
    _configure_model_cache(os.path.join(REMOTE_ROOT, "data"))


def _modal_cli(*args, capture=False):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}   # child writes UTF-8
    return subprocess.run([sys.executable, "-m", "modal", *args],          # parent decodes UTF-8 too
                          capture_output=capture, text=True, env=env,
                          encoding="utf-8", errors="replace")              # not Windows cp1252 (byte 0x90 crash)


def _modal_volume_has(path):
    """True if `path` already exists on the Volume (skip re-upload -> resume)."""
    return _modal_cli("volume", "ls", MODAL_VOLUME, path, capture=True).returncode == 0


def _modal_push(paths):
    # MODAL_FORCE_UPLOAD=comma,paths -> delete any stale copy on the Volume and re-upload fresh
    # (used to overwrite a stale substrate, e.g. the old 40-partition metaqa on some accounts).
    force = {x for x in os.environ.get("MODAL_FORCE_UPLOAD", "").split(",") if x}
    for p in paths:
        if not os.path.exists(p):
            continue
        forced = any(p == f or p.startswith(f.rstrip("/") + "/") for f in force)
        if forced and _modal_volume_has(p):
            log.info("[modal] FORCE-refresh: removing stale %s from Volume", p)
            _modal_cli("volume", "rm", MODAL_VOLUME, p, "-r", capture=True)
        elif _modal_volume_has(p):
            log.info("[modal] present on Volume, skipping upload: %s", p)
            continue
        log.info("[modal] uploading %s -> Volume", p)
        _modal_cli("volume", "put", MODAL_VOLUME, p, p)


def _modal_pull(paths):
    for p in paths:
        parent = os.path.dirname(p) or "."
        os.makedirs(parent, exist_ok=True)
        log.info("[modal] pulling %s <- Volume", p)
        _modal_cli("volume", "get", MODAL_VOLUME, p, parent, "--force", capture=True)


# Module-scope Modal app + remote functions. They MUST be defined at GLOBAL scope:
# Modal rejects a nested @app.function unless serialized=True, and serialized functions
# require the local Python to match the image's (we're 3.13 local vs 3.11 image, so that
# path errors). Global functions avoid serialization entirely. Guarded so this module
# still imports (and LocalBackend still works) when modal isn't installed.
try:
    import modal as _modal_sdk

    _MODAL_APP = _modal_sdk.App("crag")
    _MODAL_VOL = _modal_sdk.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
    _MODAL_IMG = _modal_image(_modal_sdk)
    _MODAL_REVIEW_IMG = _modal_review_image(_modal_sdk)

    def _modal_body_gpu(task_name: str, task_argv: list):
        _symlink_storage(None)
        from src.experiments import tasks as _tasks
        _tasks.get(task_name).body(task_argv)
        _MODAL_VOL.commit()

    def _modal_body_cpu(task_name: str, task_argv: list):
        _symlink_storage(None)
        from src.experiments import tasks as _tasks
        _tasks.get(task_name).body(task_argv)
        _MODAL_VOL.commit()

    def _modal_body_review(task_name: str, task_argv: list):
        _symlink_storage(None)
        from src.experiments import tasks as _tasks
        _tasks.get(task_name).body(task_argv)
        _MODAL_VOL.commit()

    _VOLS = {f"{REMOTE_ROOT}/storage": _MODAL_VOL}
    _modal_remote_gpu = _MODAL_APP.function(image=_MODAL_IMG, gpu="A10G", volumes=_VOLS, timeout=86400)(_modal_body_gpu)
    _modal_remote_cpu = _MODAL_APP.function(image=_MODAL_IMG, volumes=_VOLS, timeout=86400)(_modal_body_cpu)
    _modal_remote_review = _MODAL_APP.function(
        image=_MODAL_REVIEW_IMG, gpu="A10G", volumes=_VOLS, timeout=86400
    )(_modal_body_review)
except Exception:                                        # modal absent / not configured
    _modal_sdk = None
    _MODAL_APP = None


class ModalBackend(Backend):
    name = "modal"

    def launch(self, task, argv, gpu, dry_run=False, account=None):
        if dry_run:
            log.info("[dry-run] modal: gpu=%s task=%s (Volume=%s)", gpu, task.name, MODAL_VOLUME)
            log.info("  upload (skip-if-present): %s", sync.required_inputs(task.name, argv))
            log.info("  run: tasks['%s'].body(%s)", task.name, argv)
            log.info("  pull results: %s", sync.result_outputs(task.name, argv))
            return
        _run_with_rotation("modal", lambda cred: self._run(task, argv, gpu), account=account)

    def _run(self, task, argv, gpu):
        if _MODAL_APP is None:
            raise RuntimeError("modal is not installed/importable; cannot use the Modal backend")
        # Stage only the inputs this task needs (resume-friendly: skip if present).
        _modal_push(sync.required_inputs(task.name, argv))
        if task.name == "review-metric":
            fn = _modal_remote_review
        else:
            fn = _modal_remote_gpu if gpu else _modal_remote_cpu  # global funcs (no serialization)
        log.info("Launching '%s' on Modal (gpu=%s)...", task.name, gpu)
        # Detached apps survive terminal/thread reconnects while ``remote`` still
        # blocks here so result synchronization happens exactly once on success.
        with _MODAL_APP.run(detach=True):
            fn.remote(task.name, list(argv))
        # Pull the results back locally.
        _modal_pull(sync.result_outputs(task.name, argv))
        log.info("Modal task '%s' complete; outputs synced to the working tree.", task.name)


# ── Lightning AI (CPU-preferred) ──────────────────────────────────────────────
# Default Lightning machine names (valid in lightning_sdk Machine). GPU defaults to
# the cheap L4; override per account via machine_cpu/machine_gpu in the config.
_LIGHTNING_DEFAULT_CPU = "CPU"
_LIGHTNING_DEFAULT_GPU = "L4"


def _resolve_lightning_machine(Machine, name):
    """Resolve a machine-name string to a lightning_sdk.Machine (version-robust)."""
    name = str(name)
    for candidate in (name, name.upper()):
        if hasattr(Machine, candidate):
            return getattr(Machine, candidate)
    if hasattr(Machine, "from_str"):
        return Machine.from_str(name)
    raise ValueError(
        f"Unknown Lightning machine {name!r}. Valid names include CPU, T4, L4, A100, H100 "
        f"(see `from lightning_sdk import Machine; dir(Machine)`)."
    )


def _lightning_put(studio, path):
    """Upload a file/folder to a Lightning Studio via a SINGLE tar.gz, extracted
    remotely. This sidesteps two lightning_sdk failures seen on this box: HTTP 501 on
    zero-byte uploads (empty __init__.py) and HTTP 403 rate-limiting on many rapid
    per-file uploads. One tarball = one upload = neither issue; tar preserves empty
    files and POSIX paths (same 'bundle it' approach as the subgnn Modal uploader)."""
    import tarfile
    import tempfile
    if os.path.isfile(path):
        # Non-empty single file uploads fine; 0-byte -> create remotely.
        if os.path.getsize(path) == 0:
            studio.run_with_exit_code(f"mkdir -p '{os.path.dirname(path).replace(os.sep, '/') or '.'}' "
                                      f"&& touch '{path.replace(os.sep, '/')}'")
        else:
            studio.upload_file(path, path.replace(os.sep, "/"))
        return
    handle = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    handle.close()
    try:
        with tarfile.open(handle.name, "w:gz") as tar:
            tar.add(path, arcname=path.replace(os.sep, "/"))   # POSIX arcnames, keeps empty files
        remote_tar = f"__upload_{os.path.basename(path.rstrip(os.sep))}.tar.gz"
        studio.upload_file(handle.name, remote_tar)
        studio.run_with_exit_code(f"tar xzf {remote_tar} && rm -f {remote_tar}")
    finally:
        os.unlink(handle.name)


class LightningBackend(Backend):
    name = "lightning"

    CODE_PATHS = ["src", "configs"]  # small; refreshed every run so code changes propagate

    def launch(self, task, argv, gpu, dry_run=False, account=None):
        cmd = self._command(task, argv)
        if dry_run:
            label = _LIGHTNING_DEFAULT_GPU if gpu else _LIGHTNING_DEFAULT_CPU
            log.info("[dry-run] lightning: auto-create Studio (create_ok), machine=%s (configurable)",
                     label)
            log.info("  upload code: %s + experiments.py", self.CODE_PATHS)
            log.info("  upload inputs (skip-if-present): %s", sync.required_inputs(task.name, argv))
            log.info("  run: %r", cmd)
            log.info("  download results: %s", sync.result_outputs(task.name, argv))
            return
        _run_with_rotation("lightning", lambda cred: self._run(task, cmd, gpu, cred, argv), account=account)

    @staticmethod
    def _command(task, argv):
        base = f"python experiments.py exec {task.name}"
        command = f"{base} -- {' '.join(argv)}" if argv else base
        cache_root = "data/ukb_storage/_models"
        cache_env = (
            f"HF_HOME={cache_root}/huggingface "
            f"SENTENCE_TRANSFORMERS_HOME={cache_root}/sentence_transformers "
            f"TORCH_HOME={cache_root}/torch"
        )
        # Fresh Lightning Studios start bare (Modal bakes deps into its image). Install
        # our core deps once — guarded by an import check so re-runs skip it (the Studio
        # FS persists installed packages). faiss-cpu (CPU studio), numpy<2 for torch2.2.
        deps = ("faiss-cpu \"numpy<2.0\" torch scipy \"sentence-transformers<3.0\" "
                "\"transformers<5.0.0\" networkx rank_bm25 pyyaml pandas tqdm")
        ensure_deps = (
            "python -c 'import faiss,torch,scipy,sentence_transformers,networkx,rank_bm25' "
            f"2>/dev/null || pip install -q {deps}"
        )
        return (
            f"{ensure_deps}"
            f" && mkdir -p {cache_root}/huggingface "
            f"{cache_root}/sentence_transformers {cache_root}/torch"
            f" && {cache_env} {command}"
        )

    def _run(self, task, cmd, gpu, cred, argv):
        from lightning_sdk import Machine, Studio
        # Auto-create the Studio if it doesn't exist (create_ok=True) — no manual
        # setup required. The Studio's persistent filesystem is our shared drive:
        # uploaded inputs + trained checkpoints survive across jobs, so re-runs
        # resume (train_mlp / benchmarks skip work whose outputs already exist).
        studio_name = cred.extra.get("studio") or os.environ.get("LIGHTNING_STUDIO") or "crag"
        teamspace = cred.extra.get("teamspace") or os.environ.get("LIGHTNING_TEAMSPACE")
        machine_name = (
            (cred.extra.get("machine_gpu") or os.environ.get("LIGHTNING_MACHINE_GPU") or _LIGHTNING_DEFAULT_GPU)
            if gpu else
            (cred.extra.get("machine_cpu") or os.environ.get("LIGHTNING_MACHINE_CPU") or _LIGHTNING_DEFAULT_CPU)
        )
        machine = _resolve_lightning_machine(Machine, machine_name)

        studio = Studio(name=studio_name, teamspace=teamspace, create_ok=True)
        log.info("Lightning studio='%s' (status=%s); ensuring machine=%s...",
                 studio_name, self._safe_status(studio), machine_name)
        try:
            studio.start(machine=machine)
        except Exception as e:  # already running / already started
            log.info("studio.start note: %s; trying switch_machine.", e)
            try:
                studio.switch_machine(machine)
            except Exception as e2:
                log.info("switch_machine note: %s (continuing on current machine).", e2)

        # 1) Upload code (small, always refresh so code edits propagate).
        for folder in self.CODE_PATHS:
            if os.path.isdir(folder):
                _lightning_put(studio, folder)
        if os.path.exists("experiments.py"):
            _lightning_put(studio, "experiments.py")

        # 2) Upload only the required inputs, skipping any already on the Studio FS.
        for p in sync.required_inputs(task.name, argv):
            if not os.path.exists(p):
                continue
            out, _ = studio.run_with_exit_code(f'test -e "{p}" && echo __HAVE__ || echo __MISS__')
            if "__HAVE__" in out:
                log.info("[lightning] present on Studio, skipping upload: %s", p)
                continue
            log.info("[lightning] uploading %s -> Studio", p)
            _lightning_put(studio, p)

        # 3) Run the task on the Studio (synchronous; reads/writes the persistent FS).
        log.info("Running on Lightning studio: %r", cmd)
        out, code = studio.run_with_exit_code(cmd)
        if out:
            log.info(out)
        if code != 0:
            raise RuntimeError(f"Lightning run for '{task.name}' exited with code {code}")

        # 4) Pull outputs (results + trained checkpoints) back to the working tree.
        for p in sync.result_outputs(task.name, argv):
            try:
                os.makedirs(p if not os.path.splitext(p)[1] else os.path.dirname(p) or ".", exist_ok=True)
                studio.download_folder(p, p)
                log.info("[lightning] pulled %s <- Studio", p)
            except Exception as e:
                log.warning("[lightning] could not download %s: %s", p, e)
        log.info("Lightning task '%s' complete; outputs synced to the working tree.", task.name)

    @staticmethod
    def _safe_status(studio):
        try:
            return str(studio.status)
        except Exception:
            return "?"


_BACKENDS = {"local": LocalBackend, "modal": ModalBackend, "lightning": LightningBackend}


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend {name!r}. Available: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]()


def resolve_backend(task: Task, override: str = "auto") -> str:
    """Auto-route by resource: gpu -> modal, cpu -> lightning; else honor override."""
    if override and override != "auto":
        return override
    return "modal" if task.resource == "gpu" else "lightning"
