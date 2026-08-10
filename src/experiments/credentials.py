"""
Compute-account pool + rotation.
================================
Accounts live in a git-ignored `configs/compute.local.yaml` (never committed).
Copy `configs/compute.local.yaml.example` and fill in your own accounts. Each
backend can list multiple accounts; the runner rotates to the next one when the
current account runs out of credits / hits a quota or auth error.

Example file:

    modal:
      - name: personal
        token_id: ak-xxxx
        token_secret: as-xxxx
      - name: lab
        token_id: ak-yyyy
        token_secret: as-yyyy
    lightning:
      - name: personal
        user_id: xxxx
        api_key: xxxx
        teamspace: my-teamspace
        studio: crag            # persistent Studio holding the repo + data
      - name: backup
        user_id: yyyy
        api_key: yyyy
        teamspace: backup-ts
        studio: crag

Credentials are applied by exporting the standard SDK environment variables, so
no secret is ever passed on a command line or written outside the local file.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger("experiments.credentials")

CONFIG_PATH = os.environ.get("CRAG_COMPUTE_CONFIG", "configs/compute.local.yaml")

# Map a friendly YAML field -> the SDK environment variable the backend reads.
_ENV_MAP: Dict[str, Dict[str, str]] = {
    "modal": {
        "token_id": "MODAL_TOKEN_ID",
        "token_secret": "MODAL_TOKEN_SECRET",
        "profile": "MODAL_PROFILE",
    },
    "lightning": {
        "user_id": "LIGHTNING_USER_ID",
        "api_key": "LIGHTNING_API_KEY",
        "teamspace": "LIGHTNING_TEAMSPACE",
        "studio": "LIGHTNING_STUDIO",
        "org": "LIGHTNING_ORG",
        "cloud_account": "LIGHTNING_CLOUD_ACCOUNT",
    },
}

# Substrings that indicate "this account is out of credits / not usable" — rotate.
_ROTATE_SIGNALS = (
    "quota", "credit", "insufficient", "exhausted", "payment required",
    "402", "429", "rate limit", "too many requests", "unauthor", "forbidden",
    "not enough", "billing", "limit exceeded", "no capacity",
    "spend limit", "resourceexhausted", "exceeded its",   # Modal: "exceeded its spend limit"
)


@dataclass
class Credential:
    backend: str
    name: str
    env: Dict[str, str] = field(default_factory=dict)      # SDK env vars to export
    extra: Dict[str, str] = field(default_factory=dict)    # non-secret hints (studio, teamspace)

    def activate(self) -> None:
        """Export this account's SDK environment variables into the process."""
        for k, v in self.env.items():
            if v is not None:
                os.environ[k] = str(v)
        log.info(f"[{self.backend}] activated account '{self.name}'")


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        log.warning("pyyaml not installed; cannot read %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pool(backend: str, path: str = CONFIG_PATH) -> List[Credential]:
    """Return the ordered list of accounts for a backend (empty if none configured).

    If the config file is absent, returns a single empty Credential so the backend
    falls back to whatever ambient auth already exists (e.g. `modal token set`,
    `lightning login`, or existing env vars) — the runner still works with one
    account and no config file.
    """
    if backend not in _ENV_MAP:
        raise ValueError(f"Unknown backend {backend!r}")
    cfg = _load_yaml(path)
    entries = cfg.get(backend, []) or []
    env_map = _ENV_MAP[backend]

    pool: List[Credential] = []
    for i, entry in enumerate(entries):
        name = str(entry.get("name", f"{backend}-{i}"))
        env = {}
        extra = {}
        for field_name, value in entry.items():
            if field_name == "name":
                continue
            if field_name in env_map:
                env[env_map[field_name]] = value
                if field_name in ("teamspace", "studio", "org", "cloud_account"):
                    extra[field_name] = value
            else:
                extra[field_name] = value
        pool.append(Credential(backend=backend, name=name, env=env, extra=extra))

    if not pool:
        # Ambient-auth fallback: one account, no env overrides.
        pool = [Credential(backend=backend, name="ambient", env={}, extra={})]
    return pool


def should_rotate(exc: Exception) -> bool:
    """Heuristic: does this error mean the current account is unusable (rotate)?"""
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(sig in msg for sig in _ROTATE_SIGNALS)
