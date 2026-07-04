"""Path + credential resolution for remote compute (CLAUDE.md §5, §7).

Two inputs, cleanly separated:

* **`.env`** populates the *environment* — the corpus location
  (``DAGGER_DATA_ROOT``, a mounted-volume path) and any gated-corpus credential
  (``DAGGER_WSJ0_ACCESS_KEY``). Loaded once via :func:`load_env`.
* **`configs/phase0*.yaml`** holds the literal dataset *fields* (name, metadata
  path, split, n_src, limit), read by exact key.

Corpus audio never lives in the repo, so nothing here is committed.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """Load a ``.env`` file into ``os.environ`` if one is present.

    A no-op when ``python-dotenv`` is not installed or no ``.env`` exists, so the
    synthetic path (which needs no data root) still runs. On remote compute the
    orchestration may set the env directly instead of shipping a ``.env``.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()  # searches the cwd and parents for the nearest .env


def resolve_data_root() -> Path:
    """Return the mounted corpus root from ``DAGGER_DATA_ROOT``.

    Raises if unset or the path does not exist — that env var is how remote
    compute points the loaders at the mounted volume.
    """
    root = os.environ.get("DAGGER_DATA_ROOT", "").strip()
    if not root:
        raise ValueError(
            "DAGGER_DATA_ROOT is not set. Put it in .env (or export it) so the "
            "loaders can find the mounted corpus volume. See .env.example."
        )
    path = Path(root).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"DAGGER_DATA_ROOT={path!r} does not exist. On remote compute, mount "
            f"the corpus volume there."
        )
    return path


def get_credential(name: str) -> str | None:
    """Return the credential in env var ``name`` (or ``None`` if unset/blank).

    Used by gated corpora to authorize a fetch of licensed data hosted behind a
    private endpoint. When the data already sits on the mounted volume the
    credential is unused and this returns ``None`` — see
    :func:`dagger.data.wsj0mix.ensure_access`.
    """
    value = os.environ.get(name, "").strip()
    return value or None
