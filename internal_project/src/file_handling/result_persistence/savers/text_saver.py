from __future__ import annotations
from pathlib import Path

from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver
from file_handling.result_persistence.utils import slugify

class TextSaver(Saver):
    """Saves plain text."""
    def save(self, artifact: str, run_dir: Path, policy: SavePolicy) -> Path:
        # name = slugify(getattr(artifact, "name", "text"))
        # meta = dict(getattr(artifact, "metadata", {}) or {})
        # ext = meta.get("ext", "txt")
        # text_dir = run_dir / "text"
        # text_dir.mkdir(parents=True, exist_ok=True)

        # payload = getattr(artifact, "payload", None)
        # if payload is None:
        #     raise ValueError("TextArtifact payload is None.")
        # if not isinstance(payload, str):
        #     payload = str(payload)

        # out = text_dir / f"{name}.{ext}"
        # with out.open("w", encoding="utf-8") as f:
        #     f.write(payload)
        # return out
        pass # TODO

register_saver(TextSaver())
