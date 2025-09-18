from __future__ import annotations
import json
from pathlib import Path

from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver
from file_handling.result_persistence.utils import slugify

class JsonSaver(Saver):
    """Saves JSON-like (dict/list) payloads."""
    kind = "json"

    def save(self, artifact: dict | list, run_dir: Path, policy: SavePolicy) -> Path:
        # name = slugify(getattr(artifact, "name", "data"))
        # meta = dict(getattr(artifact, "metadata", {}) or {})
        # indent = int(meta.get("indent", policy.json_indent))
        # json_dir = run_dir / "json"
        # json_dir.mkdir(parents=True, exist_ok=True)

        # payload = getattr(artifact, "payload", None)
        # if payload is None:
        #     raise ValueError("JsonArtifact payload is None.")

        # out = json_dir / f"{name}.json"
        # with out.open("w", encoding="utf-8") as f:
        #     json.dump(payload, f, indent=indent, ensure_ascii=False)
        # return out
        pass # TODO

register_saver(JsonSaver)
