from __future__ import annotations
from pathlib import Path

from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver
from file_handling.result_persistence.utils import slugify

class BinarySaver(Saver):
    """
    Saves opaque bytes payloads.
    Can be used as fallback for overly domain-specific objects.
    """
    kind = "binary"

    def save(self, , run_dir: Path, policy: SavePolicy) -> Path:
        # name = slugify(getattr(artifact, "name", "binary"))
        # meta = dict(getattr(artifact, "metadata", {}) or {})
        # ext = meta.get("ext", "bin")
        # bin_dir = run_dir / "binary"
        # bin_dir.mkdir(parents=True, exist_ok=True)

        # payload = getattr(artifact, "payload", None)
        # if payload is None:
        #     raise ValueError("BinaryArtifact payload is None.")
        # if not isinstance(payload, (bytes, bytearray, memoryview)):
        #     raise TypeError("BinaryArtifact payload must be bytes-like.")

        # out = bin_dir / f"{name}.{ext}"
        # with out.open("wb") as f:
        #     f.write(bytes(payload))
        # return out
        pass # TODO

register_saver(BinarySaver)
