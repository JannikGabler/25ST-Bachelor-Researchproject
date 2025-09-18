from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from data_classes.plot_template.plot_template import PlotTemplate

from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import get_saver

# Import savers to ensure they self-register
# ! DO NOT REMOVE THESE IMPORTS !
from file_handling.result_persistence.savers import plot_saver   # noqa: F401
from file_handling.result_persistence.savers import json_saver    # noqa: F401
from file_handling.result_persistence.savers import text_saver    # noqa: F401
from file_handling.result_persistence.savers import binary_saver  # noqa: F401


class FilesystemResultStore:
    """
    Orchestrates persistence:
    - prepares run directory according to SavePolicy
    - delegates saving to registered savers by 'kind'
    - writes a manifest if needed
    """
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    ################################
    ### Run directory management ###
    ################################

    def prepare_run_dir(self, policy: SavePolicy) -> Path:
        """
        Create and return the run directory based on the policy.
        Soft-state uses <output_root>/runs/latest (with optional rotation);
        snapshot uses a new timestamped directory.
        """
        if policy.mode == "soft-state":
            run_dir = self.runs_root / "latest"
            if run_dir.exists() and policy.keep_soft_state_n > 1:
                self._rotate_soft_state(run_dir, policy.keep_soft_state_n)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        elif policy.mode == "snapshot":
            run_dir = self._new_timestamped_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unknown SavePolicy.mode: {policy.mode}")
        # Per-kind subfolders are created by savers on demand.
        return run_dir

    def promote_latest_to_snapshot(self) -> Path:
        """
        Copy runs/latest into a new timestamped snapshot directory and return the destination path.
        """
        src = self.runs_root / "latest"
        if not src.exists():
            raise FileNotFoundError("No 'latest' directory to promote.")
        dst = self._new_timestamped_dir()
        shutil.copytree(src, dst)
        return dst

    #########################
    ### Delegated saving  ###
    #########################

    def _save_artifact(self, artifact: Any, run_dir: Path, policy: SavePolicy) -> Path | None:
        """
        Try to save artifact:
        - PlotTemplate -> 'plot' saver
        - list/tuple -> recurse into items; if none saved, save the whole container via 'binary'
        - everything else -> 'binary' saver
        Returns the first successfully saved Path, or None.
        """
        plot_saver = get_saver("plot")
        bin_saver  = get_saver("binary")

        match artifact:
            # single plot
            case PlotTemplate():
                return plot_saver.save(artifact, run_dir, policy)

            # handle lists/tuples recursively
            case list() | tuple():
                primary: str = None
                for item in artifact:
                    p = self._save_artifact(item, run_dir, policy) # TODO maybe put them in a sub-folder
                    if p is not None:
                        primary = p
                # nothing inside saved -> fallback to saving the container as a blob
                return primary if primary is not None else bin_saver.save(artifact, run_dir, policy)

            # fallback
            case _:
                return bin_saver.save(artifact, run_dir, policy)

    # ################ TODO remove manifest
    # ### Manifest ###
    # ################

    # def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    #     """
    #     Write a run_manifest.json into the run directory and return its path.
    #     """
    #     path = run_dir / "run_manifest.json"
    #     with path.open("w", encoding="utf-8") as f:
    #         json.dump(manifest, f, indent=2, ensure_ascii=False)
    #     return path

    #################
    ### Internals ###
    #################

    def _new_timestamped_dir(self) -> Path:
        """
        Create a new unique timestamped directory name under runs/.
        """
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        path = self.runs_root / ts
        idx = 1
        while path.exists():
            path = self.runs_root / f"{ts}_{idx}"
            idx += 1
        return path

    @staticmethod
    def _rotate_soft_state(latest_dir: Path, keep_n: int) -> None:
        """
        Rotate soft-state directories: latest -> latest-1 -> ... up to keep_n-1.
        Removes the oldest if it would exceed keep_n.
        """
        base = latest_dir.parent

        # Remove the oldest if present
        oldest = base / f"latest-{keep_n-1}"
        if oldest.exists():
            shutil.rmtree(oldest)

        # Shift downward
        for i in range(keep_n - 2, -1, -1):
            src_name = "latest" if i == 0 else f"latest-{i}"
            src = base / src_name
            if src.exists():
                dst = base / f"latest-{i+1}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
