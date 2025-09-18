from __future__ import annotations
from pathlib import Path
from data_classes.plot_template.plot_template import PlotTemplate, FigureWrapper
from file_handling.result_persistence.save_policy import SavePolicy
from file_handling.result_persistence.savers.base import Saver, register_saver
from file_handling.result_persistence.utils import slugify 

class PlotSaver(Saver): 
    """Saves matplotlib figures (or axes) to disk."""
    kind = "plot"
    
    def save(self, plot_template: PlotTemplate, run_dir: Path, policy: SavePolicy) -> Path:
         name = slugify("plot") # TODO better naming 
         exts: list[str] = list(policy.plot_formats) # file extensions 
         
         # TODO default values should be in save policy 
         dpi = 150 
         transparent = False

         plot_dir = run_dir / "plots" 
         plot_dir.mkdir(parents=True, exist_ok=True) 

         fig_wrapper: FigureWrapper = plot_template.fig()

         primary_path: Path | None = None 
         for ext in exts: 
            out = plot_dir / f"{name}.{ext}" 
            try: 
                fig_wrapper.savefig(out, dpi=dpi, format=ext, bbox_inches="tight", transparent=transparent) 
                if primary_path is None: 
                    primary_path = out 
            except Exception:
                print(f"⚠️ Warning: Could not save plot {name} with extension {ext}.")
                continue
            if primary_path is None: 
                raise RuntimeError(f"Could not save matplotlib figure for {name} with formats {exts}.") 
            return primary_path

register_saver(PlotSaver)
