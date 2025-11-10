import importlib
from pathlib import Path
from typing import Any

from exceptions.directory_not_found_error import DirectoryNotFoundError
from exceptions.evaluation_error import EvaluationError






# def dynamically_load_files(folder: str | Path, recursive: bool=False) -> None:
#     if isinstance(folder, str):
#         folder: Path = Path(folder)
#
#     if not folder.exists() or not folder.is_dir():
#         raise DirectoryNotFoundError(f"The folder {folder} was not found.")
#
#     __register_components_from_folder__(folder, recursive)
#
#
#
# def register_component_from_file(file: str | Path) -> None:
#     if isinstance(file, str):
#         file: Path = Path(file)
#
#     if not file.exists() or not file.is_file():
#         raise FileNotFoundError(f"The file {file} was not found.")
#
#     __register_component_from_file__(file)
#
#
# def __register_components_from_folder__(folder: Path, recursive: bool) -> None:
#     if recursive:
#         for elem in folder.glob("*.py"):
#             if elem.is_file():
#                 __register_component_from_file__(elem)
#     else:
#         for elem in folder.iterdir():
#             if elem.is_file() and elem.suffix == ".py":
#                 __register_component_from_file__(elem)
#
#
#
# def __register_component_from_file__(file: Path) -> None:
#     module_name = file.stem  # z.B. "tmpngbxth15"
#     spec = importlib.util.spec_from_file_location(module_name, str(file))
#
#     if spec is None or spec.loader is None:
#         raise ImportError(f"Could not load spec for module {module_name} from file {file}.")
#
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#
#     importlib.import_module(file.name)