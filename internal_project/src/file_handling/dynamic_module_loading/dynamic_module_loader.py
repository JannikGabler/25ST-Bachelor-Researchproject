import importlib.util
import sys
from pathlib import Path
from typing import Any

from exceptions.directory_not_found_error import DirectoryNotFoundError
from exceptions.module_already_loaded_exception import ModuleAlreadyLoadedException
from exceptions.not_instantiable_error import NotInstantiableError


class DynamicModuleLoader:
    """
    Utility class for dynamically loading Python modules from a directory. This class is a static utility and is not instantiable.
    """

    ##########################
    ### Attribute of class ###
    ##########################
    _loaded_modules_: dict[str, Any] = {}


    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when attempting to instantiate the class.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.'")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_directory(path: Path):
        """
        Load all Python modules found under the given directory.

        Args:
            path (Path): Root directory to scan for files.

        Raises:
            DirectoryNotFoundError: If the provided path does not exist or is not a directory.
            ModuleAlreadyLoadedException: If a module with the same name has already been loaded.
            ImportError: If a file cannot be loaded or an exception occurs during execution.
        """

        path = path.resolve()

        if not (path.is_dir() and path.exists()):
            raise DirectoryNotFoundError(f"Couldn't find a directory at '{path}'.")

        for py_file in path.rglob("*.py"):
            DynamicModuleLoader._load_py_file_(py_file, path)


    @staticmethod
    def get_entity(fully_qualified_name: str):
        """
        Return an attribute from a previously loaded module.

        Args:
            fully_qualified_name (str): Dotted path to the target.

        Returns:
            Any: The resolved attribute from the loaded module.

        Raises:
            ValueError: If no loaded module/attribute matches the given name.
        """

        parts = fully_qualified_name.split(".")

        for i in range(len(parts), 0, -1):
            module_name = ".".join(parts[:i])
            attr_path = parts[i:]
            module = sys.modules.get(module_name)

            if module:
                entity = module
                for attr in attr_path:
                    entity = getattr(entity, attr)

                return entity

        raise ValueError(f"There is not entity loaded with the name '{fully_qualified_name}'.")


    @staticmethod
    def get_module_namespace() -> dict[str, Any]:
        """
        Return the internal namespace of loaded modules.

        Returns:
            dict[str, Any]: Mapping of module names to loaded module objects.
        """

        return DynamicModuleLoader._loaded_modules_


    @staticmethod
    def unload_all_modules() -> None:
        """
        Unload all modules that were loaded via this loader and clear the namespace.

        Returns:
            None
        """

        for module_name in list(DynamicModuleLoader._loaded_modules_):
            if module_name in sys.modules:
                del sys.modules[module_name]

        DynamicModuleLoader._loaded_modules_.clear()


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _load_py_file_(file_path: Path, directory_path: Path) -> None:
        rel_path = file_path.relative_to(directory_path).with_suffix("")
        module_name = ".".join(rel_path.parts)

        if module_name in sys.modules:
            raise ModuleAlreadyLoadedException(f"A module with the name {module_name} is already loaded.")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Couldn't load the file '{file_path}'.")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Exception while loading the file '{file_path}': {str(e)}.")

        sys.modules[module_name] = module
        DynamicModuleLoader._loaded_modules_[module_name] = module
