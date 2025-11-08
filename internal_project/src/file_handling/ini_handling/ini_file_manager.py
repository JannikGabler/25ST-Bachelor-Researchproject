from pathlib import Path

from exceptions.not_instantiable_error import NotInstantiableError
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from utils.ini_format_utils import INIFormatUtils


class INIFileManager:
    """
    Utility class for reading and writing INI files. This class is a static utility and is not instantiable.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """
        Raises:
            NotInstantiableError: Always raised when attempting to instantiate the class.
        """

        raise NotInstantiableError(f"Class '{self.__class__.__name__}' cannot be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_file_as_entry_list(path: Path) -> list[str]:
        """
        Load a file and return its content split into logical entries.

        Args:
            path (Path): Path to the file.

        Returns:
            list[str]: The list of parsed entry strings.

        Raises:
            FileNotFoundError: If the path does not exist or is not a file.
        """

        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        file_content: str
        with open(path, "r", encoding="utf-8") as file:
            file_content: str = file.read()

        return INIFormatUtils.split_into_entries(file_content)


    @staticmethod
    def load_file_as_key_value_pairs(path: Path) -> dict[str, str]:
        """
        Load a file and return its key–value pairs.

        Args:
            path (Path): Path to the file.

        Returns:
            dict[str, str]: Mapping of keys to values parsed from the file.

        Raises:
            FileNotFoundError: If the path does not exist or is not a file.
        """

        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        entries: list[str] = INIFileManager.load_file_as_entry_list(path)
        return INIFormatUtils.split_entries_into_key_value_pairs(entries)


    @staticmethod
    def save_to_file(to_save: PipelineInput | PipelineInputData, path: Path) -> None:
        """
        Save a pipeline input object to a file.

        Args:
            to_save (PipelineInput | PipelineInputData): The object to serialize.
            path (Path): Target file path.

        Returns:
            None
        """

        pass
