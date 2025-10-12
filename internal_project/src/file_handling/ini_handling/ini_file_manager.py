from pathlib import Path

from exceptions.not_instantiable_error import NotInstantiableError
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from utils.ini_format_utils import INIFormatUtils


class INIFileManager:

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(
            f"Class '{self.__class__.__name__}' cannot be instantiated."
        )

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_file_as_entry_list(path: Path) -> list[str]:
        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        file_content: str
        with open(path, "r", encoding="utf-8") as file:
            file_content: str = file.read()

        return INIFormatUtils.split_into_entries(file_content)

    @staticmethod
    def load_file_as_key_value_pairs(path: Path) -> dict[str, str]:
        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        entries: list[str] = INIFileManager.load_file_as_entry_list(path)
        return INIFormatUtils.split_entries_into_key_value_pairs(entries)

    # TODO
    @staticmethod
    def save_to_file(to_save: PipelineInput | PipelineInputData, path: Path) -> None:
        pass
