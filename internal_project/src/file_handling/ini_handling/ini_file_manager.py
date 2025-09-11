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
        raise NotInstantiableError(f"Class '{self.__class__.__name__}' cannot be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_file_as_entry_list(path: Path) -> list[str]:
        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        file_content: str
        with open(path, 'r', encoding='utf-8') as file:
            file_content: str = file.read()

        return INIFormatUtils.split_into_entries(file_content)



    @staticmethod
    def load_file_as_key_value_pairs(path: Path) -> dict[str, str]:
        if not (path.is_file() and path.exists()):
            raise FileNotFoundError(f"File '{str(path)}' does not exists.")

        entries: list[str] = INIFileManager.load_file_as_entry_list(path)
        return INIFormatUtils.split_entries_into_key_value_pairs(entries)



    # @staticmethod
    # def load_file(path: Path) -> dict[str, str]:
    #     if not (path.is_file() and path.exists()):
    #         raise FileNotFoundError(f"File '{str(path)}' does not exists.")
    #
    #     entry_list: list[str] = INIFileManager._collect_entries_from_file_(path)
    #     return INIFileManager._split_entry_list_into_dict_(entry_list)



    # TODO
    @staticmethod
    def save_to_file(to_save: PipelineInput | PipelineInputData, path: Path) -> None:
        pass



    #######################
    ### Private methods ###
    #######################
    # @staticmethod
    # def _collect_entries_from_file_(path: Path) -> list[str]:
    #     entry_strings: list[str] = []
    #     current_string: str = ""
    #
    #     with open(path, 'r', encoding='utf-8') as file:
    #         for line_number, raw_line in enumerate(file, start=1):
    #             current_string, add_to_entries = INIFileManager._process_line_while_collecting_(raw_line, current_string)
    #
    #             if add_to_entries:
    #                 entry_strings.append(current_string)
    #                 current_string = ""
    #
    #     return entry_strings
    #
    #
    #
    # @staticmethod
    # def _process_line_while_collecting_(raw_line: str, current_accumulation: str) -> tuple[str, bool]:
    #     leading_stripped: str = raw_line.strip()
    #     trailing_stripped: str = raw_line.rstrip()
    #
    #     if not leading_stripped.startswith(('#', '//', '[')) and not raw_line.isspace():
    #         if current_accumulation:
    #             string_to_add: str = "\n" + trailing_stripped
    #         else:
    #             string_to_add: str = raw_line.strip()
    #
    #         if string_to_add.endswith('\\'):
    #             return current_accumulation + string_to_add[:-1].rstrip(), False
    #         else:
    #             return current_accumulation + string_to_add, True
    #     else:
    #         return current_accumulation, False




