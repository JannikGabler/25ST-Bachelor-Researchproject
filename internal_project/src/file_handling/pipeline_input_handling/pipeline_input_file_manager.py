from dataclasses import fields
from pathlib import Path

from exceptions.not_instantiable_error import NotInstantiableError
from file_handling.ini_handling.ini_file_manager import INIFileManager
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData


class PipelineInputFileManager:

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"{self.__class__.__name__} cannot be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_from_file(path: Path) -> PipelineInputData:
        data: PipelineInputData = PipelineInputData()

        entries: dict[str, str] = INIFileManager.load_file(path)

        for key, value in entries.items():
            if key.startswith('§'):
                PipelineInputFileManager._handle_direct_injecting_key_value_pair_in_loading_(key, value, data)
            else:
                PipelineInputFileManager._handle_regular_key_value_pair_in_loading_(key, value, data)

        return data



    # TODO
    @staticmethod
    def save_to_file(to_save: PipelineInput | PipelineInputData, path: Path) -> None:
        pass



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _handle_regular_key_value_pair_in_loading_(key: str, value: str, data: PipelineInputData) -> None:
        if any(field.name == key for field in fields(PipelineInputData)) and key not in ["additional_values", "additional_directly_injected_values"]:
            setattr(data, key, value)
        else:
            data.additional_values[key] = value



    @staticmethod
    def _handle_direct_injecting_key_value_pair_in_loading_(key: str, value: str, data: PipelineInputData) -> None:
        key = key[1:]
        data.additional_directly_injected_values[key] = value



    # TODO: delete
    # @staticmethod
    # def _handle_line_in_loading_(line_number: int, line: str, path: Path, data: PipelineInputData) -> None:
    #     if line and not line.startswith("#"):
    #         key, value = PipelineInputFileManager._extract_key_value_from_line_(line_number, line, path)
    #
    #         if not key:
    #             raise ValueError(f"Line {line_number} of file '{str(path)}' contains invalid syntax. The key is empty.")
    #
    #         if key.startswith('§'):
    #             PipelineInputFileManager._handle_direct_injecting_key_value_pair_in_loading_(key, value, data)
    #         else:
    #             PipelineInputFileManager._handle_regular_key_value_pair_in_loading_(key, value, data)



    # @staticmethod
    # def _extract_key_value_from_line_(line_number: int, line: str, path: Path) -> tuple[str, str]:
    #     if '=' not in line:
    #         raise ValueError(f"Line {line_number} of file '{str(path)}' contains invalid syntax. There was no '=' character found.")
    #
    #     raw_key, raw_val = line.split('=', 1)
    #     key: str = raw_key.strip()
    #     value: str = raw_val.strip()
    #
    #     return key, value