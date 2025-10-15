from dataclasses import fields
from pathlib import Path

from exceptions.not_instantiable_error import NotInstantiableError
from file_handling.ini_handling.ini_file_manager import INIFileManager
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData


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

        entries: dict[str, str] = INIFileManager.load_file_as_key_value_pairs(path)

        for key, value in entries.items():
            if key.startswith("§"):
                PipelineInputFileManager._handle_direct_injecting_key_value_pair_in_loading_(key, value, data)
            else:
                PipelineInputFileManager._handle_regular_key_value_pair_in_loading_(key, value, data)

        return data


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
