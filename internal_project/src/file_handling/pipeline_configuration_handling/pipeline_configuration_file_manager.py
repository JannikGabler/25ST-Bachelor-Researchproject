from dataclasses import fields
from pathlib import Path

from exceptions.not_instantiable_error import NotInstantiableError
from file_handling.ini_handling.ini_file_manager import INIFileManager
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data import PipelineConfigurationData
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput
from pipeline_entities.pipeline_input.pipeline_input_data import PipelineInputData


class PipelineConfigurationFileManager:

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"{self.__class__.__name__} cannot be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def load_from_file(path: Path) -> PipelineConfigurationData:
        data: PipelineConfigurationData = PipelineConfigurationData()

        entries: dict[str, str] = INIFileManager.load_file(path)

        for key, value in entries.items():
            PipelineConfigurationFileManager._handle_key_value_pair_in_loading_(key, value, data)

        return data



    # TODO
    @staticmethod
    def save_to_file(to_save: PipelineInput | PipelineInputData, path: Path) -> None:
        pass



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _handle_key_value_pair_in_loading_(key: str, value: str, data: PipelineConfigurationData) -> None:
        if any(field.name == key for field in fields(PipelineConfigurationData)) and key != "additional_values":
            setattr(data, key, value)
        else:
            data.additional_values[key] = value

