from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration
from pipeline_entities.pipeline_input.pipeline_input import PipelineInput


class PipelineBuilder:


    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"The class '{self.__class__.__name__}' cannot be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def build(pipeline_configuration: PipelineConfiguration, pipeline_input: PipelineInput) -> Pipeline:
        return Pipeline(pipeline_configuration, pipeline_input)