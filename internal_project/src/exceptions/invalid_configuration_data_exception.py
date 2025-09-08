# TODO: delete

# from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration_data_irregularity import \
#     PipelineConfigurationDataIrregularity
#
#
# class InvalidConfigurationDataException(Exception):
#     ###############################
#     ### Attributes of instances ###
#     ###############################
#     __irregularity__: PipelineConfigurationDataIrregularity
#
#
#
#     ###################
#     ### Constructor ###
#     ###################
#     def __init__(self, irregularity: PipelineConfigurationDataIrregularity) -> None:
#         super().__init__(str(irregularity))
#         self.__irregularity__ = irregularity
#
#
#
#     #########################
#     ### Getters & setters ###
#     #########################
#     @property
#     def irregularity(self) -> PipelineConfigurationDataIrregularity:
#         return self.__irregularity__