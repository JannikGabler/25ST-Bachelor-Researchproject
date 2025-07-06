from data_structures.interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.component_meta_info.default_component_meta_infos.interpolation_cores.aitken_neville_interpolation_core_meta_info import \
    aitken_neville_interpolation_core_meta_info
from pipeline_entities.components.abstracts.interpolation_core import InterpolationCore

from pipeline_entities.components.decorators.pipeline_component import pipeline_component
from pipeline_entities.data_transfer.additional_component_execution_data import AdditionalComponentExecutionData
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(id="aitken neville interpolation", type=InterpolationCore, meta_info=aitken_neville_interpolation_core_meta_info)
class AitkenNevilleInterpolationCore(InterpolationCore):
    """
    TODO
    """
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable



    ###################
    ### Constructor ###
    ###################
    def __init__(self, pipeline_data: list[PipelineData], additional_execution_data: AdditionalComponentExecutionData) -> None:
        super().__init__(pipeline_data, additional_execution_data)



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        interpolant = AitkenNevilleInterpolant(
            nodes=pipeline_data.interpolation_nodes,
            values=pipeline_data.interpolation_values,
        )

        pipeline_data.interpolant = interpolant

        return pipeline_data



