from interpolants.default_interpolants.aitken_neville_interpolant import AitkenNevilleInterpolant
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.interpolation_cores.aitken_neville_interpolation_core_meta_info import \
    aitken_neville_interpolation_core_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


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
        data: PipelineData = pipeline_data[0]

        if data.data_type is not None:
            data.interpolation_nodes = data.interpolation_nodes.astype(data.data_type)
            data.interpolation_values = data.interpolation_values.astype(data.data_type)



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



