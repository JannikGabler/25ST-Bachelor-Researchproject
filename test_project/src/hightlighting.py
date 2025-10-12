import jax

from jax import numpy as jnp

from pipeline_entities.component_meta_info.default_component_meta_infos.node_generators.second_type_chebyshev_node_generator_meta_info import (
    second_type_chebyshev_node_generator_meta_info,
)
from pipeline_entities.components.abstracts.node_generator import NodeGenerator
from pipeline_entities.components.decorators.pipeline_component import (
    pipeline_component,
)
from pipeline_entities.data_transfer.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)
from pipeline_entities.data_transfer.pipeline_data import PipelineData


@pipeline_component(
    id="chebyshev2 node generator",
    type=NodeGenerator,
    meta_info=second_type_chebyshev_node_generator_meta_info,
)
class SecondTypeChebyshevNodeGenerator(NodeGenerator):
    ###############################
    ### Attributes of instances ###
    ###############################
    _compiled_jax_callable_: callable

    ###################
    ### Constructor ###
    ###################
    def __init__(
        self,
        pipeline_data: list[PipelineData],
        additional_execution_data: AdditionalComponentExecutionData,
    ) -> None:
        super().__init__(pipeline_data, additional_execution_data)
        data: PipelineData = pipeline_data[0]

        data_type: type = data.data_type
        node_count: int = data.node_count
        interpolation_interval: jnp.ndarray = data.interpolation_interval

        self._compiled_jax_callable_ = self._create_compiled_callable_(
            data_type, node_count, interpolation_interval
        )

    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        pipeline_data: PipelineData = self._pipeline_data_[0]

        nodes: jnp.ndarray = self._compiled_jax_callable_()

        pipeline_data.interpolation_nodes = nodes
        return pipeline_data

    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _create_compiled_callable_(
        data_type: type, node_count: int, interpolation_interval: jnp.ndarray
    ) -> callable:

        def _internal_perform_action_() -> jnp.ndarray:
            nodes: jnp.ndarray = jnp.arange(0, node_count, dtype=data_type)
            nodes = jnp.multiply(nodes, jnp.pi / (node_count - 1))
            nodes = jnp.cos(nodes)

            do_rescale = jnp.logical_or(
                interpolation_interval[0] != -1, interpolation_interval[1] != 1
            )

            def rescale_nodes():
                old_length = 2
                new_length = interpolation_interval[1] - interpolation_interval[0]
                length_ratio = new_length / old_length

                rescaled_nodes = jnp.multiply(nodes, length_ratio)
                return jnp.add(rescaled_nodes, interpolation_interval[0] + length_ratio)

            return jax.lax.cond(do_rescale, rescale_nodes, lambda: nodes)

        return jax.jit(_internal_perform_action_).lower().compile()


from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import (
    ComponentMetaInfo,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.post_dynamic_constraints.pipeline_data_dtype_required_post_constraint import (
    PipelineDataDtypeRequiredPostConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.attribute_required_constraint import (
    AttributeRequiredConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.max_predecessors_constraint import (
    MaxPredecessorsConstraint,
)
from pipeline_entities.pipeline.component_entities.constraints.default_constraints.static_constraints.min_predecessors_constraint import (
    MinPredecessorsConstraint,
)


second_type_chebyshev_node_generator_meta_info: ComponentMetaInfo = ComponentMetaInfo(
    attributes_modifying={"interpolation_nodes"},
    attributes_allowed_to_be_overridden={
        "data_type",
        "node_count",
        "interpolation_interval",
    },
    pre_dynamic_constraints=[],
    post_dynamic_constraints=[
        PipelineDataDtypeRequiredPostConstraint("interpolation_nodes")
    ],
    static_constraints=[
        AttributeRequiredConstraint("data_type"),
        AttributeRequiredConstraint("node_count"),
        AttributeRequiredConstraint("interpolation_interval"),
        MinPredecessorsConstraint(1),
        MaxPredecessorsConstraint(1),
    ],
)
