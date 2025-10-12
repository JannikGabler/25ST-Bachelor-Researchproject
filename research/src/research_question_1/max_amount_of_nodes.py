import jax.numpy as jnp

from jax.typing import DTypeLike

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.second_type_chebyshev_node_generator import (
    SecondTypeChebyshevNodeGenerator,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)


def create_input_data(
    amount_of_nodes: int, data_type: DTypeLike
) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
    pd: PipelineData = PipelineData()
    pd.node_count = amount_of_nodes
    pd.data_type = data_type
    pd.interpolation_interval = jnp.array([-1.0, 1.0], dtype=data_type)

    return [pd], AdditionalComponentExecutionData(None, None, None, None, None)


def create_chebyshev_2_nodes(amount_of_nodes: int, data_type: DTypeLike) -> jnp.ndarray:
    pipeline_data, additional_execution_data = create_input_data(
        amount_of_nodes, data_type
    )

    generator: SecondTypeChebyshevNodeGenerator = SecondTypeChebyshevNodeGenerator(
        pipeline_data, additional_execution_data
    )

    result_data: PipelineData = generator.perform_action()
    return result_data.interpolation_nodes


def contains_duplicate_nodes(amount_of_nodes: int, data_type: DTypeLike) -> bool:
    nodes: jnp.ndarray = create_chebyshev_2_nodes(amount_of_nodes, data_type)

    for i in range(amount_of_nodes - 1):
        if jnp.array_equal(nodes[i], nodes[i + 1]):
            return True

    return False


def exponential_search(data_type: DTypeLike) -> int:
    node_count: int = 1
    while not contains_duplicate_nodes(node_count, data_type):
        node_count *= 2
        print(f"{node_count}", end=" ")

    print("")
    return node_count


def binary_search(data_type: DTypeLike, exponential_result: int) -> int:
    lower_bound: int = exponential_result // 2 + 1
    upper_bound: int = exponential_result

    while lower_bound < upper_bound:
        print(f"({lower_bound} - {upper_bound})", end=" ")
        mid: int = (lower_bound + upper_bound) // 2
        if contains_duplicate_nodes(mid, data_type):
            upper_bound = mid
        else:
            lower_bound = mid + 1

    print("")
    return upper_bound


def get_amount(data_type: DTypeLike) -> int:
    exponential_result: int = exponential_search(data_type)
    return binary_search(data_type, exponential_result)


print(f"Chebyshev 2, float32: {get_amount(jnp.float32)}")
print(f"Chebyshev 2, float16: {get_amount(jnp.float16)}")
print(f"Chebyshev 2, bfloat16: {get_amount(jnp.bfloat16)}")
