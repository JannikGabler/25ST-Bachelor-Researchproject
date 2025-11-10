from functools import partial

import jax
import jax.numpy as jnp

from jax.typing import DTypeLike

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.equidistant_node_generator import \
    EquidistantNodeGenerator
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.first_type_chebyshev_node_generator import \
    FirstTypeChebyshevNodeGenerator
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators.second_type_chebyshev_node_generator import \
    SecondTypeChebyshevNodeGenerator
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import \
    AdditionalComponentExecutionData


def create_input_data(amount_of_nodes: int, data_type: DTypeLike) -> tuple[list[PipelineData], AdditionalComponentExecutionData]:
    pd: PipelineData = PipelineData()
    pd.node_count = amount_of_nodes
    pd.data_type = data_type
    pd.interpolation_interval = jnp.array([-1.0, 1.0], dtype=data_type)

    return [pd], AdditionalComponentExecutionData(None, None, None, None, None)


def create_equidistant_nodes(amount_of_nodes: int, data_type: DTypeLike) -> jnp.ndarray:
    pipeline_data, additional_execution_data = create_input_data(amount_of_nodes, data_type)

    generator: EquidistantNodeGenerator = EquidistantNodeGenerator(pipeline_data, additional_execution_data)

    result_data: PipelineData = generator.perform_action()
    return result_data.interpolation_nodes


def create_chebyshev_1_nodes(amount_of_nodes: int, data_type: DTypeLike) -> jnp.ndarray:
    pipeline_data, additional_execution_data = create_input_data(amount_of_nodes, data_type)

    generator: FirstTypeChebyshevNodeGenerator = FirstTypeChebyshevNodeGenerator(pipeline_data, additional_execution_data)

    result_data: PipelineData = generator.perform_action()
    return result_data.interpolation_nodes


def create_chebyshev_2_nodes(amount_of_nodes: int, data_type: DTypeLike) -> jnp.ndarray:
    pipeline_data, additional_execution_data = create_input_data(amount_of_nodes, data_type)

    generator: SecondTypeChebyshevNodeGenerator = SecondTypeChebyshevNodeGenerator(pipeline_data, additional_execution_data)

    result_data: PipelineData = generator.perform_action()
    return result_data.interpolation_nodes



# @partial(jax.jit, static_argnames=['amount_of_nodes', "data_type", "node_type"])
def contains_duplicate_nodes(amount_of_nodes: int, data_type: DTypeLike, node_type: str) -> bool:
    nodes: jnp.ndarray

    if node_type == "equidistant":
        nodes = create_equidistant_nodes(amount_of_nodes, data_type)
    elif node_type == "chebyshev_1":
        nodes = create_chebyshev_1_nodes(amount_of_nodes, data_type)
    elif node_type == "chebyshev_2":
        nodes = create_chebyshev_2_nodes(amount_of_nodes, data_type)

    if len(nodes) < 2:
        return False

    return check_for_duplicates(nodes)


@jax.jit
def check_for_duplicates(nodes: jnp.ndarray) -> bool:
    n = nodes.shape[0]

    def body_fun(i, found_duplicate):
        def check_next(_):
            return jnp.logical_or(found_duplicate, jnp.all(nodes[i] == nodes[i + 1]))
        return jax.lax.cond(found_duplicate, lambda _: found_duplicate, check_next, operand=None)

    return jax.lax.fori_loop(0, n - 1, body_fun, False)


def exponential_search(data_type: DTypeLike, node_type: str) -> int:
    node_count: int = 1
    while not contains_duplicate_nodes(node_count, data_type, node_type):
        node_count *= 10
        print(f"{node_count}", end=" ")

    print("")
    return node_count


def binary_search(data_type: DTypeLike, exponential_result: int, node_type: str) -> int:
    lower_bound: int = exponential_result // 10 + 1
    upper_bound: int = exponential_result

    while lower_bound < upper_bound:
        print(f"({lower_bound} - {upper_bound})", end=" ")
        mid: int = (lower_bound + upper_bound) // 2
        if contains_duplicate_nodes(mid, data_type, node_type):
            upper_bound = mid
        else:
            lower_bound = mid + 1

    print("")
    return upper_bound - 1



def get_amount(data_type: DTypeLike, node_type: str) -> int:
    exponential_result: int = exponential_search(data_type, node_type)
    return binary_search(data_type, exponential_result, node_type)



if __name__ == "__main__":
    for node_type in ["equidistant", "chebyshev_1", "chebyshev_2"]:
        for data_type in [jnp.float16, jnp.bfloat16, jnp.float32]:
            print(f"{node_type} {data_type}: {get_amount(data_type, node_type)}")


# print(f"Chebyshev 2, float32: {get_amount(jnp.float32, "chebyshev_2")}")
# print(f"Chebyshev 2, float16: {get_amount(jnp.float16, "chebyshev_2")}")
# print(f"Chebyshev 2, bfloat16: {get_amount(jnp.bfloat16, "chebyshev_2")}")