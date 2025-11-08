import unittest
import jax.numpy as jnp

from data_classes.pipeline_data.pipeline_data import PipelineData
from pipeline_entities.pipeline.component_entities.default_components.default_evaluation_components.aitken_neville_evaluator import (
    AitkenNevilleEvaluator,
)
from pipeline_entities.pipeline_execution.dataclasses.additional_component_execution_data import (
    AdditionalComponentExecutionData,
)


class PerformAction(unittest.TestCase):
    def test_something(self):
        def create_pipeline_data(
            node_count: int, evaluation_points_count: int
        ) -> PipelineData:
            pipeline_data: PipelineData = PipelineData()
            pipeline_data.data_type = jnp.float32

            pipeline_data.node_count = node_count
            pipeline_data.interpolation_nodes = jnp.linspace(
                -1, 1, node_count, dtype=pipeline_data.data_type
            )
            pipeline_data.interpolation_values = (
                pipeline_data.interpolation_nodes**3 - pipeline_data.interpolation_nodes
            )
            pipeline_data.interpolant_evaluation_points = jnp.linspace(
                -1, 1, evaluation_points_count, dtype=pipeline_data.data_type
            )

            return pipeline_data

        def create_additional_execution_data() -> AdditionalComponentExecutionData:
            return AdditionalComponentExecutionData({}, None, None, None, None)

        for node_count in range(4, 100):
            evaluation_points_count = 100
            pipeline_data: list[PipelineData] = [
                create_pipeline_data(node_count, evaluation_points_count)
            ]
            additional_execution_data: AdditionalComponentExecutionData = (
                create_additional_execution_data()
            )

            aitken_neville_evaluator: AitkenNevilleEvaluator = AitkenNevilleEvaluator(
                pipeline_data, additional_execution_data
            )

            result_data: PipelineData = aitken_neville_evaluator.perform_action()

            for i, evaluation_point in enumerate(
                result_data.interpolant_evaluation_points
            ):
                expected_result: float = evaluation_point**3 - evaluation_point

                self.assertAlmostEqual(
                    expected_result, result_data.interpolant_values[i], places=1
                )


if __name__ == "__main__":
    unittest.main()
