import textwrap
import tempfile

from pathlib import Path

from file_handling.pipeline_input_handling.pipeline_input_file_manager import (
    PipelineInputFileManager,
)
from pipeline_entities.pipeline_execution.output.pipeline_component_execution_report import (
    PipelineComponentExecutionReport,
)
from data_classes.pipeline_configuration.pipeline_configuration_data import (
    PipelineConfigurationData,
)
from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import (
    PipelineConfigurationFileManager,
)
from data_classes.pipeline_configuration.pipeline_configuration import (
    PipelineConfiguration,
)
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.data_classes.pipeline import Pipeline
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import (
    PipelineManager,
)
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager

InternalLogicSetupManager.setup()

pipeline_configuration_file_content: bytes = textwrap.dedent(
    """\
    name="DemoPipeline"
    supported_program_version=Version(\"1.0.0\")
    components=DirectionalAcyclicGraph(\"\"\"
        0=Base Input
        1=Equidistant Node Generator
            predecessors=["0"]
        2=Function Expression Input
            predecessors=["1"]
        \"\"\")
    extra_value=True
    """
).encode("utf-8")


temp_dir = tempfile.TemporaryDirectory()
temp_pipeline_configuration_file = Path(temp_dir.name + "/pipeline_configuration.ini")

with open(temp_pipeline_configuration_file, "wb") as f:
    f.write(pipeline_configuration_file_content)

print(temp_dir.name)


pipeline_configuration_data: PipelineConfigurationData = (
    PipelineConfigurationFileManager.load_from_file(temp_pipeline_configuration_file)
)

print(pipeline_configuration_data)


pipeline_configuration: PipelineConfiguration = PipelineConfiguration(
    pipeline_configuration_data
)

print(pipeline_configuration)

pipeline_input_file_content: bytes = textwrap.dedent(
    """\
    name="TestPipeline"
    data_type=jax.numpy.float32
    node_count=5
    interpolation_interval=jax.numpy.array([-1, 1])
    function_expression="x**2 + 1"
    piecewise_function_expressions=[((0,1), 'x'), ((1,2), 'x**2')]
    sympy_function_expression_simplification=True
    function_callable=lambda x: x**2 + 3
    function_values=jax.numpy.array([0.0, 1.0, 4.0, 9.0, 16.0])
    §secret_token="abc123"
    extra_value=[1, 2, 3]
    """
).encode("utf-8")


temp_pipeline_input_file = Path(temp_dir.name + "/pipeline_input.ini")

with open(temp_pipeline_input_file, "wb") as f:
    f.write(pipeline_input_file_content)


pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(
    temp_pipeline_input_file
)

print(pipeline_input_data)

pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)

print(pipeline_input)

pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)

pipeline_manager: PipelineManager = PipelineManager(pipeline)

pipeline_manager.execute_all()


pipeline_manager: PipelineManager = PipelineManager(pipeline)

while not pipeline_manager.is_completely_executed:
    report: PipelineComponentExecutionReport = pipeline_manager.execute_next_component()
    node_name: str = report.component_instantiation_info.component_name
    node_id: str = report.component_instantiation_info.component.component_id

    print(
        f"###### Report from node {report.component_instantiation_info.component_name} ({report.component_instantiation_info.component.component_id}) ######"
    )
    print(report)
    print("\n\n")
