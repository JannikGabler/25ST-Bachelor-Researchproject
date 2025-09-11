import tempfile
import textwrap
from pathlib import Path

from file_handling.pipeline_configuration_handling.pipeline_configuration_file_manager import \
    PipelineConfigurationFileManager
from file_handling.pipeline_input_handling.pipeline_input_file_manager import PipelineInputFileManager
from data_classes.pipeline_configuration.pipeline_configuration import PipelineConfiguration
from data_classes.pipeline_configuration.pipeline_configuration_data import \
    PipelineConfigurationData
from data_classes.pipeline_input.pipeline_input import PipelineInput
from data_classes.pipeline_input.pipeline_input_data import PipelineInputData
from pipeline_entities.pipeline.pipeline import Pipeline
from pipeline_entities.pipeline.pipeline_builder.pipeline_builder import PipelineBuilder
from pipeline_entities.pipeline_execution.pipeline_manager.pipeline_manager import PipelineManager
from setup_manager.internal_logic_setup_manager import InternalLogicSetupManager

InternalLogicSetupManager.setup()

temp_dir = tempfile.TemporaryDirectory()



pipeline_configuration_file_content: bytes = textwrap.dedent("""\
    name="DemoPipeline"
    supported_program_version=Version(\"1.0.0\")
    components=DirectionalAcyclicGraph(\"\"\"
        0=Base Input
        1=Equidistant Node Generator
            predecessors=["0"]
        2=Function Expression Input
            predecessors=["1"]
        3=Interpolation Values Evaluator
            predecessors=["2"]
        4=Aitken Neville Interpolation
            predecessors=["3"]
        5=Newton Interpolation
            predecessors=["3"]
        6=Barycentric1 Interpolation
            predecessors=["3"]
        7=Barycentric2 Interpolation
            predecessors=["3"]
        8=Interpolant Plotter
            predecessors=["4","5","6","7"]
        9=Absolute Error Plotter
            predecessors=["4","5","6","7"]
            y_limit=5
        \"\"\")

    runs_for_component_execution_time_measurements=1
    """).encode("utf-8")

temp_pipeline_configuration_file = Path(temp_dir.name + "/pipeline_configuration.ini")

with open(temp_pipeline_configuration_file, "wb") as f:
    f.write(pipeline_configuration_file_content)

pipeline_configuration_data: PipelineConfigurationData = PipelineConfigurationFileManager.load_from_file(
    temp_pipeline_configuration_file)
# print(pipeline_configuration_data)

pipeline_configuration: PipelineConfiguration = PipelineConfiguration(pipeline_configuration_data)
# print(pipeline_configuration)


pipeline_input_file_content: bytes = textwrap.dedent("""\
    name="TestPipeline"
    data_type=jax.numpy.float32
    node_count=36
    interpolation_interval=jax.numpy.array([-1, 1])
    function_expression="sin(10*x)"
    interpolant_evaluation_points=jax.numpy.empty(0)
    sympy_function_expression_simplification=True
    """).encode("utf-8")

temp_pipeline_input_file = Path(temp_dir.name + "/pipeline_input.ini")

with open(temp_pipeline_input_file, "wb") as f:
    f.write(pipeline_input_file_content)

pipeline_input_data: PipelineInputData = PipelineInputFileManager.load_from_file(temp_pipeline_input_file)
# print(pipeline_input_data)

pipeline_input: PipelineInput = PipelineInput(pipeline_input_data)
# print(pipeline_input)


pipeline: Pipeline = PipelineBuilder.build(pipeline_configuration, pipeline_input)

pipeline_manager: PipelineManager = PipelineManager(pipeline)
pipeline_manager.execute_all()

print("a")

