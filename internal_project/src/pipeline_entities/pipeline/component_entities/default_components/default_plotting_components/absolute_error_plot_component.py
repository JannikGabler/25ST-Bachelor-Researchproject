import textwrap

import dill
import subprocess
import sys
import tempfile

from constants.internal_logic_constants import AbsoluteErrorPlotComponentConstants
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.absolute_error_plot_component_meta_info import \
    absolut_error_plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from utils.absolute_error_plot_component_utils import AbsoluteErrorPlotComponentUtils


@pipeline_component(id="absolute error plotter", type=InterpolationCore, meta_info=absolut_error_plot_component_meta_info)
class AbsoluteErrorPlotComponent(InterpolationCore):
    SUB_PROCESS_CODE = textwrap.dedent("""
        import os
        import sys
        import dill

        from utils.absolute_error_plot_component_utils import AbsoluteErrorPlotComponentUtils


        if __name__ == "__main__":
            # Check if we run in the child process
            if len(sys.argv) == 3 and sys.argv[1] == "--child":
                data_file = sys.argv[2]

                with open(data_file, "rb") as f:
                    pipeline_data, additional_execution_info = dill.load(f)

                try:
                    os.remove(data_file)
                except FileNotFoundError:
                    pass

                AbsoluteErrorPlotComponentUtils.plot_data(pipeline_data, additional_execution_info)""")



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        if AbsoluteErrorPlotComponentConstants.SHOW_PLOT_IN_SEPARATE_PROCESS:
            self._start_sub_process_()
        else:
            AbsoluteErrorPlotComponentUtils.plot_data(self._pipeline_data_, self._additional_execution_info_)

        return self._pipeline_data_[0]



    #######################
    ### Private methods ###
    #######################
    def _start_sub_process_(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as file:
            dill.dump((self._pipeline_data_, self._additional_execution_info_), file)
            data_file = file.name

        subprocess.Popen([sys.executable, "-c", self.SUB_PROCESS_CODE, "--child", data_file])

