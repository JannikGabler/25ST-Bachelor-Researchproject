import textwrap

import dill
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt

from constants.internal_logic_constants import AbsoluteRoundOffErrorPlotComponentConstants
from data_classes.plot_template.plot_template import PlotTemplate
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.absolute_round_off_error_plot_component import \
    absolute_round_off_error_plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore
from pipeline_entities.pipeline.component_entities.default_component_types.plot_component import PlotComponent

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import \
    pipeline_component
from data_classes.pipeline_data.pipeline_data import PipelineData
from utils.absolute_round_off_error_plot_component_utils import AbsoluteRoundOffErrorPlotComponentUtils


@pipeline_component(id="absolute round off error plotter", type=PlotComponent, meta_info=absolute_round_off_error_plot_component_meta_info)
class AbsoluteRoundOffErrorPlotComponent(InterpolationCore):
    SUB_PROCESS_CODE = textwrap.dedent("""
        import os
        import sys
        import dill
        import matplotlib.pyplot as plt


        if __name__ == "__main__":
            # Check if we run in the child process
            if len(sys.argv) == 3 and sys.argv[1] == "--child":
                data_file = sys.argv[2]

                with open(data_file, "rb") as f:
                    template = dill.load(f)

                try:
                    os.remove(data_file)
                except FileNotFoundError:
                    pass

                template.fig.show()
                plt.show(block=True)""")



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        template: PlotTemplate = AbsoluteRoundOffErrorPlotComponentUtils.plot_data(self._pipeline_data_, self._additional_execution_info_)

        if AbsoluteRoundOffErrorPlotComponentConstants.SHOW_PLOT_IN_SEPARATE_PROCESS:
            self._start_sub_process_(template)
        else:
            template.fig.show()
            plt.show(block=True)

        self._pipeline_data_[0].plots = [template]
        return self._pipeline_data_[0]



    #######################
    ### Private methods ###
    #######################
    def _start_sub_process_(self, template: PlotTemplate) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as file:
            dill.dump(template, file)
            data_file = file.name

        subprocess.Popen([sys.executable, "-c", self.SUB_PROCESS_CODE, "--child", data_file])






