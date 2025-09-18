from abc import ABC

import textwrap
import dill
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from data_classes.pipeline_data.pipeline_data import PipelineData
from data_classes.plot_template.plot_template import PlotTemplate

class PlotComponent(PipelineComponent, ABC):
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
    
    #######################################
    ### To be implemented by subclasses ###
    #######################################
    PLOT_COMPONENT_CONSTANTS_CLASS = None
    PLOT_COMPONENT_UTILS_CLASS = None

    
    def perform_action(self) -> PipelineData:
        if not self.PLOT_COMPONENT_CONSTANTS_CLASS or not self.PLOT_COMPONENT_UTILS_CLASS:
            raise Exception("Plot components must specify PLOT_COMPONENT_UTILS_CLASS and PLOT_COMPONENT_CONSTANTS_CLASS.")

        template: PlotTemplate = self.PLOT_COMPONENT_UTILS_CLASS.plot_data(self._pipeline_data_, self._additional_execution_info_)

        if self.PLOT_COMPONENT_CONSTANTS_CLASS.SHOW_PLOT_IN_SEPARATE_PROCESS:
            self._start_sub_process_(template)
        else:
            template.fig.show()
            plt.show(block=True)

        self._pipeline_data_[0].plots = [template]
        return self._pipeline_data_[0]

    #######################
    ### Private methods ###
    #######################
    def _start_sub_process_(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as file:
            dill.dump((self._pipeline_data_, self._additional_execution_info_), file)
            data_file = file.name

        subprocess.Popen([sys.executable, "-c", self.SUB_PROCESS_CODE, "--child", data_file])
        # subprocess.Popen([sys.executable, __file__, "--child", data_file])