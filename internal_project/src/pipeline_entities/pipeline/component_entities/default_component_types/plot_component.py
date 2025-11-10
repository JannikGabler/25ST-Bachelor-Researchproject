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
    """
    Abstract base class for pipeline plot components that render a PlotTemplate and display it.
    """


    SUB_PROCESS_CODE = textwrap.dedent(
        """
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
                plt.show(block=True)"""
    )


    #######################################
    ### To be implemented by subclasses ###
    #######################################
    PLOT_COMPONENT_CONSTANTS_CLASS = None
    PLOT_COMPONENT_UTILS_CLASS = None



    def perform_action(self) -> PipelineData:
        """
        Render the plot via the configured utils, display it (optionally in a subprocess), attach the PlotTemplate to the pipeline data, and return it.

        Returns:
            PipelineData: The pipeline data with the created plot attached.
        """

        if self.PLOT_COMPONENT_CONSTANTS_CLASS is None or self.PLOT_COMPONENT_UTILS_CLASS is None:
            raise Exception("Plot components must specify PLOT_COMPONENT_UTILS_CLASS and PLOT_COMPONENT_CONSTANTS_CLASS.")

        template: PlotTemplate = self.PLOT_COMPONENT_UTILS_CLASS.plot_data(self._pipeline_data_, self._additional_execution_info_)
        if template is None:
            raise Exception(f"PlotTemplate of {self.__class__.__name__} is None!\n(object: {self.__repr__()})")

        if self.PLOT_COMPONENT_CONSTANTS_CLASS.SHOW_PLOT:
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
    def _start_sub_process_(self, template: PlotTemplate) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as file:
            dill.dump(template, file)
            data_file = file.name

        subprocess.Popen([sys.executable, "-c", self.SUB_PROCESS_CODE, "--child", data_file])
