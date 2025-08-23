import os
import pickle
import subprocess
import sys
import tempfile

from constants.internal_logic_constants import InterpolantsPlotComponentConstants
from pipeline_entities.large_data_classes.plotting_data.interpolants_plot_component_data import \
    InterpolantsPlotComponentData
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    interpolants_plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from utils.interpolants_plot_component_utils import InterpolantsPlotComponentUtils


@pipeline_component(id="absolute error plotter", type=InterpolationCore, meta_info=interpolants_plot_component_meta_info)
class AbsoluteErrorPlotComponent(InterpolationCore):


    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        plot_data: InterpolantsPlotComponentData = InterpolantsPlotComponentData.create_from(self._pipeline_data_, self._additional_execution_info_)

        if InterpolantsPlotComponentConstants.SHOW_PLOT_IN_SEPARATE_PROCESS:
            self.start_sub_process(plot_data)
        else:
            InterpolantsPlotComponentUtils.plot_data(plot_data)

        return self._pipeline_data_[0]



    #######################
    ### Private methods ###
    #######################
    def _start_sub_process_(self, plot_data: InterpolantsPlotComponentData) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as file:
            pickle.dump(plot_data, file)
            data_file = file.name

        # Eigenständigen Python-Prozess starten, der völlig unabhängig läuft
        subprocess.Popen([sys.executable, __file__, "--child", data_file])



if __name__ == "__main__":
    # Check if we run in the child process
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        data_file = sys.argv[2]

        with open(data_file, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, InterpolantsPlotComponentData):
            raise TypeError(f"Unpickled data is not of type '{InterpolantsPlotComponentData.__name__}'.")

        try:
            os.remove(data_file)
        except FileNotFoundError:
            pass

        InterpolantsPlotComponentUtils.plot_data(data)
