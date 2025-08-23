import os
import sys

import dill

from utils.interpolants_plot_component_utils import InterpolantsPlotComponentUtils


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

        InterpolantsPlotComponentUtils.plot_data(pipeline_data, additional_execution_info)