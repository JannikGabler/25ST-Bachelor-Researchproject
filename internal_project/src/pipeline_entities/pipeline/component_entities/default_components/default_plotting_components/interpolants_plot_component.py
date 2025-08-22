import textwrap

import dill
import subprocess
import sys
import tempfile

from constants.internal_logic_constants import InterpolantsPlotComponentConstants
from pipeline_entities.pipeline.component_entities.component_meta_info.defaults.plot_components.interpolants_plot_component_meta_info import \
    interpolants_plot_component_meta_info
from pipeline_entities.pipeline.component_entities.default_component_types.interpolation_core import InterpolationCore

from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component_decorator import pipeline_component
from pipeline_entities.large_data_classes.pipeline_data.pipeline_data import PipelineData
from utils.interpolants_plot_component_utils import InterpolantsPlotComponentUtils


@pipeline_component(id="interpolant plotter", type=InterpolationCore, meta_info=interpolants_plot_component_meta_info)
class InterpolantsPlotComponent(InterpolationCore):
    SUB_PROCESS_CODE = textwrap.dedent("""
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
        
                InterpolantsPlotComponentUtils.plot_data(pipeline_data, additional_execution_info)""")



    ######################
    ### Public methods ###
    ######################
    def perform_action(self) -> PipelineData:
        if InterpolantsPlotComponentConstants.SHOW_PLOT_IN_SEPARATE_PROCESS:
            self._start_sub_process_()
        else:
            InterpolantsPlotComponentUtils.plot_data(self._pipeline_data_, self._additional_execution_info_)

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







    # def create_sub_process_(self, data: InterpolantsPlotData) -> None:
    #     # Required for some platforms
    #     try:
    #         multiprocessing.set_start_method("spawn", force=True)
    #     except RuntimeError:
    #         pass  # Start method has already been set.
    #
    #     plot_operation = InterpolantsPlotComponentUtils.plot
    #     p = multiprocessing.Process(target=plot_operation,
    #                                 args=data)
    #     p.start()







        # # TODO (multiple function callables)
        # interval: jnp.ndarray = self._pipeline_data_[0].interpolation_interval
        # original_function: callable = self._pipeline_data_[0].function_callable
        # x_array: jnp.ndarray = InterpolantsPlotComponentUtils.create_x_array(interval)
        # original_function_values: jnp.ndarray = original_function(x_array)
        # original_function_values = PlotUtils.replace_nan_with_inf(original_function_values)
        #
        # y_limits: tuple[jnp.ndarray, jnp.ndarray] = InterpolantsPlotComponentUtils.calc_limits(original_function_values)
        #
        # #fig, ax = plt.subplots(figsize=(10, 6))
        # InterpolantsPlotComponentUtils.init_plot()
        #
        # InterpolantsPlotComponentUtils.plot_interpolation_points(self._pipeline_data_[0].interpolation_nodes, self._pipeline_data_[0].interpolation_values)
        # InterpolantsPlotComponentUtils.plot_original_function(x_array, original_function_values, y_limits)
        # InterpolantsPlotComponentUtils.plot_interpolants(self._pipeline_data_, x_array, y_limits)
        #
        # InterpolantsPlotComponentUtils.finish_up_plot(self._pipeline_data_, self._additional_execution_info_)
        #
        # return self._pipeline_data_[0]







    # def perform_action(self) -> PipelineData:
    #     all_data = self._pipeline_data_
    #     reference_data = next((d for d in all_data if d.function_callable is not None), all_data[0])
    #
    #     nodes = reference_data.interpolation_nodes
    #     interval = reference_data.interpolation_interval
    #     f = reference_data.function_callable
    #
    #     x_plot = jnp.linspace(interval[0], interval[1], 500)
    #     y_true = f(x_plot) if f else None
    #
    #     plt.figure(figsize=(10, 6))
    #
    #     if y_true is not None:
    #         plt.plot(x_plot, y_true, '--', label="Original Function", linewidth=2.5, color='black', zorder=1)
    #
    #     if nodes is not None and f is not None:
    #         plt.scatter(nodes, f(nodes), color='red', s=50, label="Interpolation Nodes", zorder=10)
    #
    #     colors = ['blue', 'green', 'orange', 'purple', 'brown']
    #
    #     for i, data in enumerate(all_data):
    #         interpolant = data.interpolant
    #         if interpolant is None:
    #             continue
    #
    #         compiled_interpolant: CompiledInterpolant = interpolant.compile(500, data.data_type)
    #
    #         x_eval = jnp.linspace(interval[0], interval[1], 500).astype(compiled_interpolant.used_data_type)
    #         x_eval = x_eval.reshape(compiled_interpolant.required_evaluation_points_shape)
    #
    #         raw_name = type(interpolant).__name__.replace("Interpolant", "")
    #         name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)
    #
    #         try:
    #             y_interp = compiled_interpolant.evaluate(x_eval)
    #         except Exception:
    #             print(f"[SKIP] {name}: Evaluation failed (exception).")
    #             continue
    #
    #         is_finite = jnp.isfinite(y_interp)
    #         if not jnp.any(is_finite):
    #             print(f"[SKIP] {name}: All values are NaN or Inf.")
    #             continue
    #
    #         x_plot_filtered = x_eval[is_finite]
    #         y_plot_filtered = y_interp[is_finite]
    #
    #         plt.plot(
    #             x_plot_filtered,
    #             y_plot_filtered,
    #             label=f"{name} Interpolant",
    #             color=colors[i % len(colors)],
    #             linestyle='-',
    #             linewidth=1.8
    #         )
    #
    #     if y_true is not None:
    #         max_value = float(jnp.max(y_true))
    #         min_value = float(jnp.min(y_true))
    #         difference = 7 * (max_value - min_value)
    #         plt.ylim(min_value - difference, max_value + difference)
    #     else:
    #         plt.ylim(-1, 1)
    #
    #     plt.title(f"Original Function and Interpolation Curves on [{interval[0]}, {interval[1]}]")
    #     plt.xlabel("x")
    #     plt.ylabel("f(x)")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
    #
    #     return reference_data