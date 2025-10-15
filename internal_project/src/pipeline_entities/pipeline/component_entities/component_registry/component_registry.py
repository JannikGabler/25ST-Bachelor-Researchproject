from exceptions.none_error import NoneError
from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.pipeline.component_entities.default_components.default_evaluation_components import __register_default_evaluation_components__
from pipeline_entities.pipeline.component_entities.default_components.default_input_components import __register_default_input_components__
from pipeline_entities.pipeline.component_entities.default_components.default_interpolation_cores import __register_default_interpolation_cores__
from pipeline_entities.pipeline.component_entities.default_components.default_node_generators import __register_default_node_generators__
from pipeline_entities.pipeline.component_entities.default_components.default_plotting_components import __register_default_plotting_components__
from pipeline_entities.pipeline.component_entities.default_components.default_test_components import __register_default_test_components__


class ComponentRegistry:


    ###########################
    ### Attributes of class ###
    ###########################
    __dictionary__: dict[str, PipelineComponentInfo] = {}


    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError("The class 'ComponentRegistry' can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def register_default_components() -> None:
        ComponentRegistry._register_default_input_components_()
        ComponentRegistry._register_default_evaluation_components_()
        ComponentRegistry._register_default_node_generators_()
        ComponentRegistry._register_default_interpolation_cores_()
        ComponentRegistry._register_default_test_components_()
        ComponentRegistry._register_default_ploting_components_()


    @staticmethod
    def register_component(component_info: PipelineComponentInfo) -> None:
        if component_info is None:
            raise NoneError("The parameter 'component_info' can not be None.")
        elif ComponentRegistry.is_component_id_registered(component_info.component_id):
            raise NoneError(f"A component with the ID '{component_info.component_id}' is already registered.")

        key: str = component_info.component_id.lower()
        ComponentRegistry.__dictionary__[key] = component_info


    @staticmethod
    def is_component_id_registered(component_id: str) -> bool:
        return component_id.lower() in ComponentRegistry.__dictionary__


    @staticmethod
    def get_component(component_id: str) -> PipelineComponentInfo | None:
        return ComponentRegistry.__dictionary__.get(component_id.lower())


    @staticmethod
    def get_all_components() -> list[PipelineComponentInfo]:
        return list(ComponentRegistry.__dictionary__.values())


    @staticmethod
    def clear() -> None:
        ComponentRegistry.__dictionary__.clear()


    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _register_default_input_components_():
        __register_default_input_components__.register()


    @staticmethod
    def _register_default_evaluation_components_():
        __register_default_evaluation_components__.register()


    @staticmethod
    def _register_default_node_generators_():
        __register_default_node_generators__.register()


    @staticmethod
    def _register_default_interpolation_cores_():
        __register_default_interpolation_cores__.register()


    @staticmethod
    def _register_default_test_components_():
        __register_default_test_components__.register()


    @staticmethod
    def _register_default_ploting_components_():
        __register_default_plotting_components__.register_plotting_components()
