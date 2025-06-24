from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.components.dynamic_management.component_registry import ComponentRegistry


class InternalLogicSetupManager:
    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError("The class 'InternalLogicSetupManager' can not be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def setup():
        ComponentRegistry.register_default_components()