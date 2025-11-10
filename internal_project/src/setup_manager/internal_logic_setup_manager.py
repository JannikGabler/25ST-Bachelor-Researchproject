from exceptions.not_instantiable_error import NotInstantiableError
from pipeline_entities.pipeline.component_entities.component_registry.component_registry import ComponentRegistry


class InternalLogicSetupManager:
    """
    Utility helpers for initializing internal logic. This class is not meant to be instantiated.
    """


    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """
        raise NotInstantiableError("The class 'InternalLogicSetupManager' can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def setup():
        """
        Register default pipeline components for internal logic initialization.
        """

        ComponentRegistry.register_default_components()