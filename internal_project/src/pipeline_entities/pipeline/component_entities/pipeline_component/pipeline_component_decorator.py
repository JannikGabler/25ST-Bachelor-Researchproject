from exceptions.none_error import NoneError
from pipeline_entities.pipeline.component_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.pipeline.component_entities.component_meta_info.dataclasses.component_meta_info import ComponentMetaInfo
from pipeline_entities.pipeline.component_entities.pipeline_component.pipeline_component import PipelineComponent
from pipeline_entities.pipeline.component_entities.component_registry.component_registry import ComponentRegistry


def pipeline_component(id: str, type: type, meta_info: ComponentMetaInfo):
    """
    Class decorator to register a pipeline component.

    Args:
        id (str): Component identifier.
        type (type): Component type.
        meta_info (ComponentMetaInfo): Metadata describing the component.

    Returns:
        type[PipelineComponent]: The decorated class with attached PipelineComponentInfo.

    Raises:
        TypeError: If the decorated class is not a subclass of `PipelineComponent`.
        NoneError: If any of the arguments (`id`, `type`, `meta_info`) is None.
    """

    _registered_classes_ = set()

    def decorator(cls: type):
        if not issubclass(cls, PipelineComponent):
            raise TypeError("The decorated class must be a subclass of 'PipelineComponent'.")

        if str is None:
            raise NoneError("The argument 'str' cannot be None.")
        if type is None:
            raise NoneError("The argument 'type' cannot be None.")
        if meta_info is None:
            raise NoneError("The argument 'meta_info' cannot be None.")

        component_info: PipelineComponentInfo = PipelineComponentInfo(id, type, cls, meta_info)

        cls._info_ = component_info
        ComponentRegistry.register_component(component_info)

        return cls

    return decorator
