from dataclasses import fields

from data_structures.tree.tree_node import TreeNode
from pipeline_entities.component_info.dataclasses.pipeline_component_info import PipelineComponentInfo
from pipeline_entities.constraints.abstracts.static_constraint import StaticConstraint
from pipeline_entities.data_transfer.pipeline_data import PipelineData
from pipeline_entities.pipeline_configuration.dataclasses.pipeline_configuration import PipelineConfiguration


class AttributeRequiredConstraint(StaticConstraint):
    ##############################
    ### Attributs of instances ###
    ##############################
    __attribute_name__: str



    ###################
    ### Constructor ###
    ###################
    def __init__(self, attribute_name: str):
        if not any(field.name == attribute_name for field in fields(PipelineData)):
            raise AttributeError(f"The class 'PipelineData' has no attribute with the name '{attribute_name}'")

        self.__attribute_name__ = attribute_name



    ##########################
    ### Overridden methods ###
    ##########################
    def evaluate(self, own_tree_node: TreeNode[PipelineComponentInfo], pipeline_configuration: PipelineConfiguration) -> bool:
        current_node: TreeNode[PipelineComponentInfo] = own_tree_node.parent_node

        while current_node:
            component_info: PipelineComponentInfo = current_node.value
            attributes_modifying: set[str] = component_info.component_meta_info.attributes_modifying

            if self.__attribute_name__ in attributes_modifying:
                return True

            current_node = current_node.parent_node

        return False



    def __eq__(self, other):
        if not isinstance(other, AttributeRequiredConstraint):   # Covers None
            return False
        else:
            return self.__attribute_name__ == other.__attribute_name__



    def __hash__(self):
        return hash(self.__attribute_name__)



    def __repr__(self) -> str:
        return f"AttributRequiredConstraint<attribute_name='{self.__attribute_name__}'>"





