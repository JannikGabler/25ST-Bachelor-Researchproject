from abc import ABC, abstractmethod

from exceptions.frozen_instance_error import FrozenInstanceError


class Freezable(ABC):
    ###############################
    ### Attributes of instances ###
    ###############################
    __frozen__: bool = False



    #########################
    ### Getters & setters ###
    #########################
    def freeze(self):
        self.__frozen__ = True



    #######################
    ### Private methods ###
    #######################
    def __setattr__(self, name: str, value: any) -> None:
        if self.__frozen__:
            raise FrozenInstanceError(f"Cannot modify attribute '{name}'.")
        else:
            super().__setattr__(name, value)
