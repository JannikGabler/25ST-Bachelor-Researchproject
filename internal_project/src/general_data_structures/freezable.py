from abc import ABC, abstractmethod

from exceptions.frozen_instance_error import FrozenInstanceError


class Freezable(ABC):
    ###############################
    ### Attributes of instances ###
    ###############################
    _frozen_: bool = False



    #########################
    ### Getters & setters ###
    #########################
    def freeze(self):
        self._frozen_ = True



    #######################
    ### Private methods ###
    #######################
    def __setattr__(self, name: str, value: any) -> None:
        if self._frozen_:
            raise FrozenInstanceError(f"Cannot modify attribute '{name}'.")
        else:
            super().__setattr__(name, value)
