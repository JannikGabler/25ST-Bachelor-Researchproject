from abc import ABC

from exceptions.frozen_instance_error import FrozenInstanceError


class Freezable(ABC):
    """
    Abstract base class for objects that can be frozen to prevent modifications.
    """


    ###############################
    ### Attributes of instances ###
    ###############################
    _frozen_: bool = False


    #########################
    ### Getters & setters ###
    #########################
    def freeze(self):
        """
        Freeze the instance to prevent further modifications.
        """
        self._frozen_ = True


    #######################
    ### Private methods ###
    #######################
    def __setattr__(self, name: str, value: any) -> None:
        if self._frozen_:
            raise FrozenInstanceError(f"Cannot modify attribute {repr(name)}.")
        else:
            super().__setattr__(name, value)
