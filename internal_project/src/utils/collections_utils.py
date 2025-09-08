from typing import Collection, Any

from exceptions.not_instantiable_error import NotInstantiableError


class CollectionsUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError("The class 'JaxUtils' can not be instantiated.")



    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def is_exact_element_in_collection(element: Any, collection: Collection) -> bool:
        for e in collection:
            if e is element:
                return True

        return False