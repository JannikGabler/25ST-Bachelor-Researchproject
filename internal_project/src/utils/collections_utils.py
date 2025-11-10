from typing import Collection, Any

from exceptions.not_instantiable_error import NotInstantiableError


class CollectionsUtils:
    """
    Utility class for collection-related helper functions. This class cannot be instantiated.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError("The class 'JaxUtils' can not be instantiated.")

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def is_exact_element_in_collection(element: Any, collection: Collection) -> bool:
        """
        Check whether the given element is contained in the collection.

        Args:
            element (Any): The element to search for.
            collection (Collection): The collection to search in.

        Returns:
            bool: True if the element is found by identity, False otherwise.
        """

        for e in collection:
            if e is element:
                return True

        return False
