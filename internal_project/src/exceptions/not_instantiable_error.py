class NotInstantiableError(Exception):
    """
    Exception raised when an attempt is made to instantiate a class or object that is not instantiable.
    """

    ###################
    ### Constructor ###
    ###################
    def __init__(self, message: str) -> None:
        """
        Args:
            message (str): Description of the error.
        """

        super().__init__(message)
