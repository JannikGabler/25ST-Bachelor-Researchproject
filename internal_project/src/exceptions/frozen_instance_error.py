class FrozenInstanceError(Exception):
    """
    Exception raised when an attempt is made to modify an immutable (frozen) instance.
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
