class NoneError(Exception):
    """
    Exception raised when an unexpected None value is encountered.
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
