class DirectoryNotFoundError(Exception):
    """
    Exception raised when a specified directory cannot be found.
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
