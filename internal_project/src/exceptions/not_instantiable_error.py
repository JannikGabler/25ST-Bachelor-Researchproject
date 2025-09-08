
class NotInstantiableError(Exception):

    ###################
    ### Constructor ###
    ###################
    def __init__(self, message: str) -> None:
        super().__init__(message)