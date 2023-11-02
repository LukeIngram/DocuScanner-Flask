# errors.py 

class ModelError(Exception): 
    
    """TODO DOCSTRING"""

    def __init__(self, message="A model-related exception occurred") -> None:
        self.messsage = message
        super().__init__(message)



class FileError(Exception): 

    """TODO DOCSTRING"""

    def __init__(self, message="A file upload or download exception occured") -> None:
        self.messsage = message
        super().__init__(message)