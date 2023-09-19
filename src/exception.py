import sys
from src.log import logging

def error_msg(e,error:sys):
    _,_,exc = error.exc_info()
    filename = exc.tb_frame.f_code.co_filename
    Emsg = f"Error occurred in Python script name [{filename}] line number [{exc.tb_lineno}] error message [{str(e)}]"
    return Emsg

class CustomException (Exception):

    def __init__(self, Emsg, error:sys):
        super().__init__(Emsg)
        self.Emsg = error_msg(Emsg, error=error)

    def __str__(self):
        return self.Emsg