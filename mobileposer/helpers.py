import os


def getenv(key:str, default=0): 
    return type(default)(os.getenv(key, default))