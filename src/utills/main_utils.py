import yaml
from src.exception import PhisingException
from src.logger import logging
import os,sys
import numpy as np
import dill
import numpy as np

def read_yaml_file(file_path: str) ->dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise PhisingException(e,sys)

def write_yaml_file(file_path: str,content: object, replace: bool = False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise PhisingException(e,sys)


def save_numpy_array(file_path: str, array:np.array):
    """ 
    save numpy array to desired location
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb")as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise PhisingException(e,sys)
    
def load_numpy_array(file_path:str)->np.array:
    """ 
    function for loading numpy array from desired location
    """
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise PhisingException(e,sys)

def save_object(file_path:str, obj:object):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise PhisingException(e,sys)
    
def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"{file_path} doesn't exist")
        with  open(file_path,"rb")as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise PhisingException(e,sys)
