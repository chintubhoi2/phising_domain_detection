import yaml
from src.exception import PhisingException
from src.logger import logging
import os,sys
import numpy as np
import dill

def read_yaml_file(file_path: str) ->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise PhisingException(e,sys) 