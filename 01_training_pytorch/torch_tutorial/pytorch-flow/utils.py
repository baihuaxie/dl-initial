"""
    Contains utility classes / functions

    - class Params(): loading / saving parameters from / to a json file
    - set_logger: set the logger to log info in terminal and into 'log_path'

"""

import json
import logging
import torch

class Params():
    """
    Class definition for loading hyperparameters from a json file
    - json file needs to contain dict-like definitions for hyperparameters, e.g., {"learning_rate": 0.001}

    Example:

    params = Params(/path/to/json)
    print(params.learning_rate)

    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """ save parameters to json file """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        
    def update(self, json_path):
        """ update parameters from json file """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    @property
    def dict(self):
        """ return hyperparameters as a dictionary by Params.dict['hyperparameter name'] """
        return self.__dict__


def set_logger(log_path):
    """
    Set the logger to log info in terminal and into 'log_path'

    Save every output to the terminal into a permant file

    Args:
        log_path: (string) path to log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # log into a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # log into a console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
