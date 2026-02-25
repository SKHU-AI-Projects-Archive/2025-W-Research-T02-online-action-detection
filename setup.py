import os
import ast
from configparser import ConfigParser

class Configuration(object):
    def __init__(self, *file_names):
        parser = ConfigParser()
        parser.optionxform = str
        found = parser.read(file_names)
        if not found:
            raise ValueError(f'No config file found! Checked paths: {file_names}')
        for name in parser.sections():
            self.__dict__.update({item[0]: ast.literal_eval(item[1]) for item in parser.items(name)})