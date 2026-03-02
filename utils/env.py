import os
import random
import numpy as np
import torch
import ast
from configparser import ConfigParser
from pathlib import Path
from datetime import datetime

class Configuration(object):
    def __init__(self, *file_names):
        parser = ConfigParser()
        parser.optionxform = str
        found = parser.read(file_names)
        if not found:
            raise ValueError(f'No config file found! Checked paths: {file_names}')
        for name in parser.sections():
            self.__dict__.update({item[0]: ast.literal_eval(item[1]) for item in parser.items(name)})

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def generate_log_dir(config):
    ROOT_DIR = Path(__file__).resolve().parent.parent
    dataset_name = getattr(config, 'dataset', 'UnknownDataset')
    model_name = getattr(config, 'model', 'UnknownModel')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_name = f"{dataset_name}_{model_name}_{timestamp}"
    
    base_work_dir = ROOT_DIR / config.work_dir
    log_dir = base_work_dir / exp_name
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)