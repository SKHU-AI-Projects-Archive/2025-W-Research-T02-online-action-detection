import logging
import os
import sys

def build_logger(config, log_filename='train.log'):
    logger = logging.getLogger('AI_Project')
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(config.log_level.upper())
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if hasattr(config, 'log_dir') and config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        log_file = os.path.join(config.log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger