from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent

def generate_log_dir(config):
    dataset_name = getattr(config, 'dataset', 'UnknownDataset')
    model_name = getattr(config, 'model', 'UnknownModel')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_name = f"{dataset_name}_{model_name}_{timestamp}"
    
    base_work_dir = ROOT_DIR / config.work_dir
    log_dir = base_work_dir / exp_name
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)