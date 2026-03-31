from .gatv2 import GATv2
from .gatv2_node_tsm import GATv2NodeTSM
from models.gatv2_baseline import GATv2Baseline
from .model_hypergraph import HypergraphModel  

def build_model(config):
    models = {
        'GATv2': GATv2,
        'GATv2NodeTSM': GATv2NodeTSM,
        'GATv2Baseline': GATv2Baseline,
        'HypergraphModel': HypergraphModel, 
    }
    if config.model not in models:
        raise ValueError(f"Unsupported model: {config.model}. Supported models are: {list(models.keys())}")
    
    return models[config.model](config)