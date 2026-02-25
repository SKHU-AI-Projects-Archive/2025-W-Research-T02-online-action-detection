from .gatv2 import GATv2

def build_model(config):
    models = {
        'GATv2': GATv2
    }
    if config.model not in models:
        raise ValueError(f"Unsupported model: {config.model}. Supported models are: {list(models.keys())}")
    
    return models[config.model](config)