import torch.nn as nn

def build_criterion(config):
    try:
        criterion_class = getattr(nn, config.loss_fn)
        kwargs = getattr(config, 'loss_kwargs', {})
        return criterion_class(**kwargs)
    except AttributeError:
        raise ValueError(f"Loss function '{config.loss_fn}' not found in torch.nn")
