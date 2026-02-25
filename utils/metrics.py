import torch

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=-1)
    return (preds == targets).sum().item() / targets.numel()

def build_metrics(config):
    metric_fns = {
        'accuracy': accuracy
    }
    
    return {m: metric_fns[m] for m in config.metrics}