
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def build_optimizer(model, config):
    try:
        optimizer_class = getattr(optim, config.optimizer)
        kwargs = getattr(config, 'optimizer_kwargs', {})
        return optimizer_class(model.parameters(), lr=config.learning_rate, **kwargs)
    except AttributeError:
        raise ValueError(f"Optimizer '{config.optimizer}' not found in torch.optim")

def build_scheduler(optimizer, config):
    if not hasattr(config, 'scheduler') or config.scheduler in ['None', None, '']:
        return None
        
    try:
        scheduler_class = getattr(lr_scheduler, config.scheduler)
        kwargs = getattr(config, 'scheduler_kwargs', {})
        return scheduler_class(optimizer, **kwargs)
    except AttributeError:
        raise ValueError(f"Scheduler '{config.scheduler}' not found in torch.optim.lr_scheduler")