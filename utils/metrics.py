import torch

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=-1)
    return (preds == targets).sum().item() / targets.numel()

def mean_class_accuracy(outputs, targets):
    """클래스별 accuracy 평균 - background 편향 제거"""
    _, preds = torch.max(outputs, dim=-1)
    num_classes = outputs.shape[-1]
    class_acc = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == targets[mask]).float().mean().item()
        class_acc.append(acc)
    return sum(class_acc) / len(class_acc)

def action_accuracy(outputs, targets):
    """background(0) 제외한 action 프레임만 accuracy"""
    _, preds = torch.max(outputs, dim=-1)
    mask = targets != 0  # background 제외
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()

def build_metrics(config):
    metric_fns = {
        'accuracy': accuracy,
        'mean_class_accuracy': mean_class_accuracy,
        'action_accuracy': action_accuracy,
    }
    return {m: metric_fns[m] for m in config.metrics}