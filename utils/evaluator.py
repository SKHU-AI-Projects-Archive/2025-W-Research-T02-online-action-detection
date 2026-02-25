import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, test_loader, criterion, metrics, device, config, logger):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.config = config
        self.logger = logger

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        metric_results = {name: 0.0 for name in self.metrics.keys()}
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="[Evaluate]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                batch_size = targets.size(0)
                total_samples += batch_size
                
                for name, metric_fn in self.metrics.items():
                    metric_results[name] += metric_fn(outputs, targets) * batch_size
                    
                pbar.set_postfix({'loss': test_loss / (pbar.n + 1)})
                
        avg_loss = test_loss / len(self.test_loader)
        final_metrics = {name: val / total_samples for name, val in metric_results.items()}
        
        log_str = f"Evaluation Completed. Test Loss: {avg_loss:.4f}"
        for name, val in final_metrics.items():
            log_str += f" | {name.capitalize()}: {val:.4f}"
            
        self.logger.info(log_str)
        
        return avg_loss, final_metrics