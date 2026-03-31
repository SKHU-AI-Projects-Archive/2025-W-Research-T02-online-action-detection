import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, test_loader, criterion, device, config, logger, background_class_idx=0):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        self.background_class_idx = background_class_idx

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="[Evaluate]")
            for inputs, targets in pbar:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]  # (B, C)

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                scores = torch.softmax(outputs, dim=-1)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                pbar.set_postfix({'loss': test_loss / (pbar.n + 1)})

        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        num_classes = all_scores.shape[1]
        ap_per_class = []
        for c in range(num_classes):
            if c == self.background_class_idx:
                continue
            binary_labels = (all_labels == c).astype(int)
            if binary_labels.sum() == 0:
                continue
            ap = average_precision_score(binary_labels, all_scores[:, c])
            ap_per_class.append(ap)

        mAP = float(np.mean(ap_per_class)) if ap_per_class else 0.0
        avg_loss = test_loss / len(self.test_loader)

        self.logger.info(
            f"Evaluation Completed. Test Loss: {avg_loss:.4f} | mAP: {mAP:.4f} "
            f"(Evaluated: {len(ap_per_class)}/{num_classes - 1} classes)"
        )
        return avg_loss, {"mAP": mAP, "ap_per_class": ap_per_class}