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
                inputs  = inputs.to(self.device)   # (B, T, C, N)
                targets = targets.to(self.device)  # (B, T)

                outputs = self.model(inputs)        # (B, T, num_classes)

                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]),  # (B*T, C)
                    targets.reshape(-1)                      # (B*T,)
                )
                test_loss += loss.item()

                # Last frame only for evaluation (SSNet protocol)
                last_scores  = torch.softmax(outputs[:, -1, :], dim=-1)  # (B, C)
                last_targets = targets[:, -1]                             # (B,)

                all_scores.append(last_scores.cpu().numpy())
                all_labels.append(last_targets.cpu().numpy())
                pbar.set_postfix({'loss': test_loss / (pbar.n + 1)})

        all_scores = np.concatenate(all_scores, axis=0)  # (N_total, C)
        all_labels = np.concatenate(all_labels, axis=0)  # (N_total,)

        num_classes = all_scores.shape[1]

        # mAP (background excluded)
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

        # Frame-level accuracy (background included, SSNet protocol)
        all_preds = np.argmax(all_scores, axis=1)
        accuracy  = float((all_preds == all_labels).mean())

        avg_loss = test_loss / len(self.test_loader)

        self.logger.info(
            f"Evaluation Completed. Test Loss: {avg_loss:.4f} | "
            f"mAP: {mAP:.4f} | Accuracy: {accuracy:.4f} "
            f"(Evaluated: {len(ap_per_class)}/{num_classes - 1} classes)"
        )
        return avg_loss, {"mAP": mAP, "accuracy": accuracy, "ap_per_class": ap_per_class}