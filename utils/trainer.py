import os
import copy
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_weights = copy.deepcopy(model.state_dict())

    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.total_epoch} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]  # (B, C)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        return train_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.total_epoch} [Valid]")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]  # (B, C)

                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                scores = torch.softmax(outputs, dim=-1)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(targets.cpu().numpy())
                pbar.set_postfix({'val_loss': val_loss / (pbar.n + 1)})

        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        num_classes = all_scores.shape[1]
        ap_per_class = []
        for c in range(num_classes):
            if c == 0:
                continue
            binary_labels = (all_labels == c).astype(int)
            if binary_labels.sum() == 0:
                continue
            ap = average_precision_score(binary_labels, all_scores[:, c])
            ap_per_class.append(ap)

        mAP = float(np.mean(ap_per_class)) if ap_per_class else 0.0
        avg_loss = val_loss / len(self.val_loader)

        return avg_loss, {"mAP": mAP}

    def fit(self):
        for epoch in range(1, self.config.total_epoch + 1):
            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)

            log_str = f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            for name, val in val_metrics.items():
                log_str += f" | Val {name.capitalize()}: {val:.4f}"
            self.logger.info(log_str)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_weights, os.path.join(self.config.log_dir, 'best_model.pth'))
                self.logger.info("  [*] Best model saved.")
            else:
                self.early_stop_counter += 1
                if getattr(self.config, 'early_stopping', True) and self.early_stop_counter >= getattr(self.config, 'patience', 30):
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        self.model.load_state_dict(self.best_weights)
        self.logger.info(f"Training completed. Best Validation Loss: {self.best_val_loss:.4f}")