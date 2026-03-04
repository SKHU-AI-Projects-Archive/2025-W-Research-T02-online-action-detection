import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from setup import Configuration
from datasets import build_dataloader
from models import build_model
from utils.seed import set_seed
from utils.logger import build_logger
from utils.losses import build_criterion
from utils.metrics import build_metrics
from utils.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI Model")
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

def visualize_predictions(model, test_loader, device, config, save_dir, num_samples=5):
    model.eval()
    samples_saved = 0

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / config.num_classes) for i in range(config.num_classes)]

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # (B, T)
            targets = targets.numpy()                            # (B, T)

            for i in range(len(targets)):
                gt = targets[i]
                pred = preds[i]

                # action 프레임이 있는 샘플만 시각화 -> 다 back만 시각화해서 
                if (gt > 0).sum() == 0:
                    continue

                T = len(gt)
                fig, axes = plt.subplots(2, 1, figsize=(14, 3))

                for t in range(T):
                    axes[0].barh(0, 1, left=t, color=colors[gt[t]], edgecolor='none')
                    axes[1].barh(0, 1, left=t, color=colors[pred[t]], edgecolor='none')

                axes[0].set_xlim(0, T)
                axes[1].set_xlim(0, T)
                axes[0].set_yticks([0]); axes[0].set_yticklabels(['GT'])
                axes[1].set_yticks([0]); axes[1].set_yticklabels(['Pred'])
                axes[0].set_title(f'Sample {samples_saved + 1} - Ground Truth vs Prediction')
                axes[1].set_xlabel('Frame')

                unique_classes = np.unique(np.concatenate([gt, pred]))
                patches = [mpatches.Patch(color=colors[c], label=f'Class {c}') for c in unique_classes]
                fig.legend(handles=patches, loc='right', bbox_to_anchor=(1.12, 0.5))

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'sample_{samples_saved + 1}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()

                samples_saved += 1
                if samples_saved >= num_samples:
                    return

def main():
    args = parse_args()
    config = Configuration(args.config)
    set_seed(config.seed)
    config.log_dir = config.exp_dir
    logger = build_logger(config, log_filename='eval.log')
    logger.info(f"Evaluation started! experiment directory: {config.exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Building DataLoaders...")
    test_loader = build_dataloader(config, split='test')

    logger.info("Building Model, Loss, and Metrics...")
    model = build_model(config).to(device)
    criterion = build_criterion(config)
    metrics = build_metrics(config)

    weight_path = os.path.join(config.log_dir, 'best_model.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))

    logger.info("Initializing Evaluator...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        metrics=metrics,
        device=device,
        config=config,
        logger=logger
    )

    logger.info("Starting evaluation...")
    evaluator.evaluate()

    logger.info("Saving prediction visualizations...")
    vis_dir = os.path.join(config.log_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, test_loader, device, config, vis_dir, num_samples=5)
    logger.info(f"Visualizations saved to: {vis_dir}")

if __name__ == '__main__':
    main()