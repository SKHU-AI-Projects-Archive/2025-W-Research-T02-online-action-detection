import os
import argparse
import torch

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

if __name__ == '__main__':
    main()