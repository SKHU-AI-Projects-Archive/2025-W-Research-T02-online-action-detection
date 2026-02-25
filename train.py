import argparse
import torch

from setup import Configuration
from datasets import build_dataloader
from models import build_model
from utils.paths import generate_log_dir
from utils.seed import set_seed
from utils.logger import build_logger
from utils.losses import build_criterion
from utils.optim import build_optimizer, build_scheduler
from utils.metrics import build_metrics
from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Configuration('configs/base.ini', args.config)
    set_seed(config.seed)

    config.log_dir = generate_log_dir(config)
    logger = build_logger(config, log_filename='train.log')
    logger.info(f"Training started! Logging directory: {config.log_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Building DataLoaders...")
    train_loader = build_dataloader(config, split='train')
    val_loader   = build_dataloader(config, split='val')
    
    logger.info("Building Model, Loss, Metrics, Optimizer, and Scheduler...")
    model     = build_model(config).to(device)
    criterion = build_criterion(config)
    metrics   = build_metrics(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger
    )
    
    logger.info("Starting training loop...")
    trainer.fit()

if __name__ == '__main__':
    main()