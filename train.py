import argparse
from logging import config
import torch
import shutil
import os

from datasets import build_dataloader
from models import build_model
from utils.logger import build_logger
from utils.losses import build_criterion
from utils.optim import build_optimizer, build_scheduler
from utils.metrics import build_metrics
from utils.env import Configuration, set_seed, generate_log_dir
from utils.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Configuration(args.config)
    set_seed(config.seed)

    config.log_dir = generate_log_dir(config)
    logger = build_logger(config, log_filename='train.log')
    logger.info(f"Training started! Logging directory: {config.log_dir}")

    backup_path = os.path.join(config.log_dir, 'config_backup.ini')
    shutil.copy(args.config, backup_path)
    logger.info(f"Config backup saved to: {backup_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Building DataLoaders...")
    train_loader = build_dataloader(config, split='train')
    val_loader   = build_dataloader(config, split='val')
    
    logger.info("Building Model, Loss, Optimizer, and Scheduler...")
    model     = build_model(config).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
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