import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from datasets import build_dataloader
from models import build_model
from utils.logger import build_logger
from utils.losses import build_criterion
from utils.env import Configuration, set_seed
from utils.evaluator import Evaluator

KINECT_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (5, 6), (6, 7), (7, 8),
    (2, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (7, 21), (7, 22), (11, 23), (11, 24),
]

JOINT_COLORS = {
    'spine':     [0, 1, 2, 3],
    'left_arm':  [5, 6, 7, 8, 21, 22],
    'right_arm': [9, 10, 11, 12, 23, 24],
    'left_leg':  [13, 14, 15, 16],
    'right_leg': [17, 18, 19, 20],
    'head':      [4],
}
COLOR_MAP = {
    'spine':     'royalblue',
    'left_arm':  'limegreen',
    'right_arm': 'red',
    'left_leg':  'orange',
    'right_leg': 'mediumpurple',
    'head':      'yellow',
}

def get_joint_color(node_idx):
    for part, joints in JOINT_COLORS.items():
        if node_idx in joints:
            return COLOR_MAP[part]
    return 'gray'

def get_edge_color(src, dst):
    for part, joints in JOINT_COLORS.items():
        if src in joints and dst in joints:
            return COLOR_MAP[part]
    return 'gray'

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI Model")
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

def draw_skeleton_on_ax(ax, joints, title=""):
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]

    for node_idx in range(len(joints)):
        c = get_joint_color(node_idx)
        ax.scatter(x[node_idx], y[node_idx], z[node_idx], c=c, s=30, depthshade=False)

    for src, dst in KINECT_EDGES:
        c = get_edge_color(src, dst)
        ax.plot(
            [x[src], x[dst]],
            [y[src], y[dst]],
            [z[src], z[dst]],
            c=c, linewidth=1.5
        )

    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=0, azim=-90)

def save_skeleton_gif(joints_seq, gt_labels, save_path):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    all_x = joints_seq[:, :, 0].flatten()
    all_y = joints_seq[:, :, 1].flatten()
    all_z = joints_seq[:, :, 2].flatten()

    def update(t):
        ax.cla()
        draw_skeleton_on_ax(ax, joints_seq[t], title=f'Frame {t} | GT cls: {gt_labels[t]}')
        ax.set_xlim(all_x.min(), all_x.max())
        ax.set_ylim(all_y.min(), all_y.max())
        ax.set_zlim(all_z.min(), all_z.max())

    ani = animation.FuncAnimation(fig, update, frames=len(joints_seq), interval=200)
    ani.save(save_path, writer='pillow', fps=5)
    plt.close(fig)

def save_skeleton_png(joints_seq, gt_labels, save_path):
    T = len(joints_seq)
    cols = 8
    rows = (T + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 2, rows * 2.5))
    fig.suptitle('Skeleton (GT label per frame)', fontsize=10)

    all_x = joints_seq[:, :, 0].flatten()
    all_y = joints_seq[:, :, 1].flatten()
    all_z = joints_seq[:, :, 2].flatten()

    for t in range(T):
        ax = fig.add_subplot(rows, cols, t + 1, projection='3d')
        draw_skeleton_on_ax(ax, joints_seq[t], title=f'f{t} cls:{gt_labels[t]}')
        ax.set_xlim(all_x.min(), all_x.max())
        ax.set_ylim(all_y.min(), all_y.max())
        ax.set_zlim(all_z.min(), all_z.max())

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def visualize_predictions(model, test_loader, device, config, save_dir, num_samples=5):
    model.eval()
    samples_saved = 0

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / config.num_classes) for i in range(config.num_classes)]

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if outputs.dim() == 3:
                last_outputs = outputs[:, -1, :]  # (B, C)
            else:
                last_outputs = outputs

            preds   = torch.argmax(last_outputs, dim=-1).cpu().numpy()  # (B,)
            targets = targets.cpu().numpy()                              # (B,)

            # (B, T, C, N) -> (B, T, N, C)
            skeleton = inputs.cpu().numpy().transpose(0, 1, 3, 2)

            for i in range(len(targets)):
                gt_label   = int(targets[i])
                pred_label = int(preds[i])

                if gt_label == 0:
                    continue

                # ── 1. Prediction bar ──
                fig1, ax = plt.subplots(figsize=(6, 2))
                ax.barh(0, 1, color=colors[gt_label],   label=f'GT: {gt_label}')
                ax.barh(1, 1, color=colors[pred_label], label=f'Pred: {pred_label}')
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['GT', 'Pred'])
                ax.set_title(f'Sample {samples_saved + 1} | GT={gt_label} Pred={pred_label}')
                ax.legend(loc='right')
                plt.tight_layout()
                pred_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_prediction.png')
                plt.savefig(pred_path, bbox_inches='tight', dpi=150)
                plt.close(fig1)

                # ── 2. Skeleton PNG ──
                png_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_skeleton.png')
                save_skeleton_png(skeleton[i], [gt_label] * skeleton[i].shape[0], png_path)

                # ── 3. Skeleton GIF ──
                gif_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_skeleton.gif')
                save_skeleton_gif(skeleton[i], [gt_label] * skeleton[i].shape[0], gif_path)

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

    logger.info("Building Model and Loss...")
    model     = build_model(config).to(device)
    criterion = build_criterion(config)

    weight_path = os.path.join(config.log_dir, 'best_model.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))

    logger.info("Initializing Evaluator...")
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
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