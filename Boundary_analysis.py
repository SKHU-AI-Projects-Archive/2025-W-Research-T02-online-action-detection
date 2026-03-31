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
from utils.metrics import build_metrics
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

# Body part to joint index mapping
JOINT_COLORS = {
    'spine':    [0, 1, 2, 3],
    'left_arm': [5, 6, 7, 8, 21, 22],
    'right_arm':[9, 10, 11, 12, 23, 24],
    'left_leg': [13, 14, 15, 16],
    'right_leg':[17, 18, 19, 20],
    'head':     [4],
}
# Color per body part
COLOR_MAP = {
    'spine':    'royalblue',
    'left_arm': 'limegreen',
    'right_arm':'red',
    'left_leg': 'orange',
    'right_leg':'mediumpurple',
    'head':     'yellow',
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
    """
    Draw a single skeleton frame on a 3D axis.
    joints: (N, 3) - raw (x, y, z) coordinates
    """
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
            c=c,
            linewidth=1.5
        )

    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=0, azim=-90)

def save_skeleton_gif(joints_seq, gt_labels, save_path):
    """
    Save skeleton sequence as an animated GIF.
    joints_seq: (T, N, 3)
    gt_labels: (T,)
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Fix axis range across all frames
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
    """
    Save skeleton sequence as a static grid image (one subplot per frame).
    joints_seq: (T, N, 3)
    gt_labels: (T,)
    """
    T = len(joints_seq)
    cols = 8
    rows = (T + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 2, rows * 2.5))
    fig.suptitle('Skeleton (GT label per frame)', fontsize=10)

    # Fix axis range across all frames
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
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # (B, T)
            targets = targets.numpy()                            # (B, T)

            # (B, T, C, N) -> (B, T, N, C)
            skeleton = inputs.cpu().numpy().transpose(0, 1, 3, 2)

            for i in range(len(targets)):
                gt = targets[i]
                pred = preds[i]

                # Skip background-only windows
                if (gt > 0).sum() == 0:
                    continue

                T = len(gt)

                # ── 1. Prediction bar chart (GT vs Pred) ──────────
                fig1, axes = plt.subplots(2, 1, figsize=(14, 3))
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
                fig1.legend(handles=patches, loc='right', bbox_to_anchor=(1.12, 0.5))
                plt.tight_layout()
                pred_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_prediction.png')
                plt.savefig(pred_path, bbox_inches='tight', dpi=150)
                plt.close(fig1)

                # ── 2. Skeleton PNG (16-frame grid) ───────────────
                png_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_skeleton.png')
                save_skeleton_png(skeleton[i], gt, png_path)

                # ── 3. Skeleton GIF (animated) ────────────────────
                gif_path = os.path.join(save_dir, f'sample_{samples_saved + 1}_skeleton.gif')
                save_skeleton_gif(skeleton[i], gt, gif_path)

                samples_saved += 1
                if samples_saved >= num_samples:
                    return

def analyze_boundary_errors(model, test_loader, device, config, save_dir, window=5):
    """
    경계 구간 vs 내부 구간 accuracy 비교 분석
    → Transition-aware motivation 증명용
    """
    model.eval()

    all_gt = []
    all_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # (B, T)
            targets = targets.numpy()                            # (B, T)

            for i in range(len(targets)):
                all_gt.append(targets[i])
                all_pred.append(preds[i])

    # ── 경계 프레임 마킹 ──────────────────────────────────────────
    def get_boundary_mask(gt_seq, window=5):
        mask = np.zeros(len(gt_seq), dtype=bool)
        for f in range(1, len(gt_seq)):
            if gt_seq[f] != gt_seq[f - 1]:
                for w in range(-window, window + 1):
                    if 0 <= f + w < len(gt_seq):
                        mask[f + w] = True
        return mask

    boundary_correct = []
    boundary_total = []
    interior_correct = []
    interior_total = []

    for gt_seq, pred_seq in zip(all_gt, all_pred):
        b_mask = get_boundary_mask(gt_seq, window=window)
        i_mask = ~b_mask

        # background(0) 제외하고 계산
        action_mask = gt_seq > 0

        b = b_mask & action_mask
        i = i_mask & action_mask

        if b.sum() > 0:
            boundary_correct.append((pred_seq[b] == gt_seq[b]).sum())
            boundary_total.append(b.sum())

        if i.sum() > 0:
            interior_correct.append((pred_seq[i] == gt_seq[i]).sum())
            interior_total.append(i.sum())

    b_acc = sum(boundary_correct) / sum(boundary_total) if sum(boundary_total) > 0 else 0
    i_acc = sum(interior_correct) / sum(interior_total) if sum(interior_total) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  Boundary Analysis (window={window})")
    print(f"{'='*50}")
    print(f"  경계 구간 accuracy : {b_acc:.3f}  ({sum(boundary_total)} frames)")
    print(f"  내부 구간 accuracy : {i_acc:.3f}  ({sum(interior_total)} frames)")
    print(f"  GAP (interior - boundary) : {i_acc - b_acc:.3f}")
    print(f"{'='*50}\n")

    # ── 시각화 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ['Boundary frames', 'Interior frames'],
        [b_acc * 100, i_acc * 100],
        color=['#E8593C', '#3B8BD4'],
        width=0.4
    )
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Boundary vs Interior Accuracy (window={window})')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [b_acc * 100, i_acc * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boundary_vs_interior.png'), dpi=150)
    plt.close(fig)
    print(f"  → 그래프 저장: {save_dir}/boundary_vs_interior.png")

    return b_acc, i_acc


# ── main() 안 evaluator.evaluate() 아래에 이 줄 추가 ──────────────
# analyze_boundary_errors(model, test_loader, device, config, vis_dir, window=5)
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