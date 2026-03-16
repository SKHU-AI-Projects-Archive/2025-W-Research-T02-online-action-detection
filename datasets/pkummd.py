import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class PKUMMD(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = os.path.normpath(config.data_dir)
        self.num_frames = int(config.num_frames)
        self.num_nodes = int(config.num_nodes)
        self.num_classes = int(config.num_classes)
        self.in_channels = int(config.in_channels)  # 6 (xyz + velocity)
        self.data_path = os.path.join(self.data_dir, "Data")
        self.label_path = os.path.join(self.data_dir, "Label")

        split_path = os.path.join(self.data_dir, "Split", "cross-subject.txt")
        with open(split_path, "r") as f:
            content = f.read()

        train_part = content.split("Training videos:")[1].split("Validataion videos:")[0]
        val_part = content.split("Validataion videos:")[1]

        if split == "train":
            raw = train_part
        else:
            raw = val_part

        file_list = [
            f.strip() + ".txt"
            for f in raw.split(",")
            if f.strip()
        ]

        self.stride = self.num_frames // 2

        self._data = []
        self._labels = []
        self._lengths = []
        self._files = []

        for file_name in file_list:
            skeleton_file = os.path.join(self.data_path, file_name)
            label_file = os.path.join(self.label_path, file_name)

            if not os.path.exists(skeleton_file):
                continue

            data = np.loadtxt(skeleton_file).astype(np.float32)
            if data.ndim == 1:
                data = data[None, :]
            T = int(data.shape[0])
            data = data.reshape(T, 2, self.num_nodes, 3)  # raw xyz (C=3 고정)

            labels = self._load_labels(label_file, T)

            # skeleton은 두 사람 중 하나 선택
            person = random.randint(0, 1)
            clip_data = data[:, person]     # (T, N, 3)

            self._data.append(clip_data)
            self._labels.append(labels)
            self._lengths.append(T)
            self._files.append(file_name)

        # Build sliding windows with last-frame labeling
        action_windows = []
        bg_windows = []

        for file_id, T in enumerate(self._lengths):
            if T >= self.num_frames:
                for start in range(0, T - self.num_frames + 1, self.stride):
                    # last-frame labeling: 마지막 프레임의 라벨 기준
                    last_label = self._labels[file_id][start + self.num_frames - 1]
                    if last_label > 0:
                        action_windows.append((file_id, start))
                    else:
                        bg_windows.append((file_id, start))
            else:
                last_label = self._labels[file_id][-1]
                if last_label > 0:
                    action_windows.append((file_id, 0))
                else:
                    bg_windows.append((file_id, 0))

        self.windows = action_windows + bg_windows
        random.shuffle(self.windows)

        total_frames = sum(len(l) for l in self._labels)
        action_frames = sum((l > 0).sum() for l in self._labels)
        bg_frames = total_frames - action_frames
        print(f"[DEBUG] total: {total_frames} | action: {action_frames} ({action_frames/total_frames*100:.1f}%) | bg: {bg_frames} ({bg_frames/total_frames*100:.1f}%)")
        print(f"[DEBUG] PKUMMD split={split} → action_windows: {len(action_windows)} | bg_windows: {len(bg_windows)} | total: {len(self.windows)} | stride: {self.stride}")

    def __len__(self):
        return len(self.windows)

    def _load_labels(self, label_file, T):
        """
        label 파일 형식: cls, start, end, confidence
        4번째 컬럼은 confidence (1=약함, 2=강함)
        """
        labels = np.zeros(T, dtype=np.int64)

        if not os.path.exists(label_file):
            return labels

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                cls, start, end, confidence = map(int, parts)
                start_f = max(start - 1, 0)
                end_f = min(end, T)
                labels[start_f:end_f] = cls

        return labels

    @staticmethod
    def _normalize_skeleton(clip):
        """
        View-invariant skeleton normalization.

        입력: clip (T, N, 3) — numpy float32
        출력: clip (T, N, 3) — 정규화된 numpy float32

        1) SpineBase(joint 0) 원점 정규화
        2) 어깨 너비 스케일 정규화 (ShoulderLeft=5, ShoulderRight=9)
        """
        # 1) SpineBase 원점 정규화
        spine_base = clip[:, 0:1, :]        # (T, 1, 3)
        clip = clip - spine_base

        # 2) 어깨 너비 스케일 정규화
        shoulder_vec = clip[:, 5, :] - clip[:, 9, :]
        shoulder_dist = np.linalg.norm(shoulder_vec, axis=1)
        mean_dist = shoulder_dist.mean() + 1e-6
        clip = clip / mean_dist

        return clip

    @staticmethod
    def _compute_velocity(clip):
        """
        Compute per-joint backward velocity (causal).

        입력: clip (T, N, 3) — 정규화된 xyz
        출력: vel  (T, N, 3) — v_t = x_t - x_{t-1}, t=0 -> 0

        Backward difference only -> causal, no future leakage.
        """
        vel = np.zeros_like(clip)           # (T, N, 3), t=0 -> 0
        vel[1:] = clip[1:] - clip[:-1]     # v_t = x_t - x_{t-1}
        return vel

    def __getitem__(self, idx):
        file_id, start_idx = self.windows[idx]
        data = self._data[file_id]
        labels = self._labels[file_id]
        T = self._lengths[file_id]

        if T >= self.num_frames:
            clip = data[start_idx:start_idx + self.num_frames]         # (T, N, 3)
            clip_labels = labels[start_idx:start_idx + self.num_frames]
        else:
            pad_len = self.num_frames - T
            clip = np.concatenate(
                [data, np.zeros((pad_len, self.num_nodes, 3), dtype=np.float32)],
                axis=0
            )
            clip_labels = np.concatenate(
                [labels, np.zeros((pad_len,), dtype=np.int64)],
                axis=0
            )

        # 1) Skeleton 정규화 (xyz 기반, permute 전)
        clip = self._normalize_skeleton(clip)   # (T, N, 3)

        # 2) Velocity 계산 (정규화된 좌표 기반, causal)
        vel = self._compute_velocity(clip)      # (T, N, 3)

        # 3) xyz + velocity concat -> (T, N, 6)
        clip = np.concatenate([clip, vel], axis=-1)  # (T, N, 6)

        # (T, N, 6) → (T, 6, N)
        clip = torch.from_numpy(clip).permute(0, 2, 1).contiguous()
        clip_labels = torch.from_numpy(clip_labels).long()

        return clip, clip_labels