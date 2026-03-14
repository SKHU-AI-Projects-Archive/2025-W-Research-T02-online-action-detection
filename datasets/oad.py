import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class OAD(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = os.path.normpath(config.data_dir)
        self.num_frames = int(config.num_frames)
        self.num_nodes = int(config.num_nodes)
        self.in_channels = int(config.in_channels)
        self.data_path = os.path.join(self.data_dir, "Data")
        self.label_path = os.path.join(self.data_dir, "Label")

        split_file = os.path.join(self.data_dir, "Split", "Split.txt")
        target = "training" if split == "train" else "testing"

        file_list = []
        with open(split_file, "r") as f:
            lines = [l.strip() for l in f.readlines()]
        for i, line in enumerate(lines):
            if line.lower().strip().rstrip(":") == target:
                indices = list(map(int, lines[i + 1].replace(" ", "").split(",")))
                file_list = [f"{idx}.txt" for idx in indices]
                break

        self.stride = self.num_frames // 2

        self._data = []
        self._labels = []
        self._lengths = []
        self._files = []

        for file_name in file_list:
            skeleton_file = os.path.join(self.data_path, file_name)
            data = np.loadtxt(skeleton_file).astype(np.float32)
            T = int(data.shape[0])
            data = data.reshape(T, self.num_nodes, self.in_channels)

            labels = np.zeros(T, dtype=np.int64)
            label_file = os.path.join(self.label_path, file_name)
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        cls, start, end = map(int, ln.split(","))
                        start = max(0, min(start, T - 1))
                        end = max(0, min(end, T - 1))
                        if end >= start:
                            labels[start:end + 1] = cls

            self._data.append(data)
            self._labels.append(labels)
            self._lengths.append(T)
            self._files.append(file_name)

        # Build sliding windows with majority voting
        action_windows = []
        bg_windows = []

        for file_id, T in enumerate(self._lengths):
            if T >= self.num_frames:
                for start in range(0, T - self.num_frames + 1, self.stride):
                    clip_labels = self._labels[file_id][start:start + self.num_frames]
                    # majority voting: 절반 이상이 action이면 action window
                    if (clip_labels > 0).sum() > len(clip_labels) // 2:
                        action_windows.append((file_id, start))
                    else:
                        bg_windows.append((file_id, start))
            else:
                clip_labels = self._labels[file_id]
                if (clip_labels > 0).sum() > len(clip_labels) // 2:
                    action_windows.append((file_id, 0))
                else:
                    bg_windows.append((file_id, 0))

        self.windows = action_windows + bg_windows
        random.shuffle(self.windows)

        print(f"[DEBUG] OAD split={split} → action_windows: {len(action_windows)} | bg_windows: {len(bg_windows)} | total: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _normalize_skeleton(clip):
        """
        View-invariant skeleton normalization.

        입력: clip (T, N, C) — numpy float32
        출력: clip (T, N, C) — 정규화된 numpy float32

        1) SpineBase(joint 0) 원점 정규화
           모든 프레임, 모든 관절에서 SpineBase 좌표를 빼서
           카메라/피실험자 위치 변화에 무관하게 만듦

        2) 어깨 너비 스케일 정규화
           ShoulderLeft(joint 5) ↔ ShoulderRight(joint 9) 거리의
           프레임 평균으로 나눠서 신체 크기 차이를 제거
           (1e-6 안전값으로 zero-division 방지)
        """
        # 1) SpineBase 원점 정규화
        spine_base = clip[:, 0:1, :]        # (T, 1, C)
        clip = clip - spine_base            # (T, N, C)

        # 2) 어깨 너비 스케일 정규화
        # ShoulderLeft=5, ShoulderRight=9
        shoulder_vec = clip[:, 5, :] - clip[:, 9, :]       # (T, C)
        shoulder_dist = np.linalg.norm(shoulder_vec, axis=1)  # (T,)
        mean_dist = shoulder_dist.mean() + 1e-6             # scalar
        clip = clip / mean_dist                             # (T, N, C)

        return clip

    def __getitem__(self, idx):
        file_id, start_idx = self.windows[idx]
        data = self._data[file_id]
        labels = self._labels[file_id]
        T = self._lengths[file_id]

        if T >= self.num_frames:
            clip = data[start_idx:start_idx + self.num_frames]         # (T, N, C)
            clip_labels = labels[start_idx:start_idx + self.num_frames]
        else:
            pad_len = self.num_frames - T
            clip = np.concatenate(
                [data, np.zeros((pad_len, self.num_nodes, self.in_channels), dtype=np.float32)],
                axis=0
            )                                                           # (T, N, C)
            clip_labels = np.concatenate(
                [labels, np.zeros((pad_len,), dtype=np.int64)],
                axis=0
            )

        # ── Skeleton 정규화 (permute 전 numpy 단계에서 수행) ──────────────
        # clip shape: (T, N, C) → 정규화 → 동일 shape 유지
        clip = self._normalize_skeleton(clip)

        # (T, N, C) → (T, C, N)
        clip = torch.from_numpy(clip).permute(0, 2, 1).contiguous()
        clip_labels = torch.from_numpy(clip_labels).long()

        return clip, clip_labels