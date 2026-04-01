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

        if self.split == "train":
            self.stride = max(1, self.num_frames // 8)
        elif self.split == "val":
            self.stride = 8
        else:  # test
            self.stride = 1

        self._data = []
        self._labels = []
        self._lengths = []
        self._files = []

        for file_name in file_list:
            skeleton_file = os.path.join(self.data_path, file_name)
            data = np.loadtxt(skeleton_file).astype(np.float32)
            T = int(data.shape[0])
            data = data.reshape(T, self.num_nodes, 3)

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
                        end   = max(0, min(end, T - 1))
                        if end >= start:
                            labels[start:end + 1] = cls

            self._data.append(data)
            self._labels.append(labels)
            self._lengths.append(T)
            self._files.append(file_name)

        self.windows = []
        for file_id, T in enumerate(self._lengths):
            if T >= self.num_frames:
                for start in range(0, T - self.num_frames + 1, self.stride):
                    self.windows.append((file_id, start))
            else:
                self.windows.append((file_id, 0))

        if self.split == "train":
            random.shuffle(self.windows)

        print(f"[DEBUG] OAD split={split} | Windows: {len(self.windows)} | Stride: {self.stride}")

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _normalize_skeleton(clip):
        clip = clip - clip[:, 0:1, :]
        shoulder_dist = np.linalg.norm(clip[:, 5, :] - clip[:, 9, :], axis=1).mean() + 1e-6
        return clip / shoulder_dist

    @staticmethod
    def _compute_velocity(clip):
        vel = np.zeros_like(clip)
        vel[1:] = clip[1:] - clip[:-1]
        return vel

    def __getitem__(self, idx):
        file_id, start_idx = self.windows[idx]
        data   = self._data[file_id]
        labels = self._labels[file_id]
        T      = self._lengths[file_id]

        if T >= self.num_frames:
            clip        = data[start_idx:start_idx + self.num_frames]
            clip_labels = labels[start_idx:start_idx + self.num_frames]
        else:
            pad_len = self.num_frames - T
            clip        = np.pad(data,   ((0, pad_len), (0, 0), (0, 0)), mode='constant')
            clip_labels = np.pad(labels, (0, pad_len),                   mode='constant')

        clip = self._normalize_skeleton(clip)
        vel  = self._compute_velocity(clip)
        clip = np.concatenate([clip, vel], axis=-1)           # (T, N, 6)
        clip = torch.from_numpy(clip).permute(0, 2, 1).contiguous()  # (T, 6, N)

        # Return full frame labels (T,)
        clip_labels = torch.from_numpy(clip_labels).long()

        return clip, clip_labels