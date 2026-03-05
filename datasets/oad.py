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
        self.balanced_sampling = getattr(config, 'balanced_sampling', 'false').lower() == 'true'

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

        # Build sliding windows
        action_windows = []
        bg_windows = []

        for file_id, T in enumerate(self._lengths):
            if T >= self.num_frames:
                for start in range(0, T - self.num_frames + 1, self.stride):
                    clip_labels = self._labels[file_id][start:start + self.num_frames]
                    if (clip_labels > 0).any():
                        action_windows.append((file_id, start))
                    else:
                        bg_windows.append((file_id, start))
            else:
                clip_labels = self._labels[file_id]
                if (clip_labels > 0).any():
                    action_windows.append((file_id, 0))
                else:
                    bg_windows.append((file_id, 0))

        if self.balanced_sampling:
            # 1:1 balanced sampling: sample bg windows to match action count
            n_action = len(action_windows)
            sampled_bg = random.sample(bg_windows, min(n_action, len(bg_windows)))
            self.windows = action_windows + sampled_bg
        else:
            # Use all windows (original distribution)
            self.windows = action_windows + bg_windows

        random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_id, start_idx = self.windows[idx]
        data = self._data[file_id]
        labels = self._labels[file_id]
        T = self._lengths[file_id]

        if T >= self.num_frames:
            clip = data[start_idx:start_idx + self.num_frames]
            clip_labels = labels[start_idx:start_idx + self.num_frames]
        else:
            pad_len = self.num_frames - T
            clip = np.concatenate(
                [data, np.zeros((pad_len, self.num_nodes, self.in_channels), dtype=np.float32)],
                axis=0
            )
            clip_labels = np.concatenate(
                [labels, np.zeros((pad_len,), dtype=np.int64)],
                axis=0
            )

        clip = torch.from_numpy(clip).permute(0, 2, 1).contiguous()
        clip_labels = torch.from_numpy(clip_labels).long()

        return clip, clip_labels