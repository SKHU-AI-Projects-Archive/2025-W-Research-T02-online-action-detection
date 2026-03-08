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
        self.in_channels = int(config.in_channels)
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
            data = data.reshape(T, 2, self.num_nodes, self.in_channels)

            # label은 영상 전체에 대한 것 (사람 구분 없음)
            labels = self._load_labels(label_file, T)

            # skeleton은 두 사람 중 하나 선택
            person = random.randint(0, 1)
            clip_data = data[:, person]

            self._data.append(clip_data)
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
        4번째 컬럼은 confidence (1=약함, 2=강함)이며 사람 번호가 아님
        모든 label을 영상 전체에 적용
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